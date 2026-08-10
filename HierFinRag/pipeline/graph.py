"""
pipeline/graph.py
=================
Table-Text Graph Neural Network (TTGNN) for the HierFinRAG pipeline.

This module provides two components:

GraphBuilder
    Converts the flat list of hierarchical chunks produced by parser.py into a
    heterogeneous PyTorch Geometric graph with three edge types:

    ┌──────────────────────────────────────────────────────────────────┐
    │  Node types:  Section · Paragraph · Table · Cell · Image         │
    │  Edge types:                                                       │
    │    Structural — explicit parent ↔ child links from children_ids   │
    │    Semantic   — cosine similarity above a configurable threshold   │
    │    Cross-ref  — keyword overlap between paragraph text and table   │
    │                 headers (≥2 shared tokens)                         │
    └──────────────────────────────────────────────────────────────────┘

TTGNN
    A full PyTorch Geometric GATv2-based Graph Attention Network that enriches
    each node's raw BGE-M3 embedding with structural and relational context from
    its graph neighbourhood.  Architecture matches the HierFinRAG reference
    implementation (hierfinrag/graph/ttgnn.py).

enrich_embeddings_with_gnn()
    Convenience wrapper that runs a TTGNN forward pass on a built graph and
    returns a chunk_id → enriched-embedding mapping ready for Qdrant upsert.
"""

import re
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GATv2Conv


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

NODE_TYPES = {"section": 0, "text": 1, "table": 2, "cell": 3, "image": 4}
EDGE_TYPES = {
    "semantic":   0,  # cosine similarity above threshold
    "structural": 1,  # explicit parent ↔ child hierarchy
    "cross_ref":  2,  # paragraph ↔ table keyword overlap
}

SEMANTIC_THRESHOLD = 0.75  # Minimum cosine similarity for a semantic edge


# ---------------------------------------------------------------------------
# GraphBuilder
# ---------------------------------------------------------------------------

class GraphBuilder:
    """Build a PyTorch Geometric graph from hierarchical document chunks.

    Args:
        embeddings    : Mapping of chunk_id → numpy embedding vector.
                        Produced by the BGE-M3 model in embedder.py.
        sim_threshold : Minimum cosine similarity to add a semantic edge.
                        Default: 0.75.

    Example::

        builder = GraphBuilder(embeddings_dict)
        graph   = builder.build(chunks)
    """

    def __init__(
        self,
        embeddings:    Dict[str, np.ndarray],
        sim_threshold: float = SEMANTIC_THRESHOLD,
    ):
        self.embeddings    = embeddings
        self.sim_threshold = sim_threshold

    def build(self, chunks: List[dict]) -> Data:
        """Build a PyG Data object from a list of document chunks.

        Args:
            chunks : Flat list of chunk dicts from parser.py.

        Returns:
            ``torch_geometric.data.Data`` with fields:

            x          — node feature matrix [N, D]
            edge_index — COO edge list [2, E]
            edge_attr  — edge type indices [E]
            node_types — node type indices [N]
            chunk_ids  — list of chunk_id strings (length N)
        """
        # Only include chunks whose embeddings are available
        valid_chunks = [c for c in chunks if c["chunk_id"] in self.embeddings]
        id_to_idx: Dict[str, int] = {
            c["chunk_id"]: i for i, c in enumerate(valid_chunks)
        }

        # Build node feature matrix and type index
        node_features = [
            torch.tensor(self.embeddings[c["chunk_id"]], dtype=torch.float32)
            for c in valid_chunks
        ]
        node_type_list = [NODE_TYPES.get(c["type"], 1) for c in valid_chunks]

        x                = torch.stack(node_features)
        node_types_tensor = torch.tensor(node_type_list, dtype=torch.long)

        src_list, dst_list, attr_list = [], [], []

        def add_edge(u_id: str, v_id: str, etype: int) -> None:
            """Add a bidirectional edge between two nodes if both exist."""
            if u_id in id_to_idx and v_id in id_to_idx:
                u, v = id_to_idx[u_id], id_to_idx[v_id]
                src_list.extend([u, v])
                dst_list.extend([v, u])
                attr_list.extend([etype, etype])

        # Structural edges — explicit parent ↔ child links
        for chunk in valid_chunks:
            for child_id in chunk.get("children_ids", []):
                add_edge(chunk["chunk_id"], child_id, EDGE_TYPES["structural"])

        # Semantic edges — cosine similarity above threshold
        # O(N²) — only computed for small documents to remain tractable
        if len(valid_chunks) <= 500:
            emb_matrix = x.numpy()
            norms      = np.linalg.norm(emb_matrix, axis=1, keepdims=True) + 1e-9
            normed     = emb_matrix / norms
            sim_matrix = normed @ normed.T

            for i in range(len(valid_chunks)):
                for j in range(i + 1, len(valid_chunks)):
                    if sim_matrix[i, j] >= self.sim_threshold:
                        src_list.extend([i, j])
                        dst_list.extend([j, i])
                        attr_list.extend([EDGE_TYPES["semantic"]] * 2)

        # Cross-reference edges — paragraph text mentions table headers
        table_chunks = [c for c in valid_chunks if c["type"] == "table"]
        text_chunks  = [c for c in valid_chunks if c["type"] == "text"]

        for table in table_chunks:
            header_tokens: set = set()
            for h in table.get("col_headers", []) + table.get("row_headers", []):
                header_tokens.update(re.findall(r"\b\w{4,}\b", h.lower()))

            for para in text_chunks:
                overlap = sum(1 for tok in header_tokens if tok in para["content"].lower())
                if overlap >= 2:
                    add_edge(para["chunk_id"], table["chunk_id"], EDGE_TYPES["cross_ref"])

        if src_list:
            edge_index = torch.tensor([src_list, dst_list], dtype=torch.long)
            edge_attr  = torch.tensor(attr_list,            dtype=torch.long)
        else:
            edge_index = torch.empty((2, 0), dtype=torch.long)
            edge_attr  = torch.empty((0,),   dtype=torch.long)

        return Data(
            x          = x,
            edge_index = edge_index,
            edge_attr  = edge_attr,
            node_types = node_types_tensor,
            chunk_ids  = [c["chunk_id"] for c in valid_chunks],
        )


# ---------------------------------------------------------------------------
# TTGNN — Table-Text Graph Neural Network
# ---------------------------------------------------------------------------

class TTGNN(nn.Module):
    """GATv2-based graph attention network for context-enriched embeddings.

    Architecture:
        1. Linear projection  : input_dim → hidden_dim
        2. Node-type fusion   : adds a learnable node-type embedding
        3. GATv2 layers       : N stacked layers with edge-type embeddings,
                                residual connections, and dropout
        4. Output projection  : hidden_dim → hidden_dim

    Args:
        input_dim  : Dimensionality of input node features (1024 for BGE-M3).
        hidden_dim : Width of all hidden layers.  Must be divisible by num_heads.
        num_layers : Number of GATv2 message-passing layers.
        num_heads  : Number of attention heads per layer.
        dropout    : Dropout rate applied after each GATv2 layer.
    """

    def __init__(
        self,
        input_dim:  int   = 1024,
        hidden_dim: int   = 512,
        num_layers: int   = 3,
        num_heads:  int   = 8,
        dropout:    float = 0.1,
    ):
        super().__init__()
        assert hidden_dim % num_heads == 0, (
            f"hidden_dim ({hidden_dim}) must be divisible by num_heads ({num_heads})."
        )
        self.hidden_dim = hidden_dim

        self.input_proj    = nn.Linear(input_dim, hidden_dim)
        self.node_type_emb = nn.Embedding(5, hidden_dim)   # 5 node types
        self.edge_type_emb = nn.Embedding(3, hidden_dim)   # 3 edge types

        head_dim = hidden_dim // num_heads
        self.gnn_layers = nn.ModuleList([
            GATv2Conv(
                in_channels    = hidden_dim,
                out_channels   = head_dim,
                heads          = num_heads,
                edge_dim       = hidden_dim,
                add_self_loops = True,
                concat         = True,   # output size = heads × head_dim = hidden_dim
            )
            for _ in range(num_layers)
        ])

        self.output_proj = nn.Linear(hidden_dim, hidden_dim)
        self.dropout     = nn.Dropout(dropout)

    def forward(
        self,
        x:          torch.Tensor,  # [N, input_dim]
        edge_index: torch.Tensor,  # [2, E]
        edge_attr:  torch.Tensor,  # [E]   edge type indices
        node_types: torch.Tensor,  # [N]   node type indices
    ) -> torch.Tensor:
        """Forward pass through all GATv2 layers with residual connections.

        Returns:
            h : Context-enriched node embeddings of shape [N, hidden_dim].
        """
        h        = self.input_proj(x) + self.node_type_emb(node_types)
        edge_emb = self.edge_type_emb(edge_attr)

        for layer in self.gnn_layers:
            h_prev = h
            h      = layer(h, edge_index, edge_attr=edge_emb)
            h      = F.relu(h)
            h      = self.dropout(h)
            h      = h + h_prev   # residual connection

        return self.output_proj(h)


# ---------------------------------------------------------------------------
# Convenience wrapper
# ---------------------------------------------------------------------------

def enrich_embeddings_with_gnn(
    graph:      Data,
    input_dim:  int           = 1024,
    hidden_dim: int           = 512,
    device:     Optional[str] = None,
) -> Dict[str, np.ndarray]:
    """Run a TTGNN forward pass and return context-enriched embeddings.

    Produces graph-context-aware embeddings that encode both semantic content
    and document-structure signals (hierarchy, co-occurrence, cross-references).
    These enriched embeddings can replace the raw BGE-M3 vectors in Qdrant for
    improved retrieval precision.

    Args:
        graph      : PyG Data object built by GraphBuilder.build().
        input_dim  : Must match the embedding model's output dimensionality.
        hidden_dim : TTGNN hidden dimension.
        device     : ``'cuda'``, ``'cpu'``, or None (auto-detect).

    Returns:
        Dict mapping each chunk_id to its [hidden_dim] enriched embedding
        as a numpy array.
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    model = TTGNN(input_dim=input_dim, hidden_dim=hidden_dim).to(device)
    model.eval()
    graph = graph.to(device)

    with torch.no_grad():
        enriched = model(graph.x, graph.edge_index, graph.edge_attr, graph.node_types)

    enriched_np = enriched.cpu().numpy()
    return {
        chunk_id: enriched_np[i]
        for i, chunk_id in enumerate(graph.chunk_ids)
    }
