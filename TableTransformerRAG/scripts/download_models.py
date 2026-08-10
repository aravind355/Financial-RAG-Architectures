"""
scripts/download_models.py
==========================
Pre-download TATR models from HuggingFace so parsing runs fully offline.

Fixes the 'dilation: null' strict validation error in transformers 5.x /
huggingface_hub 3.x by downloading to a local directory and patching the
config.json before loading.

Usage
-----
    python scripts/download_models.py
"""

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from huggingface_hub import snapshot_download
from transformers import AutoImageProcessor, TableTransformerForObjectDetection
import config


MODELS_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "data", "models",
)


def _download_and_patch(model_id: str, local_name: str) -> str:
    """Download model to local dir, patch config.json, return local path."""
    local_dir = os.path.join(MODELS_DIR, local_name)

    print(f"  Downloading {model_id} → {local_dir}")
    snapshot_download(
        repo_id   = model_id,
        local_dir = local_dir,
    )

    # Patch config.json in the local directory
    config_path = os.path.join(local_dir, "config.json")
    if os.path.exists(config_path):
        with open(config_path, "r", encoding="utf-8") as f:
            cfg = json.load(f)

        changed = False

        # Fix 1: dilation at the TOP LEVEL of the config (TableTransformerConfig field)
        if "dilation" in cfg and cfg["dilation"] is None:
            cfg["dilation"] = False
            changed = True
            print(f"  [patch] top-level dilation: null → false")

        # Fix 2: dilation inside backbone_config (nested ResNet config)
        bb = cfg.get("backbone_config")
        if isinstance(bb, dict) and bb.get("dilation") is None:
            bb["dilation"] = False
            cfg["backbone_config"] = bb
            changed = True
            print(f"  [patch] backbone_config.dilation: null → false")

        if changed:
            with open(config_path, "w", encoding="utf-8") as f:
                json.dump(cfg, f, indent=2)

    return local_dir


def download() -> None:
    os.makedirs(MODELS_DIR, exist_ok=True)

    # ── Detection model ──────────────────────────────────────────────────────
    print("1. TATR detection model:")
    det_dir = _download_and_patch(config.TATR_DETECTION_MODEL, "detection")
    print("  Loading model to verify...")
    AutoImageProcessor.from_pretrained(det_dir)
    TableTransformerForObjectDetection.from_pretrained(det_dir)
    print(f"  OK: {config.TATR_DETECTION_MODEL}\n")

    # ── Structure model ──────────────────────────────────────────────────────
    print("2. TATR structure model:")
    str_dir = _download_and_patch(config.TATR_STRUCTURE_MODEL, "structure")
    print("  Loading model to verify...")
    AutoImageProcessor.from_pretrained(str_dir)
    TableTransformerForObjectDetection.from_pretrained(str_dir)
    print(f"  OK: {config.TATR_STRUCTURE_MODEL}\n")

    print("Both models downloaded, patched, and verified.")
    print(f"Models stored in: {MODELS_DIR}")


if __name__ == "__main__":
    download()
