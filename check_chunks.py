# save as check_chunks.py
import json

with open("data/extracted/chunks.json", encoding="utf-8") as f:
    chunks = json.load(f)

from collections import Counter
types = Counter(c["type"] for c in chunks)

print(f"Total chunks: {len(chunks)}")
print(f"  Text   : {types.get('text', 0)}")
print(f"  Tables : {types.get('table', 0)}")
print(f"  Images : {types.get('image', 0)}")

print("\n--- Sample text chunk ---")
text_sample = next(c for c in chunks if c["type"] == "text")
print(f"Page {text_sample['page']}: {text_sample['content'][:200]}")

print("\n--- Sample table chunk ---")
table_sample = next(c for c in chunks if c["type"] == "table")
print(f"Page {table_sample['page']}:\n{table_sample['content'][:300]}")

print("\n--- Sample image chunk ---")
img_sample = next(c for c in chunks if c["type"] == "image")
print(f"Page {img_sample['page']}: {img_sample['image_path']}")