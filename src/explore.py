"""Explore the extracted Stanford fetal dataset structure."""
import os
import sys

import pandas as pd

root = sys.argv[1] if len(sys.argv) > 1 else os.path.expanduser("~/fetal_ga/data/fetal-brain")

entries = sorted(os.listdir(root))
print("top-level entries:", len(entries))
print(entries[:10], "...", entries[-5:] if len(entries) > 10 else "")

# find labels file
for f in entries:
    if "label" in f.lower() or f.endswith((".xlsx", ".xls", ".csv")):
        print("\n=== labels file:", f, "===")
        df = pd.read_excel(os.path.join(root, f)) if f.endswith((".xlsx", ".xls")) else pd.read_csv(os.path.join(root, f))
        print("shape:", df.shape)
        print("columns:", list(df.columns))
        print(df.head(8).to_string())
        break

# explore one case folder
case_dirs = [e for e in entries if os.path.isdir(os.path.join(root, e))]
if case_dirs:
    c = case_dirs[0]
    print("\n=== case:", c, "===")
    for sub in sorted(os.listdir(os.path.join(root, c))):
        p = os.path.join(root, c, sub)
        if os.path.isdir(p):
            files = sorted(os.listdir(p))
            print(f"  {sub}/  ({len(files)} files) e.g.", files[:5])
        else:
            print(f"  {sub}  ({os.path.getsize(p)} bytes)")

# image size of one sample
from PIL import Image
for dirpath, _, fnames in os.walk(root):
    jpgs = [f for f in fnames if f.lower().endswith((".jpg", ".jpeg", ".png"))]
    if jpgs:
        img = Image.open(os.path.join(dirpath, jpgs[len(jpgs) // 2]))
        print("\nsample image:", os.path.join(dirpath, jpgs[len(jpgs) // 2]))
        print("size:", img.size, "mode:", img.mode)
        break
