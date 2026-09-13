"""Smoke test for detect.py: run cpsam and sam_vit_b on two real frames, save overlays."""
import os
import time
import numpy as np
import SimpleITK as sitk
from PIL import Image, ImageDraw
from detect import detect, LAST_TIME_S

os.makedirs("results", exist_ok=True)

IMAGES = [
    "/home/visilant/CLEAR-EC/data/train_mha/0323-23.mha",
    "/home/visilant/CLEAR-EC/data/train_mha/0344-23.mha",
]


def load_u8(path):
    img = sitk.GetArrayFromImage(sitk.ReadImage(path))
    img = np.squeeze(img)
    return img.astype(np.uint8)


def save_overlay(img, pts, out_path, title):
    rgb = Image.fromarray(img).convert("RGB")
    draw = ImageDraw.Draw(rgb)
    r = 3
    for x, y in pts:
        draw.ellipse([x - r, y - r, x + r, y + r], outline=(255, 0, 0), width=1)
    draw.text((10, 10), title, fill=(255, 255, 0))
    rgb.save(out_path)


for model_name in ["cpsam", "sam_vit_b"]:
    print(f"=== {model_name} ===")
    for path in IMAGES:
        img = load_u8(path)
        t0 = time.time()
        pts = detect(model_name, img)
        dt = time.time() - t0
        base = os.path.splitext(os.path.basename(path))[0]
        out_path = f"results/smoke_{model_name}_{base}.png"
        save_overlay(img, pts, out_path, f"{model_name} {base}: N={len(pts)}")
        print(f"{path}: N={len(pts)} time={dt:.2f}s (LAST_TIME_S={LAST_TIME_S.get(model_name):.2f}s) -> {out_path}")
