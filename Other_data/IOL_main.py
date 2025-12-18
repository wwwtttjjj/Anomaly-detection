import random
import json
from pathlib import Path
from PIL import Image
import uuid
import numpy as np

# ===== 参数范围 =====
MIN_GRID = 5
MAX_GRID = 9

MIN_CELL_SIZE = 60
MAX_CELL_SIZE = 80

MIN_GAP = 10
MAX_GAP = 50

MIN_ODD = 1
MAX_ODD = 5

MIN_MARGIN = 30
MAX_MARGIN = 50

BG_COLOR = (255, 255, 255)

def add_gaussian_noise_pil(pil_img, sigma=0.02):
    """
    pil_img: PIL.Image (RGB or L), uint8 [0,255]
    sigma: noise std in [0,1] space
    return: PIL.Image, uint8 [0,255]
    """
    img = np.asarray(pil_img).astype(np.float32) / 255.0
    noise = np.random.normal(0, sigma, img.shape).astype(np.float32)
    out = img + noise
    out = np.clip(out, 0.0, 1.0)
    out = (out * 255.0).astype(np.uint8)
    return Image.fromarray(out)

def load_digit_pool(png_root: Path):
    pool = {}
    for d in range(10):
        ddir = png_root / str(d)
        imgs = sorted(ddir.glob("*.png"))
        if not imgs:
            raise RuntimeError(f"No png in {ddir}")
        pool[d] = imgs
    return pool


def generate_single_iol(digit_pool: dict):
    digit = random.choice(list(digit_pool.keys()))
    paths = digit_pool[digit]
    noise_sigma = random.uniform(0.03, 0.05)   # 你可以调，比如 0–0.05

    if len(paths) < 2:
        raise RuntimeError(f"Digit {digit} must have >=2 images")

    # ===== 随机 grid 形状（不一定正方）=====
    rows = random.randint(MIN_GRID, MAX_GRID)
    cols = random.randint(MIN_GRID, MAX_GRID)
    num_cells = rows * cols

    odd_k = random.randint(MIN_ODD, min(MAX_ODD, num_cells))

    base_path = random.choice(paths)

    all_indices = list(range(num_cells))
    odd_indices = set(random.sample(all_indices, odd_k))

    candidates = [p for p in paths if p != base_path]
    if len(candidates) >= odd_k:
        odd_paths = random.sample(candidates, odd_k)
    else:
        odd_paths = [random.choice(candidates) for _ in range(odd_k)]

    # ===== 全图共享的版式参数 =====
    cell_size = random.randint(MIN_CELL_SIZE, MAX_CELL_SIZE)
    gap = random.randint(MIN_GAP, MAX_GAP)
    margin = random.randint(MIN_MARGIN, MAX_MARGIN)

    grid_w = cols * cell_size + (cols - 1) * gap
    grid_h = rows * cell_size + (rows - 1) * gap

    W = grid_w + 2 * margin
    H = grid_h + 2 * margin

    canvas = Image.new("RGB", (W, H), color=BG_COLOR)

    base_img = Image.open(base_path).convert("RGB") \
        .resize((cell_size, cell_size), Image.BILINEAR)

    odd_ptr = 0
    for idx in range(num_cells):
        r = idx // cols
        c = idx % cols

        x = margin + c * (cell_size + gap)
        y = margin + r * (cell_size + gap)

        if idx in odd_indices:
            op = odd_paths[odd_ptr]
            odd_ptr += 1
            odd_img = Image.open(op).convert("RGB") \
                .resize((cell_size, cell_size), Image.BILINEAR)

            odd_img = add_gaussian_noise_pil(odd_img, sigma=noise_sigma)

            canvas.paste(odd_img, (x, y))
        else:
            base_img = Image.open(base_path).convert("RGB") \
                .resize((cell_size, cell_size), Image.BILINEAR)

            base_img = add_gaussian_noise_pil(base_img, sigma=noise_sigma)

            canvas.paste(base_img, (x, y))

    odd_positions = sorted([[i // cols + 1, i % cols + 1] for i in odd_indices])


    meta = {
        "id":str(uuid.uuid4()),
        "odd_count": odd_k,
        "odd_list":[],
        "class": digit,
        "odd_rows_cols": odd_positions,
        "grid_size": [rows, cols],
        "gap": gap,
        "margin": margin,
        "base_image": base_path.name,
        "odd_images": [p.name for p in odd_paths],
    }

    return canvas, meta
import shutil

def generate_dataset(
    png_root: str,
    out_dir: str,
    samples: int = 1000,
    seed: int = 0
):
    import shutil
    random.seed(seed)

    png_root = Path(png_root)
    out_dir = Path(out_dir)

    # ===== 清空输出目录 =====
    if out_dir.exists():
        shutil.rmtree(out_dir)

    out_dir.mkdir(parents=True, exist_ok=True)
    img_dir = out_dir / "images"
    img_dir.mkdir(parents=True, exist_ok=True)
    # ========================

    digit_pool = load_digit_pool(png_root)

    annotations = []   # ⭐ 关键

    for i in range(samples):
        img, meta = generate_single_iol(digit_pool)
        img = add_gaussian_noise_pil(img, sigma=0.01)
        name = f"image_{i}.png"
        img.save(img_dir / name)
        W, H = img.size
        meta["image_size"] = [W, H]
        meta["image"] = f"{name}"
        annotations.append(meta)

        if (i + 1) % 100 == 0:
            print(f"[{i+1}/{samples}] generated")

    # ===== 一次性写成 list of dict =====
    ann_path = out_dir / "iol_test_data.json"
    with ann_path.open("w", encoding="utf-8") as f:
        json.dump(annotations, f, ensure_ascii=False, indent=2)

import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--png_root",
        type=str,
        default="./mnist/mnist_png",
        help="输入数据根目录（如 ./hanzi/hanzi_png, ./mnist/mnist_png）",
    )

    parser.add_argument(
        "--samples",
        type=int,
        default=100,
        help="每个类别采样数量",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="随机种子（不传则随机）",
    )

    args = parser.parse_args()
    
    
    args.out_dir = "/".join(args.png_root.split("/")[0:-1]) + "/iol_test_data"
    
    seed = args.seed if args.seed is not None else random.randint(0, 10000)

    generate_dataset(
        png_root=args.png_root,
        out_dir=args.out_dir,
        samples=args.samples,
        seed=seed,
    )
