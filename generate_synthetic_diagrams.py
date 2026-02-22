# pip install pillow numpy
# (para treinar) pip install ultralytics

"""
Gera diagramas sintéticos para treinar YOLO (ícones AWS/Azure/etc) com mais "cara de diagrama":
- split train/val/test (evita val = train)
- variação de escala/rotação por ícone
- "clutter": caixas (containers), linhas/setas, rótulos de texto, ruído/blur, compressão JPEG

Estrutura gerada:
output/
  images/{train,val,test}/synthetic_XXXX.jpg
  labels/{train,val,test}/synthetic_XXXX.txt
aws.yaml

Uso:
python generate_synthetic_diagrams.py
yolo detect train model=yolov8n.pt data=aws.yaml epochs=50 imgsz=640
"""

import os
import random
import shutil
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

# ---------------- Configurações ----------------

@dataclass
class Cfg:
    img_size: int = 640
    icon_base: int = 100
    num_images: int = 600

    # distribuição
    min_icons: int = 3
    max_icons: int = 8

    # splits
    train_ratio: float = 0.80
    val_ratio: float = 0.10  # resto vira test

    # augmentations
    scale_min: float = 0.70
    scale_max: float = 1.25
    rot_min: int = -12
    rot_max: int = 12

    # clutter
    max_containers: int = 2
    min_lines: int = 2
    max_lines: int = 8
    label_prob: float = 0.70

    # degradação
    blur_prob: float = 0.35
    blur_min: float = 0.2
    blur_max: float = 1.2
    noise_prob: float = 0.35
    noise_std_min: float = 3.0
    noise_std_max: float = 12.0
    jpeg_q_min: int = 65
    jpeg_q_max: int = 95

    # controle de sobreposição
    overlap_padding: int = 6
    max_place_tries: int = 30

    # reprodutibilidade (opcional)
    seed: int | None = None


CFG = Cfg()

# Se quiser reprodutibilidade, defina SEED por env: SEED=123 python generate_synthetic_diagrams.py
_seed_env = os.getenv("SEED")
if _seed_env and _seed_env.strip().isdigit():
    CFG.seed = int(_seed_env.strip())

if CFG.seed is not None:
    random.seed(CFG.seed)
    np.random.seed(CFG.seed)

# ---------------- Paths ----------------

BASE_DIR = Path(__file__).resolve().parent
ICONS_DIR = BASE_DIR / "icons"
OUT_DIR = BASE_DIR / "output"
YAML_PATH = BASE_DIR / "aws.yaml"


def ensure_icons():
    if not ICONS_DIR.exists():
        raise FileNotFoundError(f"Pasta de ícones não encontrada: {ICONS_DIR}")
    pngs = [p for p in ICONS_DIR.iterdir() if p.suffix.lower() == ".png"]
    if not pngs:
        raise FileNotFoundError(f"Nenhum .png encontrado em: {ICONS_DIR}")
    return pngs


def clean_output():
    shutil.rmtree(OUT_DIR, ignore_errors=True)
    for split in ("train", "val", "test"):
        (OUT_DIR / f"images/{split}").mkdir(parents=True, exist_ok=True)
        (OUT_DIR / f"labels/{split}").mkdir(parents=True, exist_ok=True)


def build_class_map(pngs: list[Path]) -> dict[str, int]:
    classes: dict[str, int] = {}
    for p in sorted(pngs, key=lambda x: x.name.lower()):
        cname = p.stem.lower()
        if cname not in classes:
            classes[cname] = len(classes)
    return classes


def pick_split(i: int, total: int) -> str:
    # Determinístico por índice (bom para reproduzir)
    r = i / max(1, total - 1)
    if r < CFG.train_ratio:
        return "train"
    if r < CFG.train_ratio + CFG.val_ratio:
        return "val"
    return "test"


def overlaps(box1, box2, pad: int = 0) -> bool:
    # box = (x, y, w, h)
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2

    x1a, y1a, x1b, y1b = x1 - pad, y1 - pad, x1 + w1 + pad, y1 + h1 + pad
    x2a, y2a, x2b, y2b = x2 - pad, y2 - pad, x2 + w2 + pad, y2 + h2 + pad

    return not (x1b < x2a or x1a > x2b or y1b < y2a or y1a > y2b)


def draw_arrow(draw: ImageDraw.ImageDraw, x1: int, y1: int, x2: int, y2: int, width: int = 2):
    # linha principal
    draw.line([x1, y1, x2, y2], fill=(120, 120, 120), width=width)

    # "ponta" simples
    dx, dy = x2 - x1, y2 - y1
    norm = (dx * dx + dy * dy) ** 0.5
    if norm < 1:
        return

    ux, uy = dx / norm, dy / norm
    # tamanho da ponta
    ah = 10
    aw = 6
    px, py = x2 - ux * ah, y2 - uy * ah

    # perpendicular
    vx, vy = -uy, ux
    p1 = (int(px + vx * aw), int(py + vy * aw))
    p2 = (int(px - vx * aw), int(py - vy * aw))

    draw.polygon([p1, p2, (x2, y2)], fill=(120, 120, 120))


def add_noise(img: Image.Image, std: float) -> Image.Image:
    arr = np.array(img).astype(np.int16)
    noise = np.random.normal(0, std, arr.shape).astype(np.int16)
    arr = np.clip(arr + noise, 0, 255).astype(np.uint8)
    return Image.fromarray(arr, "RGB")


def main():
    pngs = ensure_icons()
    clean_output()

    classes = build_class_map(pngs)
    icon_files = list(classes.keys())

    for i in range(CFG.num_images):
        img = Image.new("RGB", (CFG.img_size, CFG.img_size), (255, 255, 255))
        draw = ImageDraw.Draw(img)

        labels: list[str] = []
        placed_boxes: list[tuple[int, int, int, int]] = []

        # Containers (caixas tipo VPC/VNet/Subnet) - só pra dar contexto
        for _ in range(random.randint(0, CFG.max_containers)):
            x1, y1 = random.randint(10, 80), random.randint(10, 80)
            x2, y2 = random.randint(350, CFG.img_size - 10), random.randint(350, CFG.img_size - 10)
            draw.rectangle([x1, y1, x2, y2], outline=(180, 180, 180), width=2)

        # Linhas/setas para "poluir" como diagrama real
        for _ in range(random.randint(CFG.min_lines, CFG.max_lines)):
            x1, y1 = random.randint(0, CFG.img_size), random.randint(0, CFG.img_size)
            x2, y2 = random.randint(0, CFG.img_size), random.randint(0, CFG.img_size)
            draw_arrow(draw, x1, y1, x2, y2, width=random.randint(1, 3))

        # Ícones: com reposição (diagrama real tem repetição)
        num_icons = random.randint(CFG.min_icons, CFG.max_icons)
        icons_selected = random.choices(icon_files, k=num_icons)

        for cname in icons_selected:
            icon_path = ICONS_DIR / f"{cname}.png"
            icon = Image.open(icon_path).convert("RGBA")

            # escala + rotação
            scale = random.uniform(CFG.scale_min, CFG.scale_max)
            size = max(24, int(CFG.icon_base * scale))
            icon = icon.resize((size, size), resample=Image.BICUBIC)
            angle = random.randint(CFG.rot_min, CFG.rot_max)
            icon = icon.rotate(angle, expand=True)

            w_px, h_px = icon.size

            # posicionamento tentando evitar overlaps
            tries = 0
            while True:
                x = random.randint(0, CFG.img_size - w_px)
                y = random.randint(0, CFG.img_size - h_px)
                new_box = (x, y, w_px, h_px)

                if all(not overlaps(new_box, b, pad=CFG.overlap_padding) for b in placed_boxes):
                    break
                tries += 1
                if tries >= CFG.max_place_tries:
                    break

            img.paste(icon, (x, y), mask=icon)
            placed_boxes.append(new_box)

            # rótulo textual próximo (simula nome de serviço)
            if random.random() < CFG.label_prob:
                label_txt = cname.upper().replace("_", " ")[:18]
                tx = min(CFG.img_size - 5, x + 2)
                ty = min(CFG.img_size - 15, y + h_px + 2)
                draw.text((tx, ty), label_txt, fill=(40, 40, 40))

            # label YOLO (x_center, y_center, w, h) normalizado
            cx = (x + w_px / 2) / CFG.img_size
            cy = (y + h_px / 2) / CFG.img_size
            w = w_px / CFG.img_size
            h = h_px / CFG.img_size

            class_id = classes[cname]
            labels.append(f"{class_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")

        # degradações globais
        if random.random() < CFG.blur_prob:
            img = img.filter(ImageFilter.GaussianBlur(radius=random.uniform(CFG.blur_min, CFG.blur_max)))

        if random.random() < CFG.noise_prob:
            img = add_noise(img, std=random.uniform(CFG.noise_std_min, CFG.noise_std_max))

        # salvar no split
        split = pick_split(i, CFG.num_images)
        img_filename = f"synthetic_{i:04d}.jpg"
        lbl_filename = f"synthetic_{i:04d}.txt"

        q = random.randint(CFG.jpeg_q_min, CFG.jpeg_q_max)
        img.save(OUT_DIR / f"images/{split}/{img_filename}", quality=q)

        with open(OUT_DIR / f"labels/{split}/{lbl_filename}", "w", encoding="utf-8") as f:
            f.write("\n".join(labels))

    # aws.yaml
    with open(YAML_PATH, "w", encoding="utf-8") as f:
        f.write("path: output\n")
        f.write("train: images/train\n")
        f.write("val: images/val\n")
        f.write("test: images/test\n")
        f.write(f"nc: {len(classes)}\n")
        f.write("names:\n")
        for cname, idx in sorted(classes.items(), key=lambda x: x[1]):
            f.write(f"  {idx}: {cname}\n")

    print(f"[✓] {CFG.num_images} imagens sintéticas geradas em output/images/(train|val|test)")
    print(f"[✓] Labels YOLOv8 geradas em output/labels/(train|val|test)")
    print(f"[✓] Arquivo aws.yaml gerado com {len(classes)} classes")
    print("Treino sugerido:")
    print("  yolo detect train model=yolov8n.pt data=aws.yaml epochs=50 imgsz=640")


if __name__ == "__main__":
    main()
