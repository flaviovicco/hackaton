# pip install ultralytics opencv-python reportlab python-dotenv google-genai
# (opcional para fallback embeddings) pip install open_clip_torch torch pillow
#
# Como autenticar (Gemini Developer API):
#   - Crie uma API Key no Google AI Studio e exporte como GEMINI_API_KEY
#
# Observação: "gratuito" depende do Free Tier e limites da sua conta/projeto no AI Studio.

import os
import time
import json
import logging
import re
from pathlib import Path
import mimetypes

from dotenv import load_dotenv
from ultralytics import YOLO
import cv2

from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer

from google import genai
from google.genai import types

try:
    from azure.cognitiveservices.vision.computervision import ComputerVisionClient
    from msrest.authentication import CognitiveServicesCredentials
except Exception:
    ComputerVisionClient = None
    CognitiveServicesCredentials = None

# ─────────── Configurações ───────────
MODEL_PATH = os.getenv("MODEL_PATH", "runs/detect/train/weights/best.pt")
IMAGE_PATH = os.getenv("IMAGE_PATH", "DiagramaTest.png")
ICONS_DIR = os.getenv("ICONS_DIR", "icons")

YOLO_CONF = float(os.getenv("YOLO_CONF", "0.25"))
YOLO_IMGSZ = int(os.getenv("YOLO_IMGSZ", "640"))
OCR_TIMEOUT_S = float(os.getenv("OCR_TIMEOUT_S", "25"))

# Gemini model (Developer API)
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")

# fallback embeddings (opcional)
USE_EMBEDDINGS = os.getenv("USE_EMBEDDINGS", "0").strip() in ("1", "true", "True", "yes", "YES")
CONF_FALLBACK = float(os.getenv("CONF_FALLBACK", "0.55"))
SCORE_ACCEPT = float(os.getenv("SCORE_ACCEPT", "0.28"))

# ─────────── Logging ───────────
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

# ─────────── Credenciais ───────────
load_dotenv()  # permite carregar GEMINI_API_KEY de um .env, se você preferir

# Gemini client pega a API key automaticamente de GEMINI_API_KEY ou GOOGLE_API_KEY
gemini_client = genai.Client()

# OCR Azure (opcional, mantém compatibilidade do pipeline)
client_cv = None
if ComputerVisionClient and CognitiveServicesCredentials:
    az_endpoint = os.getenv("AZURE_ENDPOINT")
    az_key = os.getenv("AZURE_KEY")
    if az_endpoint and az_key:
        client_cv = ComputerVisionClient(az_endpoint, CognitiveServicesCredentials(az_key))
    else:
        log.warning("AZURE_ENDPOINT/AZURE_KEY não configurados. OCR será pulado.")
else:
    log.warning("Dependências Azure OCR não instaladas. OCR será pulado.")

FENCE_RX = re.compile(r"^```(?:json)?\s*|\s*```$", re.I | re.M)

# ---------------- Embeddings fallback (opcional) ----------------
ICON_INDEX = None

def build_icon_index(icon_dir: str):
    """
    Indexa ícones (png) usando embeddings CLIP.
    Requer: open_clip_torch + torch + pillow
    """
    try:
        import torch
        import open_clip
        from PIL import Image
    except Exception as e:
        log.warning("Embeddings fallback indisponível (deps faltando): %s", e)
        return None

    icon_dir_path = Path(icon_dir)
    if not icon_dir_path.exists():
        log.warning("ICONS_DIR não existe (%s). Fallback embeddings desabilitado.", icon_dir)
        return None

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, _, preprocess = open_clip.create_model_and_transforms("ViT-B-32", pretrained="openai")
    model = model.to(device).eval()

    names, embs = [], []
    for fn in sorted(icon_dir_path.glob("*.png")):
        name = fn.stem.lower()
        img = preprocess(Image.open(fn).convert("RGB")).unsqueeze(0).to(device)
        with torch.no_grad():
            emb = model.encode_image(img)
            emb = emb / emb.norm(dim=-1, keepdim=True)
        names.append(name)
        embs.append(emb.cpu())

    if not embs:
        log.warning("Nenhum ícone PNG encontrado em %s. Fallback embeddings desabilitado.", icon_dir)
        return None

    import torch as _torch
    embs = _torch.cat(embs, dim=0)  # [N, D]
    return {"names": names, "embs": embs, "device": device, "model": model, "preprocess": preprocess}


def classify_crop_with_embeddings(crop_bgr, index):
    """
    crop_bgr: recorte OpenCV (BGR)
    retorna (name, score cosine)
    """
    if index is None:
        return None, None

    try:
        import torch
        from PIL import Image
    except Exception:
        return None, None

    crop_rgb = crop_bgr[:, :, ::-1]
    pil = Image.fromarray(crop_rgb)

    img = index["preprocess"](pil).unsqueeze(0).to(index["device"])
    model = index["model"]

    with torch.no_grad():
        emb = model.encode_image(img)
        emb = emb / emb.norm(dim=-1, keepdim=True)
        sims = (emb.cpu() @ index["embs"].T).squeeze(0)
        best = int(torch.argmax(sims).item())
        return index["names"][best], float(sims[best].item())

# ---------------- Helpers ----------------

def safe_json_loads(s: str):
    return json.loads(FENCE_RX.sub("", s).strip())

def clamp_bbox_xyxy(b, w, h):
    x1, y1, x2, y2 = b
    x1 = max(0, min(int(x1), w - 1))
    y1 = max(0, min(int(y1), h - 1))
    x2 = max(0, min(int(x2), w))
    y2 = max(0, min(int(y2), h))
    if x2 <= x1 or y2 <= y1:
        return None
    return [x1, y1, x2, y2]

def guess_mime(path: str) -> str:
    mt, _ = mimetypes.guess_type(path)
    return mt or "image/png"

# ─────────── Função: Detectar Componentes ───────────
def list_components(image_path: str, model_path: str) -> list[dict]:
    # 1) YOLO
    log.info("Detectando ícones com YOLO... (conf=%.2f, imgsz=%d)", YOLO_CONF, YOLO_IMGSZ)
    model = YOLO(model_path)
    results = model(image_path, conf=YOLO_CONF, imgsz=YOLO_IMGSZ)

    # salva imagem com boxes
    for r in results:
        annotated = r.plot()
        output_path = "output_diagrama_detected.jpg"
        cv2.imwrite(output_path, annotated)
        log.info("Imagem com bounding boxes salva em %s", output_path)

    detections = []
    names = model.names
    if len(results) and results[0].boxes is not None:
        for box in results[0].boxes:
            class_id = int(box.cls[0])
            class_name = names.get(class_id, str(class_id))
            conf = float(box.conf[0])
            x1, y1, x2, y2 = [float(v) for v in box.xyxy[0].tolist()]
            detections.append({
                "class": class_name,
                "conf": round(conf, 4),
                "bbox_xyxy": [round(x1, 1), round(y1, 1), round(x2, 1), round(y2, 1)]
            })

    log.info("YOLO detectou %d instâncias.", len(detections))

    # 2) OCR Azure (opcional)
    ocr_lines = []
    if client_cv is not None:
        log.info("Executando OCR via Azure...")
        with open(image_path, "rb") as f:
            op = client_cv.read_in_stream(f, raw=True)
        op_id = op.headers["Operation-Location"].split("/")[-1]

        start = time.time()
        while True:
            result = client_cv.get_read_result(op_id)
            if result.status not in ("notStarted", "running"):
                break
            if (time.time() - start) > OCR_TIMEOUT_S:
                log.warning("OCR timeout (%.1fs). Prosseguindo sem OCR.", OCR_TIMEOUT_S)
                result = None
                break
            time.sleep(0.4)

        if result and getattr(result, "analyze_result", None):
            ocr_lines = [ln.text for pg in result.analyze_result.read_results for ln in pg.lines]
            log.info("OCR detectou %d linhas.", len(ocr_lines))
    else:
        log.info("OCR desabilitado.")

    # 3) Fallback embeddings (opcional) em detecções fracas
    if USE_EMBEDDINGS and detections:
        global ICON_INDEX
        if ICON_INDEX is None:
            ICON_INDEX = build_icon_index(ICONS_DIR)

        img_cv = cv2.imread(image_path)
        if img_cv is not None and ICON_INDEX is not None:
            H, W = img_cv.shape[:2]
            for det in detections:
                if det["conf"] >= CONF_FALLBACK:
                    continue
                bb = clamp_bbox_xyxy(det["bbox_xyxy"], W, H)
                if not bb:
                    continue
                x1, y1, x2, y2 = bb
                crop = img_cv[y1:y2, x1:x2]
                if crop.size == 0:
                    continue

                fb, score = classify_crop_with_embeddings(crop, ICON_INDEX)
                if fb and score and score >= SCORE_ACCEPT:
                    det["fallback_class"] = fb
                    det["fallback_score"] = round(score, 4)

            log.info("Fallback embeddings aplicado (para conf < %.2f).", CONF_FALLBACK)
        else:
            log.warning("Falha ao carregar imagem/índices para embeddings; fallback ignorado.")

    # 4) GEMINI - componente estruturado (JSON)
    log.info("Enviando dados para Gemini (%s)...", GEMINI_MODEL)

    schema = {
        "type": "object",
        "properties": {
            "components": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string"},
                        "provider": {"type": "string"},
                        "type": {"type": "string"},
                        "confidence": {"type": "number"},
                        "source": {"type": "string"},
                        "bbox_xyxy": {"type": "array", "items": {"type": "number"}}
                    },
                    "required": ["name", "provider", "type"]
                }
            }
        },
        "required": ["components"]
    }

    prompt = (
        "Você receberá:\n"
        "1) Texto OCR (se houver)\n"
        "2) Detecções YOLO (instâncias com bbox+conf) e possível fallback por embeddings\n\n"
        "Tarefa:\n"
        "- Produza APENAS um JSON válido seguindo o schema.\n"
        "- Use as detecções como base. Se houver fallback_class em uma detecção fraca, prefira esse rótulo.\n"
        "- Não invente componentes que não apareçam no OCR/YOLO.\n"
        "- Para cada componente, preencha: name (curto e humano), provider (aws|azure|gcp|onprem|unknown), type (categoria).\n\n"
        "OCR:\n"
        + ("\n".join(ocr_lines) if ocr_lines else "(nenhum texto)")
        + "\n\nDETECTIONS:\n"
        + json.dumps(detections, ensure_ascii=False, indent=2)
    )

    mime = guess_mime(image_path)
    with open(image_path, "rb") as f:
        image_bytes = f.read()

    # FIX: Part.from_text usa keyword argument "text"
    resp = gemini_client.models.generate_content(
        model=GEMINI_MODEL,
        contents=[
            types.Part.from_text(text=prompt),
            types.Part.from_bytes(data=image_bytes, mime_type=mime),
        ],
        config=types.GenerateContentConfig(
            temperature=0,
            response_mime_type="application/json",
            response_schema=schema,
        ),
    )

    text = resp.text or ""
    if not text.strip():
        return []

    try:
        data = safe_json_loads(text)
        return data.get("components", [])
    except Exception as e:
        log.warning("Falha ao parsear JSON do Gemini: %s", e)
        return []

# ─────────── STRIDE via Gemini ───────────
def gerar_relatorio_stride(components: list[dict]) -> str:
    prompt = (
        "Com base na lista de componentes abaixo, gere um relatório de modelagem de ameaças "
        "utilizando a metodologia STRIDE. Para cada componente, identifique ameaças potenciais "
        "relacionadas a: Spoofing, Tampering, Repudiation, Information Disclosure, Denial of Service, "
        "e Elevation of Privilege. Sugira também contramedidas para cada ameaça.\n\n"
        "Responda com um texto estruturado, claro, técnico, e em português.\n\n"
        f"Lista de componentes:\n{json.dumps(components, indent=2, ensure_ascii=False)}"
    )

    # Passe string direto (o SDK converte internamente para Part text)
    resp = gemini_client.models.generate_content(
        model=GEMINI_MODEL,
        contents=prompt,
        config=types.GenerateContentConfig(temperature=0.4),
    )
    return (resp.text or "").strip()

# ─────────── PDF ───────────
def gerar_pdf_relatorio(texto: str, output_file: str = "relatorio_stride.pdf"):
    doc = SimpleDocTemplate(output_file, pagesize=A4, rightMargin=30, leftMargin=30, topMargin=30, bottomMargin=30)
    doc.title = "Relatório STRIDE"
    styles = getSampleStyleSheet()

    title_style = ParagraphStyle("Title", parent=styles["Heading1"], fontSize=16, spaceAfter=14)
    section_style = ParagraphStyle("Section", parent=styles["Heading2"], fontSize=13, spaceAfter=10)
    label_style = ParagraphStyle("Label", parent=styles["Normal"], fontName="Helvetica-Bold", leftIndent=10, spaceAfter=2)
    text_style = ParagraphStyle("Text", parent=styles["Normal"], leftIndent=20, spaceAfter=6)

    flowables = []
    for line in (texto or "").splitlines():
        line = line.strip()
        if not line:
            continue

        if line.startswith("# "):
            flowables.append(Paragraph(line[2:].strip(), title_style))
            flowables.append(Spacer(1, 12))
        elif line.startswith("## "):
            flowables.append(Paragraph(line[3:].strip(), section_style))
        elif line.startswith("### "):
            flowables.append(Paragraph(f"<b>{line[4:].strip()}</b>", label_style))
        elif line.startswith("- **"):
            m = re.match(r"- \*\*(.+?)\*\*: (.+)", line)
            if m:
                label, content = m.groups()
                flowables.append(Paragraph(f"<b>{label}:</b>", label_style))
                flowables.append(Paragraph(content.strip(), text_style))
            else:
                flowables.append(Paragraph(line, text_style))
        else:
            flowables.append(Paragraph(line, text_style))

    doc.build(flowables)
    log.info("Relatório STRIDE salvo em: %s", output_file)

# ─────────── Main ───────────
if __name__ == "__main__":
    components = list_components(IMAGE_PATH, MODEL_PATH)

    if not components:
        log.warning("Nenhum componente retornado pelo Gemini.")
    else:
        print("\nComponentes estruturados detectados:")
        print(json.dumps(components, indent=2, ensure_ascii=False))

        log.info("Gerando relatório STRIDE com Gemini...")
        relatorio_texto = gerar_relatorio_stride(components)

        print("\nRelatório STRIDE:\n")
        print(relatorio_texto)

        gerar_pdf_relatorio(relatorio_texto)
