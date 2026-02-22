# ui_streamlit.py
# Interface Streamlit para escolher imagem e executar o pipeline YOLO + Gemini + STRIDE + PDF.
#
# Requisitos:
#   pip install streamlit
#   pip install google-genai python-dotenv ultralytics opencv-python reportlab numpy
#   (se usar OCR Azure) pip install azure-cognitiveservices-vision-computervision msrest
#
# Execução:
#   streamlit run ui_streamlit.py
#
# Variáveis de ambiente (.env recomendado):
#   GEMINI_API_KEY=...
#   MODEL_PATH=... (caminho do best.pt)
#   ICONS_DIR=icons
#   (opcionais) YOLO_CONF, YOLO_IMGSZ, USE_NATIVE_EMBED_FALLBACK, CONF_FALLBACK, SIM_ACCEPT, MAX_FALLBACKS
#
# Observação:
# - Este UI chama funções do arquivo: app_gemini_native_embeddings_v2_fixed.py
# - Mantenha ambos na mesma pasta (ou ajuste o import).

import os
import json
import time
from pathlib import Path

import streamlit as st

# Importa as rotinas do pipeline
from app_gemini_native_embeddings_v2_fixed import (
    list_components,
    gerar_relatorio_stride,
    gerar_pdf_relatorio,
)

APP_TITLE = "Threat Modeling (STRIDE) a partir de Diagrama (YOLO + Gemini)"

def save_uploaded_file(uploaded, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    suffix = Path(uploaded.name).suffix.lower() or ".png"
    out_path = out_dir / f"input_{int(time.time())}{suffix}"
    out_path.write_bytes(uploaded.getbuffer())
    return out_path

def read_bytes(path: Path) -> bytes:
    return path.read_bytes() if path.exists() else b""

st.set_page_config(page_title=APP_TITLE, layout="wide")
st.title(APP_TITLE)

with st.sidebar:
    st.header("Configuração")
    model_path = st.text_input("MODEL_PATH (YOLO weights)", value=os.getenv("MODEL_PATH", "runs/detect/train/weights/best.pt"))
    icons_dir = st.text_input("ICONS_DIR", value=os.getenv("ICONS_DIR", "icons"))

    yolo_conf = st.slider("YOLO_CONF", 0.05, 0.95, float(os.getenv("YOLO_CONF", "0.25")), 0.01)
    yolo_imgsz = st.selectbox("YOLO_IMGSZ", [416, 512, 640, 768, 896, 1024], index=[416,512,640,768,896,1024].index(int(os.getenv("YOLO_IMGSZ","640"))) if int(os.getenv("YOLO_IMGSZ","640")) in [416,512,640,768,896,1024] else 2)

    use_fallback = st.toggle("USE_NATIVE_EMBED_FALLBACK", value=os.getenv("USE_NATIVE_EMBED_FALLBACK", "1").lower() in ("1","true","yes"))
    conf_fallback = st.slider("CONF_FALLBACK (ativa fallback abaixo)", 0.05, 0.95, float(os.getenv("CONF_FALLBACK", "0.55")), 0.01)
    sim_accept = st.slider("SIM_ACCEPT (limiar similaridade)", 0.05, 0.95, float(os.getenv("SIM_ACCEPT", "0.30")), 0.01)
    max_fallbacks = st.number_input("MAX_FALLBACKS", min_value=0, max_value=200, value=int(os.getenv("MAX_FALLBACKS","25")), step=1)

    st.caption("Dica: coloque GEMINI_API_KEY no .env. Não versionar .env.")

# Propaga configs para o pipeline via env (sem alterar o código do pipeline)
os.environ["MODEL_PATH"] = model_path
os.environ["ICONS_DIR"] = icons_dir
os.environ["YOLO_CONF"] = str(yolo_conf)
os.environ["YOLO_IMGSZ"] = str(yolo_imgsz)
os.environ["USE_NATIVE_EMBED_FALLBACK"] = "1" if use_fallback else "0"
os.environ["CONF_FALLBACK"] = str(conf_fallback)
os.environ["SIM_ACCEPT"] = str(sim_accept)
os.environ["MAX_FALLBACKS"] = str(max_fallbacks)

st.subheader("1) Selecione a imagem do diagrama")
uploaded = st.file_uploader("Envie um arquivo .png/.jpg", type=["png", "jpg", "jpeg"])

colA, colB = st.columns([1, 1], gap="large")

if uploaded:
    input_path = save_uploaded_file(uploaded, Path("uploads"))
    with colA:
        st.markdown("### Imagem de entrada")
        st.image(str(input_path), use_container_width=True)
        st.write(f"Arquivo: `{input_path}`")

    run = st.button("▶️ Analisar", type="primary")

    if run:
        if not Path(model_path).exists() and not Path(model_path).name.startswith("yolov8"):
            st.error(f"MODEL_PATH não encontrado: {model_path}")
            st.stop()

        with st.status("Executando pipeline...", expanded=True) as status:
            st.write("🔎 Rodando YOLO e extraindo componentes (Gemini)...")
            try:
                components = list_components(str(input_path), model_path)
            except Exception as e:
                status.update(label="Falha no pipeline", state="error")
                st.exception(e)
                st.stop()

            if not components:
                status.update(label="Concluído, mas sem componentes", state="error")
                st.warning("Nenhum componente retornado. Verifique logs/limiares ou imagem.")
                st.stop()

            st.write("🧠 Gerando relatório STRIDE (Gemini)...")
            try:
                report_text = gerar_relatorio_stride(components)
            except Exception as e:
                status.update(label="Falha ao gerar STRIDE", state="error")
                st.exception(e)
                st.stop()

            st.write("📄 Gerando PDF...")
            pdf_path = Path("relatorio_stride.pdf")
            try:
                gerar_pdf_relatorio(report_text, str(pdf_path))
            except Exception as e:
                status.update(label="Falha ao gerar PDF", state="error")
                st.exception(e)
                st.stop()

            status.update(label="Concluído ✅", state="complete")

        annotated_path = Path("output_diagrama_detected.jpg")

        with colB:
            st.markdown("### Imagem anotada (YOLO)")
            if annotated_path.exists():
                st.image(str(annotated_path), use_container_width=True)
                st.download_button(
                    "⬇️ Baixar imagem anotada",
                    data=read_bytes(annotated_path),
                    file_name=annotated_path.name,
                    mime="image/jpeg",
                )
            else:
                st.info("Imagem anotada não encontrada (output_diagrama_detected.jpg).")

        st.markdown("## 2) Componentes detectados (JSON)")
        st.json(components)

        st.markdown("## 3) Relatório STRIDE")
        st.write(report_text)

        st.download_button(
            "⬇️ Baixar relatório PDF",
            data=read_bytes(Path("relatorio_stride.pdf")),
            file_name="relatorio_stride.pdf",
            mime="application/pdf",
        )

        st.download_button(
            "⬇️ Baixar componentes (JSON)",
            data=json.dumps(components, ensure_ascii=False, indent=2).encode("utf-8"),
            file_name="components.json",
            mime="application/json",
        )

else:
    st.info("Envie uma imagem para habilitar a análise.")
