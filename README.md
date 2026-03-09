  # Hackathn - 5IADT
Hackaton - Fase 5: Pos Tech em IA para Devs da FIAP, 2025.
Um dos desafios é utilizar a Inteligência Artificial para realizar automaticamente
a modelagem de ameaças, baseado na metodologia STRIDE de um sistema a partir
de um diagrama de arquitetura de software em imagem. A empresa tem o objetivo de
validar a viabilidade dessa Feature, e para isso, será necessário fazer um MVP para
detecção supervisionada de ameaças.

Flavio Luiz Vicco - RM 361664

https://youtu.be/6YbMQ2dw1iw

# Hackaton – Modelagem de Ameaças (YOLO + Gemini + STRIDE) 🚀

> **Objetivo:** transformar uma **imagem de diagrama de arquitetura** (AWS/Azure/Google) em:
> - ✅ **Componentes estruturados (JSON)**
> - ✅ **Relatório STRIDE completo**
> - ✅ **PDF pronto para entrega**
> - ✅ **Imagem anotada** com bounding boxes do YOLO

---

## ✨ O que a solução faz

### 🖼️ 1) Você envia uma imagem do diagrama
- Formatos: **PNG/JPG**
- Ex.: diagramas de arquitetura cloud com ícones (AWS/Azure/Google)

### 🎯 2) O YOLO detecta os ícones
- Retorna: **classe**, **confiança**, **bbox** (x1,y1,x2,y2)
- Gera artefato: `output_diagrama_detected.jpg` (imagem com boxes)

### 🧠 3) O Gemini estrutura os componentes (JSON)
- O sistema envia ao Gemini:
  - lista de detecções do YOLO
  - (opcional) texto via OCR Azure
- Recebe de volta um **JSON** com componentes:
  - `name`, `provider`, `type` (+ opcionais)

### 🛡️ 4) O Gemini gera STRIDE e contramedidas
- Saída textual com:
  - Spoofing, Tampering, Repudiation, Info Disclosure, DoS, EoP
  - contramedidas recomendadas por componente

### 📄 5) Exporta PDF
- Artefato: `relatorio_stride.pdf`

---

## 🧩 Modos disponíveis

### ✅ Utilização — `app.py` (fallback CLIP opcional)
- Pipeline:
  - YOLO ✅
  - OCR Azure (opcional) ✅
  - Fallback embeddings **CLIP** (opcional) ✅ *(usa torch/open_clip)*
  - Gemini JSON ✅
  - Gemini STRIDE ✅
  - PDF ✅

---

## 🖥️ Interface (UI) com upload de imagem

A UI em Streamlit permite:
- ✅ escolher a imagem
- ✅ rodar análise
- ✅ ver imagem anotada + JSON + STRIDE
- ✅ baixar PDF e JSON

Arquivo: `ui_streamlit.py`

---

## 📁 Estrutura de pastas (sugestão)

```
Hackaton/
  app.py
  ui_streamlit.py
  generate_synthetic_diagrams.py
  icons/
    aws_s3.png
    aws_lambda.png
    azure_api_management.png
    ...
  runs/
    detect/
      train/
        weights/
          best.pt
  DiagramaTest.png
  .env
```

---

## 🔑 Pré-requisitos (credenciais)

### 1) Gemini API Key (Google AI Studio)
Crie uma key no Google AI Studio e salve como variável de ambiente:

- `GEMINI_API_KEY=...`

> Dica: use `.env` e **não versiona** esse arquivo.

### 2) (Opcional) OCR Azure
Se quiser OCR:
- `AZURE_ENDPOINT=...`
- `AZURE_KEY=...`

---

## ⚙️ Instalação (Windows / Linux / macOS)

### 1) Criar ambiente (opcional, recomendado)
```bash
python -m venv .venv
```

**Windows (PowerShell):**
```powershell
.\.venv\Scripts\Activate.ps1
```

**Linux/macOS:**
```bash
source .venv/bin/activate
```

### 2) Instalar dependências base
```bash
pip install google-genai python-dotenv ultralytics opencv-python reportlab numpy
```

### 3) Instalar UI (Streamlit)
```bash
pip install streamlit
```

### 4) (Opcional) OCR Azure
```bash
pip install azure-cognitiveservices-vision-computervision msrest
```

### 5) (Opcional) Fallback CLIP
```bash
pip install torch open_clip_torch pillow
```

---

## 🧾 Configuração do `.env`

Crie um arquivo `.env` na raiz do projeto:

```env
# Gemini
GEMINI_API_KEY=COLE_SUA_KEY_AQUI

# YOLO
MODEL_PATH=runs/detect/train/weights/best.pt
IMAGE_PATH=DiagramaTest.png

# Ícones
ICONS_DIR=icons

# Ajustes (opcional)
YOLO_CONF=0.25
YOLO_IMGSZ=640

# Fallback nativo (recomendado)
USE_NATIVE_EMBED_FALLBACK=1
CONF_FALLBACK=0.55
SIM_ACCEPT=0.30
MAX_FALLBACKS=25
```

---

## ▶️ Como executar

### ✅ Opção 1: Rodar via Streamlit (recomendado)
```bash id="d2vwkj"
streamlit run ui_streamlit.py
```
<img width="639" height="336" alt="streamlit" src="https://github.com/user-attachments/assets/dfbdbec2-dc2a-4f2f-9592-c09c66ab0f8a" />

Depois:
1. Faça upload da imagem
2. Clique em **Analisar**
3. Baixe o PDF e o JSON

### ✅ Opção 2: Rodar via script (sem UI)
```bash
python app.py
```

---

## 📦 Saídas geradas (artefatos)

| Arquivo | O que é |
|---|---|
| `output_diagrama_detected.jpg` | imagem original com boxes/labels do YOLO |
| `relatorio_stride.pdf` | relatório STRIDE em PDF |
| `components.json` | (na UI) JSON dos componentes para integração |

---
<img width="453" height="320" alt="relatorio_stride" src="https://github.com/user-attachments/assets/38bb3fe3-0ed7-4e19-9814-53668badf108" />

## 🧠 Como funciona o fallback (e por que isso importa)

### 🎯 Problema
Ícones muito parecidos (ou baixa resolução) podem gerar **baixa confiança** no YOLO.

## 🧩 Ícones (como montar a pasta `icons/`) 🧷

> Você precisa de ícones `.png` para treinar/usar fallback.  
> Recomendação: padronize nomes em **snake_case** e com prefixo do provider:

✅ Exemplos:
- `aws_s3.png`
- `aws_cloudfront.png`
- `azure_api_management.png`
- `azure_blob_storage.png`

### Onde baixar ícones

#### AWS
- Catálogo oficial de ícones/arquitetura (download do pacote):  
  https://aws.amazon.com/architecture/icons/

#### Azure
- Azure Architecture Icons (Microsoft Learn):  
  https://learn.microsoft.com/azure/architecture/icons/

> Se os pacotes vierem em **SVG**, converta para **PNG** (256×256 ou 512×512).

### Conversão SVG → PNG (Windows, rápido)
Opção prática: **Inkscape** (GUI) ou linha de comando.

**Inkscape (CLI) – exemplo:**
```bash
inkscape input.svg --export-type=png --export-filename=output.png --export-width=256 --export-height=256
```

---

## 🧯 Troubleshooting

### 1) `FileNotFoundError: best.pt`
Seu modelo YOLO não está no caminho definido em `MODEL_PATH`.

➡️ Procure o arquivo:
**PowerShell:**
```powershell
Get-ChildItem -Recurse -Filter best.pt | Select-Object FullName
```

Depois ajuste no `.env` ou exporte:
```powershell
$env:MODEL_PATH="C:\caminho\completo\best.pt"
```

---

### 2) “Nenhum componente retornado pelo Gemini”
Causas comuns:
- Resposta do Gemini veio com JSON inválido (raro, mas acontece)
- Detecções YOLO vazias ou muito ruins
- Imagem muito diferente do dataset de treino

Ações rápidas:
- confira `output_diagrama_detected.jpg`
- diminua `YOLO_CONF` (ex.: `0.20`)
- ative OCR (Azure) se seu diagrama tiver muito texto

---

### 3) Erros do `google-genai` com `Part.from_text`
Use as versões **_fixed.py** (já corrigidas para `types.Part.from_text(text=...)`).

---

## 🧪 Dicas de qualidade (para subir acurácia em 1 sprint)

✅ Ajustes rápidos:
- Padronize nomes dos ícones (sem espaço, sem acentos)
- Aumente dataset sintético com variações (escala, rotação, blur leve)
- Treine com `imgsz=640` e aumente `epochs` gradualmente
- Faça validação com 10–20 diagramas reais (não só sintético)

---

## 👥 Créditos e licença
- Ícones e marcas AWS/Azure/Google pertencem aos respectivos owners.
- Use conforme termos oficiais dos providers.

---

