FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    HOST=0.0.0.0 \
    PORT=8011 \
    # Threading control: optimized for 8-core CPU
    OMP_THREAD_LIMIT=6 \
    OMP_NUM_THREADS=4 \
    OPENBLAS_NUM_THREADS=4 \
    MKL_NUM_THREADS=4 \
    NUMEXPR_NUM_THREADS=4 \
    OMP_WAIT_POLICY=PASSIVE \
    OMP_DYNAMIC=FALSE
 
WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    tesseract-ocr \
    tesseract-ocr-ind \
    tesseract-ocr-eng \
    curl \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/* \
    && TESSDATA_DIR=$(dirname "$(find / -iname 'ind.traineddata' 2>/dev/null | head -n1)") \
    && echo "Mengganti trained data di: $TESSDATA_DIR" \
    && curl -sL -o "$TESSDATA_DIR/ind.traineddata" \
       "https://raw.githubusercontent.com/tesseract-ocr/tessdata_best/main/ind.traineddata" \
    && curl -sL -o "$TESSDATA_DIR/eng.traineddata" \
       "https://raw.githubusercontent.com/tesseract-ocr/tessdata_best/main/eng.traineddata" \
    && ls -lh "$TESSDATA_DIR/ind.traineddata" "$TESSDATA_DIR/eng.traineddata"

COPY requirements.txt .
RUN pip install --no-cache-dir --timeout=60 --retries=5 --root-user-action=ignore -r requirements.txt

# Pre-download & warm up PaddleOCR V2 models, then export exact English model to ONNX for V3
RUN python -c "from paddleocr import PaddleOCR; import numpy as np, cv2; ocr = PaddleOCR(use_textline_orientation=False, lang='en', enable_mkldnn=False, det_limit_side_len=960, rec_batch_num=6); img = np.ones((100, 300, 3), dtype=np.uint8) * 255; cv2.putText(img, 'WARMUP', (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2); ocr.ocr(img)" && \
    mkdir -p /app/models_onnx && \
    python -c "import os, glob, paddle2onnx; \
rec_candidates = [d for d in glob.glob('/root/**/*', recursive=True) if os.path.isdir(d) and ('rec' in d.lower() or 'en' in d.lower()) and any(f.endswith('.pdmodel') for f in os.listdir(d))]; \
model_dir = rec_candidates[0] if rec_candidates else None; \
print('Converting model dir:', model_dir); \
if model_dir: \
    m_files = [f for f in os.listdir(model_dir) if f.endswith('.pdmodel')]; \
    p_files = [f for f in os.listdir(model_dir) if f.endswith('.pdiparams')]; \
    if m_files and p_files: \
        paddle2onnx.command.c_paddle_to_onnx(model_file=os.path.join(model_dir, m_files[0]), params_file=os.path.join(model_dir, p_files[0]), save_file='/app/models_onnx/en_PP-OCRv4_rec.onnx', opset_version=11, enable_onnx_checker=True)" && \
    curl -sL -o /app/models_onnx/en_dict.txt "https://raw.githubusercontent.com/PaddlePaddle/PaddleOCR/release/2.7/ppocr/utils/en_dict.txt" && \
    python -c "import os; from rapidocr_onnxruntime import RapidOCR; import numpy as np, cv2; engine = RapidOCR(rec_model_path='/app/models_onnx/en_PP-OCRv4_rec.onnx', rec_keys_path='/app/models_onnx/en_dict.txt', text_score=0.5) if os.path.exists('/app/models_onnx/en_PP-OCRv4_rec.onnx') else RapidOCR(text_score=0.5); img = np.ones((100, 300, 3), dtype=np.uint8) * 255; cv2.putText(img, 'WARMUP ONNX', (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2); engine(img, use_cls=False)"

COPY . .

EXPOSE ${PORT}

CMD uvicorn main:app --host ${HOST} --port ${PORT}