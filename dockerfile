FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    HOST=0.0.0.0 \
    PORT=8011 \
    # Threading control: prevent nested parallelism from Tesseract/NumPy/BLAS
    OMP_THREAD_LIMIT=1 \
    OMP_NUM_THREADS=1 \
    OPENBLAS_NUM_THREADS=1 \
    MKL_NUM_THREADS=1 \
    NUMEXPR_NUM_THREADS=1 \
    OMP_WAIT_POLICY=PASSIVE
 
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

# Pre-download & warm up PaddleOCR models during container build
RUN python -c "from paddleocr import PaddleOCR; PaddleOCR(use_textline_orientation=True, lang='en')"

COPY . .

EXPOSE ${PORT}

CMD uvicorn main:app --host ${HOST} --port ${PORT}