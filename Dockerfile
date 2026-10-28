FROM python:3.14-slim

WORKDIR /app
ENV PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1

COPY requirements-api.txt .
# CPU-only PyTorch keeps the image much smaller
RUN pip install torch --index-url https://download.pytorch.org/whl/cpu \
    && pip install -r requirements-api.txt

COPY src ./src
COPY app ./app
COPY models ./models

ENV PORT=8000
EXPOSE 8000
CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT}"]
