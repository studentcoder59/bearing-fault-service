FROM python:3.14-slim

# Hugging Face Spaces runs containers as user 1000 and expects port 7860
RUN useradd -m -u 1000 user
USER user
ENV PATH="/home/user/.local/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PORT=7860
WORKDIR /home/user/app

COPY --chown=user requirements-api.txt .
# CPU-only PyTorch keeps the image much smaller; versions match the ones the models were trained with
RUN pip install --user torch==2.14.1 --index-url https://download.pytorch.org/whl/cpu \
    && pip install --user -r requirements-api.txt

COPY --chown=user src ./src
COPY --chown=user app ./app
COPY --chown=user models ./models

EXPOSE 7860
CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT}"]
