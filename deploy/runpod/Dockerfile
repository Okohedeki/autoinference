# RunPod Serverless worker for the boltz2-cloud backend (provider "runpod").
# Build from the repository root:
#   docker build -f deploy/runpod/Dockerfile -t <registry>/autoinference-boltz2:<tag> .
FROM python:3.11-slim

ARG BOLTZ_VERSION=2.2.1
ENV PYTHONUNBUFFERED=1 BOLTZ_CACHE=/opt/boltz-cache

RUN pip install --no-cache-dir "boltz[cuda]==${BOLTZ_VERSION}" runpod \
        "safetensors>=0.4" "transformers>=4.51" "huggingface_hub>=0.24"

# Bake the weights into the image so cold workers never download them.
RUN mkdir -p "$BOLTZ_CACHE" && python -c "from pathlib import Path; from boltz.main import download_boltz2; download_boltz2(Path('$BOLTZ_CACHE'))"

WORKDIR /app
COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install --no-cache-dir --no-deps .

CMD ["python", "-u", "-m", "autoinference.bio.runpod_worker"]
