#!/usr/bin/env sh
# OCR 방식 비교 실험 서버. 호스트에 Tesseract가 없어도 되도록 기존 Python 이미지(한국어 Tesseract 포함)에서 돌린다.
# 키는 backend/.env에서 컨테이너 환경으로만 전달되고 화면이나 로그에 찍지 않는다.
#   sh experiments/ocr_test/run.sh   →  http://127.0.0.1:8090
set -e
HERE="$(cd "$(dirname "$0")" && pwd)"
ENV_FILE="${ENV_FILE:-$HERE/../../../backend/.env}"
IMAGE="${CAPSTONE_IMAGE_PREFIX:-beauty-capstone}-python:dev"
docker run --rm --init --name ocr-test -p 127.0.0.1:${OCR_TEST_PORT:-8090}:8090 \
  --env-file "$ENV_FILE" -e OCR_TEST_PORT=8090 \
  -v "$HERE:/app/experiments/ocr_test:ro" \
  --entrypoint /app/.venv/bin/python "$IMAGE" /app/experiments/ocr_test/app.py
