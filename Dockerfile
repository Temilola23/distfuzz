FROM python:3.12-slim

ARG TORCH_SPEC="torch==2.14.0"
ARG TORCH_INDEX="https://download.pytorch.org/whl/cpu"
RUN pip install --no-cache-dir --pre "${TORCH_SPEC}" --index-url "${TORCH_INDEX}" \
 && pip install --no-cache-dir safetensors numpy

WORKDIR /src
COPY pyproject.toml README.md ./
COPY src src
RUN pip install --no-cache-dir -e ".[dev]"
COPY . .

ENV DISTFUZZ_MULTIRANK=1 PYTHONUNBUFFERED=1
CMD ["python", "-m", "pytest", "-m", "multirank"]
