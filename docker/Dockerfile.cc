# Adds a C++ compiler to an existing distfuzz image so torch.compile(backend="inductor") works on CPU.
# The base image is not modified.
#   docker build -f docker/Dockerfile.cc --build-arg BASE=distfuzz:cpu -t distfuzz:cpu-cc docker
#   docker build -f docker/Dockerfile.cc --build-arg BASE=distfuzz:nightly-20261103 -t distfuzz:nightly-20261103-cc docker
ARG BASE=distfuzz:cpu
FROM ${BASE}
RUN apt-get update && apt-get install -y --no-install-recommends g++ && rm -rf /var/lib/apt/lists/*
