FROM --platform=linux/amd64 pytorch/pytorch:2.1.2-cuda12.1-cudnn8-runtime AS clear_ec_algorithm_amd64

ENV PYTHONUNBUFFERED=1

RUN groupadd -r user && useradd -m --no-log-init -r -g user user

WORKDIR /opt/app
ENV PATH="/home/user/.local/bin:${PATH}"

# Install system packages needed by OpenCV / scikit-image / matplotlib backends.
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 \
        libglib2.0-0 \
        libsm6 \
        libxext6 \
        libxrender1 \
    && rm -rf /var/lib/apt/lists/*

USER user

COPY --chown=user:user requirements.txt /opt/app/

RUN python -m pip install \
    --user \
    --no-cache-dir \
    --no-color \
    --requirement /opt/app/requirements.txt

# Pre-download Cellpose v1.0 'cyto' weights so the offline runtime container
# (--network none) can load them. Cellpose stores them under ~/.cellpose/models/.
RUN python -c "from cellpose import models; models.Cellpose(gpu=False, model_type='cyto')"

COPY --chown=user:user src /opt/app/src
COPY --chown=user:user inference.py /opt/app/

ENTRYPOINT ["python", "inference.py"]
