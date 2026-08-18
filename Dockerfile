FROM --platform=linux/amd64 pytorch/pytorch:2.13.0-cuda12.6-cudnn9-runtime AS example_algorithm_amd64

ENV PYTHONUNBUFFERED=1

RUN groupadd -r user && useradd -m --no-log-init -r -g user user
USER user

WORKDIR /opt/app

# Install Python dependencies
COPY --chown=user:user requirements.txt /opt/app/
RUN python -m pip install \
    --no-cache-dir \
    --no-color \
    --break-system-packages \
    --requirement /opt/app/requirements.txt

# Install nnU-Net from bundled source (includes custom ISLES trainers)
COPY --chown=user:user nnUNet /opt/app/nnUNet
RUN python -m pip install \
    --no-cache-dir \
    --no-color \
    --break-system-packages \
    /opt/app/nnUNet

# Copy application code
COPY --chown=user:user app.py /opt/app/
COPY --chown=user:user inference.py /opt/app/

# This label is required — Grand Challenge uses it to detect that the container
# implements the invoke API.
LABEL org.grand-challenge.api-method="invoke"
EXPOSE 4743
ENTRYPOINT ["python", "app.py"]
