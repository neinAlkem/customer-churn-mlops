FROM python:3.9-slim

ENV PYTHONUNBUFFERED=1
ENV PROJECT_DIR=/app
ENV MINIO_ACCESS_KEY=minioadmin
ENV MINIO_SECRET_KEY=minioadmin
ENV MINIO_URL=http://minio:9000
ENV DVC_BUCKET_NAME=${DVC_BUCKET_NAME}
ENV DVC_PATH=${DVC_PATH}
WORKDIR ${PROJECT_DIR}

RUN apt-get update && apt-get install -y git && rm -rf /var/lib/apt/lists/*
RUN pip install dvc

COPY requirements.txt $PROJECT_DIR
RUN pip install -r requirements.txt
COPY . $PROJECT_DIR
RUN chmod +x $PROJECT_DIR/scripts/*.py

# Initialize DVC repository
RUN dvc init

# Set up the DVC remote
RUN dvc remote add -d myremote s3://${DVC_BUCKET_NAME}/${DVC_PATH}

# Use .dvc/config.local to set credentials
RUN echo "[remote \"myremote\"]" > .dvc/config.local && \
    echo "    url = s3://${DVC_BUCKET_NAME}/${DVC_PATH}" >> .dvc/config.local && \
    echo "    access_key_id = ${MINIO_ACCESS_KEY}" >> .dvc/config.local && \
    echo "    secret_access_key = ${MINIO_SECRET_KEY}" >> .dvc/config.local && \
    echo "    endpointurl = ${MINIO_URL}" >> .dvc/config.local && \
    echo "    region = us-east-1" >> .dvc/config.local  # Optional


CMD ["bash", "-c", "dvc pull && python scripts/data_preprocess.py && python scripts/model_train.py && python scripts/model_evaluation.py"]
