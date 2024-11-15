FROM python:3.9-slim

ENV PYTHONUNBUFFERED=1
ENV PROJECT_DIR=/app
ENV MINIO_ROOT_USER=admin123
ENV MINIO_ROOT_PASSWORD=admin123
ENV MINIO_URL=http://minio:9000
ENV DVC_BUCKET_NAME=project-mlops
ENV DVC_PATH=data

WORKDIR ${PROJECT_DIR}

RUN apt-get update && apt-get install -y git && rm -rf /var/lib/apt/lists/*

RUN pip install dvc
COPY requirements.txt $PROJECT_DIR
RUN pip install -r requirements.txt

COPY . $PROJECT_DIR
RUN chmod +x $PROJECT_DIR/scripts/*.py

RUN dvc remote add -d myremote s3://${DVC_BUCKET_NAME} && \
    dvc remote modify myremote endpointurl $MINIO_URL && \
    dvc remote modify myremote access_key_id $MINIO_ROOT_USER && \
    dvc remote modify myremote secret_access_key $MINIO_ROOT_PASSWORD && \
    dvc remote modify myremote region us-east-1

RUN dvc remote default myremote

CMD ["bash", "-c", "dvc pull && python scripts/data_preprocess.py && python scripts/model_train.py && python scripts/model_evaluation.py"]
