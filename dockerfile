FROM python:3.9-slim

ENV PYTHONUNBUFFERD = 1
ENV PROEJCT_DIR=/app
WORKDIR ${PROEJCT_DIR}

RUN apt-get update && apt-get install -y git && rm -rf /var/lib/apt/list/*
RUN pip install dvc

COPY requirements.txt $PROEJCT_DIR
RUN pip install -r requirements.txt
COPY . $PROEJCT_DIR
RUN chmod +x $PROJECT_DIR/src/*.py

CMD ["dvc", "repro"]
