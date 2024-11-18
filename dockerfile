# Base image
FROM python:3.8-slim

# Set the working directory
WORKDIR /app

# Copy the requirements file and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy all necessary scripts and configuration files
COPY scripts/data_preprocess.py scripts/model_train.py config.yaml /app/

# Install additional packages (if needed)
RUN apt-get update && apt-get install -y \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Expose the port (for Flask/FastAPI)
EXPOSE 5000

# Command to run the API (assuming it's in model_train.py or another file)
CMD ["python", "model_train.py"]