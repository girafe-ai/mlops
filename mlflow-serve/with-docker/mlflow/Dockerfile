FROM python:3.10-slim

WORKDIR /mlflow

COPY . .

RUN pip install mlflow torch torchvision pytorch-lightning

CMD ["mlflow", "models", "serve", "-m", "/mlflow/cats_dogs_model", "-p", "8888", "--host", "0.0.0.0", "--no-conda"]
