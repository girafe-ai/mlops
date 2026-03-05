FROM python:3.10-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

RUN dd if=/dev/zero of=/tmp/dummy.bin bs=1M count=10

COPY train.py .

CMD ["python", "train.py"]
