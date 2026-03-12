FROM python:3.11-slim

WORKDIR /app
RUN pip install --no-cache-dir kafka-python psycopg2-binary

COPY pusher.py .
CMD ["python", "-u", "pusher.py"]
