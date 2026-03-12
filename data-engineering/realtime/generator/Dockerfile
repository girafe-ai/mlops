FROM python:3.11-slim

WORKDIR /app
RUN pip install --no-cache-dir faker psycopg2-binary

COPY generate_data.py .
CMD ["python", "-u", "generate_data.py"]
