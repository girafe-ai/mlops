FROM python:3.12

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache -r requirements.txt
COPY main.py .

ENTRYPOINT [ "fastapi", "dev", "main.py", "--host", "0.0.0.0" ]
