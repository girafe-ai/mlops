FROM python:3.12

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache -r requirements.txt

COPY cats_and_dogs cats_and_dogs
COPY commands.py .

ENTRYPOINT [ "python", "commands.py" ]
