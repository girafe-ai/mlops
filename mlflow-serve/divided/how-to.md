## MLFlow models serve

Для того, чтобы запустить сервер с предсказаниями модели и отрисовкой в HTML странице, необходимо проделать следующие шаги.

1. Запустить proxy сервер:
```
python proxy_server.py
```

2. Сохранить артефакт модели в mlflow:
```
python mlflow_model.py
```

3. Запустить mlflow models serve:
```
mlflow models serve -m cats_dogs_model/ --port 8888 --host 0.0.0.0 --no-conda
```

4. Запустить HTML-страницу:
```
python -m http.server 9000
```

Пример запроса в поднятый сервер:
```
curl -X POST http://localhost:8888/invocations      -H "Content-Type: application/json"      -d '{
         "inputs": [
             {
                 "data": "/home/vl-naumov/mlflow-serve/uploads/5d3e805c-6257-4460-9beb-93eec7f3fd4d_dog.jpg"
             }
         ]
     }'
```