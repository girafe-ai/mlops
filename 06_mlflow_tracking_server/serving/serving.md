
```bash
# MLFlow сохраняет артефакт
python run.py

# Создает контейнер с нужными зависимостями
mlflow models build-docker -m model_dir -n my_model_image

# Запускам контейнер
docker-compose up

# Тестовый запрос
curl -X POST -H "Content-Type:application/json" --data '{"dataframe_records": [{"a": 5, "b": 3}]}' http://localhost:13414/invocations

curl -X POST -H "Content-Type: application/json" --data '{"dataframe_records": [{"a": 5, "b": 3}]}' http://serving.mlflow.buran.team/invocations
```


Опционально nginx:

```
server {
    listen 80;
    server_name serving.mlflow.buran.team;

    location / {
        proxy_pass http://localhost:13414;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
    }
}
```

```bash 
sudo nano /etc/nginx/sites-available/serving.mlflow.buran.team
sudo ln -s /etc/nginx/sites-available/serving.mlflow.buran.team /etc/nginx/sites-enabled/
sudo nginx -t
sudo systemctl reload nginx
```
