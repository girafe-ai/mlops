# Set Up


### Запуск контейнера
```bash
docker-compose build
docker-compose up -d
```


### Регистрируем DNS запись в Cloudfare
Добавляем A запись, выбираем доменное имя, вставляем белый айпишник сервера, ставим галочку на проксирование
(подробнее было на семинаре)


### Заводим запись в nginx

```bash
sudo nano /etc/nginx/sites-available/mlflow.conf
```

и вставляем что-то вроде (mlflow.buran.team меняем на то, что указывали в A записи):
```
server {
    listen 80;
    server_name mlflow.buran.team;

    location / {
        proxy_pass http://localhost:13412;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }
}
```

сохраняем и делаем симлинку:
```bash
sudo ln -s /etc/nginx/sites-available/mlflow.conf /etc/nginx/sites-enabled/
```

чекаем что конфигурация nginx корректна: `sudo nginx -t`, перезапускаем nginx `sudo systemctl reload nginx`.