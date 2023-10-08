# MinIO


[Официальный гайд по установке](https://min.io/docs/minio/linux/operations/installation.html)

[Простой гайд со скриншотами](http://www.sefidian.com/2022/04/08/deploy-standalone-minio-using-docker-compose/)

[Мониторинг и алерты](https://min.io/docs/minio/linux/operations/monitoring.html)


## Deploy standalone (Single Node)

1. Сначала в `docker-compose.yaml` необходимо настроить под себя порты, volume и credentials
1. Далее поднимаем контейнер: `docker-compose up` (или `up -d` для detached режима)
1. Не забудьте пробросить порты, если поднимаете на удаленной машине
1. Логинимся в консоль http://localhost:9091/
1. Заводим бакет, конфиругируем свойства


P.S. Для работы через различных клиентов
- `MINIO_ROOT_USER` <=> `access_key`
- `MINIO_ROOT_PASSWORD` <=> `secret_key`
