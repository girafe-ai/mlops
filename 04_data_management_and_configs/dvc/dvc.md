# DVC
[Документация](https://dvc.org/doc)


- [Cheat Sheet](#cheat-sheet)
    - [Basics](#basics)
    - [The pipeline](#the-pipeline)
    - [Other commands](#other-commands)
- [FAQ](#faq)
    - [Подключение хранилища](#подключение-хранилища)
        - [Как подключить хранилище на диске (локально)](#как-подключить-хранилище-на-диске-локально)
        - [Как подключить гугл диск в качестве хранилища?](#как-подключить-гугл-диск-в-качестве-хранилища)
        - [Как подключить S3 бакет?](#как-подключить-s3-бакет)
        - [Как подключить удаленное SSH хранилище?](#как-подключить-удаленное-ssh-хранилище)
    - [Работа с хранилищем](#работа-с-хранилищем)
        - [Как работать с несколькими хранилищами?](#как-работать-с-несколькими-хранилищами)
    - [Общие вопросы](#общие-вопросы)
        - [Что происходит после команды dvc init (в деталях)?](#что-происходит-после-команды-dvc-init-в-деталях)
        - [Зачем нужен "dvc commit" если есть "dvc add && git add && git commit"?](#зачем-нужен-dvc-commit-если-есть-dvc-add--git-add--git-commit)
        - [Случайно добавил в отслеживание, как убрать?](#случайно-добавил-в-отслеживание-как-убрать)



# Cheat sheet

## Basics
| Section | Command Description | Syntax | Notes |
|---------|---------------------|--------|-------|
| **Initializing** | Initialize a DVC environment | `$ dvc init` | |
| **Remote** | Set up a remote to keep and share data files | `$ dvc remote add -d myremote /path` | *Possible remotes include local, s3, gs, azure, ssh, hdfs, and http* |
| | Show all available remotes | `$ dvc remote list` | |
| | Modify remote settings | `$ dvc remote modify myremote` | *Use if remote requires extra configuration* |
| **Adding Files** | Add files under DVC control | `$ dvc add filename` | *Use `--no-commit` to stop adding the file to the cache* |
| **Share Data** | Push all data files to the remote storage | `$ dvc push` | |
| | Push outputs of a specific .dvc file | `$ dvc push filename.dvc` | |
| **Retrieve Data** | Download files from the remote storage | `$ dvc pull` | |
| | Download files from a specific .dvc file | `$ dvc pull filename.dvc` | |
| | Checkout files from cache into working space | `$ dvc checkout` | |


## The pipeline
| Section | Command Description | Syntax | Notes |
|---------|---------------------|--------|-------|
| | Add transformations and generate a stage file from a given command | `$ dvc run -d dependencyfile -o outputfile python command.py` | Use `--file` to specify the name of the generated .dvc file. Use `--metrics` to output a file containing the metric |
| **Metrics** | Collect and display project metrics | `$ dvc metrics show` | Use `=all` to show the metrics in all branches |
| **Visualizing** | Show stages in a pipeline | `$ dvc pipeline show --ascii file.dvc` | Add `--commands` or `o r` to show more detail |
| | Show connected pipelines of DVC stages | `$ dvc pipeline list` | |
| **Reproducing** | Reproduce outputs defined in .dvc file | `$ dvc repro filename.dvc` | Name a .dvc file "a/dile" to be use by `dvc repro` by default |



## Other commands
| Section | Command Description | Syntax | Notes |
|---------|---------------------|--------|-------|
| | Set/unset cache directory location | `$ dvc cache dir /path` | |
| | Commit outputs to cache | `$ dvc commit` | Use if you specified `--no-commit` in `dvc add/run/repro` |
| | Config repository or global options | `$ dvc config` | Config the default remote using `core.remote myremote`. Config `core (loglevel, remote), cache` and `state` settings |
| | Fetch files from the remote to the local cache | `$ dvc fetch file.dvc` | |
| | Remove unused objects from cache | `$ dvc gc` | |
| | Import file from URL to local directory | `$ dvc import url /path` | Supported schemes include `local`, `s3`, `gs`, `azure`, `ssh`, `hdfs`, and `http`. |
| | Remove data files tracked by dvc | `$ dvc remove filename.dvc` | |
| | Show changed stages in the pipeline | `$ dvc status` | |



# FAQ


## Подключение хранилища

### Как подключить хранилище на диске (локально)

```bash
dvc remote add -d storage /path/to/your/storage
```

### Как подключить гугл диск в качестве хранилища?
Для начала необходимы плагины для аутентификации:

```bash
pip install dvc[gdrive]
```

Затем созадаем у себя на диске директорию (называйте как хотите) и переходим в нее, 
далее смотрим на урл и копируем последнюю часть, например в
`https://drive.google.com/drive/folders/1Z3JfbS00SLrhHPVh7igikSy4Dbbug-z`
часть *1Z3JfbS00SLrhHPVh7igikSy4Dbbug-z* и является идентификатором.

Далее:
```bash
dvc remote add -d storage gdrive://1Z3JfbS00SLrhHPVh7igikSy4Dbbug-z
```

В терминале появится урл авторизации, переходим по нему и кликаем доступы.


### Как подключить S3 бакет?

Для начала понадобятся плагины для работы с s3:
```bash
pip install dvc[s3]
```

Добавляем хранилище:
```bash
dvc remote add -d myminio s3://mlops/
```

Далее понадобятся некоторые настройки:
```bash
dvc remote modify myminio endpointurl http://localhost:9000
dvc remote modify myminio access_key_id <...>
dvc remote modify myminio secret_access_key <...>
```


### Как подключить удаленное SSH хранилище?

Добавляем:
```bash
dvc remote add -d mysshremote ssh://user@hostname/path/to/directory
```

Если подключение идет по ssh ключику:
```bash
dvc remote modify mysshremote keyfile /path/to/private/key
```

Если логинишься по паролю, то:
```bash
dvc remote modify mysshremote password "парольчик"
```

Не забываем указать порт (если не знаешь порт, попробуй `22`):
```bash
dvc remote modify mysshremote port 777
```


## Работа с хранилищем
### Как работать с несколькими хранилищами?

Если у вас несколько хранилищ, то команда `dvc push` запушит в определенное хранилище
(то есть в то, которое вы заводили с флагом `-d`). 

Если у вас настроено несколько удалённых хранилищ и хочется отправить данные в 
конкретное из них, то этого можно достичь, указав имя нужного  хранилища с 
использованием опции `-r` (или `--remote`) в команде `dvc push`:
```bash
dvc push -r storagename
```

Имена удаленных хранилищ всегда можно посмотреть командой:
```bash
dvc remote list
```


## Общие вопросы

### Что происходит после команды dvc init (в деталях)?
Этапы (под капотом):
- Валидация
    - проверятся находится ли директория в проекте, в котором уже инициализирован git
    - проверяется что DVC ранее не был инициализирован в проекте
- Создание директории .dvc/
    - в корне проекта создается директория с именем .dvc/. Она будет использоваться для хранения всей конфигурации DVC, временных файлов и кеша
- Конфигурация локальной директории для кеша:
    - внутри .dvc/, DVC настраивает кеш-директорию для хранения ссылок на фактические файлы, которые вы потом будете отслеживать с помощью DVC
- Генерация файлов конфигурации
    - .dvc/config: создается для хранения конфигурационной информации для DVC. Он может включать в себя детали о удаленном хранилище, местоположении кеш-директорий и других настройках
- Интеграия с гитом:
    - меняется .gitignore чтобы убедиться, что файлы данных и внутренние файлы DVC не трекаются гитом


### Зачем нужен "dvc commit" если есть "dvc add && git add && git commit"?
Давайте разберемся:

- `dvc add` используется когда нужно чтобы DVC стал отслеживать файл. Команда перемещает данные в кэш DVC и создает DVC-файл (.dvc файл), который указывает на данные.

- `dvc commit` используется для сохранения обновленной версии файла в кэше DVC без его повторного добавления. Эта команда обновляет DVC-файлы и сохраняет новую версию файла в вашем кэше DVC.

- `git add & git commit` используются для сохранения изменений в коде, DVC-файлах или других файлах, отслеживаемых git.


Предположим, что вы модифицируете какой либо файл (уже отслеживаемый в dvc):

- `dvc status` покажет, что файл изменен
- Чтобы отразить это изменение в DVC, вы могли бы снова использовать dvc add, но это будет неэффективно для крупных наборов данных.


### Случайно добавил в отслеживание, как убрать?
Можно отменить `dvc add` через `dvc remove`. 
Команда удаляет файл .dvc (и соответствующую запись в .gitignore). 
Файл данных теперь больше не отслеживается после этого:

```bash
dvc remove data.csv.dvc
```

```bash
git status
    Untracked files:
        data.csv
```

Чтобы почистить кеш, можно выполнить команду `dvc gc` с опцией `-w` (при этом так же будут удалени и все предыдущие версии файла, если они есть):

```bash
dvc gc -w
```

