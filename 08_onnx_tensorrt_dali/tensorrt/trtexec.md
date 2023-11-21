# TensorRT


## TensorRT docker

Сперва нам потребуется контейнер с нужной версией tensorrt. 
Рекомендую делать именно через контейнеры:

```bash
docker pull nvcr.io/nvidia/tensorrt:23.04-py3
```

Далее запускаем контейнер в интерактивном режиме монтируя директорию с onnx моделями
(вместо device=2 поставьте индекс нужного gpu):

```bash
docker run -it --rm --gpus '"device=2"' -v ./models:/models nvcr.io/nvidia/tensorrt:23.04-py3
```

## ONNX Operators

Список поддерживаемых операторов и их версии (аргумент opset_version в torch.onnx.export) можно посмотреть [тут](https://github.com/onnx/onnx/blob/main/docs/Operators.md).

Релизы onnx [тут](https://github.com/onnx/onnx/releases).


## Steps

Сначала нам нужен `.onnx` файл модели. Способы компилирования в trt напрямую из pytorch не рекомендую,
так как на практике вероятнее всего вам придется работать с несколькими рантайм-окружениями с разными версиями
драйверов, а так же с разными версиями фреймворков которые запускают скомпилированную модель.

Да и в целом, запуск в изолированном окружении как правило удобнее, и более того, так как мы работаем с официальным контейнером от Nvidia, это гарантирует нам, что окружение в нем не консистентное.


Шаги в целом можно описать такой последовательностью:

1. Экспорт модели в `ONNX`
    - необходим класс отнаследованный от `torch.nn.Module`
    - определиться с выбранными динамическими осями
    - сгенерировать семпл-тензор для построения графа модели
        - подойдет случайный тензор
        - даже если батч предполагается динамическим, тут его можно ставить в 1
        - если модель принимает на вход кортеж, то нужен кортеж из тензоров
        - девайс всегда cpu
    - запустить torch.onnx.export, дождаться завершения
    - если баги
        - если бесконечно долго работает
            - такое бывает на двухсокетных системах, запустить через `taskset -c 0-16 python script.py`
        - если падает с непонятными ошибками
            - проверьте версии операторов `opset_version=...`
                - некоторые операторы не поддерживаются на старых версиях
            - если модель композитная, попробуйте разбить ее на части и сконвертировать части отдельно друг от друга
            - бывают случаи со сложностью экспортирования моделей с ГСЧ, решатся гуглом и просмотром issues на гите
    - Опционально можно изучить как выглядит граф модели через UI [netron.app](https://netron.app/)
2. Запуск trtexec
    - Запускам докер контейнер в интерактивном режиме как показано выше
    - Смотрим хелпу команды trtexec: `trtexec -h`. Если запускаете впервые, то рекомендую внимательно изучить что там написано.
    - Пример команды для компиляции в .plan в fp32 и в fp16 ниже.


```bash
# FP32
trtexec --onnx=/models/resnet50.onnx \
--saveEngine=/models/resnet50_FP32_AMPERE+.plan \
--minShapes=IMAGES:1x3x224x224 \
--optShapes=IMAGES:8x3x224x224 \
--maxShapes=IMAGES:16x3x224x224 \
--inputIOFormats=fp32:chw \
--profilingVerbosity=detailed \
--builderOptimizationLevel=5 \
--hardwareCompatibilityLevel=ampere+

# FP16
trtexec --onnx=/models/resnet50.onnx \
--saveEngine=/models/resnet50_FP16_AMPERE+.plan \
--minShapes=IMAGES:1x3x224x224 \
--optShapes=IMAGES:8x3x224x224 \
--maxShapes=IMAGES:16x3x224x224 \
--inputIOFormats=fp32:chw \
--fp16 \
--profilingVerbosity=detailed \
--builderOptimizationLevel=5 \
--hardwareCompatibilityLevel=ampere+
```
