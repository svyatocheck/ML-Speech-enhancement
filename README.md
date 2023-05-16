
# CNN for speech denoising
Домашнее задание по курсу [My First Data Project](https://ods.ai/tracks/my_first_data_project).

Основная цель представленного проекта - решить задачу очистки речи от фоновых шумов, используя технологии машинного обучения.

В описании ниже представлено обоснование выбранной архитектуры, приведены результаты экспериментов по подбору оптимальных гиперпараметров, описан процесс обучения нейронной сети.

## Что изменилось? (10.05.2023)
- Добавлена новая модель - CRNN. Все еще экспериментирую с ее архитектурой и параметрами. Постараюсь в кратчайшие сроки доработать текст с приведением результатов и полного сравнения
- Поменял структуру проекта, по совету ментора.
- Добавил полный список библиотек.

## Основные библиотеки

- librosa
- numpy
- tensorflow

Подробнее в файлике requirements.txt

## Содержание

В репозитории находится три отдельных модуля:
- cnn_denoising - CNN (baseline) для обработки аудио. 
- crnn_denoising - CRNN для обработки аудио.
- dataset_creation - скрипт для наложения записей речи на различные шумы. Показалось, что логически верным будет вынести его в отдельный модуль и отдельным этапом.

## Dataset

Машинное обучение начинается с данных. Итоговый датасет для задачи был сгенерирован на основе следующих наборов:
- Common Voice (https://commonvoice.mozilla.org/ru/datasets). Имеет большое число проверенных аудиозаписей - зачитанных спикерами текстов.
- UrbanSound8K (https://www.kaggle.com/datasets/chrisfilo/urbansound8k). Большой объем разнотипных городских шумов.
- Microsoft Scalable Noisy Speech dataset  (https://github.com/microsoft/MS-SNSD). Большой объем разных звуков, записей речи. Есть разбивка на тестовые и тренировочные данные. Изначально брал для тестирования модели, но позже включил в основной датасет.

Из каждой аудиозаписи отсекается тихий участок (в начале и конце аудиозаписи). Далее, на записи голоса (1 и 3 датасеты) накладываются выбранные случайным образом (из 2 и 3 датасета) шумы, с параметром SNR в диапазоне от 0 до 40 db. Все эти шаги выполняются скриптом из dataset_creation.
Дальше, при запуске обучения одной из моделей загружаются уже обработанные аудио и с преобразованием Фурье конвертируются в STFT magnitude vectors, которые, после некоторых дополнительных шагов, подаются в сеть (детали описаны в статье, и откомментируются в коде).

*SNR (Signal-to-Distortion) - отношение мощности полезного сигнала к мощности шума.

Детали описаны в статье [1][2], ссылки в конце документа.

## Подбор гиперпараметров

Сперва стоит отметить, что автор работы столкнулся с техническими ограничениями и не смог организовать обучение моделей с большей вариативностью в значениях гиперпараметров. Эксперименты будут продолжены.

### Инструменты
В качестве основного инструмента версионирования экспериментов была взята платформа Weights and Biases. Процесс подбора гиперпараметров проводился с использованием ее особенностей [sweeps](https://docs.wandb.ai/guides/sweeps) и метода баесовской оптимизации.
Все эксперименты доступны к просмотру в проекте speech_denoising и crnn_denoising (можно найти в профиле - [ссылка](https://wandb.ai/sams3pi01?shareProfileType=copy)).

Поговорим о самом процессе: 

### Архитектура нейронной сети

#### CNN
Пара слов об архитектуре, поскольку она имеет прямое влияние на гиперпараметры. 

![image](https://user-images.githubusercontent.com/63301430/232900314-fc11581a-f943-4477-b2fe-03eed911203a.png)

Архитектура заимствована из уже упомянутой выше статьи [1] (следующие снимки оттуда) по нескольким причинам:
- Исследование, описанное в работе, проводилось с целью найти оптимальное с точки зрения числа параметров и качества решение, которое бы функционировало в условиях серьезных аппаратных ограничений (в статье - на слуховом аппарате).
- В исследовании проведено сравнение нескольких архитектур нейронных сетей, выбор был сделан по его итогам.
- CNN демонстрирует высокую производительность в том числе на задачах связанных с обработкой речи, что полезно при размещении модели на смартфоне и обучении модели.
- Уровень сложности реализации.
- Автор данной статьи проявляет интерес к computer vision и обработке изображений. 

*Сравнение работы разных архитектур нейронных сетей в статье:

![image](https://user-images.githubusercontent.com/63301430/232899470-855f6231-0a31-487b-9fbc-e93d7b498d96.png)

![image](https://user-images.githubusercontent.com/63301430/232899557-78e334ed-e9e0-44b2-a8d0-4c10be62fd6a.png)

Выбрана CR-CED (skip), т.к. модель демонстрирует неплохие результаты, имея при этом наименьшее число параметров.

Таким образом, число нейронов и число слоев, metric - rmse (широко распространенная для задач регресии, к слову), activation - adam, loss - mse взяты из статьи.

#### CRNN

<img src=https://github.com/Svyatocheck/ML-Speech-enhancement/assets/63301430/57b8640f-9a5d-4cd6-acf4-af8b273fa1d5 width=220>

Вдохновлением при проектировании данной н.с. послужило другое исследование [2]. Там не так подробно расписано о ее устройстве, поэтому есть очень много пространства для творчества. 
Причины, по которым я взял эту архитектуру:
- Опять же, возможность разместить ее на мобильном устройстве.
- Не изученные мной до этого момента слои LSTM, мне было интересно с ними поработать.
- Улучшенное качество обработанных файлов, что неудивительно, если сравнить число параметров: 2,5М vs 35k. 

Ожидаемое значение метрик качества обработанной аудио:

![image](https://github.com/Svyatocheck/ML-Speech-enhancement/assets/63301430/39019a33-7f1c-4e0a-b8a3-1ae36406800e)

*PESQ and STOI

Metric и loss функция взята из статьи, как и функция активации.

### Размер Датасета

Ориентируюсь на 5000 аудио в обоих Н.С. Датасет разбит на чанки по 512 файлов + тестовые и валидационные. 

### Batch-size

Взял 64. Таким образом, и модель учится быстрее и памяти при обучении кушается не так много. 

### Learning rate

Одно из самых важных открытий в этой части: я научился понижать значения learning rate в процессе обучения модели, после заданного числа эпох.

Выглядит это так:

![W B Chart 5_9_2023, 6 09 03 PM](https://github.com/Svyatocheck/ML-Speech-enhancement/assets/63301430/315f1196-1599-4eed-9b07-1ed31ce07a8b)

Но это не отменяет понижение стартового значения lr по мере "скармливания" чанков в н.с.

### Epochs

Взял по 10 эпох на каждый чанк в обоих моделях. Не увидел никаких проблем с этим значением. Вот график с CRNN

![W B Chart 5_9_2023, 6 16 17 PM](https://github.com/Svyatocheck/ML-Speech-enhancement/assets/63301430/040435eb-6383-4a89-b2d9-7d30562f5e1f)

Видим, что ошибка плавно снижается по мере скармливания датасета. Похожая картина и с CNN.

## Итоговые сравнения

CRNN: средние STOI: 0.9146879230592444 и PESQ: 1.3276347517967224
CNN: средние STOI: 0.8152701249957266 и PESQ: 1.212430715560913

Таким образом, четко видна разница между CNN и CRNN, в пользу последней.

## References
- https://paperswithcode.com/paper/a-fully-convolutional-neural-network-for
- https://www.ncbi.nlm.nih.gov/pmc/articles/PMC8064406/
- https://habr.com/ru/post/668518/
- https://habr.com/ru/companies/antiplagiat/articles/528384/
- https://www.youtube.com/watch?v=ZqpSb5p1xQo&list=WL&index=8&t=1072s
- https://www.mathworks.com/help/deeplearning/ug/denoise-speech-using-deep-learning-networks.html
- https://www.kaggle.com/code/danielgraham1997/speech-denoising-analysis#Metrics-Analysis
- https://www.kaggle.com/code/carlolepelaars/bidirectional-lstm-for-audio-labeling-with-keras/notebook
