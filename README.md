# store-sales-forecast
В этом репозитории будет исследование датасета и обучение модели.
За основу взяты данные из соревнования на [kaggle](https://www.kaggle.com/competitions/store-sales-time-series-forecasting/overview/citation)

## Структура

```text
store-sales-forecast/
├── data/                 # исходные CSV
├── research.ipynb        # исследование и текущая модель
├── data_preparation.py   # подготовка признаков
├── train.py              # обучение и сохранение модели
├── predict.py            # прогноз и ручная проверка на test.csv
├── app.py                # FastAPI
├── requirements.txt      # зависимости
└── README.md
```