# Store Sales Forecast

Прогнозирование продаж магазинов на данных [Kaggle Store Sales](https://www.kaggle.com/competitions/store-sales-time-series-forecasting). Прогноз доступен через FastAPI и CSV.

## Файлы проекта

- `research.ipynb` — исследование и эксперименты.
- `data_preparation.py` — общая подготовка признаков.
- `train.py` — обучение и сохранение модели.
- `predict.py` — загрузка модели и прогноз.
- `app.py` — API: `/health` и `/predict`.
- `Dockerfile`, `.dockerignore` — сборка контейнера.

## Подготовка

Нужны Python 3.14 и CSV из соревнования в папке `data/`: `train.csv`, `test.csv`, `stores.csv`, `oil.csv`, `holidays_events.csv`. Все команды выполняются из корня проекта.

Установка в PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

Если готового `model.joblib` нет, обучите модель:

```powershell
python train.py
```

Скрипт выводит RMSLE и сохраняет модель с препроцессором в `model.joblib`. Обучение настроено на CUDA; для CPU замените `device='cuda'` на `device='cpu'` в `train.py`.

## Запуск API

```powershell
python -m uvicorn app:app --reload
```

- Документация и отправка запросов: http://127.0.0.1:8000/docs
- Проверка состояния: http://127.0.0.1:8000/health

Пример тела запроса `POST /predict`:

```json
{
  "rows": [
    {"date": "2017-08-16", "store_nbr": 1, "family": "GROCERY I", "onpromotion": 0}
  ]
}
```
## Запуск через Docker

```powershell
docker build -t store-sales-api .
docker run --rm -p 127.0.0.1:8000:8000 --name store-sales-api store-sales-api
```
## Прогноз в CSV

```powershell
python predict.py
```

Читает `data/test.csv` и создаёт `submission.csv` с колонками `id,sales`.
