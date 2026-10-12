# Что здесь будет: небольшой сервер FastAPI в одном файле.
#
# Содержимое:
# 1. Импорты FastAPI, Pydantic, pandas и функций из predict.py.
# 2. app = FastAPI().
# 3. Pydantic-класс строки запроса: date (дата), store_nbr (целое > 0),
#    family (непустая строка), onpromotion (целое >= 0).
# 4. Pydantic-класс запроса с непустым списком rows.
# 5. Однократная загрузка model.joblib при запуске сервера.
#    Сначала нужно реализовать train.py и получить этот файл.
# 6. GET /health: вернуть {"status": "ok"}, если модель загружена.
# 7. POST /predict: преобразовать rows в DataFrame, вызвать predict_sales
#    и вернуть числа через predictions.tolist().
#
# Пример входного JSON:
# {"rows": [{"date": "2017-08-16", "store_nbr": 1,
#            "family": "GROCERY I", "onpromotion": 0}]}
# Пример выходного JSON: {"predictions": [123.45]}.
# Порядок чисел в ответе совпадает с порядком rows в запросе.
# sales и готовые признаки клиент не передает: они для запроса не нужны.
# Если магазин неизвестен или для даты нельзя построить признаки,
# верни понятную ошибку через HTTPException.
#
# После реализации установи fastapi и uvicorn в свое окружение.
# Запуск из корня проекта: uvicorn app:app --reload.
# Форма для ручной проверки будет по адресу http://127.0.0.1:8000/docs.
# Сейчас это только комментарии: приложение и эндпоинты еще не написаны.

import pandas as pd
from datetime import date
from fastapi import FastAPI, HTTPException
from typing import Annotated, List
from pydantic import BaseModel, field_validator, Field
from predict import load_model, predict_sales

# Инициализируем FastAPI
app = FastAPI()

# Создаем pytdantic класс строки запроса и самого запроса
class DataItem(BaseModel):
    date: date
    store_nbr: int
    family: str
    onpromotion: int
    
    @field_validator('store_nbr')
    def check_store_nbr(cls, value):
        if value <= 0:
            raise ValueError("Номер магазина должен быть больше нуля.")
        return value
    
    @field_validator('onpromotion')
    def check_onpromotion(cls, value):
        if value < 0:
            raise ValueError("onpromotion Должен быть больше либо равен нулю")
        return value
        
    @field_validator('family')
    def check_family(cls, value):
        value = value.strip()
        if not value:
            raise ValueError("Некорректная товарная категория")
        return value
class RequestModel(BaseModel):
    rows: Annotated[List[DataItem], Field(min_length=1)]
    
# Загружаем модель.
try:
    model = load_model()
except Exception as e:
    raise RuntimeError(
        "Не удалось загрузить model.joblib. Проверьте файл модели.\n Если файл отсутствует, запустите train.py"
    ) from e
    
@app.get("/health")
async def health():
    return {
        "status": "ok"
    }

@app.post("/predict")
def predict(request: RequestModel):
    
    input_data = [row.model_dump() for row in request.rows]
    df = pd.DataFrame(input_data)
    
    try:
        predictions = predict_sales(df, model)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e)) from e
    
    return {
        "predictions": predictions.tolist()
        }