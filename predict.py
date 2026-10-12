# Что здесь будет: загрузка модели и получение прогноза.
#
# Сделай две функции:
# load_model(path="model.joblib") — читает сохраненный словарь через joblib.load.
# predict_sales(df, saved_model) — возвращает прогноз для каждой строки.
#
# Вход predict_sales: DataFrame с date, store_nbr, family, onpromotion
# и словарь с model, preprocessor, feature_columns из train.py.
# Внутри predict_sales:
# 1. Вызови ту же prepare_data из data_preparation.py.
# 2. Проверь наличие нужных колонок и поставь их в порядок feature_columns.
# 3. Вызови сохраненный preprocessor.transform (не fit_transform).
# 4. Вызови model.predict, затем np.expm1 и ограничь результат снизу нулем.
# Выход: numpy-массив неотрицательных чисел в исходном порядке строк.
#
# Для простой ручной проверки под if __name__ == "__main__": прочитай data/test.csv,
# загрузи модель, получи прогноз и сохрани CSV submission.csv с колонками id,sales.
# После реализации запуск: python predict.py.
# API будет использовать эти же функции, поэтому обучение сюда не переносим.
# Сейчас это только комментарии, прогноз не реализован.
