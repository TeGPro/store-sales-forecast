import pandas as pd
import numpy as np
import joblib
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import root_mean_squared_log_error
from data_preparation import prepare_data
from xgboost import XGBRegressor

if __name__ == '__main__':
    df = pd.read_csv('data/train.csv')
    df['date'] = pd.to_datetime(df['date'])
    df = df[df['date'] >= '2013-01-02'].copy()

    # Дальше — разбиение df на train и valid
    # Затем — вызов prepare_data для каждой части
    
    validation_start = (
        df['date'].max() - pd.Timedelta(days=15)
    )

    df_train = df[
        df['date'] < validation_start
    ].copy()

    df_valid = df[
        df['date'] >= validation_start
    ].copy()
    
    x_train = prepare_data(df_train)
    x_valid = prepare_data(df_valid)
    
    y_train = df_train['sales'].copy()
    y_valid = df_valid['sales'].copy()

    y_train_log = np.log1p(y_train)
    y_valid_log = np.log1p(y_valid)
    
    categorical_cols = ['family', 'city', 'state', 'type']

    preprocessor = ColumnTransformer(
        transformers=[
            (
                'categories',
                OneHotEncoder(
                    handle_unknown='ignore',
                    sparse_output=True
                ),
                categorical_cols
            )
        ],
        remainder='passthrough',
        sparse_threshold=1.0
    )

    x_train_encoded = preprocessor.fit_transform(x_train)
    x_valid_encoded = preprocessor.transform(x_valid)

    feature_names = preprocessor.get_feature_names_out()
    
    model_xgboost = XGBRegressor(
    objective='reg:squarederror',
    n_estimators=1000,
    learning_rate=0.05,
    tree_method='hist',
    device='cuda',
    eval_metric='rmse',
    early_stopping_rounds=50,
    random_state=42
    )

    model_xgboost.fit(
        x_train_encoded,
        y_train_log,
        eval_set=[(x_valid_encoded, y_valid_log)],
        verbose=100
    )
    
    model_xgboost.set_params(device='cpu')

    predictions_log = model_xgboost.predict(x_valid_encoded)
    predictions = np.expm1(predictions_log)

    predictions = np.maximum(predictions, 0)

    model_score = root_mean_squared_log_error(
        y_valid,
        predictions
    )
    
    print('Количество деревьев на лучшей итерации:',
        model_xgboost.best_iteration + 1)
    print('XGBoost:', model_score)
    
    data_to_save = {
        "model": model_xgboost,
        "preprocessor": preprocessor,
        "feature_columns": x_train.columns.tolist()
    }
    
    with open('model.joblib', 'wb') as f:
        joblib.dump(data_to_save, f)