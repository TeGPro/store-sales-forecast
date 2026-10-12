import joblib
import numpy as np
import pandas as pd
from data_preparation import prepare_data

def load_model(path="model.joblib"):
    with open(path, 'rb') as f:
        saved_model = joblib.load(f)
    return saved_model
        
def predict_sales(df, saved_model):
    df = prepare_data(df)
    
    required_columns = saved_model['feature_columns']
    missing_columns = [col for col in required_columns if col not in df.columns]
    
    if missing_columns:
        raise ValueError(f"В датафрейме отсутствуют необходимые колонки: {missing_columns}")
    
    df = df[required_columns]
    
    preprocessor = saved_model['preprocessor']
    df_transformed = preprocessor.transform(df)
    model = saved_model['model']
    
    predictions_log = model.predict(df_transformed)
    predictions = np.expm1(predictions_log)
    predictions = np.maximum(predictions, 0)
    return predictions

if __name__ == '__main__':
    df = pd.read_csv('data/test.csv')
    saved_model = load_model()
    predictions = predict_sales(df, saved_model)

    submission = pd.DataFrame({
        'id': df['id'],
        'sales': predictions
    })
    submission.to_csv('submission.csv', index=False)
    print(submission.head())