import pandas as pd

def prepare_data(df):
    # Загрузим данные и обработаем.
    
    # df_events
    
    df_events = pd.read_csv('data/holidays_events.csv')
    df_events['date'] = pd.to_datetime(df_events['date'])
    
    # df_oil
    
    df_oil = pd.read_csv('data/oil.csv')
    df_oil['date'] = pd.to_datetime(df_oil['date'])
    df_oil = df_oil.sort_values('date')
    df_oil = df_oil.set_index('date')
    df_oil = df_oil.reset_index()

    # df_stores
    
    df_stores = pd.read_csv('data/stores.csv')
    df_stores = df_stores.set_index('store_nbr')
    
    # основной df
    
    df = df.copy()
    df['date'] = pd.to_datetime(df['date'])
    
    # далее подготовим обучающие и тестовые выборки.
    
    df_sales_stores = pd.merge(
    df,  # сохраняем id продажи как столбец
    df_stores,
    how='left',
    on='store_nbr',
    validate='many_to_one'
    )
    
    if df_sales_stores['city'].isna().any():
        raise ValueError('В запросе есть неизвестный магазин.')

    df_sales_oil = pd.merge(
    df_sales_stores,
    df_oil,
    how='left',
    on='date',
    validate='many_to_one'
    )
    
    national_events = (
    df_events[df_events['locale'] == 'National']
    .groupby('date')
    .size()
    .rename('national_event_count')
    .reset_index()
    )

    df_features = pd.merge(
        df_sales_oil,
        national_events,
        how='left',
        on='date',
        validate='many_to_one'
    )

    df_features['national_event_count'] = (
        df_features['national_event_count'].fillna(0).astype('int16')
    )
    
    local_events = (
    df_events[df_events['locale'] == 'Local']
    .groupby(['date', 'locale_name'])
    .size()
    .rename('local_event_count')
    .reset_index()
    .rename(columns={'locale_name': 'city'})
    )

    df_features = pd.merge(
        df_features,
        local_events,
        how='left',
        on=['date', 'city'],
        validate='many_to_one'
    )

    df_features['local_event_count'] = (
        df_features['local_event_count'].fillna(0)
    )
    
    regional_events = (
    df_events[df_events['locale'] == 'Regional']
    .groupby(['date', 'locale_name'])
    .size()
    .rename('regional_event_count')
    .reset_index()
    .rename(columns={'locale_name': 'state'})
    )

    df_features = pd.merge(
        df_features,
        regional_events,
        how='left',
        on=['date', 'state'],
        validate='many_to_one'
    )

    df_features['regional_event_count'] = (
        df_features['regional_event_count'].fillna(0)
    )
    
    oil_daily = (
    df_oil[['date', 'dcoilwtico']]
    .set_index('date')
    .sort_index()
    .asfreq('D')
    )

    oil_daily['dcoilwtico'] = oil_daily['dcoilwtico'].ffill()
    
    df_features = pd.merge(
    df_features.drop(columns=['dcoilwtico']),
    oil_daily.reset_index(),
    how='left',
    on='date',
    validate='many_to_one'
    )
    if df_features['dcoilwtico'].isna().any():
        raise ValueError('Для некоторых дат нет известной цены нефти')

    # Разобъем на обучающую и валидационную выборки
    
    def prepare_features(data):
        
        features = data.drop(columns=['id', 'sales'], errors='ignore').copy()

        features['year'] = features['date'].dt.year
        features['month'] = features['date'].dt.month
        features['day'] = features['date'].dt.day
        features['day_of_week'] = features['date'].dt.dayofweek

        return features.drop(columns=['date'])
    
    return prepare_features(df_features)