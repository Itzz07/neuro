import pandas as pd
import numpy as np
from neuralprophet import NeuralProphet
from sklearn.metrics import root_mean_squared_error, mean_absolute_error
import pickle

def train_and_predict():
    # Load and preprocess data
    df = pd.read_csv('data/Manual-Station.csv')
    df['Rainfall'] = df['Rainfall'].replace(' ', np.nan)
    df.dropna(subset=['Rainfall'], inplace=True)
    
    val_obs = df.copy()
    val_obs['DateTime'] = pd.to_datetime(df['Timestamp'], format='%d/%m/%Y %H:%M')
    val_obs['Rainfall'] = pd.to_numeric(val_obs['Rainfall'], errors='coerce')
    
    # Feature engineering
    daily_rainfall = val_obs.groupby('DateTime')['Rainfall'].mean().reset_index()
    daily_rainfall.dropna(inplace=True)
    
    # Train-test split
    train_data = daily_rainfall[daily_rainfall['DateTime'] <= pd.to_datetime('2020-09-30')]
    test_data = daily_rainfall[(daily_rainfall['DateTime'] > pd.to_datetime('2020-09-30')) & 
                              (daily_rainfall['DateTime'] <= pd.to_datetime('2021-03-31'))]
    
    train_data.columns = ['ds', 'y']
    test_data.columns = ['ds', 'y']
    
    # Model configuration
    confidence_level = 0.9
    boundaries = round((1 - confidence_level) / 2, 2)
    quantiles = [boundaries, confidence_level + boundaries]
    
    # Train model
    m = NeuralProphet(quantiles=quantiles)
    m.fit(train_data, freq='M')
    
    # Save model
    with open('models/trained_model.pkl', 'wb') as f:
        pickle.dump(m, f)
    
    # Make predictions
    future = m.make_future_dataframe(train_data, periods=len(test_data))
    forecast = m.predict(future)
    
    # Calculate metrics
    rmse = root_mean_squared_error(test_data['y'], forecast['yhat1'])
    mae = mean_absolute_error(test_data['y'], forecast['yhat1'])
    mape = np.mean(np.abs((test_data['y'] - forecast['yhat1'].to_list()) / test_data['y'])) * 100

    return {
        'rmse': rmse,
        'mae': mae,
        'mape': mape,
        'actual': test_data.to_dict('list'),
        'predicted': forecast.to_dict('list')
    }