import pandas as pd
import numpy as np
from neuralprophet import NeuralProphet
from sklearn.metrics import *
import joblib
import json
import time

def preprocess_data():
    df = pd.read_csv('data/Manual-Station.csv')
    df['Rainfall'] = df['Rainfall'].replace(' ', np.nan)
    df.dropna(subset=['Rainfall'], inplace=True)
    df['DateTime'] = pd.to_datetime(df['Timestamp'], format='%d/%m/%Y %H:%M')
    return df

def train_model(train_data):
    
    # Set Up Confidence Level and Quantiles 
    confidence_level = 0.9
    boundaries = round((1 - confidence_level) / 2, 2)
    quantiles = [boundaries, confidence_level + boundaries]

    m = NeuralProphet(quantiles=quantiles)
    metrics = m.fit(train_data, freq='M')
    return m, metrics

def generate_predictions(model, periods, train_data):
    future = model.make_future_dataframe(train_data, periods=periods)
    forecast = model.predict(future)
    return forecast

def save_results(results):
    joblib.dump(results, 'models/predictions.joblib')
    with open('models/metrics.json', 'w') as f:
        json.dump({
            'period_metrics': results['period_metrics'],
            'overall': results['overall'],
            'last_updated': time.time()
        }, f)

def main():
    start_time = time.time()
    
    # Data processing
    df = preprocess_data()
    daily_rainfall = df.groupby('DateTime')['Rainfall'].mean().reset_index()
    
    # Define test periods
    periods = [
        ('2020-09-30', '2021-03-31'),
        ('2021-09-30', '2022-03-31'),
        ('2022-09-30', '2023-03-31')
    ]
    
    results = {'graphs': {}, 'period_metrics': {}, 'overall': {}}
    all_actual = []
    all_predicted = []
    
    for i, (train_end, test_end) in enumerate(periods):
        # Data splitting
        train_data = daily_rainfall[daily_rainfall['DateTime'] <= pd.to_datetime(train_end)]
        test_data = daily_rainfall[(daily_rainfall['DateTime'] > pd.to_datetime(train_end)) & 
                                  (daily_rainfall['DateTime'] <= pd.to_datetime(test_end))]
        
        # Model training
        train_data_clean = train_data.dropna().rename(columns={'DateTime': 'ds', 'Rainfall': 'y'})
        test_data_clean = test_data.dropna().rename(columns={'DateTime': 'ds', 'Rainfall': 'y'})
        
        model, metrics = train_model(train_data_clean)
        forecast = generate_predictions(model, len(test_data_clean), train_data_clean)
        print(f"{forecast.columns} /n/n")

        # Store results
        period_key = f"period_{i+1}"
        results['graphs'][period_key] = {
            'actual': test_data_clean[['ds', 'y']].to_dict('list'),
            'predicted': forecast[['ds', 'yhat1']].to_dict('list'),
            'yhat1_5': forecast[['ds', 'yhat1 5.0%']].to_dict('list'),
            'yhat1_95': forecast[['ds', 'yhat1 95.0%']].to_dict('list')
        }
        
        results['period_metrics'][period_key] = {
            'rmse': root_mean_squared_error(test_data_clean['y'], forecast['yhat1']),
            'mae': mean_absolute_error(test_data_clean['y'], forecast['yhat1']),
            'mape': np.mean(np.abs((test_data_clean['y'] - forecast['yhat1'].to_list()) / test_data_clean['y'])) * 100
        }
        
        all_actual.extend(test_data_clean['y'].tolist())
        all_predicted.extend(forecast['yhat1'].tolist())
    
    # Overall metrics
    results['overall'] = {
        'accuracy': 100 - (np.mean(np.abs((np.array(all_actual) - np.array(all_predicted)) / np.array(all_actual))) * 100),
        'r2': r2_score(all_actual, all_predicted),
        'mae': mean_absolute_error(all_actual, all_predicted),
        'mape': np.mean(np.abs((np.array(all_actual) - np.array(all_predicted)) / np.array(all_actual))) * 100
    }
    
    save_results(results)
    print(results)
    print(f"Training completed in {time.time()-start_time:.2f} seconds")

if __name__ == '__main__':
    main()