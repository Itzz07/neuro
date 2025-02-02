from flask import Flask, render_template, jsonify
from flask_cors import CORS
import pandas as pd
import plotly.express as px
import joblib
import json
import time
import threading

app = Flask(__name__)
CORS(app)
training_lock = threading.Lock()
training_status = {'running': False, 'progress': 0, 'message': ''}

def load_cached_data():
    try:
        data = joblib.load('models/predictions.joblib')
        with open('models/metrics.json') as f:
            metrics = json.load(f)
        return data, metrics
    except:
        return None, None

cached_data, cached_metrics = load_cached_data()

@app.route('/')
def home():
    # Load and process data
    df = pd.read_csv('data/predictions_by_station.csv')
    df = df.dropna(subset=['LAT', 'LON', 'predictions'])
    last_rows = df.groupby(['LAT', 'LON']).last().reset_index()

    # Create the figure
    fig = px.scatter_mapbox(last_rows,
                           lat='LAT',
                           lon='LON',
                           hover_name='Station',
                           hover_data={
                               'LON': False,
                               'LAT': False,
                               'Timestamp': True,
                               'predictions': True,
                               'actual_readings': True
                           },
                           color='predictions',
                           color_continuous_scale='viridis_r',
                           size_max=10,
                           title="Station Map Predictions - March 2021")

    # Update layout
    fig.update_layout(
        mapbox_style="open-street-map",
        mapbox_zoom=5,
        mapbox_center={"lat": last_rows['LAT'].mean(), "lon": last_rows['LON'].mean()},
        height=650
    )

    # Convert plot to HTML
    plot_html = fig.to_html(full_html=False)
    
    return render_template('index.html', 
                         data=cached_data,
                         metrics=cached_metrics,plot_html=plot_html)

@app.route('/retrain', methods=['POST'])
def retrain_model():
    global training_status
    if training_lock.locked():
        return jsonify({'status': 'running', 'progress': training_status['progress']})
    
    def training_task():
        global cached_data, cached_metrics, training_status
        training_lock.acquire()
        try:
            training_status.update({
                'running': True,
                'progress': 0,
                'message': 'Preprocessing data...'
            })
            
            import subprocess
            process = subprocess.Popen(['python', 'trainer.py'], 
                                      stdout=subprocess.PIPE,
                                      stderr=subprocess.STDOUT)
            
            while True:
                output = process.stdout.readline()
                if process.poll() is not None:
                    break
                if output:
                    training_status['progress'] += 10
                    training_status['message'] = output.decode().strip()
            
            cached_data, cached_metrics = load_cached_data()
            
        finally:
            training_status.update({'running': False, 'progress': 100, 'message': 'Completed'})
            time.sleep(2)
            training_status['progress'] = 0
            training_lock.release()
    
    threading.Thread(target=training_task, daemon=True).start()
    return jsonify({'status': 'started'})

@app.route('/training-status')
def get_training_status():
    return jsonify(training_status)

if __name__ == '__main__':
    app.run(threaded=True, debug=True)
    # app.run(host='0.0.0.0', port=5000, threaded=True, debug=True)