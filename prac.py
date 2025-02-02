import plotly.graph_objects as go
from flask import Flask, render_template

app = Flask(__name__)

@app.route('/')
def index():
    # Sample data for rainfall predictions
    time = ['2025-01-01', '2025-01-02', '2025-01-03']
    rainfall = [5, 10, 8]
    
    fig = go.Figure(data=go.Scatter(x=time, y=rainfall, mode='lines', name='Rainfall'))
    graph_html = fig.to_html(full_html=False)
    
    return render_template('index.html', graph_html=graph_html)

if __name__ == '__main__':
    app.run(debug=True)
