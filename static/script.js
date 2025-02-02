let statusCheckInterval;

function startRetraining() {
    const btn = document.getElementById('retrain-btn');
    btn.disabled = true;
    btn.innerHTML = '⏳ Training...';
    
    fetch('/retrain', { method: 'POST' })
        .then(response => response.json())
        .then(data => {
            if(data.status === 'started') {
                statusCheckInterval = setInterval(updateTrainingStatus, 1000);
            }
        });
}

function updateTrainingStatus() {
    fetch('/training-status')
        .then(response => response.json())
        .then(status => {
            const progressBar = document.getElementById('training-progress');
            const statusText = document.getElementById('status-text');
            
            progressBar.value = status.progress;
            statusText.textContent = status.message;

            if(!status.running && status.progress === 0) {
                clearInterval(statusCheckInterval);
                document.getElementById('retrain-btn').disabled = false;
                document.getElementById('retrain-btn').innerHTML = '🔄 Retrain Model';
                if(status.message === 'Completed') {
                    window.location.reload();
                }
            }
        });
}

// Initial status check on page load
document.addEventListener('DOMContentLoaded', () => {
    if(document.getElementById('retrain-btn')) {
        fetch('/training-status')
            .then(response => response.json())
            .then(status => {
                if(status.running) {
                    startRetraining();
                }
            });
    }
});