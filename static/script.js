function getFormData() {
    return {
        datetime: document.getElementById('datetime').value,
        p: parseFloat(document.getElementById('p').value),
        T: parseFloat(document.getElementById('T').value),
        Tpot: parseFloat(document.getElementById('Tpot').value),
        Tdew: parseFloat(document.getElementById('Tdew').value),
        rh: parseFloat(document.getElementById('rh').value),
        VPmax: parseFloat(document.getElementById('VPmax').value),
        VPact: parseFloat(document.getElementById('VPact').value),
        VPdef: parseFloat(document.getElementById('VPdef').value),
        sh: parseFloat(document.getElementById('sh').value),
        H2OC: parseFloat(document.getElementById('H2OC').value),
        rho: parseFloat(document.getElementById('rho').value),
        wv: parseFloat(document.getElementById('wv').value),
        max_wv: parseFloat(document.getElementById('max_wv').value),
        wd: parseFloat(document.getElementById('wd').value),
        rain: parseFloat(document.getElementById('rain').value),
        raining: parseFloat(document.getElementById('raining').value || 0),
        SWDR: parseFloat(document.getElementById('SWDR').value),
        PAR: parseFloat(document.getElementById('PAR').value),
        max_PAR: parseFloat(document.getElementById('max_PAR').value),
        Tlog: parseFloat(document.getElementById('Tlog').value)
    };
}

function validateForm() {
    const datetime = document.getElementById('datetime').value;
    if (!datetime) {
        alert('⚠️ Please select date and time');
        return false;
    }

    const fields = ['p','T','Tpot','Tdew','rh','VPmax','VPact','VPdef','sh','H2OC','rho','wv','max_wv','wd','rain','SWDR','PAR','max_PAR','Tlog'];
    
    for (let field of fields) {
        const value = document.getElementById(field).value;
        if (!value || value.trim() === '') {
            alert(`⚠️ Please fill in ${field.toUpperCase()}`);
            return false;
        }
        if (isNaN(parseFloat(value))) {
            alert(`⚠️ ${field.toUpperCase()} must be a number`);
            return false;
        }
    }

    const rh = parseFloat(document.getElementById('rh').value);
    if (rh < 0 || rh > 100) {
        alert('⚠️ Humidity must be 0-100%');
        return false;
    }

    const wd = parseFloat(document.getElementById('wd').value);
    if (wd < 0 || wd > 360) {
        alert('⚠️ Wind direction must be 0-360°');
        return false;
    }

    return true;
}

function submitData() {
    if (!validateForm()) return;
    
    const loading = document.getElementById('loading');
    if (loading) loading.classList.add('active');
    
    fetch('/save', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify(getFormData())
    })
    .then(response => response.json())
    .then(data => {
        if (loading) loading.classList.remove('active');
        alert('✅ ' + data.message);
    })
    .catch(error => {
        if (loading) loading.classList.remove('active');
        alert('❌ Error: ' + error.message);
    });
}

function makePrediction() {
    if (!validateForm()) return;
    
    const loading = document.getElementById('loading');
    const resultSection = document.getElementById('resultSection');
    
    if (loading) loading.classList.add('active');
    if (resultSection) resultSection.style.display = 'none';
    
    fetch('/predict', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify(getFormData())
    })
    .then(response => {
        if (!response.ok) throw new Error('Server error');
        return response.json();
    })
    .then(data => {
        if (loading) loading.classList.remove('active');
        
        const rainfall = data.rainfall_prediction;
        let msg = '', color = '';
        
        if (rainfall > 50) {
            msg = '🚨 SEVERE WARNING: High flood risk!';
            color = '#dc3545';
        } else if (rainfall > 25) {
            msg = '⚠️ WARNING: Moderate flood risk!';
            color = '#ff9800';
        } else if (rainfall > 10) {
            msg = '⚡ CAUTION: Heavy rainfall expected';
            color = '#ffc107';
        } else {
            msg = '🌧️ Light to moderate rainfall';
            color = '#667eea';
        }
        
        document.getElementById('predictionResult').innerHTML = `
            <div style="color: ${color}; font-weight: bold; margin-bottom: 10px;">${msg}</div>
            <div>Predicted Rainfall: <strong>${rainfall.toFixed(2)} mm</strong></div>
        `;
        
        if (resultSection) resultSection.style.display = 'block';
        updateChart(rainfall);
    })
    .catch(error => {
        if (loading) loading.classList.remove('active');
        alert('❌ Error: ' + error.message);
    });
}

let rainfallChart = null;
function updateChart(rainfall) {
    const canvas = document.getElementById('rainfallChart');
    if (!canvas) return;
    
    const ctx = canvas.getContext('2d');
    if (rainfallChart) rainfallChart.destroy();
    
    rainfallChart = new Chart(ctx, {
        type: 'bar',
        data: {
            labels: ['Predicted Rainfall'],
            datasets: [{
                label: 'Rainfall (mm)',
                data: [rainfall],
                backgroundColor: rainfall > 10 ? 'rgba(245, 87, 108, 0.8)' : 'rgba(102, 126, 234, 0.8)',
                borderColor: rainfall > 10 ? 'rgba(245, 87, 108, 1)' : 'rgba(102, 126, 234, 1)',
                borderWidth: 2,
                borderRadius: 8
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: true,
            scales: {
                y: {
                    beginAtZero: true,
                    title: {display: true, text: 'Rainfall (mm)', font: {size: 14, weight: 'bold'}}
                }
            },
            plugins: {legend: {display: false}}
        }
    });
}
