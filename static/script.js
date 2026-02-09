function submitData() {
    // Validate datetime
    const datetime = document.getElementById("datetime").value;
    if (!datetime) {
        alert("❌ Please select date and time");
        document.getElementById("datetime").focus();
        return;
    }

    // Validate all required fields
    const requiredFields = ['p', 'T', 'Tpot', 'Tdew', 'rh', 'VPmax', 'VPact', 'VPdef', 'sh', 
                           'H2OC', 'rho', 'wv', 'max_wv', 'wd', 'rain', 'SWDR', 'PAR', 'max_PAR', 'Tlog'];
    
    for (let field of requiredFields) {
        const value = document.getElementById(field).value;
        if (value === '' || isNaN(parseFloat(value))) {
            alert(`❌ Please enter a valid number for ${field.toUpperCase()}`);
            document.getElementById(field).focus();
            return;
        }
    }

    const data = {
        datetime: datetime,
        p: parseFloat(document.getElementById("p").value),
        T: parseFloat(document.getElementById("T").value),
        Tpot: parseFloat(document.getElementById("Tpot").value),
        Tdew: parseFloat(document.getElementById("Tdew").value),
        rh: parseFloat(document.getElementById("rh").value),
        VPmax: parseFloat(document.getElementById("VPmax").value),
        VPact: parseFloat(document.getElementById("VPact").value),
        VPdef: parseFloat(document.getElementById("VPdef").value),
        sh: parseFloat(document.getElementById("sh").value),
        H2OC: parseFloat(document.getElementById("H2OC").value),
        rho: parseFloat(document.getElementById("rho").value),
        wv: parseFloat(document.getElementById("wv").value),
        max_wv: parseFloat(document.getElementById("max_wv").value),
        wd: parseFloat(document.getElementById("wd").value),
        rain: parseFloat(document.getElementById("rain").value),
        SWDR: parseFloat(document.getElementById("SWDR").value),
        PAR: parseFloat(document.getElementById("PAR").value),
        max_PAR: parseFloat(document.getElementById("max_PAR").value),
        Tlog: parseFloat(document.getElementById("Tlog").value)
    };

    const loading = document.getElementById('loading');
    if (loading) loading.classList.add('active');

    fetch("/save", {
        method: "POST",
        headers: {
            "Content-Type": "application/json"
        },
        body: JSON.stringify(data)
    })
    .then(response => response.json())
    .then(data => {
        if (loading) loading.classList.remove('active');
        alert("✅ Data saved successfully!");
    })
    .catch(error => {
        if (loading) loading.classList.remove('active');
        console.error("Error:", error);
        alert("❌ Error saving data: " + error);
    });
}

function makePrediction() {
    // Validate all required fields
    const requiredFields = ['p', 'T', 'Tpot', 'Tdew', 'rh', 'VPmax', 'VPact', 'VPdef', 'sh', 
                           'H2OC', 'rho', 'wv', 'max_wv', 'wd', 'rain', 'SWDR', 'PAR', 'max_PAR', 'Tlog'];
    
    for (let field of requiredFields) {
        const value = document.getElementById(field).value;
        if (value === '' || isNaN(parseFloat(value))) {
            alert(`❌ Please enter a valid number for ${field.toUpperCase()}`);
            document.getElementById(field).focus();
            return;
        }
    }

    // Validate ranges
    const rh = parseFloat(document.getElementById("rh").value);
    if (rh < 0 || rh > 100) {
        alert("❌ Relative Humidity must be between 0 and 100");
        document.getElementById("rh").focus();
        return;
    }

    const wd = parseFloat(document.getElementById("wd").value);
    if (wd < 0 || wd > 360) {
        alert("❌ Wind Direction must be between 0 and 360 degrees");
        document.getElementById("wd").focus();
        return;
    }

    const data = {
        p: parseFloat(document.getElementById("p").value),
        T: parseFloat(document.getElementById("T").value),
        Tpot: parseFloat(document.getElementById("Tpot").value),
        Tdew: parseFloat(document.getElementById("Tdew").value),
        rh: parseFloat(document.getElementById("rh").value),
        VPmax: parseFloat(document.getElementById("VPmax").value),
        VPact: parseFloat(document.getElementById("VPact").value),
        VPdef: parseFloat(document.getElementById("VPdef").value),
        sh: parseFloat(document.getElementById("sh").value),
        H2OC: parseFloat(document.getElementById("H2OC").value),
        rho: parseFloat(document.getElementById("rho").value),
        wv: parseFloat(document.getElementById("wv").value),
        max_wv: parseFloat(document.getElementById("max_wv").value),
        wd: parseFloat(document.getElementById("wd").value),
        rain: parseFloat(document.getElementById("rain").value),
        SWDR: parseFloat(document.getElementById("SWDR").value),
        PAR: parseFloat(document.getElementById("PAR").value),
        max_PAR: parseFloat(document.getElementById("max_PAR").value),
        Tlog: parseFloat(document.getElementById("Tlog").value)
    };

    const loading = document.getElementById('loading');
    const resultSection = document.getElementById('resultSection');
    
    if (loading) loading.classList.add('active');
    if (resultSection) resultSection.style.display = 'none';

    fetch("/predict", {
        method: "POST",
        headers: {
            "Content-Type": "application/json"
        },
        body: JSON.stringify(data)
    })
    .then(response => response.json())
    .then(data => {
        if (loading) loading.classList.remove('active');
        
        if (data.error) {
            alert("❌ Error: " + data.error);
            return;
        }
        
        const result = data.rainfall_prediction;
        document.getElementById("predictionResult").innerText = `Predicted Rainfall: ${result.toFixed(2)} mm`;
        
        if (resultSection) resultSection.style.display = 'block';
        updateChart(result);

        // Show warnings based on rainfall levels
        if (result > 50) {
            alert("🚨 SEVERE WARNING: Very high risk of flooding and landslides!");
        } else if (result > 25) {
            alert("⚠️ WARNING: Moderate risk of flooding. Stay alert!");
        } else if (result > 10) {
            alert("⚡ CAUTION: Heavy rainfall expected.");
        }
    })
    .catch(error => {
        if (loading) loading.classList.remove('active');
        console.error("Error:", error);
        alert("❌ Error making prediction: " + error);
    });
}

// Chart visualization
let rainfallChart = null;

function updateChart(rainfall) {
    const canvas = document.getElementById('rainfallChart');
    if (!canvas) return;
    
    const ctx = canvas.getContext('2d');
    
    if (rainfallChart) {
        rainfallChart.destroy();
    }
    
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
                    title: {
                        display: true,
                        text: 'Rainfall (mm)',
                        font: {
                            size: 14,
                            weight: 'bold'
                        }
                    }
                }
            },
            plugins: {
                legend: {
                    display: false
                }
            }
        }
    });
}