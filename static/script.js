function submitData() {
    const data = {
        datetime: document.getElementById("datetime").value,
        p: parseFloat(document.getElementById("p").value) || 0,
        T: parseFloat(document.getElementById("T").value) || 0,
        Tpot: parseFloat(document.getElementById("Tpot").value) || 0,
        Tdew: parseFloat(document.getElementById("Tdew").value) || 0,
        rh: parseFloat(document.getElementById("rh").value) || 0,
        VPmax: parseFloat(document.getElementById("VPmax").value) || 0,
        VPact: parseFloat(document.getElementById("VPact").value) || 0,
        VPdef: parseFloat(document.getElementById("VPdef").value) || 0,
        sh: parseFloat(document.getElementById("sh").value) || 0,
        H2OC: parseFloat(document.getElementById("H2OC").value) || 0,
        rho: parseFloat(document.getElementById("rho").value) || 0,
        wv: parseFloat(document.getElementById("wv").value) || 0,
        max_wv: parseFloat(document.getElementById("max_wv").value) || 0,
        wd: parseFloat(document.getElementById("wd").value) || 0,
        rain: parseFloat(document.getElementById("rain").value) || 0,
        SWDR: parseFloat(document.getElementById("SWDR").value) || 0,
        PAR: parseFloat(document.getElementById("PAR").value) || 0,
        max_PAR: parseFloat(document.getElementById("max_PAR").value) || 0,
        Tlog: parseFloat(document.getElementById("Tlog").value) || 0
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
    const data = {
        p: parseFloat(document.getElementById("p").value) || 0,
        T: parseFloat(document.getElementById("T").value) || 0,
        Tpot: parseFloat(document.getElementById("Tpot").value) || 0,
        Tdew: parseFloat(document.getElementById("Tdew").value) || 0,
        rh: parseFloat(document.getElementById("rh").value) || 0,
        VPmax: parseFloat(document.getElementById("VPmax").value) || 0,
        VPact: parseFloat(document.getElementById("VPact").value) || 0,
        VPdef: parseFloat(document.getElementById("VPdef").value) || 0,
        sh: parseFloat(document.getElementById("sh").value) || 0,
        H2OC: parseFloat(document.getElementById("H2OC").value) || 0,
        rho: parseFloat(document.getElementById("rho").value) || 0,
        wv: parseFloat(document.getElementById("wv").value) || 0,
        max_wv: parseFloat(document.getElementById("max_wv").value) || 0,
        wd: parseFloat(document.getElementById("wd").value) || 0,
        rain: parseFloat(document.getElementById("rain").value) || 0,  // ADDED THIS!
        SWDR: parseFloat(document.getElementById("SWDR").value) || 0,
        PAR: parseFloat(document.getElementById("PAR").value) || 0,
        max_PAR: parseFloat(document.getElementById("max_PAR").value) || 0,
        Tlog: parseFloat(document.getElementById("Tlog").value) || 0
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

        // Show warning if rainfall exceeds threshold
        if (result > 50) {
            alert("⚠️ Warning: High risk of flood or landslide!");
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