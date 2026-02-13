
# 🌧️ Rainfall Prediction System - Technical Documentation

**Version:** 1.0  
**Last Updated:** February 2026  
**Deployed URL:** https://ict-304.onrender.com  
**Repository:** https://github.com/diyanashamil/ict-304

---

## 📋 Table of Contents

1. [System Overview](#system-overview)
2. [Architecture](#architecture)
3. [Model Integration](#model-integration)
4. [Features](#features)
5. [Installation](#installation)
6. [Deployment](#deployment)
7. [API Documentation](#api-documentation)
8. [File Structure](#file-structure)

---

## 🎯 System Overview

An AI-powered rainfall prediction system that uses TensorFlow/Keras LSTM model to forecast rainfall based on 19 weather parameters. The system provides real-time predictions with visual charts and multi-level warnings.

**Key Capabilities:**
- ✅ Predicts rainfall in millimeters
- ✅ Stores historical weather data
- ✅ Visual chart representation
- ✅ Multi-level warning system (Caution/Warning/Severe)
- ✅ Responsive web interface
- ✅ Cloud-hosted on Render

---

## 🏗️ Architecture
```
┌─────────────────────────────────────────────────┐
│              USER INTERFACE                      │
│  HTML + CSS + JavaScript + Chart.js             │
└──────────────────┬──────────────────────────────┘
                   │ HTTP POST/GET
                   ▼
┌─────────────────────────────────────────────────┐
│           FLASK BACKEND (Python)                │
│  ┌──────────────┐  ┌──────────────┐            │
│  │ /save        │  │ /predict     │            │
│  │ Route        │  │ Route        │            │
│  └──────────────┘  └──────────────┘            │
└──────────────────┬──────────────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────────────┐
│        TENSORFLOW KERAS MODEL                   │
│  Input: 19 features × 19 timesteps              │
│  Output: Rainfall prediction (mm)               │
└─────────────────────────────────────────────────┘
```

---

## 🤖 Model Integration

### **How the Model Works**

**1. Model Download (Automatic)**
```python
def download_model():
    if not os.path.exists('rainfall_model.keras'):
        # Downloads 1.9MB model from Google Drive
        # Handles Google virus scan warning
        # Saves locally for future use
```

**Why Google Drive?**
- GitHub has 100MB file size limit
- Model is 1.9MB (too large for repo)
- Auto-downloads on first server startup
- Cached for subsequent requests

**2. Data Preprocessing**
```python
# 19 weather features
features = [p, T, Tpot, Tdew, rh, VPmax, VPact, VPdef, sh, 
            H2OC, rho, wv, max_wv, wd, SWDR, PAR, max_PAR, Tlog, rain]

# Scale to 0-1 range
scaled = MinMaxScaler().transform([features])

# Create sequence (model expects 19 timesteps)
sequence = np.array([scaled] * 19).reshape((1, 19, 19))
```

**3. Prediction**
```python
prediction = model.predict(sequence)
rainfall_mm = float(prediction[0][0])
```

**Model Architecture:**
- **Type:** LSTM (Long Short-Term Memory)
- **Input Shape:** (1, 19, 19) = 1 sample × 19 timesteps × 19 features
- **Output:** Single value (rainfall in mm)
- **Framework:** TensorFlow/Keras 2.20.0

---

## 📊 CSV Files Explained

### **File: `data/weather_data.csv`**

**Purpose:** Historical weather data storage

**How It Works:**
1. User submits weather data via `/save` endpoint
2. Data converted to pandas DataFrame
3. Appended to CSV file (creates if doesn't exist)

**Structure:**
```csv
datetime,p,T,Tpot,Tdew,rh,VPmax,VPact,VPdef,sh,H2OC,rho,wv,max_wv,wd,rain,raining,SWDR,PAR,max_PAR,Tlog
2026-02-09 10:00,1013.25,25.5,298.5,18.2,65.5,31.7,20.8,10.9,13.2,21.1,1180.5,4.5,8.2,180.5,0.5,1,450.3,920.5,1250.8,5.7
```

**Use Cases:**
- ✅ Build historical dataset
- ✅ Model retraining
- ✅ Weather pattern analysis
- ✅ Audit trail

**Note:** CSV is for storage only, NOT used in predictions.

---

## ✨ Features & Enhancements

### **Frontend Features**

| Feature | Description |
|---------|-------------|
| Modern UI | Purple gradient theme with responsive design |
| Grid Layout | 3-column layout (adapts to mobile) |
| Visual Indicators | Emoji icons for each parameter |
| Loading Animation | Spinner during API calls |
| Chart Visualization | Bar chart with Chart.js |
| Responsive Design | Mobile-friendly interface |

### **Validation System**

| Validation | Implementation |
|------------|----------------|
| DateTime Check | Must select date/time |
| Required Fields | All 19 fields must be filled |
| Number Type | Validates numeric inputs |
| Humidity Range | 0-100% validation |
| Wind Direction | 0-360° validation |
| Focus on Error | Highlights problematic field |

### **Warning System**

| Rainfall | Alert Level | Color |
|----------|-------------|-------|
| > 50mm | 🚨 SEVERE | Red |
| > 25mm | ⚠️ WARNING | Orange |
| > 10mm | ⚡ CAUTION | Yellow |
| ≤ 10mm | Normal | Blue |

### **Backend Features**

| Feature | Purpose |
|---------|---------|
| Auto Model Download | Downloads from Google Drive on startup |
| Error Handling | Try-catch blocks with logging |
| JSON Responses | Proper API response format |
| Cross-Platform Paths | Works on Windows/Mac/Linux |
| Auto Directory Creation | Creates `data/` folder automatically |

---

## 💻 Installation

### **Local Development**
```bash
# Clone repository
git clone https://github.com/diyanashamil/ict-304.git
cd ict-304

# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # Mac/Linux
# or
venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt

# Run locally
python rainfall_prediction.py
```

**Access at:** http://localhost:5000

---

## 🚀 Deployment

### **Render Platform**

**Deployment Process:**
1. Push code to GitHub
2. Render auto-detects changes
3. Builds environment (installs dependencies)
4. Downloads model from Google Drive
5. Starts Gunicorn server
6. Live at: https://ict-304.onrender.com

**Configuration Files:**
- `render.yaml` - Deployment settings
- `requirements.txt` - Python dependencies
- `.gitignore` - Git exclusions

**Environment Variables:**
- `PORT` - Auto-set by Render (default: 10000)
- `PYTHON_VERSION` - 3.13.4

---

## 📡 API Documentation

### **1. Save Weather Data**

**Endpoint:** `POST /save`

**Request Body:**
```json
{
  "datetime": "2026-02-09T10:00",
  "p": 1013.25,
  "T": 25.5,
  "Tpot": 298.5,
  "Tdew": 18.2,
  "rh": 65.5,
  "VPmax": 31.7,
  "VPact": 20.8,
  "VPdef": 10.9,
  "sh": 13.2,
  "H2OC": 21.1,
  "rho": 1180.5,
  "wv": 4.5,
  "max_wv": 8.2,
  "wd": 180.5,
  "rain": 0.5,
  "SWDR": 450.3,
  "PAR": 920.5,
  "max_PAR": 1250.8,
  "Tlog": 5.7
}
```

**Response:**
```json
{
  "message": "Data saved successfully!"
}
```

---

### **2. Predict Rainfall**

**Endpoint:** `POST /predict`

**Request Body:**
```json
{
  "p": 1013.25,
  "T": 25.5,
  "Tpot": 298.5,
  "Tdew": 18.2,
  "rh": 65.5,
  "VPmax": 31.7,
  "VPact": 20.8,
  "VPdef": 10.9,
  "sh": 13.2,
  "H2OC": 21.1,
  "rho": 1180.5,
  "wv": 4.5,
  "max_wv": 8.2,
  "wd": 180.5,
  "rain": 0.5,
  "SWDR": 450.3,
  "PAR": 920.5,
  "max_PAR": 1250.8,
  "Tlog": 5.7
}
```

**Response:**
```json
{
  "rainfall_prediction": 12.45
}
```

---

### **3. Home Page**

**Endpoint:** `GET /`

**Response:** HTML interface

---

## 📁 File Structure
```
ict-304/
├── rainfall_prediction.py      # Main Flask application
├── model.py                     # Model definition (reference)
├── requirements.txt             # Python dependencies
├── render.yaml                  # Render deployment config
├── .gitignore                   # Git exclusions
├── README.md                    # Project overview
├── DOCUMENTATION.md             # This file
├── templates/
│   └── index.html              # Frontend UI
├── static/
│   ├── script.js               # JavaScript (validation + charts)
│   └── styles.css              # CSS styling (if separated)
└── data/
    └── weather_data.csv        # Historical data storage
```

---

## 🔧 Technologies Used

| Technology | Version | Purpose |
|------------|---------|---------|
| Python | 3.13.4 | Backend language |
| Flask | 3.1.2 | Web framework |
| TensorFlow | 2.20.0 | ML model |
| Pandas | 2.3.3 | Data manipulation |
| NumPy | 2.1.0 | Numerical computing |
| Scikit-learn | 1.6.1 | Data preprocessing |
| Gunicorn | 21.2.0 | WSGI server |
| Chart.js | Latest | Visualization |
| Render | - | Cloud hosting |

---

## 🐛 Troubleshooting

### **Model Download Issues**
**Problem:** Model fails to download  
**Solution:** Check Google Drive link is public

### **Prediction Errors**
**Problem:** "could not convert string to float"  
**Solution:** Ensure all fields are filled with numbers

### **Port Conflicts**
**Problem:** Address already in use  
**Solution:** Change port in code or kill existing process

---

## 🎓 How to Use

1. **Access:** https://ict-304.onrender.com
2. **Fill all 19 weather parameters**
3. **Click "Submit Data"** to save (optional)
4. **Click "Predict Rainfall"** to get forecast
5. **View chart** and warning level

---

## 📈 Performance

- **Model Load Time:** 2-3 seconds (first request)
- **Prediction Time:** <1 second
- **Download Size:** 1.9MB (model) + 100KB (frontend)
- **Concurrent Users:** Render free tier supports multiple

---

## 🔐 Security Considerations

- ✅ No API keys required
- ✅ No sensitive data stored
- ✅ HTTPS encryption (Render default)
- ✅ Input validation on frontend and backend
- ⚠️ CSV file grows unbounded (manual cleanup needed)

---

## 🚧 Future Enhancements

- [ ] User authentication
- [ ] Data export feature
- [ ] Historical data visualization
- [ ] Model retraining pipeline
- [ ] Email alerts for severe warnings
- [ ] Mobile app version

---

## 👥 Credits

**Developer:** Diyana  
**Institution:** Murdoch University Singapore  
**Course:** ICT 304  
**Framework:** TensorFlow/Keras + Flask

---

## 📞 Support

**Issues:** https://github.com/diyanashamil/ict-304/issues  
**Email:** Contact via GitHub

---

**Last Updated:** February 10, 2026# ict-304
