from flask import Flask, render_template, request, jsonify
import numpy as np
import tensorflow as tf
import pandas as pd
import os
import requests
from sklearn.preprocessing import MinMaxScaler
import math

app = Flask(__name__)

# Download model from Google Drive if not exists
MODEL_PATH = 'rainfall_model.keras'
GDRIVE_FILE_ID = '1cSXmZJmQdOrTRX61325gJbOT5zyM-kt-'

def download_model():
    if not os.path.exists(MODEL_PATH):
        print("Downloading model from Google Drive...")
        url = f"https://drive.google.com/uc?export=download&id={GDRIVE_FILE_ID}"
        
        session = requests.Session()
        response = session.get(url, stream=True)
        
        for key, value in response.cookies.items():
            if key.startswith('download_warning'):
                params = {'id': GDRIVE_FILE_ID, 'confirm': value}
                response = session.get(url, params=params, stream=True)
        
        with open(MODEL_PATH, 'wb') as f:
            for chunk in response.iter_content(32768):
                if chunk:
                    f.write(chunk)
        print("Model downloaded successfully!")

download_model()
model = tf.keras.models.load_model(MODEL_PATH)

scaler = MinMaxScaler()
sample_data = [[0] * 19, [100] * 19]
scaler.fit(sample_data)

@app.route('/')
@app.route('/docs')
def documentation():
    return render_template('documentation.html')
def home():
    return render_template('index.html')

os.makedirs('data', exist_ok=True)
      
@app.route('/save', methods=['POST'])
def save_data():
    try:
        data = request.json
        print("Received data:", data)
        df = pd.DataFrame([data])
        print("DataFrame created:", df)
        csv_path = 'data/weather_data.csv'
        df.to_csv(csv_path, mode='a', index=False, header=not os.path.exists(csv_path))
        print("Data saved successfully to", csv_path)
        return jsonify({"message": "Data saved successfully!"})
    except Exception as e:
        print(f"Error saving data: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        print("Received data for prediction:", data)
        feature_values = [
            float(data.get('p', 0)), float(data.get('T', 0)), float(data.get('Tpot', 0)),
            float(data.get('Tdew', 0)), float(data.get('rh', 0)), float(data.get('VPmax', 0)),
            float(data.get('VPact', 0)), float(data.get('VPdef', 0)), float(data.get('sh', 0)),
            float(data.get('H2OC', 0)), float(data.get('rho', 0)), float(data.get('wv', 0)),
            float(data.get('max_wv', 0)), float(data.get('wd', 0)), float(data.get('SWDR', 0)),
            float(data.get('PAR', 0)), float(data.get('max_PAR', 0)), float(data.get('Tlog', 0)),
            float(data.get('rain', 0))
        ]
        print("Feature values:", feature_values)
        scaled_input = scaler.transform([feature_values])[0]
        print("Scaled input data:", scaled_input)
        sequence_length = 19
        input_sequence = np.array([scaled_input] * sequence_length)
        input_sequence = input_sequence.reshape((1, sequence_length, len(feature_values)))
        print("Prepared input data for model:", input_sequence)
        prediction = model.predict(input_sequence)
        predicted_rainfall = float(prediction[0][0])
        print("Predicted rainfall:", predicted_rainfall)
        return jsonify({'rainfall_prediction': predicted_rainfall})
    except Exception as e:
        print(f"Error in prediction: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/data')
def view_data():
    return render_template('data_view.html')

@app.route('/api/data')
def get_data():
    try:
        csv_path = 'data/weather_data.csv'
        page = int(request.args.get('page', 1))
        per_page = int(request.args.get('per_page', 10))
        
        if not os.path.exists(csv_path):
            return jsonify({
                'data': [],
                'total': 0,
                'page': page,
                'per_page': per_page,
                'total_pages': 0
            })
        
        df = pd.read_csv(csv_path)
        total = len(df)
        total_pages = math.ceil(total / per_page)
        
        start = (page - 1) * per_page
        end = start + per_page
        
        page_data = df.iloc[start:end]
        
        return jsonify({
            'data': page_data.to_dict('records'),
            'total': total,
            'page': page,
            'per_page': per_page,
            'total_pages': total_pages
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)