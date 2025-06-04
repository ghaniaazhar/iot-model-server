from fastapi import FastAPI, Request
import numpy as np
import joblib
import tensorflow as tf

app = FastAPI()

model = tf.keras.models.load_model("bilstm_model.keras")
scaler = joblib.load("scaler.gz")

columns = ['CO', 'NO', 'NO2', 'O3', 'SO2', 'PM2.5', 'PM10', 'NH3', 'Temperature', 'Humidity']
thresholds = {
    "PM2.5": 35, "PM10": 50, "CO": 9, "SO2": 75, "NO": 0.1,
    "O3": 100, "NH3": 400, "Temperature": 40, "Humidity": 85
}

@app.post("/predict")
async def predict(request: Request):
    data = await request.json()
    values = [data[col] for col in columns]
    scaled = scaler.transform([values])
    seq = np.expand_dims(scaled, axis=1)  # Make 3D
    prediction = model.predict(seq)[0]

    result = {columns[i]: float(round(prediction[i], 2)) for i in range(len(columns))}
    alarm_trigger = False
    alarm_reason = []

    for col in columns:
        if result[col] > thresholds[col]:
            alarm_trigger = True
            alarm_reason.append(f"{col}={result[col]} > {thresholds[col]}")

    result["alarm"] = alarm_trigger
    result["alarm_reason"] = ", ".join(alarm_reason) if alarm_trigger else "None"
    result["fan_status"] = "ON" if alarm_trigger else "OFF"

    return result