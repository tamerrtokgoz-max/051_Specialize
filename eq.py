import gradio as gr
import numpy as np
import pandas as pd
import requests
import folium
import joblib
import tensorflow as tf
from datetime import datetime, timedelta

# =========================
# ASSET YÜKLEME
# =========================
MODEL_PATH = "earthquake_model.keras"
SCALER_PATH = "scaler.pkl"

def load_assets():
    try:
        model = tf.keras.models.load_model(MODEL_PATH)
        scaler = joblib.load(SCALER_PATH)
        return model, scaler
    except Exception as e:
        return None, None

# =========================
# GENİŞ VERİ SORGULAMA
# =========================
def get_extended_data():
    end_time = datetime.utcnow()
    start_time = end_time - timedelta(days=365)
    
    url = (
        f"https://earthquake.usgs.gov/fdsnws/event/1/query?format=geojson"
        f"&starttime={start_time.isoformat()}&endtime={end_time.isoformat()}"
        f"&minmagnitude=3.0&minlatitude=36&maxlatitude=42"
        f"&minlongitude=26&maxlongitude=45"
    )
    
    try:
        res = requests.get(url, timeout=20).json()
        rows = []
        for f in res["features"]:
            p, c = f["properties"], f["geometry"]["coordinates"]
            rows.append({
                "mag": float(p["mag"]), "lat": float(c[1]), "lon": float(c[0]),
                "time": datetime.fromtimestamp(p["time"] / 1000.0), "place": p["place"]
            })
        df = pd.DataFrame(rows).sort_values("time")
        return df
    except:
        return pd.DataFrame()

# =========================
# ANALİZ VE TAHMİN MOTORU
# =========================
def analyze():
    model, scaler = load_assets()
    df = get_extended_data()
    
    if df.empty or len(df) < 12:
        return "<h3>Veri yetersiz.</h3>", {}

    # 1. Büyüklük Tahmini (LSTM)
    last_12 = df['mag'].tail(12).values.reshape(-1, 1)
    scaled_input = scaler.transform(last_12)
    pred_scaled = model.predict(np.expand_dims(scaled_input, axis=0), verbose=0)
    pred_mag = scaler.inverse_transform(pred_scaled)[0][0]

    # 2. Konum Tahmini (Sismik Odak Analizi)
    # Son 12 depremin ağırlıklı merkezi (Büyük depremlerin olduğu yere daha çok odaklanır)
    weights = df['mag'].tail(12).values
    lat_pred = np.average(df['lat'].tail(12), weights=weights)
    lon_pred = np.average(df['lon'].tail(12), weights=weights)

    # 3. Zaman Tahmini (İstatistiksel Aralık Analizi)
    # Depremler arasındaki ortalama süreyi hesaplar
    time_diffs = df['time'].diff().tail(12).dt.total_seconds() / 86400 # Gün cinsinden
    avg_diff = time_diffs.mean()
    std_diff = time_diffs.std()
    
    # Bir sonraki deprem tahmini (Son deprem zamanı + ortalama fark)
    next_date = df['time'].iloc[-1] + timedelta(days=avg_diff)
    days_remaining = (next_date - datetime.utcnow()).days
    if days_remaining < 0: days_remaining = "1-3" # Geçmişse yakındır

    # --- HARİTA ---
    m = folium.Map(location=[39.0, 35.2], zoom_start=6, tiles="CartoDB DarkMatter")
    
    # Geçmiş Depremler
    for _, r in df.tail(50).iterrows():
        folium.CircleMarker([r["lat"], r["lon"]], radius=r["mag"]*1.5, color="cyan", opacity=0.4).add_to(m)
    
    # TAHMİN NOKTASI (Kırmızı Yıldız)
    folium.Marker(
        [lat_pred, lon_pred],
        popup=f"Tahmin Edilen Odak\nBüyüklük: {pred_mag:.1f}",
        icon=folium.Icon(color="red", icon="warning-sign")
    ).add_to(m)

    results = {
        "Tahmin Edilen Büyüklük": f"{pred_mag:.2f} MW",
        "Tahmin Edilen Bölge": f"Enlem: {lat_pred:.2f}, Boylam: {lon_pred:.2f}",
        "Tahmini Zaman Aralığı": f"Önümüzdeki {int(avg_diff) + 1} gün içinde bekleniyor",
        "Analiz Güveni": f"%{max(40, 100 - std_diff*10):.1f}"
    }

    return m._repr_html_(), results

# =========================
# UI
# =========================
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🌋 TR Deprem AI - Konum ve Zaman Analizi")
    
    with gr.Row():
        with gr.Column(scale=4):
            map_out = gr.HTML()
        with gr.Column(scale=1):
            btn = gr.Button("⚡ ANALİZ ET", variant="primary")
            stats = gr.JSON(label="Tahmin Verileri")
    
    btn.click(analyze, outputs=[map_out, stats])

demo.launch()