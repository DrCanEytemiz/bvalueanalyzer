import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import requests
import json
from datetime import datetime

# --- AYARLAR ---
# Başlangıç Tarihi (Yıl-Ay-Gün Saat:Dakika:Saniye)
START_DATE = "2025-08-10T00:01:00" 

# Koordinat Kutusu (Ekran görüntünüzdeki değerler)
MIN_LAT = 38.9382
MAX_LAT = 39.4073
MIN_LON = 27.7682
MAX_LON = 28.5698

# Analiz Parametreleri
WINDOW_SIZE = 100  # Pencere büyüklüğü (N)
STEP_SIZE = 5      # Kaydırma adımı

def fetch_afad_catalog():
    url = "https://deprem.afad.gov.tr/apiserver/getEventSearchList"
    now_str = datetime.now().strftime("%Y-%m-%dT%H:%M:%S")
    
    headers = {
        "Content-Type": "application/json",
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
    }
    
    payload = {
        "Start": START_DATE,
        "End": now_str,
        "MinLat": MIN_LAT, "MaxLat": MAX_LAT,
        "MinLon": MIN_LON, "MaxLon": MAX_LON,
        "MinMag": 0.0, "MaxMag": 9.9,
        "FilterType": "Location"
    }
    
    print(f"AFAD API isteği gönderiliyor... ({START_DATE} -> {now_str})")
    try:
        response = requests.post(url, headers=headers, json=payload, timeout=60)
        if response.status_code == 200:
            data = response.json()
            events = data if isinstance(data, list) else data.get('result', [])
            
            if not events:
                print("Bu kriterlere uygun deprem bulunamadı.")
                return pd.DataFrame()

            df = pd.DataFrame(events)
            df.rename(columns={'eventDate': 'Date', 'magnitude': 'Mag', 'latitude': 'Lat', 'longitude': 'Lon'}, inplace=True)
            df['Date'] = pd.to_datetime(df['Date'])
            df['Mag'] = pd.to_numeric(df['Mag'])
            return df.sort_values(by='Date').reset_index(drop=True)
        else:
            print(f"API Hatası: {response.status_code}")
            return pd.DataFrame()
    except Exception as e:
        print(f"Hata: {e}")
        return pd.DataFrame()

def calculate_mc(magnitudes):
    mag_rounded = np.round(magnitudes, 1)
    bins = np.arange(min(mag_rounded), max(mag_rounded) + 0.2, 0.1)
    counts, edges = np.histogram(mag_rounded, bins=bins)
    return edges[np.argmax(counts)]

def analyze_and_plot(df):
    if df.empty or len(df) < WINDOW_SIZE:
        print(f"Yetersiz veri. (Gerekli: {WINDOW_SIZE}, Mevcut: {len(df)})")
        # Veri yoksa bile boş sayfa oluştur ki site çökmesin
        html_content = f"""<html><body><h2>Yeterli Veri Yok</h2><p>Bölge: {MIN_LAT}-{MAX_LAT} / {MIN_LON}-{MAX_LON}</p><p>Deprem Sayısı: {len(df)}</p><p>Son Kontrol: {datetime.now()}</p></body></html>"""
        with open("index.html", "w") as f: f.write(html_content)
        return

    mc = calculate_mc(df['Mag'])
    df_clean = df[df['Mag'] >= mc].reset_index(drop=True)
    
    if len(df_clean) < WINDOW_SIZE:
        print("Mc sonrası yetersiz veri.")
        return

    mags = df_clean['Mag'].values
    times = df_clean['Date'].values
    main_date = times[np.argmax(mags)]

    b_values, b_errors, plot_times = [], [], []
    num_windows = (len(mags) - WINDOW_SIZE) // STEP_SIZE + 1
    
    for i in range(num_windows):
        start = i * STEP_SIZE
        end = start + WINDOW_SIZE
        m_win = mags[start:end]
        mean_mag = np.mean(m_win)
        b = np.log10(np.exp(1)) / (mean_mag - (mc - 0.05))
        b_err = 2.3 * (b**2) * np.std(m_win) / np.sqrt(WINDOW_SIZE*(WINDOW_SIZE-1))
        
        b_values.append(b)
        b_errors.append(b_err)
        plot_times.append(pd.to_datetime(np.mean(times[start:end].astype(np.int64))))
    
    # Grafik Çizimi
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
    
    ax1.plot(plot_times, b_values, 'k.-', label='b-value')
    ax1.fill_between(plot_times, np.array(b_values)-np.array(b_errors), np.array(b_values)+np.array(b_errors), color='gray', alpha=0.3)
    ax1.axvline(x=main_date, color='r', linestyle='--', label='Mainshock')
    ax1.set_ylabel('b-value')
    ax1.set_title(f'Temporal Evolution of b-Value (Mc={mc:.1f}, N={WINDOW_SIZE})\nArea: {MIN_LAT}-{MAX_LAT}N / {MIN_LON}-{MAX_LON}E')
    ax1.grid(True, linestyle='--')
    ax1.legend()
    
    ax2.scatter(df_clean['Date'], df_clean['Mag'], s=15, alpha=0.6)
    ax2.axvline(x=main_date, color='r', linestyle='--')
    ax2.set_ylabel('Magnitude')
    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('b_analiz_grafik.png', dpi=150)
    
    html_content = f"""
    <!DOCTYPE html>
    <html lang="tr">
    <head><meta charset="UTF-8"><meta http-equiv="refresh" content="1800">
    <title>Canlı b-Değeri Analizi</title>
    <style>body{{font-family:sans-serif;text-align:center;padding:20px;}} img{{max-width:100%;height:auto;}}</style>
    </head>
    <body>
        <h1>Otomatik b-Değeri Analizi</h1>
        <p>Bölge: {MIN_LAT}-{MAX_LAT} K / {MIN_LON}-{MAX_LON} D | Başlangıç: {START_DATE}</p>
        <p>Son Güncelleme: {datetime.now().strftime("%d-%m-%Y %H:%M:%S")}</p>
        <img src="b_analiz_grafik.png" alt="Analiz Grafiği">
    </body>
    </html>
    """
    with open("index.html", "w", encoding="utf-8") as f: f.write(html_content)

if __name__ == "__main__":
    df = fetch_afad_catalog()
    analyze_and_plot(df)
