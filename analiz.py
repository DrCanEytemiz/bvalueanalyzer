import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import requests
from datetime import datetime

# --- AYARLAR ---
# Koordinatlar (Manisa/Balıkesir)
MIN_LAT = 38.9382
MAX_LAT = 39.4073
MIN_LON = 27.7682
MAX_LON = 28.5698

MIN_MAG = 1.0
MAX_MAG = 7.0

# TARİH FİLTRESİNİ "GENİŞ" TUTUYORUZ
# Böylece "Hangi yıldayız?" karmaşası yaşanmaz.
# 2024'ten başlayıp 2030'a kadar her şeyi istiyoruz.
FORCE_START_DATE = "2025-08-10T00:00:01"
FORCE_END_DATE   = "2026-01-22T01:30:00"

# Analize girecek son deprem sayısı (En güncel 1000 depremi alır)
MAX_EVENTS_TO_ANALYZE = 1000 
WINDOW_SIZE = 50 
STEP_SIZE = 5

def save_empty_plot(message):
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.text(0.5, 0.5, message, ha='center', va='center', fontsize=12, color='red')
    ax.set_title("Sistem Mesajı")
    ax.axis('off')
    plt.savefig('b_analiz_grafik.png')
    
    html = f"<html><body><h2>{message}</h2><p>Guncelleme: {datetime.now()}</p></body></html>"
    with open("index.html", "w") as f: f.write(html)

def fetch_brute_force_data():
    url = "https://deprem.afad.gov.tr/apiserver/getEventSearchList"
    
    print(f"Veri isteniyor: {FORCE_START_DATE} -> {FORCE_END_DATE}")
    
    headers = {"Content-Type": "application/json"}
    payload = {
        "Start": FORCE_START_DATE,
        "End": FORCE_END_DATE,
        "MinLat": MIN_LAT, "MaxLat": MAX_LAT,
        "MinLon": MIN_LON, "MaxLon": MAX_LON,
        "MinMag": MIN_MAG, "MaxMag": MAX_MAG,
        "FilterType": "Location"
    }
    
    try:
        response = requests.post(url, headers=headers, json=payload, timeout=45)
        if response.status_code == 200:
            data = response.json()
            events = data if isinstance(data, list) else data.get('result', [])
            
            if not events: return pd.DataFrame()
            
            df = pd.DataFrame(events)
            df.rename(columns={'eventDate': 'Date', 'magnitude': 'Mag', 'latitude': 'Lat', 'longitude': 'Lon'}, inplace=True)
            df['Date'] = pd.to_datetime(df['Date'])
            df['Mag'] = pd.to_numeric(df['Mag'])
            
            # Tarihe göre sırala
            df = df.sort_values(by='Date').reset_index(drop=True)
            return df
            
    except Exception as e:
        print(f"Hata: {e}")
        
    return pd.DataFrame()

def calculate_mc(magnitudes):
    mag_rounded = np.round(magnitudes, 1)
    bins = np.arange(min(mag_rounded), max(mag_rounded) + 0.2, 0.1)
    counts, edges = np.histogram(mag_rounded, bins=bins)
    return edges[np.argmax(counts)]

def analyze_and_plot(df):
    if df.empty:
        save_empty_plot(f"API {FORCE_START_DATE} tarihinden beri hic veri dondurmedi.\nKoordinat veya API sorunu olabilir.")
        return

    print(f"API'den toplam {len(df)} deprem geldi.")

    # Sadece EN SON X depremi al (Grafik çok sıkışmasın diye)
    if len(df) > MAX_EVENTS_TO_ANALYZE:
        df = df.tail(MAX_EVENTS_TO_ANALYZE).reset_index(drop=True)
        print(f"Analiz icin son {MAX_EVENTS_TO_ANALYZE} deprem kullaniliyor.")

    # Mc Hesabı
    mc = calculate_mc(df['Mag'])
    mc = max(mc, MIN_MAG) # En az 1.0 olsun
    
    df_clean = df[df['Mag'] >= mc].reset_index(drop=True)
    
    if len(df_clean) < WINDOW_SIZE:
        save_empty_plot(f"Mc ({mc}) sonrasi veri yetersiz ({len(df_clean)}).")
        return

    mags = df_clean['Mag'].values
    times = df_clean['Date'].values
    
    # İstatistikler
    first_date = times[0]
    last_date = times[-1]
    
    # Mainshock
    main_idx = np.argmax(mags)
    main_mag = mags[main_idx]
    main_date = times[main_idx]

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
    
    # GRAFİK
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), sharex=True)
    
    # Zaman ekseni formatı
    date_format = mdates.DateFormatter('%Y-%m-%d')
    
    ax1.plot(plot_times, b_values, 'k.-', label='b-value', linewidth=1)
    ax1.fill_between(plot_times, np.array(b_values)-np.array(b_errors), 
                     np.array(b_values)+np.array(b_errors), color='gray', alpha=0.3)
    ax1.axvline(x=main_date, color='r', linestyle='--', label=f'Max (M{main_mag})')
    
    ax1.set_ylabel('b-value')
    # Title'a tarih aralığını otomatik yazdırıyoruz
    title_str = f"Sismik Analiz (Son {len(df)} Deprem)\nTarih Aralığı: {str(pd.to_datetime(first_date).date())} - {str(pd.to_datetime(last_date).date())}"
    ax1.set_title(title_str, fontweight='bold')
    ax1.grid(True, linestyle='--')
    ax1.legend()
    
    ax2.scatter(df_clean['Date'], df_clean['Mag'], s=15, alpha=0.6, edgecolors='none')
    ax2.axvline(x=main_date, color='r', linestyle='--')
    ax2.set_ylabel('Buyukluk (M)')
    ax2.set_xlabel('Tarih')
    ax2.xaxis.set_major_formatter(date_format)
    plt.xticks(rotation=45)
    
    plt.tight_layout()
    plt.savefig('b_analiz_grafik.png', dpi=150)
    
    html_content = f"""
    <!DOCTYPE html>
    <html lang="tr">
    <head><meta charset="UTF-8"><meta http-equiv="refresh" content="600">
    <title>Canlı Analiz</title>
    <style>body{{font-family:sans-serif;text-align:center;padding:20px;}} img{{max-width:100%;box-shadow:0 0 10px #ccc;}}</style>
    </head>
    <body>
        <h2>Bölgesel Sismik Analiz</h2>
        <p><strong>Veri Kaynağı:</strong> {str(pd.to_datetime(first_date).date())} ile {str(pd.to_datetime(last_date).date())} arası</p>
        <p><strong>Analiz Edilen Deprem:</strong> {len(df_clean)} adet (Filtre: M{MIN_MAG}-{MAX_MAG})</p>
        <p><small>Rapor Oluşturma: {datetime.now().strftime("%Y-%m-%d %H:%M")}</small></p>
        <img src="b_analiz_grafik.png">
    </body>
    </html>
    """
    with open("index.html", "w", encoding="utf-8") as f: f.write(html_content)

if __name__ == "__main__":
    df = fetch_brute_force_data()
    analyze_and_plot(df)
