import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import requests
from datetime import datetime, timedelta

# --- AYARLAR ---
# Koordinatlar (Manisa/Balıkesir)
MIN_LAT = 38.9382
MAX_LAT = 39.4073
MIN_LON = 27.7682
MAX_LON = 28.5698

# --- YENİ FİLTRELER ---
MIN_MAG = 1.0  # İsteğiniz üzerine en az 1.0
MAX_MAG = 7.0  # İsteğiniz üzerine en çok 7.0

# Veri boş kalmasın diye süreyi 30 güne çıkardık (1.0 altını elediğimiz için veri azalır)
DAYS_BACK = 30 
WINDOW_SIZE = 30 # En az 30 deprem lazım
STEP_SIZE = 2

def save_empty_plot(message):
    """Veri yetersizse uyarı resmi kaydeder"""
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.text(0.5, 0.5, message, ha='center', va='center', fontsize=12, color='red')
    ax.set_title("Analiz Durumu")
    ax.axis('off')
    plt.savefig('b_analiz_grafik.png')
    
    html = f"<html><body><h2>{message}</h2><p>Guncelleme: {datetime.now()}</p></body></html>"
    with open("index.html", "w") as f: f.write(html)

def fetch_recent_data():
    url = "https://deprem.afad.gov.tr/apiserver/getEventSearchList"
    
    # Şu anki zamandan geriye doğru git
    end_date = datetime.now()
    start_date = end_date - timedelta(days=DAYS_BACK)
    
    print(f"Veri aralığı taranıyor: {start_date} -> {end_date}")
    
    headers = {"Content-Type": "application/json"}
    payload = {
        "Start": start_date.strftime("%Y-%m-%dT%H:%M:%S"),
        "End": end_date.strftime("%Y-%m-%dT%H:%M:%S"),
        "MinLat": MIN_LAT, "MaxLat": MAX_LAT,
        "MinLon": MIN_LON, "MaxLon": MAX_LON,
        "MinMag": MIN_MAG, "MaxMag": MAX_MAG, # BURASI GÜNCELLENDİ
        "FilterType": "Location"
    }
    
    try:
        response = requests.post(url, headers=headers, json=payload, timeout=30)
        if response.status_code == 200:
            data = response.json()
            events = data if isinstance(data, list) else data.get('result', [])
            
            if not events: return pd.DataFrame()
            
            df = pd.DataFrame(events)
            df.rename(columns={'eventDate': 'Date', 'magnitude': 'Mag', 'latitude': 'Lat', 'longitude': 'Lon'}, inplace=True)
            df['Date'] = pd.to_datetime(df['Date'])
            df['Mag'] = pd.to_numeric(df['Mag'])
            return df.sort_values(by='Date').reset_index(drop=True)
            
    except Exception as e:
        print(f"Hata: {e}")
        
    return pd.DataFrame()

def calculate_mc(magnitudes):
    # Max Curvature
    mag_rounded = np.round(magnitudes, 1)
    bins = np.arange(min(mag_rounded), max(mag_rounded) + 0.2, 0.1)
    counts, edges = np.histogram(mag_rounded, bins=bins)
    return edges[np.argmax(counts)]

def analyze_and_plot(df):
    if df.empty:
        save_empty_plot(f"Son {DAYS_BACK} gunde, {MIN_MAG}-{MAX_MAG} buyuklugunde deprem yok.")
        return

    if len(df) < WINDOW_SIZE:
        save_empty_plot(f"Son {DAYS_BACK} gunde {len(df)} deprem bulundu.\nAnaliz icin en az {WINDOW_SIZE} gerekli.")
        return

    # Mc Hesabı
    mc = calculate_mc(df['Mag'])
    
    # Mc Filtresi (Eğer otomatik Mc, bizim 1.0 sınırından küçük çıkarsa 1.0'ı baz al)
    mc = max(mc, MIN_MAG)
    
    df_clean = df[df['Mag'] >= mc].reset_index(drop=True)
    
    if len(df_clean) < WINDOW_SIZE:
        save_empty_plot(f"Mc ({mc}) sonrasi veri yetersiz ({len(df_clean)} adet).")
        return

    mags = df_clean['Mag'].values
    times = df_clean['Date'].values
    
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
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
    
    ax1.plot(plot_times, b_values, 'k.-', label='b-value')
    ax1.fill_between(plot_times, np.array(b_values)-np.array(b_errors), 
                     np.array(b_values)+np.array(b_errors), color='gray', alpha=0.3)
    ax1.axvline(x=main_date, color='r', linestyle='--', label=f'Max (M{main_mag})')
    
    ax1.set_ylabel('b-value')
    ax1.set_title(f'SON {DAYS_BACK} GUNLUK ANALIZ (Filtre: M{MIN_MAG}-{MAX_MAG})\nBolge: {MIN_LAT}-{MAX_LAT} / {MIN_LON}-{MAX_LON}', fontweight='bold')
    ax1.grid(True, linestyle='--')
    ax1.legend()
    
    ax2.scatter(df_clean['Date'], df_clean['Mag'], s=25, alpha=0.6, edgecolors='none')
    ax2.axvline(x=main_date, color='r', linestyle='--')
    ax2.set_ylabel('Buyukluk (M)')
    ax2.set_xlabel('Tarih')
    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%d-%m %H:%M'))
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
        <h2>Deprem Analizi (Son {DAYS_BACK} Gün)</h2>
        <p><strong>Filtre:</strong> M {MIN_MAG} - {MAX_MAG}</p>
        <p><strong>Analize Giren Deprem Sayısı:</strong> {len(df_clean)}</p>
        <p><small>Son Güncelleme: {datetime.now().strftime("%d-%m-%Y %H:%M")}</small></p>
        <img src="b_analiz_grafik.png">
    </body>
    </html>
    """
    with open("index.html", "w", encoding="utf-8") as f: f.write(html_content)

if __name__ == "__main__":
    df = fetch_recent_data()
    analyze_and_plot(df)
