import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import requests
import os
from datetime import datetime, timedelta
import time

# --- AYARLAR ---
START_DATE_FIXED = "2025-08-10T00:01:00"  # İlk başlangıç
ARXIV_DOSYASI = "deprem_katalog.csv"

# Koordinatlar (Manisa/Balıkesir)
MIN_LAT = 38.9382
MAX_LAT = 39.4073
MIN_LON = 27.7682
MAX_LON = 28.5698

# 24.000 deprem olduğu için pencereyi büyüttük, eğri daha düzgün çıkar
WINDOW_SIZE = 500  
STEP_SIZE = 10     

def fetch_chunk(start_str, end_str):
    """Belirli iki tarih arasını çeker"""
    url = "https://deprem.afad.gov.tr/apiserver/getEventSearchList"
    headers = {"Content-Type": "application/json"}
    payload = {
        "Start": start_str, "End": end_str,
        "MinLat": MIN_LAT, "MaxLat": MAX_LAT,
        "MinLon": MIN_LON, "MaxLon": MAX_LON,
        "MinMag": 0.0, "MaxMag": 9.9, "FilterType": "Location"
    }
    
    try:
        response = requests.post(url, headers=headers, json=payload, timeout=30)
        if response.status_code == 200:
            data = response.json()
            events = data if isinstance(data, list) else data.get('result', [])
            return events
    except:
        return []
    return []

def update_catalog():
    """Veriyi parça parça çekip arşive ekler"""
    
    # 1. Mevcut arşivi yükle veya yeni başlat
    if os.path.exists(ARXIV_DOSYASI):
        df_all = pd.read_csv(ARXIV_DOSYASI)
        df_all['Date'] = pd.to_datetime(df_all['Date'])
        # Arşivdeki en son tarih
        last_date = df_all['Date'].max()
        # Kaldığımız yerden devam et (1 saniye ekleyerek)
        current_start = last_date + timedelta(seconds=1)
        print(f"Arşiv bulundu. {len(df_all)} kayıt var. Son tarih: {last_date}")
    else:
        df_all = pd.DataFrame()
        current_start = datetime.fromisoformat(START_DATE_FIXED)
        print("Arşiv yok. Baştan başlanıyor...")

    now = datetime.now()
    
    # Eğer arşivdeki son tarih zaten güncelse işlem yapma
    if current_start >= now:
        print("Veri zaten güncel.")
        return df_all

    # 2. Döngü ile veri çek (5'er günlük dilimler)
    # Bu sayede 24.000 veriyi tıkanmadan çekeriz
    chunk_days = 5
    temp_list = []
    
    while current_start < now:
        current_end = current_start + timedelta(days=chunk_days)
        if current_end > now: current_end = now
        
        s_str = current_start.strftime("%Y-%m-%dT%H:%M:%S")
        e_str = current_end.strftime("%Y-%m-%dT%H:%M:%S")
        
        print(f"İndiriliyor: {s_str} -> {e_str}")
        events = fetch_chunk(s_str, e_str)
        
        if events:
            temp_list.extend(events)
            print(f"  -> {len(events)} deprem bulundu.")
        
        current_start = current_end
        time.sleep(0.5) # Sunucuyu kızdırmamak için bekleme

    # 3. Yeni verileri birleştir ve Kaydet
    if temp_list:
        df_new = pd.DataFrame(temp_list)
        df_new.rename(columns={'eventDate': 'Date', 'magnitude': 'Mag', 'latitude': 'Lat', 'longitude': 'Lon'}, inplace=True)
        df_new['Date'] = pd.to_datetime(df_new['Date'])
        df_new['Mag'] = pd.to_numeric(df_new['Mag'])
        
        # Sadece gerekli sütunlar
        df_new = df_new[['Date', 'Mag', 'Lat', 'Lon']]
        
        # Eski ve yeniyi birleştir
        df_all = pd.concat([df_all, df_new])
        
        # Çift kayıtları temizle (Date ve Mag aynıysa sil)
        df_all.drop_duplicates(subset=['Date', 'Lat', 'Mag'], inplace=True)
        df_all = df_all.sort_values(by='Date').reset_index(drop=True)
        
        # CSV'ye yaz
        df_all.to_csv(ARXIV_DOSYASI, index=False)
        print(f"Arşiv güncellendi! Toplam Kayıt: {len(df_all)}")
    
    return df_all

def calculate_mc(magnitudes):
    mag_rounded = np.round(magnitudes, 1)
    bins = np.arange(min(mag_rounded), max(mag_rounded) + 0.2, 0.1)
    counts, edges = np.histogram(mag_rounded, bins=bins)
    return edges[np.argmax(counts)]

def analyze_and_plot(df):
    if df.empty or len(df) < WINDOW_SIZE:
        print("Grafik için veri yetersiz.")
        # Boş grafik kaydet ki hata vermesin
        plt.figure(); plt.text(0.5,0.5,"Yetersiz Veri"); plt.savefig('b_analiz_grafik.png')
        return

    # Mc Hesabı
    mc = calculate_mc(df['Mag'])
    
    # Mc Filtresi
    df_clean = df[df['Mag'] >= mc].reset_index(drop=True)
    mags = df_clean['Mag'].values
    times = df_clean['Date'].values
    
    print(f"Mc: {mc}, Kullanılan Deprem Sayısı: {len(df_clean)}")

    if len(df_clean) < WINDOW_SIZE:
        plt.figure(); plt.text(0.5,0.5,"Mc Sonrası Yetersiz Veri"); plt.savefig('b_analiz_grafik.png')
        return

    # Mainshock
    main_idx = np.argmax(mags)
    main_date = times[main_idx]
    main_mag = mags[main_idx]

    # b-value Hesabı
    b_values, b_errors, plot_times = [], [], []
    num_windows = (len(mags) - WINDOW_SIZE) // STEP_SIZE + 1
    
    for i in range(num_windows):
        start = i * STEP_SIZE
        end = start + WINDOW_SIZE
        m_win = mags[start:end]
        mean_mag = np.mean(m_win)
        
        # Aki Formülü
        b = np.log10(np.exp(1)) / (mean_mag - (mc - 0.05))
        b_err = 2.3 * (b**2) * np.std(m_win) / np.sqrt(WINDOW_SIZE*(WINDOW_SIZE-1))
        
        b_values.append(b)
        b_errors.append(b_err)
        plot_times.append(pd.to_datetime(np.mean(times[start:end].astype(np.int64))))
    
    # GRAFİK ÇİZİMİ
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), sharex=True)
    
    # 1. Panel: b-value
    ax1.plot(plot_times, b_values, 'k-', linewidth=1.5, label='b-value')
    ax1.fill_between(plot_times, np.array(b_values)-np.array(b_errors), 
                     np.array(b_values)+np.array(b_errors), color='gray', alpha=0.3)
    ax1.axvline(x=main_date, color='r', linestyle='--', label=f'Mainshock (M{main_mag})')
    
    ax1.set_ylabel('b-value', fontsize=12)
    ax1.set_title(f'Temporal b-Value Evolution (N={len(df_clean)}, Mc={mc:.1f})\nWindow Size={WINDOW_SIZE}', 
                  fontsize=14, fontweight='bold')
    ax1.grid(True, which='both', linestyle='--', alpha=0.7)
    ax1.legend(loc='upper right')
    
    # 2. Panel: Sismisite
    ax2.scatter(df_clean['Date'], df_clean['Mag'], s=10, alpha=0.5, color='#0072BD')
    ax2.axvline(x=main_date, color='r', linestyle='--')
    
    ax2.set_ylabel('Magnitude', fontsize=12)
    ax2.set_xlabel('Date', fontsize=12)
    ax2.grid(True, which='both', linestyle='--', alpha=0.7)
    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    plt.xticks(rotation=45)
    
    plt.tight_layout()
    plt.savefig('b_analiz_grafik.png', dpi=150)
    
    # HTML Güncelle
    html_content = f"""
    <!DOCTYPE html>
    <html lang="tr">
    <head><meta charset="UTF-8"><meta http-equiv="refresh" content="1800">
    <title>Sismik Analiz</title>
    <style>body{{font-family:sans-serif;text-align:center;background:#f4f4f4;}} .box{{background:white;padding:20px;margin:auto;max-width:1200px;box-shadow:0 0 10px #ccc;}}</style>
    </head>
    <body>
        <div class="box">
            <h1>Manisa/Balıkesir Sismik b-Değeri Analizi</h1>
            <p><strong>Toplam Deprem:</strong> {len(df)} | <strong>Mc Üstü:</strong> {len(df_clean)}</p>
            <p><strong>Son Güncelleme:</strong> {datetime.now().strftime("%d-%m-%Y %H:%M")}</p>
            <img src="b_analiz_grafik.png" style="max-width:100%">
        </div>
    </body>
    </html>
    """
    with open("index.html", "w", encoding="utf-8") as f: f.write(html_content)

if __name__ == "__main__":
    df_total = update_catalog()
    analyze_and_plot(df_total)
