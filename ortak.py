"""
Ortak yardımcılar (03_oznitelik_muhendisligi.py ve 04_temel_model.py kullanır)

  - Tarih bölmeleri: test dönemi, ufuk analizi başlangıcı, genişleyen pencereli çapraz doğrulama katmanları
  - Hata ölçütleri: RMSPE, MAPE, MAE, RMSE (satışı 0 olan günler hesaba katılmaz)
  - İşlenmiş veriyi okuma, grafik stili ve Türkçe sayı biçimi (02_eda.py ile aynı görünüm)

Rapor 2.6–2.7'deki tanımlar burada tek yerde tutulur; tarih değişirse yalnızca bu dosya güncellenir.
"""
from pathlib import Path

import numpy as np
import pandas as pd

# --- rapor 2.6–2.7: tarihler ---------------------------------------------------------------------
VERI_BITIS = pd.Timestamp("2015-07-31")
VARSAYILAN_UFUK = 42                                  # H (gün): 6 hafta
TEST_BASLANGIC = pd.Timestamp("2015-06-20")           # verinin son 42 günü: 20.06.2015–31.07.2015
UFUK_ANALIZI_BASLANGIC = pd.Timestamp("2015-05-09")   # tüm H değerleri aynı başlangıçtan ölçülür
UFUK_DEGERLERI = (7, 28, 42, 84)
KATMAN_SAYISI = 3


def gun(n: int) -> pd.Timedelta:
    return pd.Timedelta(days=int(n))


def katmanlar(ufuk: int = VARSAYILAN_UFUK, degerlendirme_baslangic: pd.Timestamp = TEST_BASLANGIC,
              katman_sayisi: int = KATMAN_SAYISI) -> list:
    """Genişleyen pencereli (rolling-origin) zaman serisi çapraz doğrulama katmanları.

    Her katmanda doğrulama penceresi `ufuk` gündür ve eğitim, doğrulama başlangıcından önceki tüm
    günlerden oluşur. Son katmanın doğrulaması `degerlendirme_baslangic`tan bir gün önce biter;
    böylece hiperparametre seçimi değerlendirilecek (test / ufuk analizi) günleri hiç görmez.
    """
    sonuc = []
    for i in range(katman_sayisi):
        dog_bitis = degerlendirme_baslangic - gun(1 + i * ufuk)
        dog_bas = dog_bitis - gun(ufuk - 1)
        sonuc.append({"egitim_bitis": dog_bas - gun(1), "dogrulama_baslangic": dog_bas, "dogrulama_bitis": dog_bitis})
    sonuc.reverse()
    for k, s in enumerate(sonuc, start=1):
        s["ad"] = f"K{k}"
    return sonuc


def test_penceresi(ufuk: int = VARSAYILAN_UFUK) -> tuple:
    """Varsayılan ufuk için test penceresi (rapor 2.7)."""
    return TEST_BASLANGIC, TEST_BASLANGIC + gun(ufuk - 1)


def ufuk_penceresi(ufuk: int) -> tuple:
    """Ufuk analizi penceresi: 09.05.2015'ten başlayan `ufuk` günlük dönem."""
    return UFUK_ANALIZI_BASLANGIC, UFUK_ANALIZI_BASLANGIC + gun(ufuk - 1)


# --- hata ölçütleri (rapor 2.7) --------------------------------------------------------------------
def _ayikla(y, yhat):
    y = np.asarray(y, dtype=float)
    yhat = np.asarray(yhat, dtype=float)
    m = y > 0                      # RMSPE tanımı gereği satışı 0 olan günler hesaba katılmaz
    return y[m], yhat[m]


def rmspe(y, yhat) -> float:
    y, yhat = _ayikla(y, yhat)
    return float(np.sqrt(np.mean(((y - yhat) / y) ** 2)))


def metrikler(y, yhat) -> dict:
    y, yhat = _ayikla(y, yhat)
    e = y - yhat
    return {
        "n": int(len(y)),
        "RMSPE": float(np.sqrt(np.mean((e / y) ** 2))),
        "MAPE": float(np.mean(np.abs(e) / y)),
        "MAE": float(np.mean(np.abs(e))),
        "RMSE": float(np.sqrt(np.mean(e ** 2))),
    }


# --- veri okuma ------------------------------------------------------------------------------------
def oku_islenmis(girdi_dir: Path) -> pd.DataFrame:
    """01_veri_hazirlama.py çıktısını (tüm satırlar) okur."""
    for ad in ("islenmis_tam.parquet", "islenmis_tam.csv.gz"):
        yol = Path(girdi_dir) / ad
        if yol.exists():
            df = pd.read_parquet(yol) if ad.endswith(".parquet") else pd.read_csv(yol, parse_dates=["Date"])
            df["Date"] = pd.to_datetime(df["Date"])
            df["egitim_mi"] = df["egitim_mi"].astype(bool)
            return df.sort_values(["Store", "Date"]).reset_index(drop=True)
    raise FileNotFoundError(f"{girdi_dir} içinde islenmis_tam.* bulunamadı. Önce 01_veri_hazirlama.py çalıştırın.")


def kaydet_tablo(df: pd.DataFrame, taban: Path) -> Path:
    """Parquet varsa onu, yoksa sıkıştırılmış CSV kullanır (01_veri_hazirlama.py ile aynı kural)."""
    try:
        yol = Path(taban).with_suffix(".parquet")
        df.to_parquet(yol, index=False)
    except Exception:
        yol = Path(taban).with_suffix(".csv.gz")
        df.to_csv(yol, index=False)
    return yol


# --- biçim ve grafik stili (02_eda.py ile aynı) -----------------------------------------------------
MAVI, TURUNCU, TURKUAZ = "#2a78d6", "#eb6834", "#1baf7a"
MUREKKEP, MUREKKEP2, IZGARA = "#0b0b0b", "#52514e", "#e3e2dd"
GUNLER = ["Pzt", "Sal", "Çar", "Per", "Cum", "Cmt", "Paz"]


def sayi(v, ondalik=0) -> str:
    """Türkçe binlik ayracı ve ondalık virgül: 1.234,5"""
    return f"{v:,.{ondalik}f}".replace(",", "X").replace(".", ",").replace("X", ".")


def tarih(t) -> str:
    return pd.Timestamp(t).strftime("%d.%m.%Y")


def grafik_stili():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        "figure.facecolor": "white", "axes.facecolor": "white", "savefig.facecolor": "white",
        "axes.edgecolor": IZGARA, "axes.labelcolor": MUREKKEP2, "xtick.color": MUREKKEP2, "ytick.color": MUREKKEP2,
        "text.color": MUREKKEP, "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "axes.grid.axis": "y", "grid.color": IZGARA, "grid.linewidth": 0.8,
        "axes.axisbelow": True, "axes.titlelocation": "left", "axes.titlesize": 12, "axes.titleweight": "bold",
        "font.size": 10, "legend.frameon": False,
    })
    return plt
