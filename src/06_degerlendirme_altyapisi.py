"""
İP 7 - Değerlendirme altyapısı: kayan kökenli doğrulama, Diebold-Mariano, alt grup kırılımları

Girdi : 01_veri_hazirlama.py çıktısı (<girdi-dir>/islenmis_tam.*)
Çıktı : data/tahminler/naive_tahminleri.parquet        tüm pencerelerde tahmin sözleşmesine uygun tahminler
        raporlar/ornek_tahmin_formati.csv               1000 satırlık örnek
        raporlar/hafta7_ozet.md                         sonuç özeti
        raporlar/hafta7_sonuclar.json                   makine-okunur sonuçlar
        gorseller/model/17_kayan_kokenli_rmspe.png
        gorseller/model/18_alt_grup_hatasi.png

Diebold-Mariano (DM) testi matematiksel formülasyonu:

  Kayıp fonksiyonu (RMSPE ile tutarlı):
    L_{m, i} = ((y_i - ŷ_{m, i}) / y_i)²   (yalnızca y_i > 0 olan açık mağaza-günleri)
    Kayıp farkı: d_i = L_{Naive, i} - L_{Naive-P, i}
    Pozitif d değeri Naive-P yönteminin hatasının daha düşük olduğunu gösterir.

  (a) Pencere düzeyi DM testi (k = 1..K pencereler, K = 21):
    Her pencere k için ortalama kayıp farkı:
      d_k = (1 / N_k) Σ_{i=1}^{N_k} (L_{Naive, k, i} - L_{Naive-P, k, i})
    Pencereler arası ortalama ve örneklem standart sapması:
      d̄ = (1 / K) Σ_{k=1}^K d_k
      s = √[ (1 / (K - 1)) Σ_{k=1}^K (d_k - d̄)² ]
      SE = s / √K
      t = d̄ / SE,   serbestlik derecesi df = K - 1
      İki taraflı p değeri: p = 2 · (1 - F_t(|t|, df))

    Newey-West (Bartlett çekirdeği) lag-1 otokorelasyon duyarlılığı:
      γ̂(0) = (1 / K) Σ_{k=1}^K (d_k - d̄)²
      γ̂(1) = (1 / K) Σ_{k=1}^{K-1} (d_k - d̄)(d_{k+1} - d̄)
      LRV_{lag1} = γ̂(0) + 2 · (1 - 1/2) · γ̂(1) = γ̂(0) + γ̂(1)
      SE_{NW1} = √[ max(LRV_{lag1}, 0) / K ]
      t_{NW1} = d̄ / SE_{NW1},   p_{lag1} = 2 · (1 - F_t(|t_{NW1}|, df))

  (b) Günlük DM testi (test penceresi, t = 1..T, T = 42 gün):
    Her gün t için mağazalar üzerinden ortalama kayıp farkı:
      d_t = (1 / S_t) Σ_{s=1}^{S_t} (L_{Naive, s, t} - L_{Naive-P, s, t})
    Günlük ortalama:
      d̄ = (1 / T) Σ_{t=1}^T d_t
    Newey-West uzun dönem varyans (Bartlett çekirdeği):
      Bant genişliği: m = floor( 4 · (T / 100)^(2/9) )   (T = 42 için m = 3)
      Otokovaryanslar: γ̂(h) = (1 / T) Σ_{t=1}^{T-h} (d_t - d̄)(d_{t+h} - d̄)
      LRV = γ̂(0) + 2 · Σ_{h=1}^m [ 1 - h / (m + 1) ] · γ̂(h)
      SE_{NW} = √[ max(LRV, 0) / T ]
      t = d̄ / SE_{NW},   serbestlik derecesi df = T - 1 = 41
      İki taraflı p = 2 · (1 - F_t(|t|, df))
      Duyarlılık analizi: m ∈ {0, 3, 6, 13} için ayrı ayrı hesaplanır.

Kullanım:
  python src/06_degerlendirme_altyapisi.py --girdi-dir data/islenmis --cikti-dir data/tahminler
  python src/06_degerlendirme_altyapisi.py --girdi-dir data/islenmis --dogrula data/tahminler/naive_tahminleri.parquet
"""
import argparse
import importlib
import json
import math
import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

import numpy as np
import pandas as pd
from matplotlib.ticker import FuncFormatter

# --- modül yolunu ayarla: src/ dizinini sys.path'e ekle ---
_KOK = Path(__file__).resolve().parent
if str(_KOK) not in sys.path:
    sys.path.insert(0, str(_KOK))

_tm_mod = importlib.import_module("04_temel_model")
_pm_mod = importlib.import_module("05_promosyon_duzeltmesi")
import ortak
from ortak import (MAVI, MUREKKEP2, TURKUAZ, TURUNCU, VARSAYILAN_UFUK,
                   gun, grafik_stili, metrikler, oku_islenmis,
                   rmspe, sayi, tarih, test_penceresi)

mevsimsel_naive = _tm_mod.mevsimsel_naive
naive_p = _pm_mod.naive_p

TEST_BAS = ortak.TEST_BASLANGIC
UFUK = VARSAYILAN_UFUK  # 42
VERI_BAS = pd.Timestamp("2013-01-01")

# =============================================================================
# TASK 1 — Tahmin sözleşmesi doğrulayıcı
# =============================================================================

ZORUNLU_SUTUNLAR = {"Store", "Date", "Sales", "Prediction", "Model", "H", "Origin", "Split"}
ANAHTAR_SUTUNLAR = ["Model", "H", "Origin", "Store", "Date"]


def tahmin_dogrula(pred_df: pd.DataFrame, gercek_df: pd.DataFrame) -> list:
    """Tahmin DataFrame'ini sözleşme kurallarına göre doğrular ve sorun listesi döndürür.

    Kurallar:
      1. Zorunlu sütunlar: Store, Date, Sales, Prediction, Model, H, Origin, Split
      2. (Model, H, Origin, Store, Date) birleşik anahtarı tekil olmalı
      3. Eksik veya sonsuz tahmin yok
      4. Tahminler satış ölçeğinde (negatif olmayan)
      5. Kapalı günlerde (Open=0) tahmin tam olarak 0 olmalı
      6. Pozitif satışlı satırlarda RMSPE/MAPE hesaplanabilir olmalı
      7. Modeller arası karşılaştırmada anahtar kümeleri birebir aynı olmalı
    """
    sorunlar = []

    # 1. Zorunlu sütun kontrolü
    eksik_sut = ZORUNLU_SUTUNLAR - set(pred_df.columns)
    if eksik_sut:
        sorunlar.append(f"Eksik sütunlar: {sorted(list(eksik_sut))}")
        return sorunlar

    # 2. Tekil anahtar kontrolü
    dup = pred_df.duplicated(subset=ANAHTAR_SUTUNLAR, keep=False)
    if dup.any():
        sorunlar.append(f"Yinelenmiş anahtar: {int(dup.sum())} satır")

    # 3. Eksik ve sonsuz tahmin kontrolü
    eksik_tahmin = pred_df["Prediction"].isna().sum()
    if eksik_tahmin > 0:
        sorunlar.append(f"Eksik tahmin: {int(eksik_tahmin)} satır")
    pred_vals = pred_df["Prediction"].values.astype(float)
    sonsuz = np.isinf(pred_vals).sum()
    if sonsuz > 0:
        sorunlar.append(f"Sonsuz tahmin: {int(sonsuz)} satır")

    # 4. Satış ölçeği (negatif tahmin kontrolü)
    negatif = (pred_vals < 0).sum()
    if negatif > 0:
        sorunlar.append(f"Negatif tahmin: {int(negatif)} satır (satış ölçeğinde olmalı)")

    # 5. Kapalı günlerde (Open=0) tahmin = 0 kontrolü
    if "Open" in pred_df.columns:
        kapali = pred_df[pred_df["Open"] == 0]
    else:
        kapali_key = gercek_df[gercek_df["Open"] == 0][["Store", "Date"]].drop_duplicates()
        kapali = pred_df.merge(kapali_key, on=["Store", "Date"], how="inner")
    if len(kapali) > 0:
        sifir_olmayan = (kapali["Prediction"] != 0).sum()
        if sifir_olmayan > 0:
            sorunlar.append(f"Kapalı günlerde sıfır olmayan tahmin: {int(sifir_olmayan)} satır")

    # 6. Pozitif satış kontrolü
    pozitif = pred_df[pred_df["Sales"] > 0]
    if len(pozitif) == 0:
        sorunlar.append("Pozitif satışlı satır yok — metrik hesaplanamaz")

    # 7. Modeller arası anahtar kümesi tutarlılığı
    modeller = pred_df["Model"].unique()
    if len(modeller) > 1:
        anahtar_kumeleri = {}
        for m in modeller:
            m_df = pred_df[pred_df["Model"] == m]
            keys = set(zip(m_df["H"], m_df["Origin"], m_df["Store"], m_df["Date"]))
            anahtar_kumeleri[m] = keys
        ref_model = modeller[0]
        for m in modeller[1:]:
            if anahtar_kumeleri[m] != anahtar_kumeleri[ref_model]:
                fark = len(anahtar_kumeleri[m].symmetric_difference(anahtar_kumeleri[ref_model]))
                sorunlar.append(f"Modeller '{ref_model}' ve '{m}' arasında {fark} farklı anahtar var")

    return sorunlar


# =============================================================================
# TASK 2 — Kayan kökenli değerlendirme pencereleri
# =============================================================================

def tum_pencereler() -> list:
    """Test başlangıcından geriye doğru 42 günlük örtüşmeyen pencereler.

    K3 = 09.05-19.06.2015, K2 = 28.03-08.05.2015, K1 = 14.02-27.03.2015,
    sonra geriye devam (RO17 .. RO01).
    Origin = pencere başlangıcı - 1 gün.
    Origin'in en az 28 gün geçmişi olmalı.
    Test = 20.06-31.07.2015.
    """
    pencereler = []

    simdiki_bit = TEST_BAS - gun(1)  # K3 bitişi = 19.06.2015
    while True:
        bas = simdiki_bit - gun(UFUK - 1)
        origin = bas - gun(1)
        # Origin'den önce en az 28 gün veri olmalı
        if (origin - VERI_BAS).days + 1 < 28:
            break
        pencereler.append({
            "bas": bas,
            "bit": simdiki_bit,
            "origin": origin,
        })
        simdiki_bit = bas - gun(1)

    pencereler.reverse()

    n = len(pencereler)
    for i, p in enumerate(pencereler):
        if i >= n - 3:
            k_no = i - (n - 3) + 1  # 1, 2, 3 -> K1, K2, K3
            p["ad"] = f"K{k_no}"
        else:
            ro_no = i + 1
            p["ad"] = f"RO{ro_no:02d}"

    test_bas, test_bit = test_penceresi()
    pencereler.append({
        "ad": "Test",
        "bas": test_bas,
        "bit": test_bit,
        "origin": test_bas - gun(1),
    })

    return pencereler


def tahmin_uret(df: pd.DataFrame, baslangic: pd.Timestamp, bitis: pd.Timestamp,
                yontem: str, ad: str) -> pd.DataFrame:
    """Verilen pencere ve yöntem için tahmin üretir, sözleşme formatında."""
    origin = baslangic - gun(1)

    if yontem == "Naive":
        t = mevsimsel_naive(df, baslangic, bitis)
    elif yontem == "Naive-P":
        t = naive_p(df, baslangic, bitis)
    else:
        raise ValueError(f"Bilinmeyen yöntem: {yontem}")

    # Geçmişte hiç açık günü olmayan yeni mağazalar için zincir genel ortalaması ile tamamlama
    acik = df[df["egitim_mi"] & (df["Date"] <= origin)]
    genel_ort = float(acik["Sales"].mean())
    tahmin_duz = np.where(t["Open"] == 1, t["Tahmin"].fillna(genel_ort), 0.0)

    sonuc = pd.DataFrame({
        "Store": t["Store"].astype(int),
        "Date": pd.to_datetime(t["Date"]),
        "Sales": t["Sales"].astype(float),
        "Prediction": tahmin_duz.astype(float),
        "Open": t["Open"].astype(int),
        "Model": yontem,
        "H": int(UFUK),
        "Origin": pd.to_datetime(origin),
        "Split": ad,
        "egitim_mi": t["egitim_mi"].astype(bool),
    })
    return sonuc


def hesapla_metrikler(pred_df: pd.DataFrame) -> dict:
    """Pozitif satışlı satırlarda RMSPE, MAPE, MAE, RMSE hesaplar."""
    e = pred_df[pred_df["egitim_mi"] & (pred_df["Sales"] > 0)]
    if len(e) == 0:
        return {"rmspe": np.nan, "mape": np.nan, "mae": np.nan, "rmse": np.nan,
                "n": 0, "n_magaza": 0}
    m = metrikler(e["Sales"], e["Prediction"])
    return {
        "rmspe": float(m["RMSPE"]),
        "mape": float(m["MAPE"]),
        "mae": float(m["MAE"]),
        "rmse": float(m["RMSE"]),
        "n": int(m["n"]),
        "n_magaza": int(e["Store"].nunique()),
    }


# =============================================================================
# TASK 3 — Diebold-Mariano testleri
# =============================================================================

def newey_west_se(d: np.ndarray, lag: int) -> float:
    """Newey-West (Bartlett çekirdeği) standart hatası.

    γ̂(h) = (1/T) Σ_{t=1}^{T-h} (d_t - d̄)(d_{t+h} - d̄)
    LRV = γ̂(0) + 2 Σ_{h=1}^{m} [1 - h/(m+1)] γ̂(h)
    SE = √(max(LRV, 0) / T)
    """
    T = len(d)
    d_bar = float(d.mean())
    dd = d - d_bar

    def gamma(h):
        return float(np.sum(dd[:T - h] * dd[h:])) / T

    lrv = gamma(0)
    for h in range(1, lag + 1):
        w = 1.0 - (h / (lag + 1.0))
        lrv += 2.0 * w * gamma(h)

    return float(np.sqrt(max(lrv, 0.0) / T))


def dm_pencere(d_pencereleri: np.ndarray) -> dict:
    """Pencere düzeyinde DM testi.

    d_k = pencere k'daki ortalama kayıp farkı (Naive - Naive-P).
    H0: E[d_k] = 0, iki taraflı Student-t testi.
    """
    from scipy import stats

    K = len(d_pencereleri)
    d_bar = float(d_pencereleri.mean())

    if np.all(d_pencereleri == 0):
        return {"K": K, "ort_fark": 0.0, "se": np.nan, "stat": np.nan,
                "df": K - 1, "p": np.nan, "p_lag1": np.nan,
                "uygulanamaz": True}

    se = float(d_pencereleri.std(ddof=1) / np.sqrt(K))
    t_stat = float(d_bar / se) if se > 0 else np.nan
    p_val = float(2.0 * stats.t.sf(abs(t_stat), K - 1)) if not np.isnan(t_stat) else np.nan

    se_nw = newey_west_se(d_pencereleri, 1)
    t_nw = float(d_bar / se_nw) if se_nw > 0 else np.nan
    p_nw = float(2.0 * stats.t.sf(abs(t_nw), K - 1)) if not np.isnan(t_nw) else np.nan

    return {
        "K": K, "ort_fark": d_bar, "se": se, "stat": t_stat,
        "df": K - 1, "p": p_val, "p_lag1": p_nw,
        "uygulanamaz": False,
    }


def dm_gunluk(d_gunluk: np.ndarray) -> dict:
    """Günlük DM testi (test penceresi).

    d_t = mağazalar üzerinden ortalama kayıp farkı (gün t).
    Newey-West Bartlett, m = floor(4·(T/100)^(2/9)).
    """
    from scipy import stats

    T = len(d_gunluk)
    d_bar = float(d_gunluk.mean())
    m = int(math.floor(4.0 * (T / 100.0) ** (2.0 / 9.0)))

    if np.all(d_gunluk == 0):
        return {"T": T, "lag": m, "ort_fark": 0.0, "se": np.nan, "stat": np.nan,
                "df": T - 1, "p": np.nan,
                "duyarlilik": {f"lag{l}": np.nan for l in [0, 3, 6, 13]},
                "uygulanamaz": True}

    se = newey_west_se(d_gunluk, m)
    t_stat = float(d_bar / se) if se > 0 else np.nan
    p_val = float(2.0 * stats.t.sf(abs(t_stat), T - 1)) if not np.isnan(t_stat) else np.nan

    duyarlilik = {}
    for l in [0, 3, 6, 13]:
        se_l = newey_west_se(d_gunluk, l)
        t_l = float(d_bar / se_l) if se_l > 0 else np.nan
        p_l = float(2.0 * stats.t.sf(abs(t_l), T - 1)) if not np.isnan(t_l) else np.nan
        duyarlilik[f"lag{l}"] = p_l

    return {
        "T": T, "lag": m, "ort_fark": d_bar, "se": se, "stat": t_stat,
        "df": T - 1, "p": p_val,
        "duyarlilik": duyarlilik,
        "uygulanamaz": False,
    }


def dm_kendi_kontrol(tohum: int = 42) -> dict:
    """DM testinin kendi kendini denetlemesi.

    (i) Aynı tahmin → d=0, 'test uygulanamaz' dönmeli.
    (ii) Sentetik fark → model A bariz üstünse p < 0.05 ile ret.
    (iii) Aynı dağılım → 200 tekrarda ret oranı tipik olarak %5 civarında (< 0.15).
    """
    rng = np.random.default_rng(tohum)

    # (i) Aynı tahmin
    n = 100
    y = rng.uniform(100, 10000, n)
    yhat = y.copy()
    L1 = ((y - yhat) / y) ** 2
    d_ayni = L1 - L1
    d_pencereleri = np.array([d_ayni[i * 10:(i + 1) * 10].mean() for i in range(10)])
    sonuc_ayni = dm_pencere(d_pencereleri)
    ayni_ok = sonuc_ayni["uygulanamaz"]

    # (ii) Sentetik fark: model A bariz iyi
    n = 500
    y = rng.uniform(100, 10000, n)
    yhat_A = y + rng.normal(0, 50, n)
    yhat_B = y + rng.normal(500, 500, n)
    L_A = ((y - yhat_A) / y) ** 2
    L_B = ((y - yhat_B) / y) ** 2
    d_fark = L_B - L_A
    k = 10
    beden = n // k
    d_p = np.array([d_fark[i * beden:(i + 1) * beden].mean() for i in range(k)])
    sonuc_fark = dm_pencere(d_p)
    fark_ok = (sonuc_fark["p"] < 0.05) if not np.isnan(sonuc_fark["p"]) else False

    # (iii) Aynı dağılım: 200 tekrar
    ret_sayisi = 0
    tekrar = 200
    for r in range(tekrar):
        rng_r = np.random.default_rng(tohum + r + 1000)
        y_r = rng_r.uniform(100, 10000, n)
        yhat_1 = y_r + rng_r.normal(0, 300, n)
        yhat_2 = y_r + rng_r.normal(0, 300, n)
        L1_r = ((y_r - yhat_1) / y_r) ** 2
        L2_r = ((y_r - yhat_2) / y_r) ** 2
        d_r = L1_r - L2_r
        d_p_r = np.array([d_r[i * beden:(i + 1) * beden].mean() for i in range(k)])
        s = dm_pencere(d_p_r)
        if not np.isnan(s["p"]) and s["p"] < 0.05:
            ret_sayisi += 1
    ret_orani = float(ret_sayisi / tekrar)

    return {
        "ayni_tahmin_uygulanamaz": bool(ayni_ok),
        "sentetik_fark_red": bool(fark_ok),
        "sentetik_ayni_red_orani": ret_orani,
    }


# =============================================================================
# TASK 4 — Alt grup kırılımları
# =============================================================================

def en_kotu_magazalar(test_np: pd.DataFrame, gercek_df: pd.DataFrame, n: int = 10) -> list:
    """Naive-P test RMSPE'sine göre en kötü n mağaza."""
    e = test_np[test_np["egitim_mi"] & (test_np["Sales"] > 0)]
    mg_rmspe = (
        e.groupby("Store")
        .apply(lambda g: float(rmspe(g["Sales"], g["Prediction"])))
        .rename("naive_p_rmspe")
        .reset_index()
        .sort_values("naive_p_rmspe", ascending=False)
        .head(n)
    )

    bilgi = gercek_df[["Store", "StoreType", "Assortment", "CompetitionDistance", "Promo2"]].drop_duplicates("Store")
    mg_rmspe = mg_rmspe.merge(bilgi, on="Store", how="left")

    sonuc = []
    for _, s in mg_rmspe.iterrows():
        cd_val = float(s["CompetitionDistance"]) if pd.notna(s["CompetitionDistance"]) else None
        sonuc.append({
            "Store": int(s["Store"]),
            "naive_p_rmspe": float(s["naive_p_rmspe"]),
            "StoreType": str(s["StoreType"]),
            "Assortment": str(s["Assortment"]),
            "CompetitionDistance": cd_val,
            "Promo2": int(s["Promo2"]),
        })
    return sonuc


# =============================================================================
# Grafikler
# =============================================================================

def grafik_kayan_kokenli(plt, pencere_sonuc: list, cikti: Path):
    """17_kayan_kokenli_rmspe.png: Naive vs Naive-P per window in chronological order."""
    fig, ax = plt.subplots(figsize=(12, 5))

    adlar = [p["ad"] for p in pencere_sonuc]
    naive_vals = [p["naive"]["rmspe"] for p in pencere_sonuc]
    naive_p_vals = [p["naive_p"]["rmspe"] for p in pencere_sonuc]

    x = np.arange(len(adlar))
    w = 0.35

    b1 = ax.bar(x - w / 2, naive_vals, w, color=MAVI, label="Standart naive")
    b2 = ax.bar(x + w / 2, naive_p_vals, w, color=TURUNCU, label="Naive-P")

    ax.bar_label(b1, labels=[sayi(v, 2) for v in naive_vals], padding=3, fontsize=6.5, color=MUREKKEP2)
    ax.bar_label(b2, labels=[sayi(v, 2) for v in naive_p_vals], padding=3, fontsize=6.5, color=MUREKKEP2)

    ax.axhline(0.15, color=TURKUAZ, linestyle="--", linewidth=1.2, label="Eşik (0,15)")
    ax.set_xticks(x)
    ax.set_xticklabels(adlar, fontsize=8)

    # K1-K3 ve Test etiketlerini vurgula
    for tick_label in ax.get_xticklabels():
        txt = tick_label.get_text()
        if txt in ["K1", "K2", "K3", "Test"]:
            tick_label.set_fontweight("bold")
            tick_label.set_color("#0b0b0b")

    ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: sayi(y, 2)))
    ax.set_ylabel("RMSPE")
    max_val = max(max(naive_vals), max(naive_p_vals))
    ax.set_ylim(0, max_val * 1.15)
    ax.set_title("Kayan kökenli doğrulama: Naive ve Naive-P pencere bazında RMSPE")
    ax.legend(loc="upper left", fontsize=9)

    fig.tight_layout()
    fig.savefig(cikti, dpi=150)
    plt.close(fig)
    print(f"[OK] {cikti} kaydedildi.")


def grafik_altgrup(plt, mt_dict: dict, cikti: Path):
    """18_alt_grup_hatasi.png: Naive-P and Naive RMSPE by StoreType on test window."""
    tipler = ["a", "b", "c", "d"]
    naive_vals = [mt_dict[t]["naive"] for t in tipler]
    naive_p_vals = [mt_dict[t]["naive_p"] for t in tipler]
    n_magaza = [mt_dict[t]["n_magaza"] for t in tipler]

    x = np.arange(len(tipler))
    w = 0.35

    fig, ax = plt.subplots(figsize=(8, 5))
    b1 = ax.bar(x - w / 2, naive_vals, w, color=MAVI, label="Standart naive")
    b2 = ax.bar(x + w / 2, naive_p_vals, w, color=TURUNCU, label="Naive-P")

    ax.bar_label(b1, labels=[sayi(v, 3) for v in naive_vals], padding=3, fontsize=9, color=MUREKKEP2)
    ax.bar_label(b2, labels=[sayi(v, 3) for v in naive_p_vals], padding=3, fontsize=9, color=MUREKKEP2)

    ax.set_xticks(x)
    ax.set_xticklabels([f"Tip {t}\n({sayi(n)} mağaza)" for t, n in zip(tipler, n_magaza)], fontsize=9)
    ax.axhline(0.15, color=TURKUAZ, linestyle="--", linewidth=1.2, label="Eşik (0,15)")

    ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: sayi(y, 2)))
    ax.set_ylabel("RMSPE")
    max_val = max(max(naive_vals), max(naive_p_vals))
    ax.set_ylim(0, max_val * 1.25)
    ax.set_title("Test penceresi: mağaza tipine göre RMSPE")
    ax.legend(loc="upper right", fontsize=9)

    fig.tight_layout()
    fig.savefig(cikti, dpi=150)
    plt.close(fig)
    print(f"[OK] {cikti} kaydedildi.")


# =============================================================================
# Rapor
# =============================================================================

def rapor_yaz(json_sonuc: dict, cikti: Path, alt_grup_havuz: dict = None):
    """raporlar/hafta7_ozet.md yazar (hafta5_ozet.md stiliyle)."""
    r = ["# Değerlendirme altyapısı (Hafta 7)", "",
         "`06_degerlendirme_altyapisi.py` tarafından hesaplanmıştır.", ""]

    # Pencere tablosu
    r += ["## Kayan kökenli doğrulama pencereleri (Rolling-Origin Evaluation)", "",
          f"Toplam pencere sayısı: {json_sonuc['pencere_sayisi']} "
          f"({json_sonuc['pencere_sayisi'] - 1} doğrulama + 1 test penceresi, her biri 42 gün).", ""]

    r += ["| Pencere | Başlangıç | Bitiş | Gözlem | Mağaza | Naive RMSPE | Naive-P RMSPE | İyileşme |",
          "|---|---|---|---|---|---|---|---|"]
    for p in json_sonuc["pencereler"]:
        iy = p["iyilesme_rmspe_yuzde"]
        bas_tr = pd.Timestamp(p["bas"]).strftime("%d.%m.%Y")
        bit_tr = pd.Timestamp(p["bit"]).strftime("%d.%m.%Y")
        r.append(f"| {p['ad']} | {bas_tr} | {bit_tr} | {sayi(p['n'])} | {sayi(p['n_magaza'])} | "
                 f"{sayi(p['naive']['rmspe'], 3)} | {sayi(p['naive_p']['rmspe'], 3)} | %{sayi(iy, 1)} |")
    r.append("")

    # Özet istatistikler
    oz = json_sonuc["ozet"]
    r += ["## Özet istatistikler (RMSPE)", "",
          "| Ölçüt | Standart Naive | Naive-P |", "|---|---|---|",
          f"| Ortalama | {sayi(oz['naive']['ort'], 3)} | {sayi(oz['naive_p']['ort'], 3)} |",
          f"| Medyan | {sayi(oz['naive']['medyan'], 3)} | {sayi(oz['naive_p']['medyan'], 3)} |",
          f"| En düşük (Min) | {sayi(oz['naive']['min'], 3)} | {sayi(oz['naive_p']['min'], 3)} |",
          f"| En yüksek (Maks) | {sayi(oz['naive']['max'], 3)} | {sayi(oz['naive_p']['max'], 3)} |",
          f"| RMSPE ≤ 0,15 olan pencere sayısı | {oz['esik_alti_naive']} | {oz['esik_alti_naive_p']} |",
          f"| Naive-P'nin Naive'den iyi olduğu pencere sayısı | — | {oz['naive_p_daha_iyi']} / {json_sonuc['pencere_sayisi']} |",
          ""]

    # Diebold-Mariano testleri
    dm_p = json_sonuc["dm_pencere"]
    r += ["## Diebold-Mariano testleri", "",
          "Kayıp fonksiyonu: L = ((y - ŷ) / y)² (kare yüzde hatası, RMSPE ile tutarlı). "
          "Fark d = L(Naive) - L(Naive-P); pozitif d değeri Naive-P lehinedir.", "",
          "### (a) Pencere düzeyi DM testi (K = 21 pencere)", "",
          f"- Ortalama kayıp farkı (d̄): {sayi(dm_p['ort_fark'], 6)}",
          f"- Standart hata (SE): {sayi(dm_p['se'], 6)}",
          f"- t istatistiği: {sayi(dm_p['stat'], 3)}",
          f"- Serbestlik derecesi: {dm_p['df']}",
          f"- İki taraflı p değeri: {sayi(dm_p['p'], 4)}",
          f"- Newey-West lag-1 duyarlılığı p değeri: {sayi(dm_p['p_lag1'], 4)}",
          ""]

    dm_g = json_sonuc["dm_gunluk_test"]
    r += ["### (b) Günlük DM testi (test penceresi, T = 42 gün)", "",
          f"- Ortalama kayıp farkı (d̄): {sayi(dm_g['ort_fark'], 6)}",
          f"- Bant genişliği (m = floor(4·(T/100)^(2/9))): {dm_g['lag']}",
          f"- Newey-West standart hata (SE): {sayi(dm_g['se'], 6)}",
          f"- t istatistiği: {sayi(dm_g['stat'], 3)}",
          f"- Serbestlik derecesi: {dm_g['df']}",
          f"- İki taraflı p değeri: {sayi(dm_g['p'], 4)}", "",
          "Bant genişliği duyarlılık analizi:", "",
          "| Gecikme (lag) | p değeri |", "|---|---|"]
    for k, v in dm_g["duyarlilik"].items():
        r.append(f"| {k} | {sayi(v, 4)} |")
    r.append("")

    dm_k = json_sonuc["dm_kendi_kontrol"]
    r += ["### (c) DM testi doğrulama ve kendi kendini denetleme", "",
          f"- Özdeş tahminler testi: {'geçti (test uygulanamaz bildirildi)' if dm_k['ayni_tahmin_uygulanamaz'] else 'KALDI'}",
          f"- Belirgin hata farkı olan sentetik veri (p < 0,05 ret): {'geçti' if dm_k['sentetik_fark_red'] else 'KALDI'}",
          f"- Aynı dağılımdan sentetik veri (200 tekrar ret oranı): %{sayi(dm_k['sentetik_ayni_red_orani'] * 100, 1)} (beklenen: <%15)",
          ""]

    # Alt grup kırılımları
    r += ["## Alt grup kırılımları (Test penceresi)", ""]

    # Mağaza tipi
    mt = json_sonuc["alt_grup_test"]["magaza_tipi"]
    r += ["### Mağaza tipine göre", "",
          "| Mağaza Tipi | Mağaza Sayısı | Satır Sayısı | Standart Naive RMSPE | Naive-P RMSPE |",
          "|---|---|---|---|---|"]
    for t in ["a", "b", "c", "d"]:
        d = mt[t]
        r.append(f"| Tip {t} | {sayi(d['n_magaza'])} | {sayi(d['n_satir'])} | "
                 f"{sayi(d['naive'], 3)} | {sayi(d['naive_p'], 3)} |")
    r.append("")

    # Promosyon
    pr = json_sonuc["alt_grup_test"]["promo"]
    r += ["### Promosyon durumuna göre", "",
          "| Promosyon | Standart Naive RMSPE | Naive-P RMSPE |", "|---|---|---|",
          f"| Promo = 0 (Yok) | {sayi(pr['0']['naive'], 3)} | {sayi(pr['0']['naive_p'], 3)} |",
          f"| Promo = 1 (Var) | {sayi(pr['1']['naive'], 3)} | {sayi(pr['1']['naive_p'], 3)} |",
          ""]

    # Devlet tatili
    dt = json_sonuc["alt_grup_test"]["devlet_tatili"]
    r += ["### Devlet tatiline göre", "",
          "| Devlet Tatili | Standart Naive RMSPE | Naive-P RMSPE |", "|---|---|---|",
          f"| Tatil Yok (0) | {sayi(dt['0']['naive'], 3)} | {sayi(dt['0']['naive_p'], 3)} |",
          f"| Resmî Tatil (any) | {sayi(dt['any']['naive'], 3)} | {sayi(dt['any']['naive_p'], 3)} |",
          ""]

    # Okul tatili
    ot = json_sonuc["alt_grup_test"]["okul_tatili"]
    r += ["### Okul tatiline göre", "",
          "| Okul Tatili | Standart Naive RMSPE | Naive-P RMSPE |", "|---|---|---|",
          f"| Tatil Yok (0) | {sayi(ot['0']['naive'], 3)} | {sayi(ot['0']['naive_p'], 3)} |",
          f"| Okul Tatili (1) | {sayi(ot['1']['naive'], 3)} | {sayi(ot['1']['naive_p'], 3)} |",
          ""]

    # Haftanın günü
    hg = json_sonuc["alt_grup_test"]["haftanin_gunu"]
    gun_adlari = {"1": "Pazartesi", "2": "Salı", "3": "Çarşamba", "4": "Perşembe",
                  "5": "Cuma", "6": "Cumartesi", "7": "Pazar"}
    r += ["### Haftanın gününe göre", "",
          "| Gün | Standart Naive RMSPE | Naive-P RMSPE |", "|---|---|---|"]
    for k in [str(i) for i in range(1, 8)]:
        if k in hg:
            r.append(f"| {k} ({gun_adlari[k]}) | {sayi(hg[k]['naive'], 3)} | {sayi(hg[k]['naive_p'], 3)} |")
    r.append("")

    # Haftalık test
    r += ["### Test penceresi haftalık hata (6 hafta)", "",
          "| Hafta | Dönem | Standart Naive RMSPE | Naive-P RMSPE |", "|---|---|---|---|"]
    for h in json_sonuc["haftalik_test"]:
        h_no = h["hafta"]
        h_bas = (pd.Timestamp("2015-06-20") + gun((h_no - 1) * 7)).strftime("%d.%m")
        h_bit = (pd.Timestamp("2015-06-20") + gun(h_no * 7 - 1)).strftime("%d.%m.%Y")
        r.append(f"| {h_no}. Hafta | {h_bas}–{h_bit} | {sayi(h['naive'], 3)} | {sayi(h['naive_p'], 3)} |")
    r.append("")

    # Havuzlanmış kırılımlar
    if alt_grup_havuz is not None:
        r += ["## Alt grup kırılımları (Tüm 21 pencere havuzlanmış)", "",
              "### Mağaza tipine göre (tüm pencereler)", "",
              "| Mağaza Tipi | Mağaza Sayısı | Satır Sayısı | Standart Naive RMSPE | Naive-P RMSPE |",
              "|---|---|---|---|---|"]
        for t in ["a", "b", "c", "d"]:
            d = alt_grup_havuz["magaza_tipi"][t]
            r.append(f"| Tip {t} | {sayi(d['n_magaza'])} | {sayi(d['n_satir'])} | "
                     f"{sayi(d['naive'], 3)} | {sayi(d['naive_p'], 3)} |")
        r.append("")

        r += ["### Promosyon durumuna göre (tüm pencereler)", "",
              "| Promosyon | Standart Naive RMSPE | Naive-P RMSPE |", "|---|---|---|",
              f"| Promo = 0 (Yok) | {sayi(alt_grup_havuz['promo']['0']['naive'], 3)} | {sayi(alt_grup_havuz['promo']['0']['naive_p'], 3)} |",
              f"| Promo = 1 (Var) | {sayi(alt_grup_havuz['promo']['1']['naive'], 3)} | {sayi(alt_grup_havuz['promo']['1']['naive_p'], 3)} |",
              ""]

    # En kötü 10 mağaza
    r += ["## En kötü 10 mağaza (Naive-P test RMSPE)", "",
          "| Sıra | Mağaza | Naive-P RMSPE | Tip | Çeşit | Rakip Mesafesi (m) | Promo2 |",
          "|---|---|---|---|---|---|---|"]
    for idx, m in enumerate(json_sonuc["en_kotu_magazalar"], start=1):
        cd_str = sayi(m["CompetitionDistance"]) if m["CompetitionDistance"] is not None else "—"
        r.append(f"| {idx} | {m['Store']} | {sayi(m['naive_p_rmspe'], 3)} | {m['StoreType']} | "
                 f"{m['Assortment']} | {cd_str} | {m['Promo2']} |")
    r.append("")

    # Tahmin sözleşmesi doğrulayıcı
    dg = json_sonuc["dogrulayici"]
    r += ["## Tahmin sözleşmesi doğrulaması", "",
          f"- Toplam kontrol sayısı: {dg['kontrol_sayisi']}",
          f"- Başarıyla geçen kontrol sayısı: {dg['gecen']}",
          f"- Bilerek bozulan tahminlerin yakalanması: {'Başarılı' if dg['bilerek_bozulan_yakalandi'] else 'Başarısız'}",
          ""]

    metin = "\n".join(r) + "\n"
    cikti.write_text(metin, encoding="utf-8")
    print(f"[OK] {cikti} kaydedildi.")


# =============================================================================
# MAIN
# =============================================================================

def main():
    ap = argparse.ArgumentParser(description="İP 7 - Değerlendirme altyapısı")
    ap.add_argument("--girdi-dir", type=Path, default=Path("data/islenmis"))
    ap.add_argument("--cikti-dir", type=Path, default=Path("data/tahminler"))
    ap.add_argument("--dogrula", type=Path, default=None,
                    help="Tahmin dosyasını doğrula (parquet veya csv)")
    args = ap.parse_args()

    Path("gorseller/model").mkdir(parents=True, exist_ok=True)
    Path("raporlar").mkdir(parents=True, exist_ok=True)
    args.cikti_dir.mkdir(parents=True, exist_ok=True)

    df = oku_islenmis(args.girdi_dir)

    # --dogrula CLI seçeneği
    if args.dogrula is not None:
        yol = args.dogrula
        if yol.suffix == ".parquet":
            pred = pd.read_parquet(yol)
        else:
            pred = pd.read_csv(yol, parse_dates=["Date", "Origin"])
        pred["Date"] = pd.to_datetime(pred["Date"])
        sorunlar = tahmin_dogrula(pred, df)
        if sorunlar:
            print(f"DOĞRULAMA BAŞARISIZ ({len(sorunlar)} sorun):")
            for s in sorunlar:
                print(f"  - {s}")
        else:
            print("DOĞRULAMA BAŞARILI: Tüm sözleşme kuralları sağlandı.")
        return

    # ===========================================================================
    # TASK 2: Kayan kökenli değerlendirme
    # ===========================================================================
    print("=" * 70)
    print("TASK 2: Kayan kökenli değerlendirme")
    print("=" * 70)

    pencereler = tum_pencereler()
    print(f"\nToplam pencere sayısı: {len(pencereler)}")
    for p in pencereler:
        print(f"  {p['ad']}: {tarih(p['bas'])} – {tarih(p['bit'])} (köken: {tarih(p['origin'])})")

    tum_tahminler = []
    pencere_sonuc = []

    for p in pencereler:
        ad, bas, bit = p["ad"], p["bas"], p["bit"]

        t_naive = tahmin_uret(df, bas, bit, "Naive", ad)
        m_naive = hesapla_metrikler(t_naive)

        t_naive_p = tahmin_uret(df, bas, bit, "Naive-P", ad)
        m_naive_p = hesapla_metrikler(t_naive_p)

        tum_tahminler.append(t_naive)
        tum_tahminler.append(t_naive_p)

        iyilesme = (m_naive["rmspe"] - m_naive_p["rmspe"]) / m_naive["rmspe"] * 100.0

        pencere_sonuc.append({
            "ad": ad,
            "bas": bas.strftime("%Y-%m-%d"),
            "bit": bit.strftime("%Y-%m-%d"),
            "n": m_naive["n"],
            "n_magaza": m_naive["n_magaza"],
            "naive": {"rmspe": m_naive["rmspe"], "mape": m_naive["mape"],
                      "mae": m_naive["mae"], "rmse": m_naive["rmse"]},
            "naive_p": {"rmspe": m_naive_p["rmspe"], "mape": m_naive_p["mape"],
                        "mae": m_naive_p["mae"], "rmse": m_naive_p["rmse"]},
            "iyilesme_rmspe_yuzde": iyilesme,
        })

        print(f"  {ad}: Naive={sayi(m_naive['rmspe'], 3)} / Naive-P={sayi(m_naive_p['rmspe'], 3)} "
              f"(iyileşme %{sayi(iyilesme, 1)})")

    # Hafta 5 sayılarını doğrula
    print("\nHafta 5 doğrulaması (K1, K2, K3, Test):")
    beklenen = {
        "K1": (0.244, 0.192),
        "K2": (0.294, 0.180),
        "K3": (0.303, 0.215),
        "Test": (0.349, 0.146),
    }
    for ps in pencere_sonuc:
        ad = ps["ad"]
        if ad in beklenen:
            bek_n, bek_p = beklenen[ad]
            ger_n = round(ps["naive"]["rmspe"], 3)
            ger_p = round(ps["naive_p"]["rmspe"], 3)
            durum = "OK" if ger_n == bek_n and ger_p == bek_p else "HATA"
            print(f"  {ad}: Naive {ger_n} (beklenen {bek_n}), Naive-P {ger_p} (beklenen {bek_p}) [{durum}]")
            if durum == "HATA":
                raise SystemExit(f"DURDURULUYOR: {ad} sayıları Hafta 5 ile eşleşmiyor!")

    print("[OK] Hafta 5 sayıları başarıyla doğrulandı.")

    naive_rmspe_all = np.array([p["naive"]["rmspe"] for p in pencere_sonuc])
    naive_p_rmspe_all = np.array([p["naive_p"]["rmspe"] for p in pencere_sonuc])

    ozet = {
        "naive": {
            "ort": float(naive_rmspe_all.mean()),
            "medyan": float(np.median(naive_rmspe_all)),
            "min": float(naive_rmspe_all.min()),
            "max": float(naive_rmspe_all.max()),
        },
        "naive_p": {
            "ort": float(naive_p_rmspe_all.mean()),
            "medyan": float(np.median(naive_p_rmspe_all)),
            "min": float(naive_p_rmspe_all.min()),
            "max": float(naive_p_rmspe_all.max()),
        },
        "esik_alti_naive": int((naive_rmspe_all <= 0.15).sum()),
        "esik_alti_naive_p": int((naive_p_rmspe_all <= 0.15).sum()),
        "naive_p_daha_iyi": int((naive_p_rmspe_all < naive_rmspe_all).sum()),
    }

    # ===========================================================================
    # TASK 3: Diebold-Mariano testleri
    # ===========================================================================
    print("\n" + "=" * 70)
    print("TASK 3: Diebold-Mariano testleri")
    print("=" * 70)

    # (a) Pencere düzeyi DM testi
    d_pencereleri = []
    for p in pencere_sonuc:
        ad = p["ad"]
        naive_pred = [t for t in tum_tahminler if
                      (t["Split"].iloc[0] == ad) and (t["Model"].iloc[0] == "Naive")][0]
        naive_p_pred = [t for t in tum_tahminler if
                        (t["Split"].iloc[0] == ad) and (t["Model"].iloc[0] == "Naive-P")][0]

        m = naive_pred["egitim_mi"] & (naive_pred["Sales"] > 0)
        L_naive = ((naive_pred.loc[m, "Sales"].values - naive_pred.loc[m, "Prediction"].values) /
                   naive_pred.loc[m, "Sales"].values) ** 2
        L_naive_p = ((naive_p_pred.loc[m, "Sales"].values - naive_p_pred.loc[m, "Prediction"].values) /
                     naive_p_pred.loc[m, "Sales"].values) ** 2
        d_k = float((L_naive - L_naive_p).mean())
        d_pencereleri.append(d_k)

    d_pencereleri = np.array(d_pencereleri)
    dm_p_sonuc = dm_pencere(d_pencereleri)
    print(f"\n  Pencere DM: d̄={dm_p_sonuc['ort_fark']:.6f}, t={dm_p_sonuc['stat']:.3f}, "
          f"p={dm_p_sonuc['p']:.4f}, p_lag1={dm_p_sonuc['p_lag1']:.4f}")

    # (b) Günlük test penceresi DM testi
    test_naive = [t for t in tum_tahminler if
                  (t["Split"].iloc[0] == "Test") and (t["Model"].iloc[0] == "Naive")][0]
    test_naive_p = [t for t in tum_tahminler if
                    (t["Split"].iloc[0] == "Test") and (t["Model"].iloc[0] == "Naive-P")][0]

    m_test = test_naive["egitim_mi"] & (test_naive["Sales"] > 0)
    test_n = test_naive[m_test].copy()
    test_np = test_naive_p[m_test].copy()

    L_n = ((test_n["Sales"].values - test_n["Prediction"].values) / test_n["Sales"].values) ** 2
    L_np = ((test_np["Sales"].values - test_np["Prediction"].values) / test_np["Sales"].values) ** 2
    d_all = L_n - L_np

    test_n_df = test_n.copy()
    test_n_df["d"] = d_all
    d_gunluk = test_n_df.groupby("Date")["d"].mean().sort_index().values
    dm_g_sonuc = dm_gunluk(d_gunluk)
    print(f"  Günlük DM: d̄={dm_g_sonuc['ort_fark']:.6f}, t={dm_g_sonuc['stat']:.3f}, "
          f"p={dm_g_sonuc['p']:.4f}, lag={dm_g_sonuc['lag']}")

    # (c) Kendi kendini denetleme
    dm_k_sonuc = dm_kendi_kontrol(42)
    print(f"\n  DM kendi kontrol:")
    print(f"    Aynı tahmin uygulanamaz: {dm_k_sonuc['ayni_tahmin_uygulanamaz']}")
    print(f"    Sentetik fark ret: {dm_k_sonuc['sentetik_fark_red']}")
    print(f"    Aynı dağılım ret oranı: {dm_k_sonuc['sentetik_ayni_red_orani']:.3f}")

    # ===========================================================================
    # TASK 4: Alt grup kırılımları
    # ===========================================================================
    print("\n" + "=" * 70)
    print("TASK 4: Alt grup kırılımları")
    print("=" * 70)

    bilgi_sutunlar = ["Store", "Date", "StoreType", "Promo", "StateHoliday",
                      "SchoolHoliday", "DayOfWeek"]
    bilgi = df[bilgi_sutunlar].drop_duplicates(["Store", "Date"])

    test_naive_full = test_naive.merge(bilgi, on=["Store", "Date"], how="left")
    test_naive_p_full = test_naive_p.merge(bilgi, on=["Store", "Date"], how="left")

    # StateHoliday "0" vs "any"
    test_naive_full["DevletTatili"] = np.where(test_naive_full["StateHoliday"].astype(str) == "0", "0", "any")
    test_naive_p_full["DevletTatili"] = np.where(test_naive_p_full["StateHoliday"].astype(str) == "0", "0", "any")

    test_naive_full["OkulTatili"] = test_naive_full["SchoolHoliday"].astype(str)
    test_naive_p_full["OkulTatili"] = test_naive_p_full["SchoolHoliday"].astype(str)

    test_bas_dt = pd.Timestamp("2015-06-20")
    test_naive_full["HaftaNo"] = ((test_naive_full["Date"] - test_bas_dt).dt.days // 7) + 1
    test_naive_p_full["HaftaNo"] = ((test_naive_p_full["Date"] - test_bas_dt).dt.days // 7) + 1

    def kirilim_hesapla(naive_full, naive_p_full, sutun, degerler):
        sonuc = {}
        ne = naive_full[naive_full["egitim_mi"] & (naive_full["Sales"] > 0)]
        npe = naive_p_full[naive_p_full["egitim_mi"] & (naive_p_full["Sales"] > 0)]
        for d in degerler:
            d_str = str(d)
            gn = ne[ne[sutun].astype(str) == d_str]
            gp = npe[npe[sutun].astype(str) == d_str]
            rn = float(rmspe(gn["Sales"], gn["Prediction"])) if len(gn) > 0 else np.nan
            rp = float(rmspe(gp["Sales"], gp["Prediction"])) if len(gp) > 0 else np.nan
            sonuc[d_str] = {
                "n_magaza": int(gn["Store"].nunique()),
                "n_satir": int(len(gn)),
                "naive": rn,
                "naive_p": rp,
            }
        return sonuc

    # Test penceresi kırılımları
    mt_test = kirilim_hesapla(test_naive_full, test_naive_p_full, "StoreType", ["a", "b", "c", "d"])
    promo_test = kirilim_hesapla(test_naive_full, test_naive_p_full, "Promo", ["0", "1"])
    devlet_test = kirilim_hesapla(test_naive_full, test_naive_p_full, "DevletTatili", ["0", "any"])
    okul_test = kirilim_hesapla(test_naive_full, test_naive_p_full, "OkulTatili", ["0", "1"])
    dow_test = kirilim_hesapla(test_naive_full, test_naive_p_full, "DayOfWeek", [str(i) for i in range(1, 8)])

    # Alt grup test formatı
    alt_grup_test = {
        "magaza_tipi": {
            t: {
                "n_magaza": mt_test[t]["n_magaza"],
                "n_satir": mt_test[t]["n_satir"],
                "naive": mt_test[t]["naive"],
                "naive_p": mt_test[t]["naive_p"],
            } for t in ["a", "b", "c", "d"]
        },
        "promo": {
            k: {"naive": promo_test[k]["naive"], "naive_p": promo_test[k]["naive_p"]}
            for k in ["0", "1"]
        },
        "devlet_tatili": {
            k: {"naive": devlet_test[k]["naive"], "naive_p": devlet_test[k]["naive_p"]}
            for k in ["0", "any"]
        },
        "okul_tatili": {
            k: {"naive": okul_test[k]["naive"], "naive_p": okul_test[k]["naive_p"]}
            for k in ["0", "1"]
        },
        "haftanin_gunu": {
            k: {"naive": dow_test[k]["naive"], "naive_p": dow_test[k]["naive_p"]}
            for k in [str(i) for i in range(1, 8)]
        },
    }

    # Haftalık test (6 hafta)
    haftalik_test = []
    for w in range(1, 7):
        gn = test_naive_full[(test_naive_full["HaftaNo"] == w) &
                             test_naive_full["egitim_mi"] & (test_naive_full["Sales"] > 0)]
        gp = test_naive_p_full[(test_naive_p_full["HaftaNo"] == w) &
                               test_naive_p_full["egitim_mi"] & (test_naive_p_full["Sales"] > 0)]
        rn = float(rmspe(gn["Sales"], gn["Prediction"])) if len(gn) > 0 else np.nan
        rp = float(rmspe(gp["Sales"], gp["Prediction"])) if len(gp) > 0 else np.nan
        haftalik_test.append({"hafta": w, "naive": rn, "naive_p": rp})

    # Havuzlanmış kırılımlar (rapor için)
    tum_naive_list = [t for t in tum_tahminler if t["Model"].iloc[0] == "Naive"]
    tum_naive_p_list = [t for t in tum_tahminler if t["Model"].iloc[0] == "Naive-P"]
    all_naive_df = pd.concat(tum_naive_list, ignore_index=True).merge(bilgi, on=["Store", "Date"], how="left")
    all_naive_p_df = pd.concat(tum_naive_p_list, ignore_index=True).merge(bilgi, on=["Store", "Date"], how="left")
    all_naive_df["DevletTatili"] = np.where(all_naive_df["StateHoliday"].astype(str) == "0", "0", "any")
    all_naive_p_df["DevletTatili"] = np.where(all_naive_p_df["StateHoliday"].astype(str) == "0", "0", "any")
    all_naive_df["OkulTatili"] = all_naive_df["SchoolHoliday"].astype(str)
    all_naive_p_df["OkulTatili"] = all_naive_p_df["SchoolHoliday"].astype(str)

    mt_havuz = kirilim_hesapla(all_naive_df, all_naive_p_df, "StoreType", ["a", "b", "c", "d"])
    promo_havuz = kirilim_hesapla(all_naive_df, all_naive_p_df, "Promo", ["0", "1"])
    alt_grup_havuz = {"magaza_tipi": mt_havuz, "promo": promo_havuz}

    # En kötü 10 mağaza
    kotu_magazalar = en_kotu_magazalar(test_naive_p, df, 10)

    # ===========================================================================
    # TASK 1: Tahmin sözleşmesi — dışa aktar ve doğrula
    # ===========================================================================
    print("\n" + "=" * 70)
    print("TASK 1: Tahmin sözleşmesi doğrulayıcı")
    print("=" * 70)

    tum = pd.concat(tum_tahminler, ignore_index=True)

    # Dışa aktarma sözleşmesi: Store, Date, Sales, Prediction, Model, H, Origin, Split
    kayit_sutunlar = ["Store", "Date", "Sales", "Prediction", "Model", "H", "Origin", "Split", "Open"]
    tahmin_yol = args.cikti_dir / "naive_tahminleri.parquet"
    tum[kayit_sutunlar].to_parquet(tahmin_yol, index=False)
    print(f"  {tahmin_yol} kaydedildi ({len(tum):,} satır)")

    # 1000 satırlık örnek
    ornek = tum[kayit_sutunlar].sample(1000, random_state=42).sort_values(["Model", "Split", "Store", "Date"])
    ornek.to_csv("raporlar/ornek_tahmin_formati.csv", index=False)
    print(f"  raporlar/ornek_tahmin_formati.csv kaydedildi")

    # Doğrula
    sorunlar = tahmin_dogrula(tum, df)
    kontrol_sayisi = 7
    gecen = kontrol_sayisi - len(sorunlar)
    print(f"\n  Doğrulama: {gecen}/{kontrol_sayisi} kontrol geçti")
    if sorunlar:
        for s in sorunlar:
            print(f"    - {s}")

    # Negatif kontrol (bilerek bozulan tahminler)
    bozuk = tum.copy()
    bozuk.loc[bozuk.index[:5], "Prediction"] = np.nan
    bozuk.loc[bozuk.index[5:10], "Prediction"] = -100.0
    bozuk_sorunlar = tahmin_dogrula(bozuk, df)
    bozuk_yakalandi = len(bozuk_sorunlar) > 0
    print(f"  Bilerek bozulan tahminler yakalandı: {bozuk_yakalandi}")

    dogrulayici = {
        "kontrol_sayisi": kontrol_sayisi,
        "gecen": gecen,
        "bilerek_bozulan_yakalandi": bool(bozuk_yakalandi),
    }

    # ===========================================================================
    # TASK 5: Çıktılar
    # ===========================================================================
    print("\n" + "=" * 70)
    print("TASK 5: Çıktılar")
    print("=" * 70)

    json_sonuc = {
        "pencere_sayisi": len(pencereler),
        "pencereler": pencere_sonuc,
        "ozet": ozet,
        "dm_pencere": {
            "K": dm_p_sonuc["K"],
            "ort_fark": dm_p_sonuc["ort_fark"],
            "se": dm_p_sonuc["se"],
            "stat": dm_p_sonuc["stat"],
            "df": dm_p_sonuc["df"],
            "p": dm_p_sonuc["p"],
            "p_lag1": dm_p_sonuc["p_lag1"],
        },
        "dm_gunluk_test": {
            "T": dm_g_sonuc["T"],
            "lag": dm_g_sonuc["lag"],
            "ort_fark": dm_g_sonuc["ort_fark"],
            "se": dm_g_sonuc["se"],
            "stat": dm_g_sonuc["stat"],
            "df": dm_g_sonuc["df"],
            "p": dm_g_sonuc["p"],
            "duyarlilik": dm_g_sonuc["duyarlilik"],
        },
        "dm_kendi_kontrol": dm_k_sonuc,
        "alt_grup_test": alt_grup_test,
        "haftalik_test": haftalik_test,
        "en_kotu_magazalar": kotu_magazalar,
        "dogrulayici": dogrulayici,
    }

    def temizle_nan(obj):
        if isinstance(obj, float) and (math.isnan(obj) or np.isnan(obj)):
            return None
        elif isinstance(obj, dict):
            return {k: temizle_nan(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [temizle_nan(v) for v in obj]
        return obj

    json_sonuc = temizle_nan(json_sonuc)

    json_yol = Path("raporlar/hafta7_sonuclar.json")
    with open(json_yol, "w", encoding="utf-8") as f:
        json.dump(json_sonuc, f, indent=2, ensure_ascii=False)
    print(f"  {json_yol} kaydedildi.")

    plt = grafik_stili()
    grafik_kayan_kokenli(plt, pencere_sonuc, Path("gorseller/model/17_kayan_kokenli_rmspe.png"))
    grafik_altgrup(plt, alt_grup_test["magaza_tipi"], Path("gorseller/model/18_alt_grup_hatasi.png"))

    rapor_yaz(json_sonuc, Path("raporlar/hafta7_ozet.md"), alt_grup_havuz)

    print(f"\n{'=' * 70}")
    print("Tamamlandı.")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
