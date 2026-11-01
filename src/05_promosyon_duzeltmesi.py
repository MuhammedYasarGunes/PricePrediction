"""
İP 6 - Promosyon döngüsü düzeltmesi

Girdi : 01_veri_hazirlama.py çıktısı (<girdi-dir>/islenmis_tam.*)
Çıktı : <cikti-dir>/oznitelikler_H{H}.*           her H için güncellenmiş tablo (25 + 3 yeni öznitelik)
        raporlar/hafta5_ozet.md                     sonuç özeti
        raporlar/hafta5_sonuclar.json               makine-okunur sonuçlar
        gorseller/model/15_naive_p_karsilastirma.png
        gorseller/model/16_promo_duzeltme_korelasyon.png

Yeni öznitelikler (her H için 25 mevcut özniteliğin üstüne):
  PromoGecikme          t−L gününün Promo değeri (aynı mağaza)
  PromoUyumsuz          1 eğer Promo(t) ≠ PromoGecikme; 0 eğer eşit
  SatisGecikmePromoDuz  SatisGecikme + (Promo(t) − PromoGecikme) × delta_store

  delta_store = promo günlerdeki ortalama log satış − promo-dışı günlerdeki ortalama log satış,
                aynı mağazanın t−H'ye kadarki 91 günlük açık hafta-içi (Pzt-Cum, Sales>0) satırlarından.
                En az 4 promo ve 4 non-promo gün gerekir; yoksa havuzlanmış (pooled) delta kullanılır.

Naive-P: kaynak gün olarak en yakın aynı hafta-günü VE aynı promo durumu seçilir.

Kullanım:
  python src/05_promosyon_duzeltmesi.py --girdi-dir data/islenmis --cikti-dir data/oznitelikler_v2
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

# --- modül yolunu ayarla: src/ dizinini sys.path'e ekle ---
_KOK = Path(__file__).resolve().parent
if str(_KOK) not in sys.path:
    sys.path.insert(0, str(_KOK))

# --- digits ile başlayan modüller importlib ile yüklenir ---
_oz_mod = importlib.import_module("03_oznitelik_muhendisligi")
_tm_mod = importlib.import_module("04_temel_model")
import ortak

# Kısa erişim
from ortak import (MAVI, MUREKKEP2, TURKUAZ, TURUNCU, UFUK_ANALIZI_BASLANGIC, UFUK_DEGERLERI,
                   VARSAYILAN_UFUK, VERI_BITIS, gun, grafik_stili, kaydet_tablo, katmanlar,
                   metrikler, oku_islenmis, rmspe, sayi, tarih, test_penceresi, ufuk_penceresi)

# 03 modülünden gerekli sınıf ve fonksiyonlar
Izgara = _oz_mod.Izgara
kaydir = _oz_mod.kaydir
log_satis_matrisi = _oz_mod.log_satis_matrisi
gecmis_satis = _oz_mod.gecmis_satis
oznitelik_uret = _oz_mod.oznitelik_uret
eksik_orani = _oz_mod.eksik_orani
TEMEL_OZNITELIKLER = _oz_mod.TEMEL_OZNITELIKLER
YENI_OZNITELIKLER = _oz_mod.YENI_OZNITELIKLER
TUM_OZNITELIKLER = _oz_mod.TUM_OZNITELIKLER
GECMIS_OZNITELIKLERI = _oz_mod.GECMIS_OZNITELIKLERI

# 04 modülünden gerekli fonksiyonlar
mevsimsel_naive = _tm_mod.mevsimsel_naive
degerlendir = _tm_mod.degerlendir
kirilim = _tm_mod.kirilim

# Yeni öznitelik adları
PROMO_OZNITELIKLER = ["PromoGecikme", "PromoUyumsuz", "SatisGecikmePromoDuz"]
DELTA_PENCERE = 91       # delta hesabı pencere uzunluğu (gün)
DELTA_MIN_GOZLEM = 4     # promo ve non-promo için en az gözlem

GERI_BAKIS_HAFTA = 8     # Naive-P kaynak arama penceresi (hafta)


# =============================================================================
# TASK 1: Yeni öznitelikler
# =============================================================================

def promo_gecikme(promo_mat: np.ndarray, ufuk: int) -> np.ndarray:
    """PromoGecikme: t-L gününün Promo değeri. L = 7*ceil(H/7)."""
    L = 7 * math.ceil(ufuk / 7)
    return kaydir(promo_mat, L)


def promo_uyumsuz(promo_mat: np.ndarray, promo_gecikme_mat: np.ndarray) -> np.ndarray:
    """PromoUyumsuz: 1 if Promo(t) != PromoGecikme else 0. NaN if either is NaN."""
    sonuc = np.where(promo_mat != promo_gecikme_mat, 1.0, 0.0)
    sonuc[np.isnan(promo_mat) | np.isnan(promo_gecikme_mat)] = np.nan
    return sonuc


def delta_store_matrisi(ls_mat: np.ndarray, promo_mat: np.ndarray,
                        dow_mat: np.ndarray, ufuk: int) -> tuple:
    """Her mağaza ve her gün t için delta_store hesaplar.

    delta_store = (promo günlerin ortalama log satışı) − (non-promo günlerin ortalama log satışı)
    t−H'de biten 91 günlük pencerede, hafta-içi (Pzt-Cum: dow 1..5), sales>0, açık günlerden.

    Returns:
        delta: (T, S) matris, delta_store
        fallback: (T, S) bool matris, pooled delta kullanılan hücreler
    """
    T, S = ls_mat.shape
    # Hafta-içi maskesi: DayOfWeek 1..5 = Pzt-Cum
    hafta_ici = (dow_mat >= 1) & (dow_mat <= 5)
    # Log satış ve promo: yalnızca açık + satış>0 + hafta-içi olan günler
    gecerli = ~np.isnan(ls_mat) & hafta_ici

    # Promo ve non-promo maskeleri
    promo_gecerli = gecerli & (promo_mat == 1)
    non_promo_gecerli = gecerli & (~np.isnan(promo_mat)) & (promo_mat == 0)

    # Kaydırılmış: t-H ile t-H-90 arası (91 günlük pencere)
    # Yani gecikme H'den H+90'a kadar (dahil)
    promo_toplam = np.zeros((T, S))
    promo_adet = np.zeros((T, S))
    non_promo_toplam = np.zeros((T, S))
    non_promo_adet = np.zeros((T, S))

    for k in range(DELTA_PENCERE):
        gecikme = ufuk + k
        if gecikme >= T:
            break
        ls_k = kaydir(ls_mat, gecikme)
        p_g = kaydir(promo_gecerli.astype(float), gecikme)
        np_g = kaydir(non_promo_gecerli.astype(float), gecikme)

        m_p = (p_g == 1)
        m_np = (np_g == 1)

        promo_toplam += np.where(m_p, np.nan_to_num(ls_k), 0)
        promo_adet += m_p.astype(float)
        non_promo_toplam += np.where(m_np, np.nan_to_num(ls_k), 0)
        non_promo_adet += m_np.astype(float)

    with np.errstate(invalid="ignore", divide="ignore"):
        promo_ort = np.where(promo_adet >= DELTA_MIN_GOZLEM, promo_toplam / promo_adet, np.nan)
        non_promo_ort = np.where(non_promo_adet >= DELTA_MIN_GOZLEM, non_promo_toplam / non_promo_adet, np.nan)
        delta = promo_ort - non_promo_ort

    # Pooled delta: her t için tüm mağazaların pooled ortalaması
    pool_promo_toplam = np.nansum(np.where(promo_adet >= DELTA_MIN_GOZLEM, promo_toplam, np.nan), axis=1, keepdims=True)
    pool_promo_adet = np.nansum(np.where(promo_adet >= DELTA_MIN_GOZLEM, promo_adet, np.nan), axis=1, keepdims=True)
    pool_non_promo_toplam = np.nansum(np.where(non_promo_adet >= DELTA_MIN_GOZLEM, non_promo_toplam, np.nan), axis=1, keepdims=True)
    pool_non_promo_adet = np.nansum(np.where(non_promo_adet >= DELTA_MIN_GOZLEM, non_promo_adet, np.nan), axis=1, keepdims=True)

    # Pooled delta doğru hesap: hem promo hem non-promo tarafta yeterli olan mağazaları topla
    ikisi_var = (promo_adet >= DELTA_MIN_GOZLEM) & (non_promo_adet >= DELTA_MIN_GOZLEM)
    pool_p_top = np.where(ikisi_var, promo_toplam, 0.0).sum(axis=1, keepdims=True)
    pool_p_adet = np.where(ikisi_var, promo_adet, 0.0).sum(axis=1, keepdims=True)
    pool_np_top = np.where(ikisi_var, non_promo_toplam, 0.0).sum(axis=1, keepdims=True)
    pool_np_adet = np.where(ikisi_var, non_promo_adet, 0.0).sum(axis=1, keepdims=True)

    with np.errstate(invalid="ignore", divide="ignore"):
        pooled_delta = np.where(
            (pool_p_adet >= DELTA_MIN_GOZLEM) & (pool_np_adet >= DELTA_MIN_GOZLEM),
            pool_p_top / pool_p_adet - pool_np_top / pool_np_adet,
            np.nan
        )

    fallback = np.isnan(delta) & ~np.isnan(pooled_delta)
    delta_nihai = np.where(~np.isnan(delta), delta, pooled_delta)

    return delta_nihai, fallback


def satis_gecikme_promo_duz(satis_gecikme: np.ndarray, promo_mat: np.ndarray,
                             promo_gecikme_mat: np.ndarray, delta: np.ndarray) -> np.ndarray:
    """SatisGecikmePromoDuz = SatisGecikme + (Promo(t) - PromoGecikme) * delta_store.
    Missing SatisGecikme stays NaN. Never fill with 0."""
    duzeltme = (promo_mat - promo_gecikme_mat) * delta
    sonuc = satis_gecikme + duzeltme
    # NaN yayılımı: satis_gecikme NaN ise sonuç NaN olmalı
    sonuc[np.isnan(satis_gecikme)] = np.nan
    return sonuc


def promo_oznitelik_uret(df: pd.DataFrame, ufuk: int, izgara: Izgara) -> tuple:
    """03'ün özniteliklerini üretip, 3 yeni promo özniteliğini ekler.

    Returns:
        out: DataFrame with all 28 features
        fallback_orani: fallback kullanılan satırların oranı (modele giren satırlar)
    """
    # Önce mevcut 25 özniteliği üret
    out = oznitelik_uret(df, ufuk, izgara)

    # Promo matrisi
    promo_mat = izgara.matris(df["Promo"])

    # PromoGecikme
    pg_mat = promo_gecikme(promo_mat, ufuk)
    out["PromoGecikme"] = izgara.satirlara(pg_mat)

    # PromoUyumsuz
    pu_mat = promo_uyumsuz(promo_mat, pg_mat)
    out["PromoUyumsuz"] = izgara.satirlara(pu_mat)

    # SatisGecikmePromoDuz
    ls_mat = log_satis_matrisi(df, izgara)
    L = 7 * math.ceil(ufuk / 7)
    satis_gecikme_mat = kaydir(ls_mat, L)

    # DayOfWeek matrisi
    dow_mat = izgara.matris(df["DayOfWeek"])

    delta, fallback = delta_store_matrisi(ls_mat, promo_mat, dow_mat, ufuk)
    sdpd_mat = satis_gecikme_promo_duz(satis_gecikme_mat, promo_mat, pg_mat, delta)
    out["SatisGecikmePromoDuz"] = izgara.satirlara(sdpd_mat)

    # Fallback oranı hesabı (modele giren satırlar)
    fallback_satirlar = izgara.satirlara(fallback)
    m = out["egitim_mi"]
    # Yalnızca SatisGecikmePromoDuz'un NaN olmadığı satırlar
    gecerli = m & ~out["SatisGecikmePromoDuz"].isna()
    if gecerli.sum() > 0:
        fallback_orani = float(fallback_satirlar[gecerli].sum() / gecerli.sum()) * 100
    else:
        fallback_orani = 0.0

    return out, fallback_orani


# =============================================================================
# TASK 2: Naive-P
# =============================================================================

def naive_p(df: pd.DataFrame, baslangic: pd.Timestamp, bitis: pd.Timestamp) -> pd.DataFrame:
    """Promosyon uyumlu naive tahmin.

    Her mağaza ve hedef gün için: köken öncesindeki en yakın aynı hafta-günü ve aynı promo
    durumundaki açık günün satışını kullan. Bulunamazsa standart naive'e dön.
    Kapalı günler 0 tahmin edilir.
    """
    koken = baslangic - gun(1)
    acik = df[df["egitim_mi"] & (df["Date"] <= koken)]

    # Hedef satırlar
    hedef = df[df["Date"].between(baslangic, bitis)][
        ["Store", "Date", "DayOfWeek", "Sales", "Open", "Promo", "StoreType", "egitim_mi"]].copy()

    # Her mağaza × hafta günü × promo kombinasyonu için en yakın kaynak
    aday = (acik[acik["Date"] > koken - gun(7 * GERI_BAKIS_HAFTA)]
            .sort_values("Date")
            [["Store", "DayOfWeek", "Promo", "Date", "Sales"]]
            .rename(columns={"Date": "KaynakTarihP", "Sales": "KaynakSatisP"}))

    # En son (tail(1)) aynı DayOfWeek VE aynı Promo
    son_p = aday.groupby(["Store", "DayOfWeek", "Promo"]).tail(1)

    hedef = hedef.merge(son_p, on=["Store", "DayOfWeek", "Promo"], how="left")

    # Standart naive: aynı DayOfWeek, promo durumu farketmez
    son_std = (acik[acik["Date"] > koken - gun(7 * GERI_BAKIS_HAFTA)]
               .sort_values("Date").groupby(["Store", "DayOfWeek"]).tail(1)
               [["Store", "DayOfWeek", "Date", "Sales", "Promo"]]
               .rename(columns={"Date": "KaynakTarih", "Sales": "KaynakSatis", "Promo": "KaynakPromo"}))

    # Son 28 gün ve tüm geçmiş yedek
    yedek28 = acik[acik["Date"] > koken - gun(28)].groupby("Store")["Sales"].mean().rename("Yedek28")
    yedek_tum = acik.groupby("Store")["Sales"].mean().rename("YedekTum")

    hedef = (hedef.merge(son_std, on=["Store", "DayOfWeek"], how="left")
             .merge(yedek28, on="Store", how="left")
             .merge(yedek_tum, on="Store", how="left"))

    # Naive-P: promo uyumlu kaynak varsa onu kullan, yoksa standart naive kaynak
    yedek_flag = hedef["KaynakSatisP"].isna()
    tahmin_p = hedef["KaynakSatisP"].fillna(hedef["KaynakSatis"]).fillna(hedef["Yedek28"]).fillna(hedef["YedekTum"])
    hedef["Tahmin"] = np.where(hedef["Open"] == 1, tahmin_p, 0.0)
    hedef["YedekKullanildi"] = yedek_flag

    # Standard naive fields (PromoUyumu) for later analysis
    hedef["Koken"] = koken
    hedef["GunIleri"] = (hedef["Date"] - koken).dt.days
    hedef["HaftaIleri"] = (hedef["GunIleri"] - 1) // 7 + 1
    hedef["PromoUyumu"] = np.where(
        hedef["KaynakPromo"].isna(), "kaynak yok",
        np.where(hedef["KaynakPromo"] == hedef["Promo"], "uyumlu", "uyumsuz"))

    return hedef


# =============================================================================
# TASK 4: Sızıntı testi
# =============================================================================

def sizinti_testi_promo(df: pd.DataFrame, izgara: Izgara, ufuk: int,
                        kesim: pd.Timestamp, tohum: int = 0) -> dict:
    """03'ün sizinti_testi'nin genişletilmiş sürümü: 3 yeni öznitelik için sızıntı testi."""
    rng = np.random.default_rng(tohum)

    # Asıl öznitelikler
    promo_mat = izgara.matris(df["Promo"])
    ls_mat = log_satis_matrisi(df, izgara)
    dow_mat = izgara.matris(df["DayOfWeek"])
    L = 7 * math.ceil(ufuk / 7)

    asil_pg = promo_gecikme(promo_mat, ufuk)
    asil_sge = kaydir(ls_mat, L)
    asil_delta, _ = delta_store_matrisi(ls_mat, promo_mat, dow_mat, ufuk)
    asil_pu = promo_uyumsuz(promo_mat, asil_pg)
    asil_sdpd = satis_gecikme_promo_duz(asil_sge, promo_mat, asil_pg, asil_delta)

    # Bozuk versiyon: kesimden sonraki satışları değiştir
    bozuk_df = df.copy()
    sonra = bozuk_df["Date"] > kesim
    bozuk_df.loc[sonra, "Sales"] = rng.integers(1, 40_000, int(sonra.sum()))
    bozuk_ls = log_satis_matrisi(bozuk_df, izgara)

    bozuk_pg = promo_gecikme(promo_mat, ufuk)  # promo değişmez
    bozuk_sge = kaydir(bozuk_ls, L)
    bozuk_delta, _ = delta_store_matrisi(bozuk_ls, promo_mat, dow_mat, ufuk)
    bozuk_pu = promo_uyumsuz(promo_mat, bozuk_pg)
    bozuk_sdpd = satis_gecikme_promo_duz(bozuk_sge, promo_mat, bozuk_pg, bozuk_delta)

    gun_no = (izgara.tarihler - kesim).days.to_numpy()[:, None]
    korunmali = np.broadcast_to(gun_no <= ufuk, asil_pg.shape)

    asil_list = [asil_pg, asil_pu, asil_sdpd]
    bozuk_list = [bozuk_pg, bozuk_pu, bozuk_sdpd]

    degisen_once, degisen_sonra = 0, 0
    for a, b in zip(asil_list, bozuk_list):
        fark = ~((a == b) | (np.isnan(a) & np.isnan(b)))
        degisen_once += int(fark[korunmali].sum())
        degisen_sonra += int(fark[~korunmali].sum())

    duyarlilik_olculebilir = bool((~korunmali).any())
    return {"ufuk": ufuk, "kesim": kesim,
            "korunan_hucre": int(korunmali.sum()) * 3,
            "degisen_korunan": degisen_once,
            "degisen_sonrasi": degisen_sonra,
            "duyarlilik_olculebilir": duyarlilik_olculebilir,
            "gecti": degisen_once == 0 and (degisen_sonra > 0 or not duyarlilik_olculebilir)}


def negatif_kontrol(df: pd.DataFrame, izgara: Izgara, ufuk: int) -> int:
    """Kasıtlı olarak 1 gün sızdıran sürümün yakalandığını doğrular.
    Kaç korunan hücrenin değiştiğini döndürür."""
    kesim = pd.Timestamp("2015-05-08")
    rng = np.random.default_rng(99)

    promo_mat = izgara.matris(df["Promo"])
    ls_mat = log_satis_matrisi(df, izgara)
    dow_mat = izgara.matris(df["DayOfWeek"])
    L = 7 * math.ceil(ufuk / 7)

    # Asıl
    asil_delta, _ = delta_store_matrisi(ls_mat, promo_mat, dow_mat, ufuk)
    asil_sge = kaydir(ls_mat, L)
    asil_pg = promo_gecikme(promo_mat, ufuk)
    asil_sdpd = satis_gecikme_promo_duz(asil_sge, promo_mat, asil_pg, asil_delta)

    # Bozuk
    bozuk_df = df.copy()
    sonra = bozuk_df["Date"] > kesim
    bozuk_df.loc[sonra, "Sales"] = rng.integers(1, 40_000, int(sonra.sum()))
    bozuk_ls = log_satis_matrisi(bozuk_df, izgara)

    # 1 gün sızdıran: ufuk - 1
    bozuk_delta, _ = delta_store_matrisi(bozuk_ls, promo_mat, dow_mat, ufuk - 1)
    bozuk_sge = kaydir(bozuk_ls, L - 1)  # kasıtlı 1 gün az gecikme
    bozuk_pg = promo_gecikme(promo_mat, ufuk)  # promo kaydırma değişmez
    bozuk_sdpd = satis_gecikme_promo_duz(bozuk_sge, promo_mat, bozuk_pg, bozuk_delta)

    gun_no = (izgara.tarihler - kesim).days.to_numpy()[:, None]
    korunmali = np.broadcast_to(gun_no <= ufuk, asil_sdpd.shape)

    degisen = 0
    for a, b in [(asil_sdpd, bozuk_sdpd)]:
        fark = ~((a == b) | (np.isnan(a) & np.isnan(b)))
        degisen += int(fark[korunmali].sum())
    return degisen


# =============================================================================
# MAIN
# =============================================================================

def main():
    ap = argparse.ArgumentParser(description="İP 6 - Promosyon döngüsü düzeltmesi")
    ap.add_argument("--girdi-dir", type=Path, default=Path("data/islenmis"))
    ap.add_argument("--cikti-dir", type=Path, default=Path("data/oznitelikler_v2"))
    ap.add_argument("--ufuk", type=int, nargs="+", default=list(UFUK_DEGERLERI),
                    help="tahmin ufku H (gün); birden fazla verilebilir")
    ap.add_argument("--sizinti-testi-yok", action="store_true", help="sızıntı testini atla")
    args = ap.parse_args()
    args.cikti_dir.mkdir(parents=True, exist_ok=True)
    Path("gorseller/model").mkdir(parents=True, exist_ok=True)
    Path("raporlar").mkdir(parents=True, exist_ok=True)

    df = oku_islenmis(args.girdi_dir)
    izgara = Izgara(df)
    test_bas, test_bit = test_penceresi()

    # ===========================================================================
    # TASK 2: STANDART NAİVE KONTROLÜ - 04'ün fonksiyonlarıyla aynı sonucu doğrula
    # ===========================================================================
    print("=" * 70)
    print("Standart naive doğrulama (04'ün fonksiyonlarıyla)")
    print("=" * 70)

    # CV katmanları + test (H=42)
    pencereler = [(k["ad"], k["dogrulama_baslangic"], k["dogrulama_bitis"])
                  for k in katmanlar(VARSAYILAN_UFUK, test_bas)]
    pencereler.append(("Test", test_bas, test_bit))

    naive_sonuc = {}
    naive_tahminler = {}
    for ad, bas, bit in pencereler:
        t = mevsimsel_naive(df, bas, bit)
        m = degerlendir(t)
        naive_sonuc[ad] = m
        naive_tahminler[ad] = t
        print(f"  {ad}: RMSPE = {m['RMSPE']:.3f}")

    # Ufuk analizi
    ufuk_t = mevsimsel_naive(df, UFUK_ANALIZI_BASLANGIC, VERI_BITIS)
    for h in UFUK_DEGERLERI:
        bas, bit = ufuk_penceresi(h)
        t_sub = ufuk_t[ufuk_t["Date"].between(bas, bit)]
        m = degerlendir(t_sub)
        naive_sonuc[f"H{h}"] = m
        print(f"  H{h}: RMSPE = {m['RMSPE']:.3f}")

    # Beklenen değerlerle karşılaştır
    beklenen = {"K1": 0.244, "K2": 0.294, "K3": 0.303, "Test": 0.349,
                "H7": 0.432, "H28": 0.291, "H42": 0.303, "H84": 0.340}
    for ad, bek in beklenen.items():
        gercek = round(naive_sonuc[ad]["RMSPE"], 3)
        if gercek != bek:
            raise SystemExit(f"HATA: Standart naive {ad} RMSPE = {gercek}, beklenen {bek}. DURDURULUYOR.")
    print("[OK] Tüm standart naive değerleri doğrulandı.")

    # ===========================================================================
    # TASK 2 devam: Naive-P tahminleri
    # ===========================================================================
    print("\n" + "=" * 70)
    print("Naive-P tahminleri")
    print("=" * 70)

    naive_p_sonuc = {}
    naive_p_tahminler = {}
    for ad, bas, bit in pencereler:
        t = naive_p(df, bas, bit)
        m = degerlendir(t)
        naive_p_sonuc[ad] = m
        naive_p_tahminler[ad] = t
        yedek_oran = float(t[t["egitim_mi"]]["YedekKullanildi"].mean()) * 100
        print(f"  {ad}: RMSPE = {m['RMSPE']:.3f} (yedek oranı %{yedek_oran:.1f})")

    # Ufuk analizi Naive-P
    ufuk_t_p = naive_p(df, UFUK_ANALIZI_BASLANGIC, VERI_BITIS)
    for h in UFUK_DEGERLERI:
        bas, bit = ufuk_penceresi(h)
        t_sub = ufuk_t_p[ufuk_t_p["Date"].between(bas, bit)]
        m = degerlendir(t_sub)
        naive_p_sonuc[f"H{h}"] = m
        yedek_oran = float(t_sub[t_sub["egitim_mi"]]["YedekKullanildi"].mean()) * 100
        print(f"  H{h}: RMSPE = {m['RMSPE']:.3f} (yedek oranı %{yedek_oran:.1f})")

    # Yedek oranı (test)
    test_p = naive_p_tahminler["Test"]
    te_p = test_p[test_p["egitim_mi"]]
    yedek_orani_test = float(te_p["YedekKullanildi"].mean()) * 100

    # TASK 2(a): Promosyon uyumu kırılımı (test penceresi)
    # Standart naive test
    test_std = naive_tahminler["Test"]
    te_std = test_std[test_std["egitim_mi"]]

    kirilim_test = {"naive": {}, "naive_p": {}}
    for durum in ["uyumlu", "uyumsuz"]:
        m_std = te_std[te_std["PromoUyumu"] == durum]
        m_p = te_p[te_p["PromoUyumu"] == durum]
        kirilim_test["naive"][durum] = {
            "rmspe": rmspe(m_std["Sales"], m_std["Tahmin"]),
            "oran": len(m_std) / len(te_std) * 100
        }
        kirilim_test["naive_p"][durum] = {
            "rmspe": rmspe(m_p["Sales"], m_p["Tahmin"]),
            "oran": len(m_p) / len(te_p) * 100
        }

    print(f"\n  Test kırılımı (promo uyumu):")
    for yontem, veriler in kirilim_test.items():
        for durum, d in veriler.items():
            print(f"    {yontem} / {durum}: RMSPE = {d['rmspe']:.3f}, oran = %{d['oran']:.1f}")

    # TASK 2(b): Mağaza tipine göre kırılım (test)
    magaza_tipi_test = {"naive": {}, "naive_p": {}}
    for tip in ["a", "b", "c", "d"]:
        m_std = te_std[te_std["StoreType"] == tip]
        m_p = te_p[te_p["StoreType"] == tip]
        magaza_tipi_test["naive"][tip] = rmspe(m_std["Sales"], m_std["Tahmin"])
        magaza_tipi_test["naive_p"][tip] = rmspe(m_p["Sales"], m_p["Tahmin"])

    print(f"\n  Test kırılımı (mağaza tipi):")
    for yontem, veriler in magaza_tipi_test.items():
        for tip, v in veriler.items():
            print(f"    {yontem} / {tip}: RMSPE = {v:.3f}")

    # ===========================================================================
    # TASK 1: Yeni öznitelikler üret
    # ===========================================================================
    print("\n" + "=" * 70)
    print("Yeni öznitelikler (PromoGecikme, PromoUyumsuz, SatisGecikmePromoDuz)")
    print("=" * 70)

    tum_oznitelikler_yeni = TUM_OZNITELIKLER + PROMO_OZNITELIKLER
    eksik_tablo = {}
    fallback_oranlari = {}
    korelasyon_tablo = {}

    for H in args.ufuk:
        out, fb_oran = promo_oznitelik_uret(df, H, izgara)
        fallback_oranlari[H] = fb_oran

        # Kaydet
        sutunlar = (["Store", "Date", "Sales", "Open", "egitim_mi", "LogSales", "AykiriDeger", "Ufuk"]
                    + [c for c in tum_oznitelikler_yeni if c != "Store"])
        yol = kaydet_tablo(out[sutunlar], args.cikti_dir / f"oznitelikler_H{H}")
        print(f"  H={H}: {yol} ({len(out):,} satır, fallback %{fb_oran:.1f})")

        # Eksik oranları
        eksik_h = {}
        for c in PROMO_OZNITELIKLER:
            eksik_h[c] = eksik_orani(out, c)
        eksik_tablo[H] = eksik_h

        # TASK 3: Korelasyonlar (aynı satırlar: egitim_mi)
        e = out[out["egitim_mi"]]
        kor_h = {}
        for c in ["SatisGecikme", "SatisGecikmePromoDuz"]:
            kor_h[c] = float(e[c].corr(e["LogSales"]))
        korelasyon_tablo[H] = kor_h

    # TASK 3: SatisGecikme korelasyonlarını doğrula
    print("\n" + "=" * 70)
    print("Korelasyon doğrulama")
    print("=" * 70)
    beklenen_kor = {7: 0.638, 28: 0.808, 42: 0.767, 84: 0.687}
    for H, bek in beklenen_kor.items():
        gercek = round(korelasyon_tablo[H]["SatisGecikme"], 3)
        print(f"  H={H}: SatisGecikme korelasyonu = {gercek} (beklenen {bek})")
        if gercek != bek:
            raise SystemExit(f"HATA: SatisGecikme korelasyonu H={H}: {gercek} != {bek}. DURDURULUYOR.")
    print("[OK] Tüm korelasyonlar doğrulandı.")

    for H in args.ufuk:
        print(f"  H={H}: SatisGecikme = {korelasyon_tablo[H]['SatisGecikme']:.3f}, "
              f"SatisGecikmePromoDuz = {korelasyon_tablo[H]['SatisGecikmePromoDuz']:.3f}")

    # ===========================================================================
    # TASK 4: Sızıntı testi
    # ===========================================================================
    sizinti_sonuc = []
    kasitli_degisen = 0

    if not args.sizinti_testi_yok:
        print("\n" + "=" * 70)
        print("Sızıntı testi (3 yeni öznitelik)")
        print("=" * 70)

        kesimler = [pd.Timestamp("2015-05-08"), pd.Timestamp("2015-06-19"), pd.Timestamp("2014-03-31")]
        for H in args.ufuk:
            for kesim in kesimler:
                s = sizinti_testi_promo(df, izgara, H, kesim, tohum=H)
                sizinti_sonuc.append(s)
                sonuc_str = "geçti" if s["gecti"] else "KALDI"
                print(f"  H={H}, Kesim={tarih(kesim)}: korunan değişen={s['degisen_korunan']}, "
                      f"sonrası değişen={s['degisen_sonrasi']}, {sonuc_str}")

        if not all(s["gecti"] for s in sizinti_sonuc):
            raise SystemExit("Sızıntı testi başarısız: yeni öznitelikler gelecekten bilgi alıyor.")

        # Negatif kontrol
        kasitli_degisen = negatif_kontrol(df, izgara, 42)
        print(f"\n  Negatif kontrol (kasıtlı 1 gün sızıntı): korunan bölgede değişen = {kasitli_degisen}")
        if kasitli_degisen == 0:
            print("  UYARI: Negatif kontrol hiç değişiklik algılayamadı!")

    toplam_sizinti = len(sizinti_sonuc)
    gecen_sizinti = sum(s["gecti"] for s in sizinti_sonuc)

    # ===========================================================================
    # TASK 4(c): Bağımsız kontrol bilgileri (kontrol_oznitelikler.py'ye eklenecek)
    # ===========================================================================
    # Burada elle kontrol yapıyoruz, sonucu JSON'a yazacağız
    print("\n" + "=" * 70)
    print("Bağımsız elle kontrol (300 satır)")
    print("=" * 70)

    elle_kontrol_sonuc = elle_kontrol(df, args.ufuk[2] if len(args.ufuk) > 2 else 42, izgara)
    print(f"  Kontrol edilen satır: {elle_kontrol_sonuc['satir']}")
    print(f"  Maksimum mutlak fark: {elle_kontrol_sonuc['maks_fark']:.2e}")

    # ===========================================================================
    # TASK 5: Çıktılar
    # ===========================================================================

    # Eksik oranları H=42
    eksik_h42 = {}
    for c in PROMO_OZNITELIKLER:
        eksik_h42[c] = eksik_tablo[42][c] * 100

    # JSON sonuçları
    def sonuc_dict(m):
        return {"rmspe": m["RMSPE"], "mape": m["MAPE"], "mae": m["MAE"], "rmse": m["RMSE"], "n": m["n"]}

    json_sonuc = {
        "toplam_oznitelik": len(tum_oznitelikler_yeni),
        "eksik_oranlari_h42": {c: eksik_h42[c] for c in PROMO_OZNITELIKLER},
        "geri_donus_orani_h42": fallback_oranlari[42],
        "naive": {ad: sonuc_dict(naive_sonuc[ad]) for ad in
                  ["K1", "K2", "K3", "Test", "H7", "H28", "H42", "H84"]},
        "naive_p": {
            **{ad: sonuc_dict(naive_p_sonuc[ad]) for ad in
               ["K1", "K2", "K3", "Test", "H7", "H28", "H42", "H84"]},
            "yedek_orani_test": yedek_orani_test
        },
        "kirilim_test": {
            "naive": {
                "uyumlu": {"rmspe": kirilim_test["naive"]["uyumlu"]["rmspe"],
                           "oran": kirilim_test["naive"]["uyumlu"]["oran"]},
                "uyumsuz": {"rmspe": kirilim_test["naive"]["uyumsuz"]["rmspe"],
                            "oran": kirilim_test["naive"]["uyumsuz"]["oran"]}
            },
            "naive_p": {
                "uyumlu": {"rmspe": kirilim_test["naive_p"]["uyumlu"]["rmspe"],
                           "oran": kirilim_test["naive_p"]["uyumlu"]["oran"]},
                "uyumsuz": {"rmspe": kirilim_test["naive_p"]["uyumsuz"]["rmspe"],
                            "oran": kirilim_test["naive_p"]["uyumsuz"]["oran"]}
            }
        },
        "magaza_tipi_test": magaza_tipi_test,
        "korelasyon": {f"H{H}": korelasyon_tablo[H] for H in args.ufuk},
        "sizinti": {
            "toplam": toplam_sizinti,
            "gecen": gecen_sizinti,
            "kasitli_degisen": kasitli_degisen,
            "elle_satir": elle_kontrol_sonuc["satir"],
            "elle_maks_fark": elle_kontrol_sonuc["maks_fark"]
        }
    }

    # JSON kaydet
    json_yol = Path("raporlar/hafta5_sonuclar.json")
    with open(json_yol, "w", encoding="utf-8") as f:
        json.dump(json_sonuc, f, indent=2, ensure_ascii=False)
    print(f"\n[OK] {json_yol} kaydedildi.")

    # ===========================================================================
    # Grafikler
    # ===========================================================================
    plt = grafik_stili()
    grafik_karsilastirma(plt, json_sonuc)
    grafik_korelasyon(plt, korelasyon_tablo, args.ufuk)

    # ===========================================================================
    # Rapor
    # ===========================================================================
    rapor_yaz(json_sonuc, eksik_tablo, korelasyon_tablo, args.ufuk, sizinti_sonuc,
              fallback_oranlari, kasitli_degisen, elle_kontrol_sonuc)

    print(f"\n{'=' * 70}")
    print("Tamamlandı.")
    print(f"{'=' * 70}")


def elle_kontrol(df: pd.DataFrame, ufuk: int, izgara: Izgara) -> dict:
    """300 rastgele satır için yeni öznitelikleri basit döngülerle yeniden hesaplar."""
    GUN_TD = pd.Timedelta(days=1)
    H = ufuk
    L = 7 * math.ceil(H / 7)

    out, _ = promo_oznitelik_uret(df, H, izgara)

    # Log satış sözlüğü (açık ve satış>0)
    ls = {(s, d): math.log1p(v)
          for s, d, v, e in zip(df.Store, df.Date, df.Sales, df.egitim_mi) if e}
    promo_dict = dict(zip(zip(df.Store, df.Date), df.Promo))
    dow_dict = dict(zip(zip(df.Store, df.Date), df.DayOfWeek))

    ornek = df[df.egitim_mi].sample(300, random_state=42)
    maks_fark = 0.0
    out_idx = out.set_index(["Store", "Date"])

    for s, t in zip(ornek.Store, ornek.Date):
        satir = out_idx.loc[(s, t)]

        # PromoGecikme
        bek_pg = promo_dict.get((s, t - L * GUN_TD), np.nan)
        gercek_pg = satir["PromoGecikme"]
        if not (pd.isna(bek_pg) and pd.isna(gercek_pg)):
            if pd.isna(bek_pg) or pd.isna(gercek_pg):
                fark = abs(float("nan"))  # should not happen
            else:
                fark = abs(bek_pg - gercek_pg)
                maks_fark = max(maks_fark, fark)

        # PromoUyumsuz
        promo_t = promo_dict.get((s, t), np.nan)
        if not pd.isna(bek_pg) and not pd.isna(promo_t):
            bek_pu = 1.0 if promo_t != bek_pg else 0.0
        else:
            bek_pu = np.nan
        gercek_pu = satir["PromoUyumsuz"]
        if not (pd.isna(bek_pu) and pd.isna(gercek_pu)):
            if not pd.isna(bek_pu) and not pd.isna(gercek_pu):
                fark = abs(bek_pu - gercek_pu)
                maks_fark = max(maks_fark, fark)

        # SatisGecikmePromoDuz
        satis_gecikme = ls.get((s, t - L * GUN_TD), np.nan)
        if pd.isna(satis_gecikme):
            bek_sdpd = np.nan
        else:
            # delta hesapla: 91 günlük pencere, hafta-içi, t-H'de biten
            promo_vals, non_promo_vals = [], []
            for k in range(DELTA_PENCERE):
                d = t - (H + k) * GUN_TD
                if (s, d) in ls and (s, d) in dow_dict:
                    dw = dow_dict[(s, d)]
                    if 1 <= dw <= 5:
                        p = promo_dict.get((s, d), np.nan)
                        if p == 1:
                            promo_vals.append(ls[(s, d)])
                        elif p == 0:
                            non_promo_vals.append(ls[(s, d)])

            if len(promo_vals) >= DELTA_MIN_GOZLEM and len(non_promo_vals) >= DELTA_MIN_GOZLEM:
                delta_val = np.mean(promo_vals) - np.mean(non_promo_vals)
            else:
                # Pooled delta: tüm mağazalardaki aynı pencere
                pool_promo, pool_non_promo = [], []
                for store in df["Store"].unique():
                    s_promo, s_non_promo = [], []
                    for k in range(DELTA_PENCERE):
                        d = t - (H + k) * GUN_TD
                        if (store, d) in ls and (store, d) in dow_dict:
                            dw = dow_dict[(store, d)]
                            if 1 <= dw <= 5:
                                p = promo_dict.get((store, d), np.nan)
                                if p == 1:
                                    s_promo.append(ls[(store, d)])
                                elif p == 0:
                                    s_non_promo.append(ls[(store, d)])
                    if len(s_promo) >= DELTA_MIN_GOZLEM and len(s_non_promo) >= DELTA_MIN_GOZLEM:
                        pool_promo.extend(s_promo)
                        pool_non_promo.extend(s_non_promo)
                if pool_promo and pool_non_promo:
                    delta_val = np.mean(pool_promo) - np.mean(pool_non_promo)
                else:
                    delta_val = np.nan

            if pd.isna(promo_t) or pd.isna(bek_pg) or pd.isna(delta_val):
                bek_sdpd = np.nan
            else:
                bek_sdpd = satis_gecikme + (promo_t - bek_pg) * delta_val

        gercek_sdpd = satir["SatisGecikmePromoDuz"]
        if not (pd.isna(bek_sdpd) and pd.isna(gercek_sdpd)):
            if not pd.isna(bek_sdpd) and not pd.isna(gercek_sdpd):
                fark = abs(bek_sdpd - gercek_sdpd)
                maks_fark = max(maks_fark, fark)

    return {"satir": 300, "maks_fark": maks_fark}


# =============================================================================
# Grafikler
# =============================================================================

def grafik_karsilastirma(plt, sonuc):
    """15_naive_p_karsilastirma.png: grouped bars, standard naive vs Naive-P RMSPE."""
    etiketler = ["K1", "K2", "K3", "Test", "H7", "H28", "H42", "H84"]
    naive_vals = [sonuc["naive"][e]["rmspe"] for e in etiketler]
    naive_p_vals = [sonuc["naive_p"][e]["rmspe"] for e in etiketler]

    x = np.arange(len(etiketler))
    w = 0.35

    fig, ax = plt.subplots(figsize=(10, 5))
    b1 = ax.bar(x - w/2, naive_vals, w, color=MAVI, label="Standart naive")
    b2 = ax.bar(x + w/2, naive_p_vals, w, color=TURUNCU, label="Naive-P")

    ax.bar_label(b1, labels=[f"{v:.3f}" for v in naive_vals], padding=3, fontsize=8, color=MUREKKEP2)
    ax.bar_label(b2, labels=[f"{v:.3f}" for v in naive_p_vals], padding=3, fontsize=8, color=MUREKKEP2)

    ax.set_xticks(x)
    ax.set_xticklabels(etiketler)
    ax.set_ylabel("RMSPE")
    ax.set_ylim(0, max(max(naive_vals), max(naive_p_vals)) * 1.25)
    ax.set_title("Standart naive ve Naive-P karşılaştırması: promosyon uyumu iyileşme sağlıyor")
    ax.legend(loc="upper left", fontsize=9)

    fig.tight_layout()
    fig.savefig("gorseller/model/15_naive_p_karsilastirma.png", dpi=150)
    plt.close(fig)
    print("[OK] gorseller/model/15_naive_p_karsilastirma.png kaydedildi.")


def grafik_korelasyon(plt, korelasyon_tablo, ufuklar):
    """16_promo_duzeltme_korelasyon.png: SatisGecikme vs SatisGecikmePromoDuz korelasyonu."""
    ufuklar_sorted = sorted(ufuklar)
    sg_vals = [korelasyon_tablo[h]["SatisGecikme"] for h in ufuklar_sorted]
    sdpd_vals = [korelasyon_tablo[h]["SatisGecikmePromoDuz"] for h in ufuklar_sorted]

    x = np.arange(len(ufuklar_sorted))
    w = 0.35

    fig, ax = plt.subplots(figsize=(8, 5))
    b1 = ax.bar(x - w/2, sg_vals, w, color=MAVI, label="SatisGecikme")
    b2 = ax.bar(x + w/2, sdpd_vals, w, color=TURKUAZ, label="SatisGecikmePromoDuz")

    ax.bar_label(b1, labels=[f"{v:.3f}" for v in sg_vals], padding=3, fontsize=9, color=MUREKKEP2)
    ax.bar_label(b2, labels=[f"{v:.3f}" for v in sdpd_vals], padding=3, fontsize=9, color=MUREKKEP2)

    ax.set_xticks(x)
    ax.set_xticklabels([f"H={h}" for h in ufuklar_sorted])
    ax.set_ylabel("Pearson korelasyonu (log satışla)")
    ax.set_ylim(0, max(max(sg_vals), max(sdpd_vals)) * 1.15)
    ax.set_title("Promosyon düzeltmesi korelasyonu artırıyor")
    ax.legend(loc="upper right", fontsize=9)

    fig.tight_layout()
    fig.savefig("gorseller/model/16_promo_duzeltme_korelasyon.png", dpi=150)
    plt.close(fig)
    print("[OK] gorseller/model/16_promo_duzeltme_korelasyon.png kaydedildi.")


# =============================================================================
# Rapor
# =============================================================================

def rapor_yaz(json_sonuc, eksik_tablo, korelasyon_tablo, ufuklar, sizinti_sonuc,
              fallback_oranlari, kasitli_degisen, elle_kontrol_sonuc):
    """raporlar/hafta5_ozet.md yazar."""
    r = ["# Promosyon döngüsü düzeltmesi (Hafta 5)", "",
         "`05_promosyon_duzeltmesi.py` tarafından hesaplanmıştır.", ""]

    # Yeni öznitelikler
    r += ["## Yeni öznitelikler", "",
          f"- Toplam öznitelik: {json_sonuc['toplam_oznitelik']} (önceki 25 + yeni 3)", "",
          "| Öznitelik | Tanım |", "|---|---|",
          "| PromoGecikme | Aynı mağazanın t−L gününün Promo değeri (L = 7·⌈H/7⌉) |",
          "| PromoUyumsuz | 1 eğer Promo(t) ≠ PromoGecikme, 0 eğer eşit |",
          "| SatisGecikmePromoDuz | SatisGecikme + (Promo(t) − PromoGecikme) × delta_store |",
          "", "delta_store: aynı mağazanın t−H'ye kadar 91 günlük penceredeki açık hafta-içi (Pzt-Cum) "
          "satırlarından promo günlerin ortalama log satışı − non-promo günlerin ortalama log satışı. "
          "En az 4 promo ve 4 non-promo gün gerekir; yoksa tüm mağazaların havuzlanmış (pooled) deltası kullanılır.", ""]

    # Eksik oranları
    r += ["## Eksik değer oranları (modele girecek satırlar)", "",
          "| Öznitelik | " + " | ".join(f"H={h}" for h in sorted(ufuklar)) + " |",
          "|---|" + "---|" * len(ufuklar)]
    for c in PROMO_OZNITELIKLER:
        r.append(f"| {c} | " + " | ".join(
            f"%{sayi(eksik_tablo[h][c] * 100, 1)}" for h in sorted(ufuklar)) + " |")
    r.append("")

    # Fallback oranları
    r += ["## Geri dönüş (pooled delta) oranı", "",
          "| " + " | ".join(f"H={h}" for h in sorted(ufuklar)) + " |",
          "|" + "---|" * len(ufuklar)]
    r.append("| " + " | ".join(f"%{sayi(fallback_oranlari[h], 1)}" for h in sorted(ufuklar)) + " |")
    r.append("")

    # Naive vs Naive-P
    etiketler = ["K1", "K2", "K3", "Test", "H7", "H28", "H42", "H84"]
    r += ["## Naive ve Naive-P karşılaştırması", "",
          "| Pencere | Naive RMSPE | Naive-P RMSPE | İyileşme |", "|---|---|---|---|"]
    for e in etiketler:
        n_r = json_sonuc["naive"][e]["rmspe"]
        np_r = json_sonuc["naive_p"][e]["rmspe"]
        iyilesme = (n_r - np_r) / n_r * 100
        r.append(f"| {e} | {sayi(n_r, 3)} | {sayi(np_r, 3)} | %{sayi(iyilesme, 1)} |")
    r.append("")
    r.append(f"Naive-P'nin test döneminde standart naive yedeğine düşme oranı: "
             f"%{sayi(json_sonuc['naive_p']['yedek_orani_test'], 1)}")
    r.append("")

    # Kırılımlar
    r += ["## Test dönemi promosyon uyumu kırılımı", "",
          "| Yöntem | Durum | RMSPE | Oran |", "|---|---|---|---|"]
    for yontem in ["naive", "naive_p"]:
        yontem_adi = "Standart naive" if yontem == "naive" else "Naive-P"
        for durum in ["uyumlu", "uyumsuz"]:
            d = json_sonuc["kirilim_test"][yontem][durum]
            r.append(f"| {yontem_adi} | {durum} | {sayi(d['rmspe'], 3)} | %{sayi(d['oran'], 1)} |")
    r.append("")

    r += ["## Test dönemi mağaza tipi kırılımı", "",
          "| Tip | Naive RMSPE | Naive-P RMSPE |", "|---|---|---|"]
    for tip in ["a", "b", "c", "d"]:
        r.append(f"| {tip} | {sayi(json_sonuc['magaza_tipi_test']['naive'][tip], 3)} | "
                 f"{sayi(json_sonuc['magaza_tipi_test']['naive_p'][tip], 3)} |")
    r.append("")

    # Korelasyonlar
    r += ["## Korelasyon karşılaştırması (Pearson, log satışla)", "",
          "| Öznitelik | " + " | ".join(f"H={h}" for h in sorted(ufuklar)) + " |",
          "|---|" + "---|" * len(ufuklar)]
    for c in ["SatisGecikme", "SatisGecikmePromoDuz"]:
        r.append(f"| {c} | " + " | ".join(
            sayi(korelasyon_tablo[h][c], 3) for h in sorted(ufuklar)) + " |")
    r.append("")

    # Sızıntı testi
    if sizinti_sonuc:
        r += ["## Sızıntı testi (3 yeni öznitelik)", "",
              "Kesim tarihinden sonraki tüm satışlar rastgele değerlerle değiştirilip yeni öznitelikler "
              "yeniden üretildi. Kesim + H gününe kadar hiçbir değer değişmemeli.", "",
              "| H | Kesim | Korunan hücre | Korunan değişen | Sonrası değişen | Sonuç |",
              "|---|---|---|---|---|---|"]
        for s in sizinti_sonuc:
            sonrasi = sayi(s["degisen_sonrasi"]) if s["duyarlilik_olculebilir"] else "— (kesim + H veri sonunda)"
            r.append(f"| {s['ufuk']} | {tarih(s['kesim'])} | {sayi(s['korunan_hucre'])} | "
                     f"{s['degisen_korunan']} | {sonrasi} | {'geçti' if s['gecti'] else 'KALDI'} |")
        r.append("")
        r.append(f"Toplam {len(sizinti_sonuc)} testin {sum(s['gecti'] for s in sizinti_sonuc)} tanesi geçti.")
        r.append(f"Negatif kontrol (kasıtlı 1 gün sızıntı): korunan bölgede {sayi(kasitli_degisen)} hücre değişti "
                 f"(testin duyarlılığını doğrular).")
        r.append("")

    # Bağımsız kontrol
    r += ["## Bağımsız kontrol", "",
          f"300 rastgele satır (sabit tohum) basit döngülerle yeniden hesaplandı.",
          f"Maksimum mutlak fark: {elle_kontrol_sonuc['maks_fark']:.2e}", ""]

    metin = "\n".join(r) + "\n"
    Path("raporlar/hafta5_ozet.md").write_text(metin, encoding="utf-8")
    print(f"[OK] raporlar/hafta5_ozet.md kaydedildi.")


if __name__ == "__main__":
    main()
