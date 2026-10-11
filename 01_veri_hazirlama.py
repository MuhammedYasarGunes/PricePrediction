"""
İP 2 - Veri temini ve ön işleme (Rossmann Store Sales)

Girdi : <veri-dir>/train.csv ve <veri-dir>/store.csv   (Kaggle'dan indirilir)
Çıktı : <cikti-dir>/islenmis_tam.*        tüm satırlar + egitim_mi bayrağı
        <cikti-dir>/islenmis_egitim.*     yalnızca modele girecek satırlar
        <cikti-dir>/veri_kalite_raporu.txt

Rapor 2.4'teki kurallar:
  - Kapalı (Open=0) ve satışı sıfır günler eğitimden çıkarılır; tahminleri 0 atanır.
  - CompetitionDistance eksikleri çok büyük bir değerle doldurulur,
    rakip açılış tarihi eksikleri ayrı kategori (-1) olur.
  - Kategorik değişkenler sayısal kodlara çevrilir.
  - Hedef değişkene log(1 + Sales) uygulanır.
  - Aykırı değerler mağaza bazında MAD ile işaretlenir (silinmez).
  - Customers kullanılmaz (tahmin anında bilinmez, veri sızıntısı).

Kullanım:
  python 01_veri_hazirlama.py --veri-dir data --cikti-dir data/islenmis
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

BEKLENEN_SATIR = 1_017_209
BEKLENEN_MAGAZA = 1_115
UZAK_RAKIP_MESAFESI = 200_000      # "yakında rakip yok" anlamına gelen büyük değer (metre)
RAKIP_AYI_UST_SINIR = 120          # 1900 gibi uç açılış yıllarını sınırlamak için (ay)
MAD_ESIK = 5.0                     # |robust z| > 5 ise aykırı olarak işaretle

STORE_TYPE_KOD = {"a": 0, "b": 1, "c": 2, "d": 3}
ASSORTMENT_KOD = {"a": 0, "b": 1, "c": 2}
STATE_HOLIDAY_KOD = {"0": 0, "a": 1, "b": 2, "c": 3}
AY_KISALTMA = {1: "Jan", 2: "Feb", 3: "Mar", 4: "Apr", 5: "May", 6: "Jun",
               7: "Jul", 8: "Aug", 9: "Sep", 10: "Oct", 11: "Nov", 12: "Dec"}


def oku(veri_dir: Path):
    # StateHoliday dosyada hem 0 (sayı) hem '0' (metin) olarak geçer; metin olarak okuyoruz.
    train = pd.read_csv(veri_dir / "train.csv", dtype={"StateHoliday": str}, parse_dates=["Date"])
    store = pd.read_csv(veri_dir / "store.csv")
    train["StateHoliday"] = train["StateHoliday"].astype(str).str.strip()
    return train, store


def kalite_kontrol(train: pd.DataFrame, store: pd.DataFrame) -> list:
    """Ham veri üzerinde temel kontroller; rapor satırları döndürür."""
    r = []
    r.append("== HAM VERİ KONTROLÜ ==")
    r.append(f"train satır sayısı      : {len(train):,}  (beklenen {BEKLENEN_SATIR:,})")
    r.append(f"mağaza sayısı (train)   : {train['Store'].nunique():,}  (beklenen {BEKLENEN_MAGAZA:,})")
    r.append(f"tarih aralığı           : {train['Date'].min().date()} - {train['Date'].max().date()}")
    r.append(f"mükerrer (Store, Date)  : {int(train.duplicated(['Store', 'Date']).sum())}")

    gun_sayisi = (train["Date"].max() - train["Date"].min()).days + 1
    satir_per_magaza = train.groupby("Store").size()
    eksik = satir_per_magaza[satir_per_magaza < gun_sayisi]
    r.append(f"tam dönem gün sayısı    : {gun_sayisi}")
    r.append(f"eksik günü olan mağaza  : {len(eksik)}  (rapordaki bilinen boşluk: 180)")
    if len(eksik):
        r.append(f"  eksik gün sayısı dağılımı: {(gun_sayisi - eksik).value_counts().to_dict()}")

    r.append(f"Open=0 satır sayısı     : {int((train['Open'] == 0).sum()):,}")
    r.append(f"Open=1 ve Sales=0       : {int(((train['Open'] == 1) & (train['Sales'] == 0)).sum())}")
    r.append(f"Sales<0 satır           : {int((train['Sales'] < 0).sum())}")
    r.append(f"train eksik değerler    : {train.isna().sum()[lambda s: s > 0].to_dict()}")
    r.append(f"store eksik değerler    : {store.isna().sum()[lambda s: s > 0].to_dict()}")
    r.append(f"StateHoliday değerleri  : {sorted(train['StateHoliday'].unique().tolist())}")
    return r


def promo2_baslangic(store: pd.DataFrame) -> pd.Series:
    """Promo2SinceYear / Promo2SinceWeek (ISO hafta) -> başlangıç tarihi (mağaza düzeyinde)."""
    yil = store["Promo2SinceYear"]
    hafta = store["Promo2SinceWeek"]
    ok = yil.notna() & hafta.notna()
    baslangic = pd.Series(pd.NaT, index=store.index, dtype="datetime64[ns]")
    # ISO 8601: 1. hafta 4 Ocak'ı içeren haftadır.
    ocak4 = pd.to_datetime(yil[ok].astype(int).astype(str) + "-01-04")
    pazartesi = ocak4 - pd.to_timedelta(ocak4.dt.weekday, unit="D")
    baslangic[ok] = pazartesi + pd.to_timedelta((hafta[ok].astype(int) - 1) * 7, unit="D")
    return baslangic


def hazirla(train: pd.DataFrame, store: pd.DataFrame) -> pd.DataFrame:
    store = store.copy()
    store["Promo2Baslangic"] = promo2_baslangic(store)

    # Customers tahmin anında bilinmediği için (veri sızıntısı) hiç taşınmaz.
    df = train.drop(columns=["Customers"]).merge(store, on="Store", how="left", validate="many_to_one")
    df = df.sort_values(["Store", "Date"]).reset_index(drop=True)

    # --- rakip bilgisi ---
    df["CompetitionDistanceEksik"] = df["CompetitionDistance"].isna().astype(int)
    df["CompetitionDistance"] = df["CompetitionDistance"].fillna(UZAK_RAKIP_MESAFESI)

    acilis_var = df["CompetitionOpenSinceYear"].notna() & df["CompetitionOpenSinceMonth"].notna()
    ay_farki = ((df["Date"].dt.year - df["CompetitionOpenSinceYear"]) * 12
                + (df["Date"].dt.month - df["CompetitionOpenSinceMonth"]))
    # Açılış tarihi gelecekteyse (henüz açılmamış) 0; bilinmiyorsa -1 (ayrı kategori).
    df["CompetitionOpenMonths"] = np.where(acilis_var, ay_farki.clip(lower=0, upper=RAKIP_AYI_UST_SINIR), -1).astype(int)
    df = df.drop(columns=["CompetitionOpenSinceYear", "CompetitionOpenSinceMonth"])

    # --- Promo2: o gün için etkin mi? ---
    ay_kisa = df["Date"].dt.month.map(AY_KISALTMA)
    aralik = df["PromoInterval"].fillna("").str.replace("Sept", "Sep")
    ay_icinde = pd.Series([a in s.split(",") for a, s in zip(ay_kisa, aralik)], index=df.index)
    df["Promo2Active"] = ((df["Promo2"] == 1) & (df["Date"] >= df["Promo2Baslangic"]) & ay_icinde).astype(int)
    df = df.drop(columns=["Promo2SinceWeek", "Promo2SinceYear", "Promo2Baslangic"])

    # --- takvim öznitelikleri ---
    d = df["Date"].dt
    df["Year"] = d.year
    df["Month"] = d.month
    df["Day"] = d.day
    df["WeekOfYear"] = d.isocalendar().week.astype(int)
    df["IsMonthStart"] = d.is_month_start.astype(int)
    df["IsMonthEnd"] = d.is_month_end.astype(int)

    # --- kategorik kodlama (ham metin sütunları EDA için korunur) ---
    df["StoreType_kod"] = df["StoreType"].map(STORE_TYPE_KOD).astype(int)
    df["Assortment_kod"] = df["Assortment"].map(ASSORTMENT_KOD).astype(int)
    df["StateHoliday_kod"] = df["StateHoliday"].map(STATE_HOLIDAY_KOD).fillna(0).astype(int)

    # --- eğitime girecek satırlar ve hedef ---
    df["egitim_mi"] = (df["Open"] == 1) & (df["Sales"] > 0)
    df["LogSales"] = np.where(df["egitim_mi"], np.log1p(df["Sales"].clip(lower=0)), np.nan)

    # --- mağaza bazında MAD ile aykırı değer işareti (silinmez) ---
    g = df.loc[df["egitim_mi"]].groupby("Store")["LogSales"]
    med = g.transform("median")
    mad = (df.loc[df["egitim_mi"], "LogSales"] - med).abs().groupby(df.loc[df["egitim_mi"], "Store"]).transform("median")
    z = 0.6745 * (df.loc[df["egitim_mi"], "LogSales"] - med) / mad.replace(0, np.nan)
    df["AykiriDeger"] = 0
    df.loc[z.index, "AykiriDeger"] = (z.abs() > MAD_ESIK).astype(int)

    return df


def kaydet(df: pd.DataFrame, taban: Path) -> Path:
    """Parquet varsa onu, yoksa sıkıştırılmış CSV kullanır."""
    try:
        yol = taban.with_suffix(".parquet")
        df.to_parquet(yol, index=False)
    except Exception:
        yol = taban.with_suffix(".csv.gz")
        df.to_csv(yol, index=False)
    return yol


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--veri-dir", type=Path, default=Path("data"))
    ap.add_argument("--cikti-dir", type=Path, default=Path("data/islenmis"))
    args = ap.parse_args()
    args.cikti_dir.mkdir(parents=True, exist_ok=True)

    train, store = oku(args.veri_dir)
    rapor = kalite_kontrol(train, store)

    df = hazirla(train, store)
    egitim = df[df["egitim_mi"]].copy()

    rapor += [
        "",
        "== İŞLENMİŞ VERİ ==",
        f"toplam satır            : {len(df):,}",
        f"eğitime girecek satır   : {len(egitim):,}",
        f"çıkarılan satır         : {len(df) - len(egitim):,}  (kapalı gün veya satışı 0 olan gün)",
        f"aykırı işaretli satır   : {int(df['AykiriDeger'].sum()):,}",
        f"CompetitionDistance eksik doldurulan mağaza: {df.loc[df['CompetitionDistanceEksik'] == 1, 'Store'].nunique()}",
        f"rakip açılışı bilinmeyen mağaza            : {df.loc[df['CompetitionOpenMonths'] == -1, 'Store'].nunique()}",
        f"Promo2 etkin satır oranı: {df['Promo2Active'].mean():.3f}",
        f"son eğitim sütunları    : {list(egitim.columns)}",
    ]
    ozellikler = ["Store", "DayOfWeek", "Promo", "StateHoliday_kod", "SchoolHoliday", "StoreType_kod",
                  "Assortment_kod", "CompetitionDistance", "CompetitionOpenMonths", "Promo2", "Promo2Active",
                  "Year", "Month", "Day", "WeekOfYear", "IsMonthStart", "IsMonthEnd"]
    rapor.append(f"açıklayıcı öznitelik sayısı: {len(ozellikler)}  (hedef H1: en az 15)")

    p1 = kaydet(df, args.cikti_dir / "islenmis_tam")
    p2 = kaydet(egitim, args.cikti_dir / "islenmis_egitim")
    rapor += ["", f"kaydedildi: {p1.name}, {p2.name}"]

    metin = "\n".join(rapor)
    (args.cikti_dir / "veri_kalite_raporu.txt").write_text(metin, encoding="utf-8")
    print(metin)


if __name__ == "__main__":
    main()
