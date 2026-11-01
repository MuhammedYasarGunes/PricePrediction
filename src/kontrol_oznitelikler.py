"""
İP 4 ve İP 6 bağımsız kontrolü

1) Rastgele seçilen satırlarda 03_oznitelik_muhendisligi.py'nin ürettiği 8 yeni özniteliği, tanımlarından
   doğrudan (basit döngülerle) yeniden hesaplar ve karşılaştırır.
2) Negatif kontrol: kasıtlı olarak 1 gün sızdıran bir sürümün sızıntı testinde yakalandığını doğrular.
3) --promo-kontrol: 05_promosyon_duzeltmesi.py'nin ürettiği 3 yeni promo özniteliğini (PromoGecikme,
   PromoUyumsuz, SatisGecikmePromoDuz) 300 rastgele satırda basit döngülerle yeniden hesaplar ve karşılaştırır.

Kullanım (önce 01 ve 03 çalıştırılmış olmalı):
  python src/kontrol_oznitelikler.py --girdi-dir data/islenmis --oznitelik-dir data/oznitelikler --ufuk 42
  python src/kontrol_oznitelikler.py --promo-kontrol --girdi-dir data/islenmis --promo-oznitelik-dir data/oznitelikler_v2 --ufuk 42
"""
import argparse
import importlib.util
import math
from pathlib import Path

import numpy as np
import pandas as pd

from ortak import oku_islenmis

KOK = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location("oznitelik", KOK / "03_oznitelik_muhendisligi.py")
oz = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(oz)
GUN = pd.Timedelta(days=1)

DELTA_PENCERE = 91
DELTA_MIN_GOZLEM = 4


def esit(a, b) -> bool:
    return (pd.isna(a) and pd.isna(b)) or (not pd.isna(a) and not pd.isna(b) and abs(a - b) < 1e-9)


def beklenen_degerler(s, t, H, ls, promo, tatil, bas) -> dict:
    L = 7 * math.ceil(H / 7)
    b = {"SatisGecikme": ls.get((s, t - L * GUN), np.nan)}
    for ad, w in oz.PENCERELER.items():
        v = [ls[(s, t - (H + k) * GUN)] for k in range(w) if (s, t - (H + k) * GUN) in ls and t - (H + k) * GUN >= bas]
        b[ad] = np.mean(v) if len(v) >= oz.EN_AZ_GOZLEM[ad] else np.nan
    v = [ls[(s, t - (L + 7 * k) * GUN)] for k in range(oz.GUN_HAFTA_SAYISI) if (s, t - (L + 7 * k) * GUN) in ls]
    b["MagazaGunOrt"] = np.mean(v) if len(v) >= oz.EN_AZ_GOZLEM["MagazaGunOrt"] else np.nan
    if promo[(s, t)] == 0:
        b["PromoGunu"] = 0
    else:
        k, d = 0, t
        while (s, d) in promo and promo[(s, d)] == 1:
            k, d = k + 1, d - GUN
        b["PromoGunu"] = k if (s, d) in promo else np.nan
    for ad, yon in (("TatildenGecenGun", -1), ("TatileKalanGun", 1)):
        b[ad] = np.nan
        for k in range(oz.TATIL_UST_SINIR + 1):
            d = t + yon * k * GUN
            if k == oz.TATIL_UST_SINIR:
                b[ad] = k
                break
            if (s, d) not in tatil:
                break
            if tatil[(s, d)] == 1:
                b[ad] = k
                break
    return b


def beklenen_promo_degerler(s, t, H, ls, promo, dow_dict, tum_magazalar) -> dict:
    """3 yeni promo özniteliğini basit döngülerle yeniden hesaplar."""
    L = 7 * math.ceil(H / 7)

    # PromoGecikme
    pg = promo.get((s, t - L * GUN), np.nan)

    # PromoUyumsuz
    promo_t = promo.get((s, t), np.nan)
    if not pd.isna(pg) and not pd.isna(promo_t):
        pu = 1.0 if promo_t != pg else 0.0
    else:
        pu = np.nan

    # SatisGecikmePromoDuz
    satis_gecikme = ls.get((s, t - L * GUN), np.nan)
    if pd.isna(satis_gecikme):
        sdpd = np.nan
    else:
        # delta hesapla: 91 günlük pencere, hafta-içi, t-H'de biten
        promo_vals, non_promo_vals = [], []
        for k in range(DELTA_PENCERE):
            d = t - (H + k) * GUN
            if (s, d) in ls and (s, d) in dow_dict:
                dw = dow_dict[(s, d)]
                if 1 <= dw <= 5:
                    p = promo.get((s, d), np.nan)
                    if p == 1:
                        promo_vals.append(ls[(s, d)])
                    elif p == 0:
                        non_promo_vals.append(ls[(s, d)])

        if len(promo_vals) >= DELTA_MIN_GOZLEM and len(non_promo_vals) >= DELTA_MIN_GOZLEM:
            delta_val = np.mean(promo_vals) - np.mean(non_promo_vals)
        else:
            # Pooled delta: tüm mağazalardaki aynı pencere
            pool_promo, pool_non_promo = [], []
            for store in tum_magazalar:
                s_promo, s_non_promo = [], []
                for k2 in range(DELTA_PENCERE):
                    d2 = t - (H + k2) * GUN
                    if (store, d2) in ls and (store, d2) in dow_dict:
                        dw2 = dow_dict[(store, d2)]
                        if 1 <= dw2 <= 5:
                            p2 = promo.get((store, d2), np.nan)
                            if p2 == 1:
                                s_promo.append(ls[(store, d2)])
                            elif p2 == 0:
                                s_non_promo.append(ls[(store, d2)])
                if len(s_promo) >= DELTA_MIN_GOZLEM and len(s_non_promo) >= DELTA_MIN_GOZLEM:
                    pool_promo.extend(s_promo)
                    pool_non_promo.extend(s_non_promo)
            if pool_promo and pool_non_promo:
                delta_val = np.mean(pool_promo) - np.mean(pool_non_promo)
            else:
                delta_val = np.nan

        if pd.isna(promo_t) or pd.isna(pg) or pd.isna(delta_val):
            sdpd = np.nan
        else:
            sdpd = satis_gecikme + (promo_t - pg) * delta_val

    return {"PromoGecikme": pg, "PromoUyumsuz": pu, "SatisGecikmePromoDuz": sdpd}


def kontrol_promo(df, args):
    """05_promosyon_duzeltmesi.py'nin ürettiği 3 yeni özniteliği bağımsız olarak doğrular."""
    yol = next(args.promo_oznitelik_dir.glob(f"oznitelikler_H{args.ufuk}.*"))
    f = (pd.read_parquet(yol) if yol.suffix == ".parquet" else pd.read_csv(yol, parse_dates=["Date"]))
    f = f.set_index(["Store", "Date"])

    ls = {(s, d): math.log1p(v) for s, d, v, e in zip(df.Store, df.Date, df.Sales, df.egitim_mi) if e}
    promo = dict(zip(zip(df.Store, df.Date), df.Promo))
    dow_dict = dict(zip(zip(df.Store, df.Date), df.DayOfWeek))
    tum_magazalar = sorted(df["Store"].unique())

    ornek = df[df.egitim_mi].sample(300, random_state=42)
    maks_fark = 0.0
    hatalar = {"PromoGecikme": 0, "PromoUyumsuz": 0, "SatisGecikmePromoDuz": 0}

    for s, t in zip(ornek.Store, ornek.Date):
        b = beklenen_promo_degerler(s, t, args.ufuk, ls, promo, dow_dict, tum_magazalar)
        satir = f.loc[(s, t)]
        for c in ["PromoGecikme", "PromoUyumsuz", "SatisGecikmePromoDuz"]:
            bek = b[c]
            ger = satir[c]
            if not esit(ger, bek):
                hatalar[c] += 1
            elif not pd.isna(bek) and not pd.isna(ger):
                maks_fark = max(maks_fark, abs(ger - bek))

    print(f"3) Promo öznitelikleri bağımsız kontrolü (300 satır, H={args.ufuk}) uyuşmazlık sayıları: {hatalar}")
    print(f"   Maksimum mutlak fark: {maks_fark:.2e}")
    if any(hatalar.values()):
        raise SystemExit("Promo öznitelikleri kontrolü başarısız.")
    print("   Promo öznitelikleri kontrolü başarılı.")
    return {"satir": 300, "maks_fark": maks_fark}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--girdi-dir", type=Path, default=Path("data/islenmis"))
    ap.add_argument("--oznitelik-dir", type=Path, default=Path("data/oznitelikler"))
    ap.add_argument("--promo-oznitelik-dir", type=Path, default=Path("data/oznitelikler_v2"))
    ap.add_argument("--ufuk", type=int, default=42)
    ap.add_argument("--ornek", type=int, default=400)
    ap.add_argument("--promo-kontrol", action="store_true",
                    help="05_promosyon_duzeltmesi.py'nin 3 yeni promo özniteliğini bağımsız doğrula")
    args = ap.parse_args()

    df = oku_islenmis(args.girdi_dir)

    if args.promo_kontrol:
        kontrol_promo(df, args)
        return

    yol = next(args.oznitelik_dir.glob(f"oznitelikler_H{args.ufuk}.*"))
    f = (pd.read_parquet(yol) if yol.suffix == ".parquet" else pd.read_csv(yol, parse_dates=["Date"]))
    f = f.set_index(["Store", "Date"])

    ls = {(s, d): math.log1p(v) for s, d, v, e in zip(df.Store, df.Date, df.Sales, df.egitim_mi) if e}
    promo = dict(zip(zip(df.Store, df.Date), df.Promo))
    tatil = dict(zip(zip(df.Store, df.Date), (df.StateHoliday_kod > 0).astype(int)))
    ornek = df[df.egitim_mi].sample(args.ornek, random_state=3)
    hatalar = {c: 0 for c in oz.YENI_OZNITELIKLER}
    for s, t in zip(ornek.Store, ornek.Date):
        b = beklenen_degerler(s, t, args.ufuk, ls, promo, tatil, df.Date.min())
        satir = f.loc[(s, t)]
        for c in oz.YENI_OZNITELIKLER:
            hatalar[c] += not esit(satir[c], b[c])
    print(f"1) Bağımsız yeniden hesaplama ({args.ornek} rastgele satır, H={args.ufuk}) uyuşmazlık sayıları: {hatalar}")

    asil = oz.gecmis_satis
    oz.gecmis_satis = lambda m, ufuk: asil(m, ufuk - 1)          # kasıtlı 1 günlük sızıntı
    sonuc = oz.sizinti_testi(df, oz.Izgara(df), args.ufuk, pd.Timestamp("2015-05-08"))
    oz.gecmis_satis = asil
    print(f"2) Negatif kontrol (H-1 gecikmeli sürüm): korunan bölgede değişen hücre = {sonuc['degisen_korunan']}, "
          f"test sonucu = {'geçti (SORUN!)' if sonuc['gecti'] else 'kaldı (beklenen)'}")

    if any(hatalar.values()) or sonuc["gecti"]:
        raise SystemExit("Kontrol başarısız.")
    print("Tüm kontroller başarılı.")


if __name__ == "__main__":
    main()

