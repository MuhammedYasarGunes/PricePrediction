"""
İP 4 bağımsız kontrolü

1) Rastgele seçilen satırlarda 03_oznitelik_muhendisligi.py'nin ürettiği 8 yeni özniteliği, tanımlarından
   doğrudan (basit döngülerle) yeniden hesaplar ve karşılaştırır.
2) Negatif kontrol: kasıtlı olarak 1 gün sızdıran bir sürümün sızıntı testinde yakalandığını doğrular.

Kullanım (önce 01 ve 03 çalıştırılmış olmalı):
  python kontrol_oznitelikler.py --girdi-dir data/islenmis --oznitelik-dir data/oznitelikler --ufuk 42
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--girdi-dir", type=Path, default=Path("data/islenmis"))
    ap.add_argument("--oznitelik-dir", type=Path, default=Path("data/oznitelikler"))
    ap.add_argument("--ufuk", type=int, default=42)
    ap.add_argument("--ornek", type=int, default=400)
    args = ap.parse_args()

    df = oku_islenmis(args.girdi_dir)
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
