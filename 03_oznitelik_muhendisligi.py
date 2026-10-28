"""
İP 4 - Öznitelik mühendisliği (ufka duyarlı, sızıntısız)

Girdi : 01_veri_hazirlama.py çıktısı (<girdi-dir>/islenmis_tam.*)
Çıktı : <cikti-dir>/oznitelikler_H{H}.*      her H için modele hazır tablo (tüm satırlar + egitim_mi)
        <cikti-dir>/oznitelik_raporu.md       öznitelik listesi, eksik oranları, bölmeler, sızıntı testi

Rapor 2.3 ve 2.6'daki kurallar:
  - Tahmin ufku H tek bir parametredir (--ufuk); varsayılan 42 gün.
  - Geçmiş satıştan türetilen her öznitelik en az H gün gecikmelidir: t günü için yalnızca t-H ve
    öncesindeki satışlar kullanılır. Böylece T kökenli tahminde, T'den sonraki H günün hepsinde
    yalnızca T'ye kadar bilinen satışlar kullanılmış olur (doğrudan strateji).
  - Takvim, tatil, promosyon ve açık/kapalı bilgisinin tahmin anında önceden bilindiği varsayılır;
    bu yüzden promosyon ve tatil öznitelikleri gecikmesizdir.
  - Bilinmeyen günler (2014'teki veri boşluğu, verinin bittiği 31.07.2015'in sonrası) "tatil yok" ya da
    "satış yok" sayılmaz; ilgili öznitelik eksik (NaN) bırakılır. LightGBM eksik değerleri doğal işler.

Yeni öznitelikler (2. haftadaki 17 özniteliğe eklenir):
  PromoGunu         promosyon bloğunun kaçıncı günü (promosyon yoksa 0)
  TatileKalanGun    bir sonraki resmî tatile kalan gün (0..7; 7 = "7 gün ve üzeri")
  TatildenGecenGun  son resmî tatilden bu yana geçen gün (0..7; 7 = "7 gün ve üzeri")
  SatisGecikme      aynı haftagününün L = 7*ceil(H/7) gün önceki log satışı
  SatisOrt7/28/91   t-H'de biten 7, 28 ve 91 günlük pencerede açık günlerin ortalama log satışı
  MagazaGunOrt      aynı haftagününün t-L'de biten son 13 haftadaki ortalama log satışı

Kullanım:
  python 03_oznitelik_muhendisligi.py --girdi-dir data/islenmis --cikti-dir data/oznitelikler
  python 03_oznitelik_muhendisligi.py --ufuk 7 28 42 84        # ufuk analizi için dört H birden
"""
import argparse
import math
from pathlib import Path

import numpy as np
import pandas as pd

from ortak import (UFUK_ANALIZI_BASLANGIC, UFUK_DEGERLERI, VARSAYILAN_UFUK, gun, kaydet_tablo, katmanlar,
                   oku_islenmis, sayi, tarih, test_penceresi, ufuk_penceresi)

TATIL_UST_SINIR = 7                     # 7 = "en yakın resmî tatil en az 7 gün uzakta"
PENCERELER = {"SatisOrt7": 7, "SatisOrt28": 28, "SatisOrt91": 91}
EN_AZ_GOZLEM = {"SatisOrt7": 1, "SatisOrt28": 4, "SatisOrt91": 13, "MagazaGunOrt": 3}
GUN_HAFTA_SAYISI = 13

TEMEL_OZNITELIKLER = ["Store", "DayOfWeek", "Promo", "StateHoliday_kod", "SchoolHoliday", "StoreType_kod",
                      "Assortment_kod", "CompetitionDistance", "CompetitionOpenMonths", "Promo2", "Promo2Active",
                      "Year", "Month", "Day", "WeekOfYear", "IsMonthStart", "IsMonthEnd"]   # 2. hafta (17)
TAKVIM_OZNITELIKLERI = ["PromoGunu", "TatileKalanGun", "TatildenGecenGun"]
GECMIS_OZNITELIKLERI = ["SatisGecikme", "SatisOrt7", "SatisOrt28", "SatisOrt91", "MagazaGunOrt"]
YENI_OZNITELIKLER = TAKVIM_OZNITELIKLERI + GECMIS_OZNITELIKLERI
TUM_OZNITELIKLER = TEMEL_OZNITELIKLER + YENI_OZNITELIKLER

ACIKLAMA = {
    "PromoGunu": "Promosyon bloğunun kaçıncı günü (promosyon yoksa 0)",
    "TatileKalanGun": "Bir sonraki resmî tatile kalan gün (0–7; 7 = 7 gün ve üzeri)",
    "TatildenGecenGun": "Son resmî tatilden bu yana geçen gün (0–7; 7 = 7 gün ve üzeri)",
    "SatisGecikme": "Aynı haftagününün L = 7·⌈H/7⌉ gün önceki log satışı",
    "SatisOrt7": "t−H'de biten 7 günde açık günlerin ortalama log satışı",
    "SatisOrt28": "t−H'de biten 28 günde açık günlerin ortalama log satışı",
    "SatisOrt91": "t−H'de biten 91 günde açık günlerin ortalama log satışı",
    "MagazaGunOrt": "Aynı haftagününün t−L'de biten 13 haftalık ortalama log satışı",
}


# --- tarih x mağaza matrisleri ----------------------------------------------------------------------
class Izgara:
    """Tüm takvim günleri x tüm mağazalar. Veride satırı olmayan gün (veri boşluğu) NaN olur."""

    def __init__(self, df: pd.DataFrame):
        self.tarihler = pd.date_range(df["Date"].min(), df["Date"].max(), freq="D")
        self.magazalar = np.sort(df["Store"].unique())
        self.i = (df["Date"] - self.tarihler[0]).dt.days.to_numpy()
        self.j = pd.Index(self.magazalar).get_indexer(df["Store"])

    def matris(self, degerler) -> np.ndarray:
        m = np.full((len(self.tarihler), len(self.magazalar)), np.nan)
        m[self.i, self.j] = np.asarray(degerler, dtype=float)
        return m

    def satirlara(self, m: np.ndarray) -> np.ndarray:
        return m[self.i, self.j]


def kaydir(m: np.ndarray, k: int) -> np.ndarray:
    """k gün geriye bakan değer: sonuc[t] = m[t-k] (t-k veri başlangıcından önceyse NaN)."""
    s = np.full_like(m, np.nan)
    if k < len(m):
        s[k:] = m[:len(m) - k]
    return s


# --- takvim tabanlı öznitelikler (önceden bilinir) ---------------------------------------------------
def promo_gunu(p: np.ndarray) -> np.ndarray:
    """Ardışık promosyon günlerini sayar; bilinmeyen gün zinciri bozar (blok bitene kadar NaN)."""
    sonuc = np.full_like(p, np.nan)
    onceki = np.full(p.shape[1], np.nan)
    for t in range(len(p)):
        bugun = np.where(p[t] == 1, onceki + 1, 0.0)
        bugun[np.isnan(p[t])] = np.nan
        sonuc[t] = bugun
        onceki = bugun
    return sonuc


def tatil_uzakligi(h: np.ndarray, ust: int, ileri: bool) -> np.ndarray:
    """En yakın resmî tatile uzaklık (gün), 0..ust ile sınırlı.

    ileri=False: son tatilden bu yana geçen gün; ileri=True: sonraki tatile kalan gün.
    Değer ancak ilgili günler biliniyorsa hesaplanır; aksi halde NaN.
    """
    sira = range(len(h) - 1, -1, -1) if ileri else range(len(h))
    sonuc = np.full_like(h, np.nan)
    uzaklik = np.full(h.shape[1], np.nan)    # mevcut bilinen dönem içindeki son tatile uzaklık
    seri = np.zeros(h.shape[1])              # art arda bilinen gün sayısı
    for t in sira:
        bilinen = ~np.isnan(h[t])
        uzaklik = np.where(h[t] == 1, 0.0, uzaklik + 1)
        uzaklik[~bilinen] = np.nan
        seri = np.where(bilinen, seri + 1, 0)
        sonuc[t] = np.where(~np.isnan(uzaklik), np.minimum(uzaklik, ust), np.where(seri >= ust, ust, np.nan))
    return sonuc


# --- geçmiş satış öznitelikleri (en az H gün gecikmeli) ---------------------------------------------
def gecmis_satis(ls: np.ndarray, ufuk: int) -> dict:
    """ls: log satış matrisi (açık ve satışı > 0 olmayan günler NaN). Tüm değerler t-H ve öncesinden gelir."""
    L = 7 * math.ceil(ufuk / 7)                   # aynı haftagününe denk gelen en küçük gecikme (>= H)
    sonuc = {"SatisGecikme": kaydir(ls, L)}
    gecikmeli = pd.DataFrame(kaydir(ls, ufuk))
    for ad, w in PENCERELER.items():
        sonuc[ad] = gecikmeli.rolling(w, min_periods=EN_AZ_GOZLEM[ad]).mean().to_numpy()
    toplam = np.zeros_like(ls)
    adet = np.zeros_like(ls)
    for k in range(GUN_HAFTA_SAYISI):
        m = kaydir(ls, L + 7 * k)
        toplam += np.nan_to_num(m)
        adet += ~np.isnan(m)
    with np.errstate(invalid="ignore", divide="ignore"):
        sonuc["MagazaGunOrt"] = np.where(adet >= EN_AZ_GOZLEM["MagazaGunOrt"], toplam / adet, np.nan)
    return sonuc


def log_satis_matrisi(df: pd.DataFrame, izgara: Izgara) -> np.ndarray:
    return izgara.matris(np.where(df["egitim_mi"], np.log1p(df["Sales"].clip(lower=0)), np.nan))


def oznitelik_uret(df: pd.DataFrame, ufuk: int, izgara: Izgara = None) -> pd.DataFrame:
    izgara = izgara or Izgara(df)
    out = df.copy()
    out["PromoGunu"] = izgara.satirlara(promo_gunu(izgara.matris(df["Promo"])))
    tatil = izgara.matris((df["StateHoliday_kod"] > 0).astype(float))
    out["TatileKalanGun"] = izgara.satirlara(tatil_uzakligi(tatil, TATIL_UST_SINIR, ileri=True))
    out["TatildenGecenGun"] = izgara.satirlara(tatil_uzakligi(tatil, TATIL_UST_SINIR, ileri=False))
    for ad, m in gecmis_satis(log_satis_matrisi(df, izgara), ufuk).items():
        out[ad] = izgara.satirlara(m)
    out["Ufuk"] = ufuk
    return out


# --- sızıntı testi ------------------------------------------------------------------------------------
def sizinti_testi(df: pd.DataFrame, izgara: Izgara, ufuk: int, kesim: pd.Timestamp, tohum: int = 0) -> dict:
    """Kesim tarihinden SONRAKİ satışları rastgele değiştirip öznitelikleri yeniden üretir.

    Sızıntı yoksa t <= kesim + H olan hiçbir satırın geçmiş satış özniteliği değişmemelidir.
    t > kesim + H satırlarında değişiklik görülmesi, testin gerçekten duyarlı olduğunu gösterir.
    """
    rng = np.random.default_rng(tohum)
    asil = gecmis_satis(log_satis_matrisi(df, izgara), ufuk)
    bozuk_df = df.copy()
    sonra = bozuk_df["Date"] > kesim
    bozuk_df.loc[sonra, "Sales"] = rng.integers(1, 40_000, int(sonra.sum()))
    bozuk = gecmis_satis(log_satis_matrisi(bozuk_df, izgara), ufuk)

    gun_no = (izgara.tarihler - kesim).days.to_numpy()[:, None]
    korunmali = np.broadcast_to(gun_no <= ufuk, asil["SatisGecikme"].shape)
    degisen_once, degisen_sonra = 0, 0
    for ad in GECMIS_OZNITELIKLERI:
        a, b = asil[ad], bozuk[ad]
        fark = ~((a == b) | (np.isnan(a) & np.isnan(b)))
        degisen_once += int(fark[korunmali].sum())
        degisen_sonra += int(fark[~korunmali].sum())
    # Kesim + H verinin sonunu aşıyorsa "sonrası" bölgesi yoktur; duyarlılık o kesimde ölçülemez.
    duyarlilik_olculebilir = bool((~korunmali).any())
    return {"ufuk": ufuk, "kesim": kesim, "korunan_hucre": int(korunmali.sum()) * len(GECMIS_OZNITELIKLERI),
            "degisen_korunan": degisen_once, "degisen_sonrasi": degisen_sonra,
            "duyarlilik_olculebilir": duyarlilik_olculebilir,
            "gecti": degisen_once == 0 and (degisen_sonra > 0 or not duyarlilik_olculebilir)}


# --- rapor ----------------------------------------------------------------------------------------------
def eksik_orani(df: pd.DataFrame, sutun: str, maske=None) -> float:
    m = df["egitim_mi"] if maske is None else (df["egitim_mi"] & maske)
    return float(df.loc[m, sutun].isna().mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--girdi-dir", type=Path, default=Path("data/islenmis"))
    ap.add_argument("--cikti-dir", type=Path, default=Path("data/oznitelikler"))
    ap.add_argument("--ufuk", type=int, nargs="+", default=list(UFUK_DEGERLERI),
                    help="tahmin ufku H (gün); birden fazla verilebilir")
    ap.add_argument("--sizinti-testi-yok", action="store_true", help="sızıntı testini atla")
    args = ap.parse_args()
    args.cikti_dir.mkdir(parents=True, exist_ok=True)

    df = oku_islenmis(args.girdi_dir)
    izgara = Izgara(df)
    bosluk_magazalari = set(df.groupby("Store").size().loc[lambda s: s < len(izgara.tarihler) - 1].index)

    r = ["# Öznitelik mühendisliği raporu (İP 4)", "",
         "`03_oznitelik_muhendisligi.py` tarafından üretilmiştir. Eksik oranları yalnızca modele girecek "
         "(açık ve satışı > 0) satırlar üzerinden hesaplanır.", ""]
    r += [f"- Toplam öznitelik: {len(TUM_OZNITELIKLER)} (2. haftadan {len(TEMEL_OZNITELIKLER)}, "
          f"bu hafta eklenen {len(YENI_OZNITELIKLER)})",
          f"- Modele girecek satır: {sayi(int(df['egitim_mi'].sum()))}", ""]

    eksik_tablo, iliski_tablo, sizinti = [], [], []
    test_bas, test_bit = test_penceresi()
    for H in args.ufuk:
        out = oznitelik_uret(df, H, izgara)
        yol = kaydet_tablo(out[["Store", "Date", "Sales", "Open", "egitim_mi", "LogSales", "AykiriDeger", "Ufuk"]
                               + [c for c in TUM_OZNITELIKLER if c != "Store"]],
                           args.cikti_dir / f"oznitelikler_H{H}")
        bas, bit = (test_bas, test_bit) if H == VARSAYILAN_UFUK else ufuk_penceresi(H)
        pencere = out["Date"].between(bas, bit)
        bosluk = out["Store"].isin(bosluk_magazalari) & out["Date"].between("2015-01-01", "2015-03-31")
        satir = {"H": H, "dosya": yol.name}
        for c in YENI_OZNITELIKLER:
            satir[c] = eksik_orani(out, c)
            satir[c + "_pencere"] = eksik_orani(out, c, pencere)
            satir[c + "_bosluk"] = eksik_orani(out, c, bosluk)
        eksik_tablo.append(satir)
        e = out[out["egitim_mi"]]
        iliski_tablo.append({"H": H, **{c: float(e[c].corr(e["LogSales"])) for c in GECMIS_OZNITELIKLERI}})
        if not args.sizinti_testi_yok:
            for kesim in (UFUK_ANALIZI_BASLANGIC - gun(1), test_bas - gun(1), pd.Timestamp("2014-03-31")):
                sizinti.append(sizinti_testi(df, izgara, H, kesim, tohum=H))
        print(f"H={H}: {yol} ({len(out):,} satır)")

    r += ["## Yeni öznitelikler", "", "| Öznitelik | Tanım |", "|---|---|"]
    r += [f"| {c} | {ACIKLAMA[c]} |" for c in YENI_OZNITELIKLER]
    r += ["", "## Eksik değer oranı (modele girecek satırlar)", "",
          "Sütunlar: tüm dönem / değerlendirme penceresi (H=42 için test, diğerleri için ufuk analizi) / "
          "veri boşluğu olan 180 mağazanın Ocak–Mart 2015 satırları.", "",
          "| Öznitelik | " + " | ".join(f"H={s['H']}" for s in eksik_tablo) + " |",
          "|---|" + "---|" * len(eksik_tablo)]
    for c in YENI_OZNITELIKLER:
        r.append(f"| {c} | " + " | ".join(
            f"%{sayi(s[c] * 100, 1)} / %{sayi(s[c + '_pencere'] * 100, 1)} / %{sayi(s[c + '_bosluk'] * 100, 1)}"
            for s in eksik_tablo) + " |")
    r += ["", "## Geçmiş satış özniteliklerinin log satışla korelasyonu (Pearson)", "",
          "Ufuk uzadıkça öznitelikler daha eski bilgiye dayanır; korelasyondaki düşüş bunu gösterir.", "",
          "| Öznitelik | " + " | ".join(f"H={s['H']}" for s in iliski_tablo) + " |",
          "|---|" + "---|" * len(iliski_tablo)]
    for c in GECMIS_OZNITELIKLERI:
        r.append(f"| {c} | " + " | ".join(sayi(s[c], 3) for s in iliski_tablo) + " |")

    r += ["", "## Kronolojik bölmeler", ""]
    for H in args.ufuk:
        bas, bit = (test_bas, test_bit) if H == VARSAYILAN_UFUK else ufuk_penceresi(H)
        ad = "Test" if H == VARSAYILAN_UFUK else "Ufuk analizi"
        r.append(f"**H = {H}** ({ad}: {tarih(bas)}–{tarih(bit)})")
        r.append("")
        r.append("| Katman | Eğitim | Doğrulama | Eğitim satırı | Doğrulama satırı |")
        r.append("|---|---|---|---|---|")
        for k in katmanlar(H, bas):
            egt = int((df["egitim_mi"] & (df["Date"] <= k["egitim_bitis"])).sum())
            dog = int((df["egitim_mi"] & df["Date"].between(k["dogrulama_baslangic"], k["dogrulama_bitis"])).sum())
            r.append(f"| {k['ad']} | 01.01.2013–{tarih(k['egitim_bitis'])} | "
                     f"{tarih(k['dogrulama_baslangic'])}–{tarih(k['dogrulama_bitis'])} | {sayi(egt)} | {sayi(dog)} |")
        r.append("")

    if sizinti:
        r += ["## Sızıntı testi", "",
              "Kesim tarihinden sonraki tüm satışlar rastgele değerlerle değiştirilip geçmiş satış öznitelikleri "
              "yeniden üretildi. Kesim + H gününe kadar hiçbir değer değişmemeli (sızıntı yok); sonrasında değişmeli "
              "(test duyarlı).", "",
              "| H | Kesim | Korunan hücre | Korunan bölgede değişen | Sonrasında değişen | Sonuç |",
              "|---|---|---|---|---|---|"]
        for s in sizinti:
            sonrasi = sayi(s["degisen_sonrasi"]) if s["duyarlilik_olculebilir"] else "— (kesim + H veri sonunda)"
            r.append(f"| {s['ufuk']} | {tarih(s['kesim'])} | {sayi(s['korunan_hucre'])} | {s['degisen_korunan']} | "
                     f"{sonrasi} | {'geçti' if s['gecti'] else 'KALDI'} |")
        duyarli = {s["ufuk"] for s in sizinti if s["duyarlilik_olculebilir"] and s["degisen_sonrasi"] > 0}
        r += ["", f"Toplam {len(sizinti)} testin {sum(s['gecti'] for s in sizinti)} tanesi geçti. "
              f"Testin duyarlılığı her H için en az bir kesimde doğrulandı: "
              f"{'evet' if duyarli == {s['ufuk'] for s in sizinti} else 'HAYIR'}."]

    metin = "\n".join(r) + "\n"
    (args.cikti_dir / "oznitelik_raporu.md").write_text(metin, encoding="utf-8")
    pd.DataFrame(eksik_tablo).to_csv(args.cikti_dir / "eksik_oranlari.csv", index=False)
    pd.DataFrame(iliski_tablo).to_csv(args.cikti_dir / "korelasyonlar.csv", index=False)
    pd.DataFrame(sizinti).to_csv(args.cikti_dir / "sizinti_testi.csv", index=False)
    print(metin)
    if sizinti and not all(s["gecti"] for s in sizinti):
        raise SystemExit("Sızıntı testi başarısız: geçmiş satış öznitelikleri gelecekten bilgi alıyor.")
    if sizinti and duyarli != {s["ufuk"] for s in sizinti}:
        raise SystemExit("Sızıntı testi bazı H değerlerinde hiç duyarlılık gösteremedi; kesim tarihlerini gözden geçirin.")


if __name__ == "__main__":
    main()
