"""
İP 3 - Keşifsel veri analizi (EDA)

Girdi : 01_veri_hazirlama.py çıktısı (islenmis_tam.parquet veya islenmis_tam.csv.gz)
Çıktı : <cikti-dir>/grafikler/*.png   (10 grafik; hedef: en az 8)
        <cikti-dir>/bulgu_ozeti.md    (verideki gerçek sayılardan otomatik üretilen özet)

Kullanım:
  python src/02_eda.py --girdi-dir data/islenmis --cikti-dir eda_ciktilari

Not: Ortalamalar "açık ve satışı > 0" olan mağaza-günleri üzerinden hesaplanır.
Gruplar arası farklar (ör. promosyonlu / promosyonsuz) nedensellik kanıtı değildir;
sadece betimleyici karşılaştırmadır. Raporda böyle ifade edin.
"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

MAVI, TURUNCU, TURKUAZ = "#2a78d6", "#eb6834", "#1baf7a"   # kategorik sıra: 1, 2, 3
MUREKKEP, MUREKKEP2, IZGARA = "#0b0b0b", "#52514e", "#e3e2dd"
GUNLER = ["Pzt", "Sal", "Çar", "Per", "Cum", "Cmt", "Paz"]
AYLAR = ["Oca", "Şub", "Mar", "Nis", "May", "Haz", "Tem", "Ağu", "Eyl", "Eki", "Kas", "Ara"]

plt.rcParams.update({
    "figure.facecolor": "white", "axes.facecolor": "white", "savefig.facecolor": "white",
    "axes.edgecolor": IZGARA, "axes.labelcolor": MUREKKEP2, "xtick.color": MUREKKEP2, "ytick.color": MUREKKEP2,
    "text.color": MUREKKEP, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "axes.grid.axis": "y", "grid.color": IZGARA, "grid.linewidth": 0.8,
    "axes.axisbelow": True, "axes.titlelocation": "left", "axes.titlesize": 12, "axes.titleweight": "bold",
    "font.size": 10, "legend.frameon": False,
})


def sayi(v, ondalik=0):
    """Türkçe binlik ayracı: 1.234"""
    return f"{v:,.{ondalik}f}".replace(",", "X").replace(".", ",").replace("X", ".")


def oku_islenmis(girdi_dir: Path) -> pd.DataFrame:
    for ad in ("islenmis_tam.parquet", "islenmis_tam.csv.gz"):
        yol = girdi_dir / ad
        if yol.exists():
            return pd.read_parquet(yol) if ad.endswith(".parquet") else pd.read_csv(yol, parse_dates=["Date"])
    raise FileNotFoundError(f"{girdi_dir} içinde islenmis_tam.* bulunamadı. Önce 01_veri_hazirlama.py çalıştırın.")


def kaydet(fig, dizin: Path, ad: str):
    fig.tight_layout()
    fig.savefig(dizin / ad, dpi=150)
    plt.close(fig)


def cubuk(ax, etiketler, degerler, renk=MAVI, ondalik=0, ek_etiket=None):
    bars = ax.bar(range(len(degerler)), degerler, color=renk, width=0.6)
    ax.set_xticks(range(len(etiketler)))
    ax.set_xticklabels(etiketler)
    metin = [sayi(v, ondalik) for v in degerler]
    if ek_etiket:
        metin = [f"{m}\n{e}" for m, e in zip(metin, ek_etiket)]
    ax.bar_label(bars, labels=metin, padding=3, color=MUREKKEP2, fontsize=9)
    ax.set_ylim(0, max(degerler) * 1.18)
    return bars


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--girdi-dir", type=Path, default=Path("data/islenmis"))
    ap.add_argument("--cikti-dir", type=Path, default=Path("eda_ciktilari"))
    args = ap.parse_args()
    gdir = args.cikti_dir / "grafikler"
    gdir.mkdir(parents=True, exist_ok=True)

    df = oku_islenmis(args.girdi_dir)
    acik = df[df["egitim_mi"]].copy()
    bulgu = []   # (başlık, [cümleler])

    # 1) Zaman içinde ortalama satış --------------------------------------------------------
    aylik = acik.groupby(pd.Grouper(key="Date", freq="MS"))["Sales"].mean()
    fig, ax = plt.subplots(figsize=(9, 4.2))
    ax.plot(aylik.index, aylik.values, color=MAVI, linewidth=2)
    ax.set_title("Aylık ortalama günlük mağaza satışı (açık günler)")
    ax.set_ylabel("Ortalama satış (€)")
    ax.set_ylim(0, aylik.max() * 1.15)
    kaydet(fig, gdir, "01_aylik_ortalama_satis.png")
    zirve, dip = aylik.idxmax(), aylik.idxmin()
    bulgu.append(("Zaman içinde satış", [
        f"Açık mağaza-günlerinin ortalama satışı {sayi(acik['Sales'].mean())}, medyanı {sayi(acik['Sales'].median())}.",
        f"Aylık ortalamanın en yükseği {zirve:%m.%Y} ({sayi(aylik.max())}), en düşüğü {dip:%m.%Y} ({sayi(aylik.min())}).",
    ]))

    # 2) Haftanın günü ----------------------------------------------------------------------
    gun_ort = acik.groupby("DayOfWeek")["Sales"].mean().reindex(range(1, 8))
    acik_orani = df.groupby("DayOfWeek")["Open"].mean().reindex(range(1, 8))
    fig, ax = plt.subplots(figsize=(8, 4.2))
    cubuk(ax, GUNLER, gun_ort.values, ek_etiket=[f"açık %{v * 100:.0f}" for v in acik_orani.values])
    ax.set_title("Haftanın gününe göre ortalama satış")
    ax.set_ylabel("Ortalama satış (€)")
    kaydet(fig, gdir, "02_haftanin_gunu.png")
    bulgu.append(("Haftalık mevsimsellik", [
        f"En yüksek ortalama {GUNLER[int(gun_ort.idxmax()) - 1]} ({sayi(gun_ort.max())}), en düşük {GUNLER[int(gun_ort.idxmin()) - 1]} ({sayi(gun_ort.min())}).",
        f"Pazar günü mağazaların yalnızca %{sayi(acik_orani.loc[7] * 100, 1)}'i açık; pazar ortalaması az sayıda mağazaya dayanır.",
    ]))

    # 3) Promosyon etkisi (haftanın gününe göre) -------------------------------------------------
    pv = acik.pivot_table(index="DayOfWeek", columns="Promo", values="Sales", aggfunc="mean").reindex(range(1, 8))
    fig, ax = plt.subplots(figsize=(8, 4.2))
    ax.plot(GUNLER, pv[0], color=MAVI, marker="o", markersize=6, linewidth=2, label="Promosyon yok")
    if 1 in pv.columns:
        ax.plot(GUNLER, pv[1], color=TURUNCU, marker="o", markersize=6, linewidth=2, label="Promosyon var")
    ax.set_title("Promosyonun haftanın gününe göre satışa etkisi")
    ax.set_ylabel("Ortalama satış (€)")
    ax.set_ylim(0, np.nanmax(pv.values) * 1.15)
    ax.legend(loc="lower left")
    kaydet(fig, gdir, "03_promosyon_etkisi.png")
    p = acik.groupby("Promo")["Sales"].mean()
    fark = (p.get(1, np.nan) / p.get(0, np.nan) - 1) * 100
    bulgu.append(("Promosyon", [
        f"Promosyonlu günlerde ortalama satış {sayi(p.get(1, np.nan))}, promosyonsuz günlerde {sayi(p.get(0, np.nan))} (fark %{sayi(fark, 1)}).",
        f"Mağaza-günlerinin %{sayi(acik['Promo'].mean() * 100, 1)}'i promosyonlu.",
    ]))

    # 4) Tatil etkisi ------------------------------------------------------------------------
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 4.2), gridspec_kw={"width_ratios": [1.5, 1]})
    sh_ad = {"0": "Yok", "a": "Resmî\ntatil (a)", "b": "Paskalya\n(b)", "c": "Noel\n(c)"}
    sh = acik.groupby("StateHoliday")["Sales"].agg(["mean", "size"]).reindex(["0", "a", "b", "c"]).dropna()
    cubuk(a1, [sh_ad[i] for i in sh.index], sh["mean"].values, ek_etiket=[f"n={sayi(n)}" for n in sh["size"]])
    a1.set_title("Resmî tatil")
    a1.set_ylabel("Ortalama satış (€)")
    sc = acik.groupby("SchoolHoliday")["Sales"].mean().reindex([0, 1])
    cubuk(a2, ["Yok", "Var"], sc.values)
    a2.set_title("Okul tatili")
    kaydet(fig, gdir, "04_tatil_etkisi.png")
    bulgu.append(("Tatiller", [
        "Resmî tatil günlerinde açık mağaza-günü sayısı çok azdır; ortalamalar şu sayılara dayanır: "
        + ", ".join(f"{sh_ad[i].replace(chr(10), ' ')} n={sayi(r['size'])} ort={sayi(r['mean'])}" for i, r in sh.iterrows()) + ".",
        f"Okul tatili olan günlerde ortalama {sayi(sc.get(1, np.nan))}, olmayan günlerde {sayi(sc.get(0, np.nan))} (fark %{sayi((sc.get(1, np.nan) / sc.get(0, np.nan) - 1) * 100, 1)}).",
    ]))

    # 5) Yıl içi mevsimsellik (yıllara göre) --------------------------------------------------
    ym = acik.groupby(["Year", "Month"])["Sales"].mean().unstack(0)
    fig, ax = plt.subplots(figsize=(9, 4.2))
    for renk, yil in zip((MAVI, TURUNCU, TURKUAZ), ym.columns):
        ax.plot(range(1, 13), ym[yil].reindex(range(1, 13)), color=renk, marker="o", markersize=5, linewidth=2, label=str(yil))
    ax.set_xticks(range(1, 13))
    ax.set_xticklabels(AYLAR)
    ax.set_ylim(0, np.nanmax(ym.values) * 1.15)
    ax.set_title("Aya göre ortalama satış (yıllara göre)")
    ax.set_ylabel("Ortalama satış (€)")
    ax.legend(loc="lower right", ncol=3)
    kaydet(fig, gdir, "05_yillik_mevsimsellik.png")
    ay_ort = acik.groupby("Month")["Sales"].mean()
    bulgu.append(("Yıllık mevsimsellik", [
        f"Tüm yılların ortalamasına göre en yüksek ay {AYLAR[int(ay_ort.idxmax()) - 1]} ({sayi(ay_ort.max())}), en düşük ay {AYLAR[int(ay_ort.idxmin()) - 1]} ({sayi(ay_ort.min())}).",
        "2015 yalnızca Temmuz sonuna kadar vardır; 2015 çizgisi eksik aylar içerir.",
    ]))

    # 6) Mağaza tipi x çeşitlilik ---------------------------------------------------------------
    tp = acik.pivot_table(index="StoreType", columns="Assortment", values="Sales", aggfunc="mean")
    fig, ax = plt.subplots(figsize=(9, 4.2))
    n_grup = len(tp.columns)
    genislik = 0.8 / max(n_grup, 1)
    renkler = {"a": MAVI, "b": TURUNCU, "c": TURKUAZ}
    for j, col in enumerate(tp.columns):
        x = np.arange(len(tp.index)) + j * genislik - 0.4 + genislik / 2
        ax.bar(x, tp[col].values, width=genislik * 0.9, color=renkler.get(col, MAVI),
               label={"a": "Temel (a)", "b": "Ekstra (b)", "c": "Geniş (c)"}.get(col, col))
    ax.set_xticks(range(len(tp.index)))
    ax.set_xticklabels([f"Tip {t}" for t in tp.index])
    ax.set_title("Mağaza tipi ve ürün çeşitliliğine göre ortalama satış")
    ax.set_ylabel("Ortalama satış (€)")
    ax.set_ylim(0, np.nanmax(tp.values) * 1.3)
    ax.legend(title="Çeşitlilik", loc="upper left", ncol=3)
    kaydet(fig, gdir, "06_magaza_tipi_cesitlilik.png")
    tip = acik.groupby("StoreType")["Sales"].mean()
    bulgu.append(("Mağaza tipi", [
        "Tipe göre ortalama satış: " + ", ".join(f"{t}={sayi(v)}" for t, v in tip.items()) + ".",
        "Bazı tip–çeşitlilik kombinasyonları çok az mağazadan oluşur; çubuğu olmayan kombinasyon veride yoktur.",
    ]))

    # 7) Rekabet mesafesi ---------------------------------------------------------------------
    mg = acik.groupby("Store").agg(ort=("Sales", "mean"), mesafe=("CompetitionDistance", "first"))
    mg = mg[mg["mesafe"] < 100_000]    # "rakip yok" yerine konan büyük değerleri hariç tut
    mg["dilim"] = pd.qcut(mg["mesafe"], 8, duplicates="drop")
    dil = mg.groupby("dilim", observed=True)["ort"].mean()
    etik = [f"{iv.left / 1000:.1f}-{iv.right / 1000:.1f}" for iv in dil.index]
    fig, ax = plt.subplots(figsize=(9, 4.2))
    cubuk(ax, etik, dil.values)
    ax.set_title("En yakın rakibe uzaklığa göre mağaza ortalama satışı")
    ax.set_xlabel("Rakip mesafesi (km, eşit mağaza sayılı 8 dilim)")
    ax.set_ylabel("Mağaza ort. satışı (€)")
    kaydet(fig, gdir, "07_rekabet_mesafesi.png")
    rho = mg["ort"].corr(mg["mesafe"], method="spearman")
    bulgu.append(("Rekabet", [
        f"Rakip mesafesi ile mağaza ortalama satışı arasındaki Spearman korelasyonu {sayi(rho, 2)} "
        f"({len(mg)} mağaza; 'rakip yok' için doldurulan büyük değerler hariç).",
    ]))

    # 8) Promo2 -------------------------------------------------------------------------------
    kat = np.where(acik["Promo2"] == 0, 0, np.where(acik["Promo2Active"] == 1, 2, 1))
    p2 = acik.groupby(kat)["Sales"].agg(["mean", "size"]).reindex([0, 1, 2]).dropna()
    ad2 = {0: "Promo2'ye\ngirmeyen", 1: "Promo2 mağazası,\nbu gün pasif", 2: "Promo2\netkin gün"}
    fig, ax = plt.subplots(figsize=(8, 4.2))
    cubuk(ax, [ad2[i] for i in p2.index], p2["mean"].values, ek_etiket=[f"n={sayi(n)}" for n in p2["size"]])
    ax.set_title("Sürekli promosyonun (Promo2) ortalama satışla ilişkisi")
    ax.set_ylabel("Ortalama satış (€)")
    kaydet(fig, gdir, "08_promo2.png")
    bulgu.append(("Promo2", [
        "Ortalama satış: " + ", ".join(f"{ad2[i].replace(chr(10), ' ')}={sayi(r['mean'])}" for i, r in p2.iterrows()) + ".",
    ]))

    # 9) Satış dağılımı: ham ve log ---------------------------------------------------------------
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 4))
    a1.hist(acik["Sales"], bins=60, color=MAVI)
    a1.set_title("Satış (ham)")
    a1.set_xlabel("Satış (€)")
    a2.hist(np.log1p(acik["Sales"]), bins=60, color=MAVI)
    a2.set_title("log(1 + satış)")
    a2.set_xlabel("log(1 + satış)")
    kaydet(fig, gdir, "09_satis_dagilimi.png")
    bulgu.append(("Hedef değişken dağılımı", [
        f"Çarpıklık: ham satış {sayi(acik['Sales'].skew(), 2)}, log(1+satış) {sayi(np.log1p(acik['Sales']).skew(), 2)}. "
        "Log dönüşümü dağılımı simetriğe yaklaştırır; modelde log hedefin kullanılma gerekçesi budur.",
    ]))

    # 10) Günlük raporlayan mağaza sayısı (veri boşluğu) -----------------------------------------
    gunluk = df.groupby("Date")["Store"].nunique()
    fig, ax = plt.subplots(figsize=(9, 4.2))
    ax.plot(gunluk.index, gunluk.values, color=MAVI, linewidth=2)
    ax.set_ylim(0, gunluk.max() * 1.15)
    ax.set_title("Günlük kaydı bulunan mağaza sayısı")
    ax.set_ylabel("Mağaza sayısı")
    kaydet(fig, gdir, "10_veri_boslugu.png")
    dusuk = gunluk[gunluk < gunluk.max()]
    bulgu.append(("Veri boşluğu", [
        f"Günlük kaydı olan mağaza sayısı en fazla {gunluk.max()}, en az {gunluk.min()}.",
        (f"Eksik mağaza olan gün aralığı {dusuk.index.min():%d.%m.%Y} - {dusuk.index.max():%d.%m.%Y} "
         f"({len(dusuk)} gün). Bu dönemde gecikmeli öznitelikler eksik kalır." if len(dusuk) else "Eksik mağaza günü yok."),
    ]))

    # Bulgu özeti ------------------------------------------------------------------------------
    satirlar = ["# EDA bulgu özeti", "",
                "Sayılar `02_eda.py` tarafından veriden hesaplanmıştır. Farklar betimleyicidir, nedensellik iddiası değildir.", ""]
    for baslik, cumleler in bulgu:
        satirlar.append(f"## {baslik}")
        satirlar += [f"- {c}" for c in cumleler]
        satirlar.append("")
    (args.cikti_dir / "bulgu_ozeti.md").write_text("\n".join(satirlar), encoding="utf-8")
    print("\n".join(satirlar))
    print(f"Grafikler: {gdir}  ({len(list(gdir.glob('*.png')))} adet)")


if __name__ == "__main__":
    main()
