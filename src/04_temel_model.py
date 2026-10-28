"""
İP 5 (ilk adım) - Mevsimsel naive temel model

Girdi : 01_veri_hazirlama.py çıktısı (<girdi-dir>/islenmis_tam.*)
Çıktı : <cikti-dir>/naive_tahminler.*        tüm değerlendirme pencerelerinde satır bazlı tahminler
        <cikti-dir>/naive_sonuclari.csv       pencere bazında RMSPE / MAPE / MAE / RMSE
        <cikti-dir>/naive_kirilimlar.csv      test döneminde mağaza tipi, promosyon uyumu ve haftaya göre hata
        <cikti-dir>/temel_model_ozeti.md      sonuç özeti
        <cikti-dir>/grafikler/11–14_*.png

Yöntem (rapor 2.5): T kökenli tahminde her mağaza ve haftanın günü için T'ye kadarki son açık
(satışı > 0) aynı günün satışı, hedef penceredeki aynı günlere kopyalanır; yani son gözlenen hafta
ufuk boyunca tekrarlanır. Son haftadaki aynı gün kapalıysa (ör. tatil) bir önceki haftaya bakılır
(en fazla 8 hafta). Hiç açık aynı gün yoksa mağazanın son 28 gündeki, o da yoksa köken öncesindeki
tüm açık günlerinin ortalaması kullanılır. Kapalı günlerin tahmini 0'dır ve RMSPE'ye girmez
(açık/kapalı bilgisinin önceden bilindiği varsayılır, rapor 2.6).

Bu modelin parametresi yoktur; test dönemindeki hatası, sonraki modellerin karşılaştırılacağı
referanstır (H2). Ana modelden beklenen en az %30 iyileşme (H4) bu değere göre ölçülür.

Kullanım:
  python src/04_temel_model.py --girdi-dir data/islenmis --cikti-dir model_ciktilari
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from ortak import (MAVI, MUREKKEP2, TURKUAZ, TURUNCU, UFUK_ANALIZI_BASLANGIC, UFUK_DEGERLERI,
                   VARSAYILAN_UFUK, VERI_BITIS, gun, grafik_stili, kaydet_tablo, katmanlar, metrikler,
                   oku_islenmis, rmspe, sayi, tarih, test_penceresi, ufuk_penceresi)

GERI_BAKIS_HAFTA = 8
YEDEK_GUN = 28
H4_IYILESME = 0.30
H3_HEDEF = 0.15


def mevsimsel_naive(df: pd.DataFrame, baslangic: pd.Timestamp, bitis: pd.Timestamp) -> pd.DataFrame:
    """baslangic-1 kökenli mevsimsel naive tahmin; hedef penceredeki her satır için tahmin döndürür."""
    koken = baslangic - gun(1)
    acik = df[df["egitim_mi"] & (df["Date"] <= koken)]
    son = (acik[acik["Date"] > koken - gun(7 * GERI_BAKIS_HAFTA)]
           .sort_values("Date").groupby(["Store", "DayOfWeek"]).tail(1)
           [["Store", "DayOfWeek", "Date", "Sales", "Promo"]]
           .rename(columns={"Date": "KaynakTarih", "Sales": "KaynakSatis", "Promo": "KaynakPromo"}))
    yedek28 = acik[acik["Date"] > koken - gun(YEDEK_GUN)].groupby("Store")["Sales"].mean().rename("Yedek28")
    yedek_tum = acik.groupby("Store")["Sales"].mean().rename("YedekTum")

    hedef = df[df["Date"].between(baslangic, bitis)][
        ["Store", "Date", "DayOfWeek", "Sales", "Open", "Promo", "StoreType", "egitim_mi"]].copy()
    hedef = (hedef.merge(son, on=["Store", "DayOfWeek"], how="left")
             .merge(yedek28, on="Store", how="left").merge(yedek_tum, on="Store", how="left"))
    hedef["KaynakHafta"] = (koken - hedef["KaynakTarih"]).dt.days // 7 + 1          # 1 = son gözlenen hafta
    tahmin = hedef["KaynakSatis"].fillna(hedef["Yedek28"]).fillna(hedef["YedekTum"])
    hedef["Yontem"] = np.select(
        [hedef["KaynakHafta"] == 1, hedef["KaynakSatis"].notna(), hedef["Yedek28"].notna(), hedef["YedekTum"].notna()],
        ["son hafta", "önceki haftalar", "son 28 gün ortalaması", "tüm geçmiş ortalaması"], default="tahmin yok")
    hedef["Tahmin"] = np.where(hedef["Open"] == 1, tahmin, 0.0)
    hedef["Koken"] = koken
    hedef["GunIleri"] = (hedef["Date"] - koken).dt.days
    hedef["HaftaIleri"] = (hedef["GunIleri"] - 1) // 7 + 1
    hedef["PromoUyumu"] = np.where(hedef["KaynakPromo"].isna(), "kaynak yok",
                                   np.where(hedef["KaynakPromo"] == hedef["Promo"], "uyumlu", "uyumsuz"))
    hedef["PromoGecisi"] = (hedef["KaynakPromo"].map({0: "yok", 1: "var"}).fillna("?") + " → "
                            + hedef["Promo"].map({0: "yok", 1: "var"}))
    return hedef


def degerlendir(t: pd.DataFrame) -> dict:
    e = t[t["egitim_mi"]]
    return metrikler(e["Sales"], e["Tahmin"])


def kirilim(t: pd.DataFrame, sutun: str) -> pd.DataFrame:
    e = t[t["egitim_mi"]]
    satirlar = []
    for deger, g in e.groupby(sutun, sort=True):
        m = metrikler(g["Sales"], g["Tahmin"])
        yanlilik = float(((g["Tahmin"] - g["Sales"]) / g["Sales"]).mean())   # + ise fazla tahmin
        satirlar.append({"kirilim": sutun, "deger": deger, "magaza": int(g["Store"].nunique()),
                         "pay": len(g) / len(e), "yanlilik": yanlilik, **m})
    return pd.DataFrame(satirlar)


# --- grafikler ------------------------------------------------------------------------------------------
def grafik_bolme(plt, gdir: Path):
    test_bas, test_bit = test_penceresi()
    satirlar = [(k["ad"], k["egitim_bitis"], k["dogrulama_baslangic"], k["dogrulama_bitis"], TURUNCU)
                for k in katmanlar(VARSAYILAN_UFUK, test_bas)]
    satirlar.append(("Test", test_bas - gun(1), test_bas, test_bit, TURKUAZ))
    satirlar.append(("Ufuk\nanalizi", UFUK_ANALIZI_BASLANGIC - gun(1), UFUK_ANALIZI_BASLANGIC, VERI_BITIS, "#7fd3b5"))
    sol, sag = pd.Timestamp("2014-06-01"), pd.Timestamp("2015-11-10")
    fig, ax = plt.subplots(figsize=(10, 4.2))
    ax.axvspan(pd.Timestamp("2014-07-01"), pd.Timestamp("2014-12-31"), color="#f1f0ec", zorder=0)
    ax.text(pd.Timestamp("2014-10-01"), len(satirlar) - 0.42, "180 mağazada veri boşluğu",
            ha="center", va="center", fontsize=8.5, color=MUREKKEP2)
    for i, (ad, egt_bit, bas, bit, renk) in enumerate(satirlar):
        y = len(satirlar) - 1 - i
        ax.barh(y, (egt_bit - sol).days + 1, left=sol, height=0.5, color="#c9dcf3")
        ax.barh(y, (bit - bas).days + 1, left=bas, height=0.5, color=renk)
        ax.text(bas + (bit - bas) / 2, y, f"{(bit - bas).days + 1} gün", ha="center", va="center",
                fontsize=8, color="white", fontweight="bold")
        etiket = (f"{bas:%d.%m}–{bit:%d.%m.%Y}" if i < len(satirlar) - 1
                  else "H = " + ", ".join(str(h) for h in UFUK_DEGERLERI) + "\naynı başlangıçtan")
        ax.text(bit + gun(6), y, etiket, ha="left", va="center", fontsize=8, color=MUREKKEP2)
    ax.text(sol + gun(4), -0.62, "◀ eğitim 01.01.2013'ten başlar", fontsize=8.5, color=MUREKKEP2, va="center")
    ax.set_yticks(range(len(satirlar)))
    ax.set_yticklabels([s[0] for s in satirlar][::-1])
    ax.set_xlim(sol, sag)
    ax.set_ylim(-0.95, len(satirlar) - 0.1)
    aylar = ["Oca", "Şub", "Mar", "Nis", "May", "Haz", "Tem", "Ağu", "Eyl", "Eki", "Kas", "Ara"]
    isaretler = pd.date_range("2014-07-01", "2015-07-01", freq="2MS")
    ax.set_xticks(isaretler)
    ax.set_xticklabels([f"{aylar[d.month - 1]} {d:%Y}" for d in isaretler])
    ax.grid(axis="x", color="#e3e2dd")
    ax.grid(axis="y", visible=False)
    ax.set_title("Kronolojik bölme: genişleyen pencereli 3 katman, test ve ufuk analizi")
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color="#c9dcf3", label="Eğitim"), Patch(color=TURUNCU, label="Doğrulama"),
                       Patch(color=TURKUAZ, label="Test"), Patch(color="#7fd3b5", label="Ufuk analizi")],
              loc="upper left", ncol=4, fontsize=8.5, bbox_to_anchor=(0, -0.08))
    fig.tight_layout()
    fig.savefig(gdir / "11_bolme_semasi.png", dpi=150)
    plt.close(fig)


def grafik_ornek_magaza(plt, gdir: Path, test: pd.DataFrame, magaza: int):
    g = test[test["Store"] == magaza].sort_values("Date")
    fig, ax = plt.subplots(figsize=(10, 4.2))
    promo_gunleri = g.loc[g["Promo"] == 1, "Date"]
    yarim = pd.Timedelta(hours=12)
    for i, d in enumerate(promo_gunleri):
        ax.axvspan(d - yarim, d + yarim, color="#fbe3d6", linewidth=0, zorder=0,
                   label="Promosyon günü" if i == 0 else None)
    gercek = g["Sales"].where(g["Open"] == 1)
    tahmin = g["Tahmin"].where(g["Open"] == 1)
    ax.plot(g["Date"], gercek, color=MAVI, marker="o", markersize=4, linewidth=2, label="Gerçek satış")
    ax.plot(g["Date"], tahmin, color=TURUNCU, marker="o", markersize=3, linewidth=1.6, linestyle="--",
            label="Mevsimsel naive tahmin")
    ax.set_ylim(0, np.nanmax([gercek.max(), tahmin.max()]) * 1.18)
    ax.set_ylabel("Günlük satış (€)")
    tip = g["StoreType"].iloc[0]
    ax.set_title(f"Örnek mağaza (No. {magaza}, tip {tip}): test döneminde gerçek satış ve naive tahmin")
    ax.legend(loc="upper left", ncol=3, fontsize=9)
    import matplotlib.dates as mdates
    ax.xaxis.set_major_locator(mdates.WeekdayLocator(byweekday=mdates.MO))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%d.%m"))
    fig.tight_layout()
    fig.savefig(gdir / "12_naive_ornek_magaza.png", dpi=150)
    plt.close(fig)


def grafik_ufuk(plt, gdir: Path, ufuk_sonuc: pd.DataFrame, haftalik: pd.DataFrame):
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 4.2), gridspec_kw={"width_ratios": [1, 1.7]})
    x = np.arange(len(ufuk_sonuc))
    bars = a1.bar(x, ufuk_sonuc["RMSPE"], color=MAVI, width=0.6)
    a1.bar_label(bars, labels=[sayi(v, 3) for v in ufuk_sonuc["RMSPE"]], padding=3, color=MUREKKEP2, fontsize=9)
    a1.set_xticks(x)
    a1.set_xticklabels([f"H={h}" for h in ufuk_sonuc["H"]])
    a1.set_ylim(0, ufuk_sonuc["RMSPE"].max() * 1.25)
    a1.set_ylabel("RMSPE")
    a1.set_title("Ufka göre RMSPE (köken 08.05.2015)")
    renk = [TURUNCU if v > 0.5 else MAVI for v in haftalik["uyumsuz_pay"]]
    bars = a2.bar(haftalik["HaftaIleri"], haftalik["RMSPE"], color=renk, width=0.65)
    a2.bar_label(bars, labels=[sayi(v, 2) for v in haftalik["RMSPE"]], padding=3, color=MUREKKEP2, fontsize=8)
    a2.set_xticks(haftalik["HaftaIleri"])
    a2.set_xlabel("Kökenden sonraki hafta")
    a2.set_ylim(0, haftalik["RMSPE"].max() * 1.25)
    a2.set_title("Haftaya göre RMSPE")
    from matplotlib.patches import Patch
    a2.legend(handles=[Patch(color=MAVI, label="Promosyon durumu son haftayla aynı"),
                       Patch(color=TURUNCU, label="Promosyon durumu son haftadan farklı")],
              loc="upper left", fontsize=8.5)
    fig.tight_layout()
    fig.savefig(gdir / "13_naive_ufuk_hata.png", dpi=150)
    plt.close(fig)


def grafik_kirilim(plt, gdir: Path, tip: pd.DataFrame, uyum: pd.DataFrame):
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 4.2), gridspec_kw={"width_ratios": [1.6, 1]})
    x = np.arange(len(tip))
    bars = a1.bar(x, tip["RMSPE"], color=MAVI, width=0.6)
    a1.bar_label(bars, labels=[f"{sayi(v, 3)}\n{m} mağaza" for v, m in zip(tip["RMSPE"], tip["magaza"])],
                 padding=3, color=MUREKKEP2, fontsize=8.5)
    a1.set_xticks(x)
    a1.set_xticklabels([f"Tip {t}" for t in tip["deger"]])
    a1.set_ylim(0, tip["RMSPE"].max() * 1.3)
    a1.set_ylabel("RMSPE")
    a1.set_title("Mağaza tipine göre (test)")
    uyum = uyum[uyum["deger"].isin(["uyumlu", "uyumsuz"])].set_index("deger").loc[["uyumlu", "uyumsuz"]]
    x = np.arange(len(uyum))
    bars = a2.bar(x, uyum["RMSPE"], color=[MAVI, TURUNCU], width=0.6)
    a2.bar_label(bars, labels=[f"{sayi(v, 3)}\ngünlerin %{sayi(p * 100, 1)}'i" for v, p in zip(uyum["RMSPE"], uyum["pay"])],
                 padding=3, color=MUREKKEP2, fontsize=8.5)
    a2.set_xticks(x)
    a2.set_xticklabels(["Son haftayla\naynı", "Son haftadan\nfarklı"])
    a2.set_ylim(0, uyum["RMSPE"].max() * 1.3)
    a2.set_title("Promosyon durumuna göre (test)")
    fig.tight_layout()
    fig.savefig(gdir / "14_naive_kirilim.png", dpi=150)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--girdi-dir", type=Path, default=Path("data/islenmis"))
    ap.add_argument("--cikti-dir", type=Path, default=Path("model_ciktilari"))
    args = ap.parse_args()
    gdir = args.cikti_dir / "grafikler"
    gdir.mkdir(parents=True, exist_ok=True)

    df = oku_islenmis(args.girdi_dir)
    test_bas, test_bit = test_penceresi()

    # 1) Çapraz doğrulama katmanları ve test (H = 42) -------------------------------------------------------
    pencereler = [(k["ad"], k["dogrulama_baslangic"], k["dogrulama_bitis"]) for k in katmanlar(VARSAYILAN_UFUK, test_bas)]
    pencereler.append(("Test", test_bas, test_bit))
    tahminler, sonuc = [], []
    for ad, bas, bit in pencereler:
        t = mevsimsel_naive(df, bas, bit)
        t["Pencere"] = ad
        tahminler.append(t)
        sonuc.append({"Pencere": ad, "Baslangic": tarih(bas), "Bitis": tarih(bit), "H": VARSAYILAN_UFUK, **degerlendir(t)})
    test = tahminler[-1]

    # 2) Ufuk analizi: 08.05.2015 kökenli tek tahmin, iç içe pencereler ------------------------------------------
    ufuk_t = mevsimsel_naive(df, UFUK_ANALIZI_BASLANGIC, VERI_BITIS)
    ufuk_t["Pencere"] = "Ufuk analizi"
    tahminler.append(ufuk_t)
    ufuk_sonuc = []
    for h in UFUK_DEGERLERI:
        bas, bit = ufuk_penceresi(h)
        m = degerlendir(ufuk_t[ufuk_t["Date"].between(bas, bit)])
        ufuk_sonuc.append({"Pencere": f"Ufuk H={h}", "Baslangic": tarih(bas), "Bitis": tarih(bit), "H": h, **m})
    sonuc += ufuk_sonuc
    ufuk_sonuc = pd.DataFrame(ufuk_sonuc)

    e = ufuk_t[ufuk_t["egitim_mi"]]
    haftalik = pd.DataFrame([{"HaftaIleri": w, "RMSPE": rmspe(g["Sales"], g["Tahmin"]),
                              "uyumsuz_pay": float((g["PromoUyumu"] == "uyumsuz").mean()), "n": len(g)}
                             for w, g in e.groupby("HaftaIleri")])

    # 3) Test kırılımları -------------------------------------------------------------------------------------
    tip = kirilim(test, "StoreType")
    gecis = kirilim(test, "PromoGecisi")
    uyum = kirilim(test, "PromoUyumu")
    hafta_test = kirilim(test, "HaftaIleri")
    yontem = test[test["egitim_mi"]]["Yontem"].value_counts()
    te = test[test["egitim_mi"]]
    mg = (((te["Sales"] - te["Tahmin"]) / te["Sales"]) ** 2).groupby(te["Store"]).mean().pow(0.5)
    mg = mg.rename("RMSPE").reset_index().merge(df[["Store", "StoreType"]].drop_duplicates(), on="Store")
    medyan = mg["RMSPE"].median()
    ornek = int(mg.iloc[(mg["RMSPE"] - medyan).abs().argsort().iloc[0]]["Store"])

    # 4) Kaydet ------------------------------------------------------------------------------------------------
    sonuc = pd.DataFrame(sonuc)
    sonuc.to_csv(args.cikti_dir / "naive_sonuclari.csv", index=False)
    pd.concat([tip, gecis, uyum, hafta_test]).to_csv(args.cikti_dir / "naive_kirilimlar.csv", index=False)
    haftalik.to_csv(args.cikti_dir / "naive_haftalik_ufuk.csv", index=False)
    mg.to_csv(args.cikti_dir / "naive_magaza_rmspe_test.csv", index=False)
    kaydet_tablo(pd.concat(tahminler, ignore_index=True), args.cikti_dir / "naive_tahminler")

    plt = grafik_stili()
    grafik_bolme(plt, gdir)
    grafik_ornek_magaza(plt, gdir, test, ornek)
    grafik_ufuk(plt, gdir, ufuk_sonuc, haftalik)
    grafik_kirilim(plt, gdir, tip, uyum)

    # 5) Özet ---------------------------------------------------------------------------------------------------
    t_rmspe = float(sonuc.loc[sonuc["Pencere"] == "Test", "RMSPE"].iloc[0])
    esik = t_rmspe * (1 - H4_IYILESME)
    cv = sonuc[sonuc["Pencere"].str.startswith("K")]
    r = ["# Mevsimsel naive temel model özeti (İP 5, ilk adım)", "",
         "`04_temel_model.py` tarafından hesaplanmıştır. Hatalar yalnızca açık ve satışı > 0 olan mağaza-günleri "
         "üzerinden, tüm mağaza-günleri birlikte alınarak hesaplanır.", "",
         "## Pencere bazında sonuçlar", "",
         "| Pencere | Dönem | Gözlem | RMSPE | MAPE | MAE (€) | RMSE (€) |", "|---|---|---|---|---|---|---|"]
    for _, s in sonuc.iterrows():
        r.append(f"| {s['Pencere']} | {s['Baslangic']}–{s['Bitis']} | {sayi(s['n'])} | {sayi(s['RMSPE'], 3)} | "
                 f"%{sayi(s['MAPE'] * 100, 1)} | {sayi(s['MAE'])} | {sayi(s['RMSE'])} |")
    r += ["", "## Hedeflerle ilişkisi", "",
          f"- Referans (H2): test dönemi RMSPE = {sayi(t_rmspe, 3)}; çapraz doğrulama ortalaması "
          f"{sayi(cv['RMSPE'].mean(), 3)} (en düşük {sayi(cv['RMSPE'].min(), 3)}, en yüksek {sayi(cv['RMSPE'].max(), 3)}).",
          f"- H4 için ana modelin test RMSPE'si en fazla {sayi(esik, 3)} olmalıdır (%30 iyileşme).",
          f"- H3 hedefi (≤ {sayi(H3_HEDEF, 2)}) bu eşikten {'daha sıkıdır' if H3_HEDEF < esik else 'daha gevşektir'}.",
          "", "## Test dönemi kırılımları", "",
          "Ortalama yüzde hata = ortalama (tahmin − gerçek) / gerçek; pozitifse model fazla tahmin ediyor.", "",
          "| Kırılım | Değer | Mağaza | Gözlem payı | RMSPE | Ortalama yüzde hata |", "|---|---|---|---|---|---|"]
    for k in (tip, gecis, uyum, hafta_test):
        for _, s in k.iterrows():
            r.append(f"| {s['kirilim']} | {s['deger']} | {s['magaza']} | %{sayi(s['pay'] * 100, 1)} | "
                     f"{sayi(s['RMSPE'], 3)} | %{sayi(s['yanlilik'] * 100, 1)} |")
    r += ["", "## Ufuk analizi: kökenden sonraki haftalara göre RMSPE (köken 08.05.2015)", "",
          "| Hafta | RMSPE | Promosyon durumu son haftadan farklı gün payı |", "|---|---|---|"]
    for _, s in haftalik.iterrows():
        r.append(f"| {int(s['HaftaIleri'])} | {sayi(s['RMSPE'], 3)} | %{sayi(s['uyumsuz_pay'] * 100, 0)} |")
    r += ["", "## Tahmin kaynağı (test, modele giren satırlar)", ""]
    r += [f"- {k}: {sayi(v)} satır (%{sayi(v / yontem.sum() * 100, 1)})" for k, v in yontem.items()]
    r += ["", "## Mağaza bazında test RMSPE dağılımı", "",
          f"- Medyan {sayi(medyan, 3)}, %90'lık dilim {sayi(mg['RMSPE'].quantile(0.9), 3)}, en yüksek "
          f"{sayi(mg['RMSPE'].max(), 3)} (mağaza {int(mg.loc[mg['RMSPE'].idxmax(), 'Store'])}).",
          f"- Şekildeki örnek mağaza: {ornek} (test RMSPE'si medyana en yakın mağaza)."]
    metin = "\n".join(r) + "\n"
    (args.cikti_dir / "temel_model_ozeti.md").write_text(metin, encoding="utf-8")
    print(metin)
    print(f"Grafikler: {gdir}  ({len(list(gdir.glob('*.png')))} adet)")


if __name__ == "__main__":
    main()
