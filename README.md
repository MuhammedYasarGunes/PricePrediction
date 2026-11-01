# PricePrediction · Mağaza Satış Tahmini

Fırat Üniversitesi Bilgisayar Mühendisliği Tasarım dersi kapsamında yürütülen takım projesi. Amaç, [Rossmann Store Sales](https://www.kaggle.com/c/rossmann-store-sales) verisiyle 1.115 mağazanın günlük satışlarını tahmin etmek.

Bu repo şu an beş aşamayı içeriyor: **veri hazırlama**, **keşifsel veri analizi (EDA)**, **ufka duyarlı öznitelik mühendisliği**, **mevsimsel naive temel model** ve **promosyon döngüsü düzeltmesi**. Sırada LightGBM ve Prophet var.

> **English:** Team project on the Rossmann Store Sales dataset (1,115 stores, ~1M daily records). This repo covers data preparation (leak-free features, log-transformed target, MAD-based outlier flags), exploratory analysis with 10 charts, a horizon-aware feature pipeline (forecast horizon H as a single parameter, verified by an automatic leakage test), a seasonal naive baseline (test RMSPE 0.349), and a promo-cycle correction (Naive-P test RMSPE 0.146, 3 new features). LightGBM and Prophet come next.

<p align="center">
  <img src="gorseller/eda/03_promosyon_etkisi.png" width="720" alt="Promosyonun haftanın gününe göre satışa etkisi">
</p>

## Aşamalar

**1. Veri hazırlama** · [`01_veri_hazirlama.py`](src/01_veri_hazirlama.py)

- Kapalı (`Open=0`) ve satışı sıfır olan günler eğitimden çıkarılır; bu günler için tahmin 0 atanır.
- `CompetitionDistance` eksikleri "yakında rakip yok" anlamında büyük bir değerle doldurulur; eksik rakip açılış tarihleri ayrı bir kategori olur.
- Kategorik değişkenler sayısal kodlara çevrilir.
- Hedef değişkene `log(1 + Sales)` uygulanır.
- Aykırı değerler mağaza bazında MAD ile işaretlenir ama silinmez.
- `Customers` kullanılmaz: tahmin anında bilinmez, kullanılırsa veri sızıntısı olur.
- Çıktı olarak işlenmiş veri ve bir veri kalite raporu üretilir.

**2. Keşifsel analiz** · [`02_eda.py`](src/02_eda.py)

10 grafik üretir ve [`bulgu_ozeti.md`](raporlar/bulgu_ozeti.md) dosyasını verideki gerçek sayılardan otomatik yazar.

**3. Öznitelik mühendisliği** · [`03_oznitelik_muhendisligi.py`](src/03_oznitelik_muhendisligi.py)

- Tahmin ufku `H` tek bir parametredir (`--ufuk`, varsayılan 42 gün); dört ufuk (7, 28, 42, 84) tek komutla üretilir.
- 2. haftadaki 17 özniteliğe 8 yeni öznitelik eklenir (toplam 25): promosyon bloğunun kaçıncı günü, en yakın resmî tatile kalan / tatilden geçen gün ve geçmiş satıştan türetilen 5 öznitelik.
- Geçmiş satış öznitelikleri en az `H` gün gecikmelidir; `t` günü için yalnızca `t−H` ve öncesi kullanılır.
- **Sızıntı testi:** kesim tarihinden sonraki satışlar rastgele değiştirilir; kesim + `H` gününe kadar hiçbir öznitelik değişmemelidir. 4 ufuk × 3 kesim = 12 testin hepsi geçer.
- Sonuçlar: [`oznitelik_raporu.md`](raporlar/oznitelik_raporu.md). Bağımsız kontrol: [`kontrol_oznitelikler.py`](src/kontrol_oznitelikler.py).

**4. Mevsimsel naive temel model** · [`04_temel_model.py`](src/04_temel_model.py)

- Son gözlenen hafta, ufuk boyunca tekrarlanır; kapalı günlerin tahmini 0'dır.
- Üç çapraz doğrulama katmanı, test dönemi (20.06–31.07.2015) ve 08.05.2015 kökenli ufuk analizi için hata hesaplar.
- Sonuçlar: [`temel_model_ozeti.md`](raporlar/temel_model_ozeti.md).

**5. Promosyon döngüsü düzeltmesi** · [`05_promosyon_duzeltmesi.py`](src/05_promosyon_duzeltmesi.py)

- Promosyon döngüsü uyumsuzluğunu gidermek için Naive-P yöntemi ve 3 yeni öznitelik (`PromoGecikme`, `PromoUyumsuz`, `SatisGecikmePromoDuz`) geliştirildi (toplam 28 öznitelik).
- Naive-P, tahmin kökeni öncesindeki en yakın aynı hafta-günü ve aynı promo durumundaki açık günü kaynak alır; test RMSPE'si 0,349'dan 0,146'ya (%58,1 iyileşme) düşer.
- `SatisGecikmePromoDuz` özniteliği, mağaza bazında 91 günlük pencereden hesaplanan log promosyon etkisi (`delta_store`) ile düzeltilir; log satışla Pearson korelasyonu H=7 için 0,638'den 0,879'a, H=42 için 0,767'den 0,860'a yükselir.
- Sonuçlar: [`hafta5_ozet.md`](raporlar/hafta5_ozet.md) ve [`hafta5_sonuclar.json`](raporlar/hafta5_sonuclar.json).

Ortak tarih bölmeleri ve hata ölçütleri [`ortak.py`](src/ortak.py) dosyasındadır.

## Öne çıkan bulgular

Aşağıdaki karşılaştırmalar betimleyicidir, nedensellik iddiası taşımaz.

- **Promosyon:** promosyonlu günlerde ortalama satış %38,8 daha yüksek (8.229'a karşı 5.930).
- **Mevsimsellik:** en güçlü ay Aralık, en zayıf ay Eylül. Pazar günü mağazaların yalnızca %2,5'i açık.
- **Rekabet:** rakip mesafesi ile mağaza satışı arasında neredeyse ilişki yok (Spearman −0,03).
- **Hedef dağılımı:** log dönüşümü çarpıklığı 1,59'dan −0,11'e indiriyor; modelde log hedef kullanılmasının gerekçesi bu.
- **Veri boşluğu:** 2014'ün ikinci yarısında yaklaşık altı ay boyunca 180 mağazanın kaydı yok. Bu dönemde gecikmeli öznitelikler eksik kalıyor.
- **Temel model:** mevsimsel naive yöntemin test RMSPE'si 0,349. Ana modelin %30 iyileşme hedefi için test RMSPE'si 0,244 (0,349 × 0,70 = 0,2443).
- **Promosyon döngüsü:** promosyonlar zincir genelinde ve iki haftalık döngüyle uygulanıyor. Naive yöntem, promosyon durumu son haftayla aynı olan günlerde 0,148, farklı olan günlerde 0,513 RMSPE veriyor. Naive-P bu uyumsuzluğu tamamen gidererek test RMSPE'sini 0,146'ya düşürüyor.

| | |
|---|---|
| <img src="gorseller/eda/01_aylik_ortalama_satis.png" alt="Aylık ortalama satış"> | <img src="gorseller/eda/05_yillik_mevsimsellik.png" alt="Yıllık mevsimsellik"> |
| <img src="gorseller/eda/02_haftanin_gunu.png" alt="Haftanın gününe göre satış"> | <img src="gorseller/eda/09_satis_dagilimi.png" alt="Satış dağılımı"> |
| <img src="gorseller/model/11_bolme_semasi.png" alt="Kronolojik bölme şeması"> | <img src="gorseller/model/15_naive_p_karsilastirma.png" alt="Standart naive ve Naive-P karşılaştırması"> |

## Proje yapısı

```
PricePrediction/
├── src/                          # kodlar (sırayla çalıştırılır)
│   ├── 01_veri_hazirlama.py
│   ├── 02_eda.py
│   ├── 03_oznitelik_muhendisligi.py
│   ├── 04_temel_model.py
│   ├── 05_promosyon_duzeltmesi.py
│   ├── ortak.py                  # tarih bölmeleri, hata ölçütleri, grafik stili
│   └── kontrol_oznitelikler.py   # bağımsız kontrol
├── gorseller/
│   ├── eda/                      # 01–10: keşifsel analiz grafikleri
│   └── model/                    # 11–16: bölme şeması, temel model ve promosyon düzeltmesi grafikleri
├── raporlar/                     # kodların ürettiği bulgu ve sonuç özetleri
├── data/                         # Kaggle verisi (repoya girmez)
├── requirements.txt
└── README.md
```

## Çalıştırma

[Kaggle](https://www.kaggle.com/c/rossmann-store-sales/data) sayfasından `train.csv` ve `store.csv` dosyalarını indirip `data/` klasörüne koy:

```bash
pip install -r requirements.txt
python src/01_veri_hazirlama.py --veri-dir data --cikti-dir data/islenmis
python src/02_eda.py --girdi-dir data/islenmis --cikti-dir eda_ciktilari
python src/03_oznitelik_muhendisligi.py --girdi-dir data/islenmis --cikti-dir data/oznitelikler
python src/04_temel_model.py --girdi-dir data/islenmis --cikti-dir model_ciktilari
python src/05_promosyon_duzeltmesi.py --girdi-dir data/islenmis --cikti-dir data/oznitelikler_v2
python src/kontrol_oznitelikler.py                     # 03 bağımsız kontrolü
python src/kontrol_oznitelikler.py --promo-kontrol     # 05 promo öznitelikleri kontrolü
```

`pyarrow` kuruluysa işlenmiş veri Parquet, değilse sıkıştırılmış CSV olarak kaydedilir.
