# Promosyon döngüsü düzeltmesi (Hafta 5)

`05_promosyon_duzeltmesi.py` tarafından hesaplanmıştır.

## Yeni öznitelikler

- Toplam öznitelik: 28 (önceki 25 + yeni 3)

| Öznitelik | Tanım |
|---|---|
| PromoGecikme | Aynı mağazanın t−L gününün Promo değeri (L = 7·⌈H/7⌉) |
| PromoUyumsuz | 1 eğer Promo(t) ≠ PromoGecikme, 0 eğer eşit |
| SatisGecikmePromoDuz | SatisGecikme + (Promo(t) − PromoGecikme) × delta_store |

delta_store: aynı mağazanın t−H'ye kadar 91 günlük penceredeki açık hafta-içi (Pzt-Cum) satırlarından promo günlerin ortalama log satışı − non-promo günlerin ortalama log satışı. En az 4 promo ve 4 non-promo gün gerekir; yoksa tüm mağazaların havuzlanmış (pooled) deltası kullanılır.

## Eksik değer oranları (modele girecek satırlar)

| Öznitelik | H=7 | H=28 | H=42 | H=84 |
|---|---|---|---|---|
| PromoGecikme | %0,7 | %3,5 | %5,3 | %10,8 |
| PromoUyumsuz | %0,7 | %3,5 | %5,3 | %10,8 |
| SatisGecikmePromoDuz | %5,1 | %7,8 | %9,9 | %14,6 |

## Geri dönüş (pooled delta) oranı

| H=7 | H=28 | H=42 | H=84 |
|---|---|---|---|
| %0,8 | %0,8 | %0,8 | %0,8 |

## Naive ve Naive-P karşılaştırması

| Pencere | Naive RMSPE | Naive-P RMSPE | İyileşme |
|---|---|---|---|
| K1 | 0,244 | 0,192 | %21,1 |
| K2 | 0,294 | 0,180 | %38,8 |
| K3 | 0,303 | 0,215 | %29,0 |
| Test | 0,349 | 0,146 | %58,1 |
| H7 | 0,432 | 0,259 | %40,1 |
| H28 | 0,291 | 0,222 | %23,8 |
| H42 | 0,303 | 0,215 | %29,0 |
| H84 | 0,340 | 0,227 | %33,4 |

Naive-P'nin test döneminde standart naive yedeğine düşme oranı: %0,0

## Test dönemi promosyon uyumu kırılımı

| Yöntem | Durum | RMSPE | Oran |
|---|---|---|---|
| Standart naive | uyumlu | 0,148 | %58,5 |
| Standart naive | uyumsuz | 0,513 | %41,5 |
| Naive-P | uyumlu | 0,148 | %58,5 |
| Naive-P | uyumsuz | 0,145 | %41,5 |

## Test dönemi mağaza tipi kırılımı

| Tip | Naive RMSPE | Naive-P RMSPE |
|---|---|---|
| a | 0,340 | 0,154 |
| b | 0,178 | 0,099 |
| c | 0,350 | 0,142 |
| d | 0,372 | 0,138 |

## Korelasyon karşılaştırması (Pearson, log satışla)

| Öznitelik | H=7 | H=28 | H=42 | H=84 |
|---|---|---|---|---|
| SatisGecikme | 0,638 | 0,808 | 0,767 | 0,687 |
| SatisGecikmePromoDuz | 0,879 | 0,880 | 0,860 | 0,844 |

## Sızıntı testi (3 yeni öznitelik)

Kesim tarihinden sonraki tüm satışlar rastgele değerlerle değiştirilip yeni öznitelikler yeniden üretildi. Kesim + H gününe kadar hiçbir değer değişmemeli.

| H | Kesim | Korunan hücre | Korunan değişen | Sonrası değişen | Sonuç |
|---|---|---|---|---|---|
| 7 | 08.05.2015 | 2.893.425 | 0 | 71.044 | geçti |
| 7 | 19.06.2015 | 3.033.915 | 0 | 33.571 | geçti |
| 7 | 31.03.2014 | 1.545.390 | 0 | 414.117 | geçti |
| 28 | 08.05.2015 | 2.963.670 | 0 | 50.905 | geçti |
| 28 | 19.06.2015 | 3.104.160 | 0 | 13.437 | geçti |
| 28 | 31.03.2014 | 1.615.635 | 0 | 391.093 | geçti |
| 42 | 08.05.2015 | 3.010.500 | 0 | 37.472 | geçti |
| 42 | 19.06.2015 | 3.150.990 | 0 | — (kesim + H veri sonunda) | geçti |
| 42 | 31.03.2014 | 1.662.465 | 0 | 375.675 | geçti |
| 84 | 08.05.2015 | 3.150.990 | 0 | — (kesim + H veri sonunda) | geçti |
| 84 | 19.06.2015 | 3.150.990 | 0 | — (kesim + H veri sonunda) | geçti |
| 84 | 31.03.2014 | 1.802.955 | 0 | 332.258 | geçti |

Toplam 12 testin 12 tanesi geçti.
Negatif kontrol (kasıtlı 1 gün sızıntı): korunan bölgede 897.022 hücre değişti (testin duyarlılığını doğrular).

## Bağımsız kontrol

300 rastgele satır (sabit tohum) basit döngülerle yeniden hesaplandı.
Maksimum mutlak fark: 3.55e-15

