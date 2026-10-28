# Mevsimsel naive temel model özeti (İP 5, ilk adım)

`04_temel_model.py` tarafından hesaplanmıştır. Hatalar yalnızca açık ve satışı > 0 olan mağaza-günleri üzerinden, tüm mağaza-günleri birlikte alınarak hesaplanır.

## Pencere bazında sonuçlar

| Pencere | Dönem | Gözlem | RMSPE | MAPE | MAE (€) | RMSE (€) |
|---|---|---|---|---|---|---|
| K1 | 14.02.2015–27.03.2015 | 40.175 | 0,244 | %16,6 | 1.198 | 1.800 |
| K2 | 28.03.2015–08.05.2015 | 37.044 | 0,294 | %24,2 | 2.073 | 2.827 |
| K3 | 09.05.2015–19.06.2015 | 37.472 | 0,303 | %21,3 | 1.348 | 1.802 |
| Test | 20.06.2015–31.07.2015 | 40.282 | 0,349 | %24,2 | 1.485 | 2.084 |
| Ufuk H=7 | 09.05.2015–15.05.2015 | 5.634 | 0,432 | %29,8 | 1.763 | 2.359 |
| Ufuk H=28 | 09.05.2015–05.06.2015 | 24.034 | 0,291 | %19,3 | 1.250 | 1.710 |
| Ufuk H=42 | 09.05.2015–19.06.2015 | 37.472 | 0,303 | %21,3 | 1.348 | 1.802 |
| Ufuk H=84 | 09.05.2015–31.07.2015 | 77.754 | 0,340 | %25,1 | 1.542 | 2.016 |

## Hedeflerle ilişkisi

- Referans (H2): test dönemi RMSPE = 0,349; çapraz doğrulama ortalaması 0,280 (en düşük 0,244, en yüksek 0,303).
- H4 için ana modelin test RMSPE'si en fazla 0,245 olmalıdır (%30 iyileşme).
- H3 hedefi (≤ 0,15) bu eşikten daha sıkıdır.

## Test dönemi kırılımları

Ortalama yüzde hata = ortalama (tahmin − gerçek) / gerçek; pozitifse model fazla tahmin ediyor.

| Kırılım | Değer | Mağaza | Gözlem payı | RMSPE | Ortalama yüzde hata |
|---|---|---|---|---|---|
| StoreType | a | 602 | %53,9 | 0,340 | %17,1 |
| StoreType | b | 17 | %1,8 | 0,178 | %7,0 |
| StoreType | c | 148 | %13,2 | 0,350 | %17,5 |
| StoreType | d | 348 | %31,1 | 0,372 | %20,5 |
| PromoGecisi | var → var | 1115 | %41,4 | 0,144 | %-0,6 |
| PromoGecisi | var → yok | 1115 | %41,5 | 0,513 | %42,3 |
| PromoGecisi | yok → var | 1 | %0,0 | 0,488 | %12,4 |
| PromoGecisi | yok → yok | 1115 | %17,1 | 0,156 | %4,4 |
| PromoUyumu | uyumlu | 1115 | %58,5 | 0,148 | %0,8 |
| PromoUyumu | uyumsuz | 1115 | %41,5 | 0,513 | %42,3 |
| HaftaIleri | 1 | 1115 | %16,7 | 0,517 | %40,4 |
| HaftaIleri | 2 | 1115 | %16,7 | 0,145 | %-3,3 |
| HaftaIleri | 3 | 1114 | %16,7 | 0,429 | %31,3 |
| HaftaIleri | 4 | 1113 | %16,7 | 0,130 | %3,3 |
| HaftaIleri | 5 | 1113 | %16,7 | 0,465 | %35,7 |
| HaftaIleri | 6 | 1113 | %16,7 | 0,166 | %1,0 |

## Ufuk analizi: kökenden sonraki haftalara göre RMSPE (köken 08.05.2015)

| Hafta | RMSPE | Promosyon durumu son haftadan farklı gün payı |
|---|---|---|
| 1 | 0,432 | %80 |
| 2 | 0,189 | %0 |
| 3 | 0,315 | %80 |
| 4 | 0,176 | %0 |
| 5 | 0,412 | %83 |
| 6 | 0,200 | %0 |
| 7 | 0,507 | %83 |
| 8 | 0,230 | %0 |
| 9 | 0,429 | %83 |
| 10 | 0,222 | %0 |
| 11 | 0,459 | %83 |
| 12 | 0,273 | %0 |

## Tahmin kaynağı (test, modele giren satırlar)

- son hafta: 40.250 satır (%99,9)
- önceki haftalar: 32 satır (%0,1)

## Mağaza bazında test RMSPE dağılımı

- Medyan 0,324, %90'lık dilim 0,463, en yüksek 1,411 (mağaza 292).
- Şekildeki örnek mağaza: 90 (test RMSPE'si medyana en yakın mağaza).
