# Değerlendirme altyapısı (Hafta 7)

`06_degerlendirme_altyapisi.py` tarafından hesaplanmıştır.

## Kayan kökenli doğrulama pencereleri (Rolling-Origin Evaluation)

Toplam pencere sayısı: 21 (20 doğrulama + 1 test penceresi, her biri 42 gün).

| Pencere | Başlangıç | Bitiş | Gözlem | Mağaza | Naive RMSPE | Naive-P RMSPE | İyileşme |
|---|---|---|---|---|---|---|---|
| RO01 | 02.03.2013 | 12.04.2013 | 37.928 | 1.113 | 0,307 | 0,217 | %29,2 |
| RO02 | 13.04.2013 | 24.05.2013 | 36.908 | 1.113 | 0,315 | 0,188 | %40,3 |
| RO03 | 25.05.2013 | 05.07.2013 | 39.509 | 1.115 | 0,436 | 0,393 | %9,9 |
| RO04 | 06.07.2013 | 16.08.2013 | 40.109 | 1.115 | 0,386 | 0,217 | %43,8 |
| RO05 | 17.08.2013 | 27.09.2013 | 40.246 | 1.115 | 0,431 | 0,234 | %45,7 |
| RO06 | 28.09.2013 | 08.11.2013 | 38.503 | 1.115 | 0,271 | 0,180 | %33,7 |
| RO07 | 09.11.2013 | 20.12.2013 | 40.168 | 1.115 | 0,325 | 0,229 | %29,6 |
| RO08 | 21.12.2013 | 31.01.2014 | 36.601 | 1.115 | 0,964 | 0,428 | %55,6 |
| RO09 | 01.02.2014 | 14.03.2014 | 40.159 | 1.115 | 0,259 | 0,202 | %21,8 |
| RO10 | 15.03.2014 | 25.04.2014 | 38.066 | 1.115 | 0,305 | 0,238 | %22,0 |
| RO11 | 26.04.2014 | 06.06.2014 | 38.103 | 1.115 | 0,373 | 0,271 | %27,4 |
| RO12 | 07.06.2014 | 18.07.2014 | 35.558 | 1.114 | 0,369 | 0,233 | %36,8 |
| RO13 | 19.07.2014 | 29.08.2014 | 33.710 | 934 | 0,331 | 0,154 | %53,6 |
| RO14 | 30.08.2014 | 10.10.2014 | 32.845 | 935 | 0,285 | 0,194 | %32,1 |
| RO15 | 11.10.2014 | 21.11.2014 | 33.167 | 935 | 0,338 | 0,192 | %43,3 |
| RO16 | 22.11.2014 | 02.01.2015 | 31.203 | 1.115 | 0,685 | 0,648 | %5,3 |
| RO17 | 03.01.2015 | 13.02.2015 | 39.923 | 1.115 | 0,438 | 0,425 | %2,9 |
| K1 | 14.02.2015 | 27.03.2015 | 40.175 | 1.115 | 0,244 | 0,192 | %21,1 |
| K2 | 28.03.2015 | 08.05.2015 | 37.044 | 1.115 | 0,294 | 0,180 | %38,8 |
| K3 | 09.05.2015 | 19.06.2015 | 37.472 | 1.115 | 0,303 | 0,215 | %29,0 |
| Test | 20.06.2015 | 31.07.2015 | 40.282 | 1.115 | 0,349 | 0,146 | %58,1 |

## Özet istatistikler (RMSPE)

| Ölçüt | Standart Naive | Naive-P |
|---|---|---|
| Ortalama | 0,381 | 0,256 |
| Medyan | 0,331 | 0,217 |
| En düşük (Min) | 0,244 | 0,146 |
| En yüksek (Maks) | 0,964 | 0,648 |
| RMSPE ≤ 0,15 olan pencere sayısı | 0 | 1 |
| Naive-P'nin Naive'den iyi olduğu pencere sayısı | — | 21 / 21 |

## Diebold-Mariano testleri

Kayıp fonksiyonu: L = ((y - ŷ) / y)² (kare yüzde hatası, RMSPE ile tutarlı). Fark d = L(Naive) - L(Naive-P); pozitif d değeri Naive-P lehinedir.

### (a) Pencere düzeyi DM testi (K = 21 pencere)

- Ortalama kayıp farkı (d̄): 0,091412
- Standart hata (SE): 0,033377
- t istatistiği: 2,739
- Serbestlik derecesi: 20
- İki taraflı p değeri: 0,0127
- Newey-West lag-1 duyarlılığı p değeri: 0,0078

### (b) Günlük DM testi (test penceresi, T = 42 gün)

- Ortalama kayıp farkı (d̄): 0,086672
- Bant genişliği (m = floor(4·(T/100)^(2/9))): 3
- Newey-West standart hata (SE): 0,032420
- t istatistiği: 2,673
- Serbestlik derecesi: 41
- İki taraflı p değeri: 0,0107

Bant genişliği duyarlılık analizi:

| Gecikme (lag) | p değeri |
|---|---|
| lag0 | 0,0013 |
| lag3 | 0,0107 |
| lag6 | 0,0074 |
| lag13 | 0,0000 |

### (c) DM testi doğrulama ve kendi kendini denetleme

- Özdeş tahminler testi: geçti (test uygulanamaz bildirildi)
- Belirgin hata farkı olan sentetik veri (p < 0,05 ret): geçti
- Aynı dağılımdan sentetik veri (200 tekrar ret oranı): %2,0 (beklenen: <%15)

## Alt grup kırılımları (Test penceresi)

### Mağaza tipine göre

| Mağaza Tipi | Mağaza Sayısı | Satır Sayısı | Standart Naive RMSPE | Naive-P RMSPE |
|---|---|---|---|---|
| Tip a | 602 | 21.694 | 0,340 | 0,154 |
| Tip b | 17 | 714 | 0,178 | 0,099 |
| Tip c | 148 | 5.328 | 0,350 | 0,142 |
| Tip d | 348 | 12.546 | 0,372 | 0,138 |

### Promosyon durumuna göre

| Promosyon | Standart Naive RMSPE | Naive-P RMSPE |
|---|---|---|
| Promo = 0 (Yok) | 0,440 | 0,146 |
| Promo = 1 (Var) | 0,145 | 0,147 |

### Devlet tatiline göre

| Devlet Tatili | Standart Naive RMSPE | Naive-P RMSPE |
|---|---|---|
| Tatil Yok (0) | 0,349 | 0,146 |
| Resmî Tatil (any) | nan | nan |

### Okul tatiline göre

| Okul Tatili | Standart Naive RMSPE | Naive-P RMSPE |
|---|---|---|
| Tatil Yok (0) | 0,355 | 0,148 |
| Okul Tatili (1) | 0,338 | 0,142 |

### Haftanın gününe göre

| Gün | Standart Naive RMSPE | Naive-P RMSPE |
|---|---|---|
| 1 (Pazartesi) | 0,543 | 0,149 |
| 2 (Salı) | 0,438 | 0,156 |
| 3 (Çarşamba) | 0,313 | 0,138 |
| 4 (Perşembe) | 0,244 | 0,126 |
| 5 (Cuma) | 0,260 | 0,153 |
| 6 (Cumartesi) | 0,153 | 0,153 |
| 7 (Pazar) | 0,181 | 0,181 |

### Test penceresi haftalık hata (6 hafta)

| Hafta | Dönem | Standart Naive RMSPE | Naive-P RMSPE |
|---|---|---|---|
| 1. Hafta | 20.06–26.06.2015 | 0,517 | 0,140 |
| 2. Hafta | 27.06–03.07.2015 | 0,145 | 0,145 |
| 3. Hafta | 04.07–10.07.2015 | 0,429 | 0,152 |
| 4. Hafta | 11.07–17.07.2015 | 0,130 | 0,133 |
| 5. Hafta | 18.07–24.07.2015 | 0,465 | 0,136 |
| 6. Hafta | 25.07–31.07.2015 | 0,166 | 0,169 |

## Alt grup kırılımları (Tüm 21 pencere havuzlanmış)

### Mağaza tipine göre (tüm pencereler)

| Mağaza Tipi | Mağaza Sayısı | Satır Sayısı | Standart Naive RMSPE | Naive-P RMSPE |
|---|---|---|---|---|
| Tip a | 602 | 426.433 | 0,413 | 0,291 |
| Tip b | 17 | 14.634 | 0,272 | 0,217 |
| Tip c | 148 | 105.531 | 0,463 | 0,338 |
| Tip d | 348 | 241.081 | 0,385 | 0,223 |

### Promosyon durumuna göre (tüm pencereler)

| Promosyon | Standart Naive RMSPE | Naive-P RMSPE |
|---|---|---|
| Promo = 0 (Yok) | 0,478 | 0,294 |
| Promo = 1 (Var) | 0,305 | 0,257 |

## En kötü 10 mağaza (Naive-P test RMSPE)

| Sıra | Mağaza | Naive-P RMSPE | Tip | Çeşit | Rakip Mesafesi (m) | Promo2 |
|---|---|---|---|---|---|---|
| 1 | 292 | 1,261 | a | a | 1.100 | 0 |
| 2 | 183 | 0,855 | a | a | 9.670 | 0 |
| 3 | 909 | 0,752 | a | c | 1.680 | 1 |
| 4 | 782 | 0,507 | c | c | 5.390 | 1 |
| 5 | 876 | 0,479 | a | a | 21.790 | 1 |
| 6 | 268 | 0,432 | a | a | 4.520 | 0 |
| 7 | 550 | 0,418 | d | c | 50 | 0 |
| 8 | 269 | 0,409 | a | c | 60 | 0 |
| 9 | 534 | 0,338 | d | a | 1.200 | 1 |
| 10 | 530 | 0,307 | a | c | 18.160 | 0 |

## Tahmin sözleşmesi doğrulaması

- Toplam kontrol sayısı: 7
- Başarıyla geçen kontrol sayısı: 7
- Bilerek bozulan tahminlerin yakalanması: Başarılı

