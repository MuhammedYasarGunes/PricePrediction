# Öznitelik mühendisliği raporu (İP 4)

`03_oznitelik_muhendisligi.py` tarafından üretilmiştir. Eksik oranları yalnızca modele girecek (açık ve satışı > 0) satırlar üzerinden hesaplanır.

- Toplam öznitelik: 25 (2. haftadan 17, bu hafta eklenen 8)
- Modele girecek satır: 844.338

## Yeni öznitelikler

| Öznitelik | Tanım |
|---|---|
| PromoGunu | Promosyon bloğunun kaçıncı günü (promosyon yoksa 0) |
| TatileKalanGun | Bir sonraki resmî tatile kalan gün (0–7; 7 = 7 gün ve üzeri) |
| TatildenGecenGun | Son resmî tatilden bu yana geçen gün (0–7; 7 = 7 gün ve üzeri) |
| SatisGecikme | Aynı haftagününün L = 7·⌈H/7⌉ gün önceki log satışı |
| SatisOrt7 | t−H'de biten 7 günde açık günlerin ortalama log satışı |
| SatisOrt28 | t−H'de biten 28 günde açık günlerin ortalama log satışı |
| SatisOrt91 | t−H'de biten 91 günde açık günlerin ortalama log satışı |
| MagazaGunOrt | Aynı haftagününün t−L'de biten 13 haftalık ortalama log satışı |

## Eksik değer oranı (modele girecek satırlar)

Sütunlar: tüm dönem / değerlendirme penceresi (H=42 için test, diğerleri için ufuk analizi) / veri boşluğu olan 180 mağazanın Ocak–Mart 2015 satırları.

| Öznitelik | H=7 | H=28 | H=42 | H=84 |
|---|---|---|---|---|
| PromoGunu | %0,0 / %0,0 / %0,0 | %0,0 / %0,0 / %0,0 | %0,0 / %0,0 / %0,0 | %0,0 / %0,0 / %0,0 |
| TatileKalanGun | %0,8 / %0,0 / %0,0 | %0,8 / %0,0 / %0,0 | %0,8 / %13,9 / %0,0 | %0,8 / %7,2 / %0,0 |
| TatildenGecenGun | %0,0 / %0,0 / %0,0 | %0,0 / %0,0 / %0,0 | %0,0 / %0,0 / %0,0 | %0,0 / %0,0 / %0,0 |
| SatisGecikme | %4,1 / %0,0 / %8,1 | %6,9 / %4,6 / %32,1 | %9,0 / %7,1 / %48,1 | %13,9 / %4,4 / %96,0 |
| SatisOrt7 | %0,9 / %0,0 / %6,7 | %3,7 / %0,0 / %30,7 | %5,6 / %0,0 / %46,7 | %11,1 / %0,1 / %94,6 |
| SatisOrt28 | %1,4 / %0,0 / %12,1 | %4,2 / %0,0 / %36,1 | %6,0 / %0,0 / %52,1 | %11,4 / %0,0 / %100,0 |
| SatisOrt91 | %2,8 / %0,0 / %24,0 | %5,5 / %0,0 / %48,1 | %7,4 / %0,0 / %64,0 | %12,6 / %0,0 / %100,0 |
| MagazaGunOrt | %2,8 / %0,0 / %24,0 | %5,6 / %0,0 / %48,1 | %7,4 / %0,0 / %64,1 | %12,7 / %0,1 / %100,0 |

## Geçmiş satış özniteliklerinin log satışla korelasyonu (Pearson)

Ufuk uzadıkça öznitelikler daha eski bilgiye dayanır; korelasyondaki düşüş bunu gösterir.

| Öznitelik | H=7 | H=28 | H=42 | H=84 |
|---|---|---|---|---|
| SatisGecikme | 0,638 | 0,808 | 0,767 | 0,687 |
| SatisOrt7 | 0,678 | 0,737 | 0,714 | 0,681 |
| SatisOrt28 | 0,759 | 0,745 | 0,742 | 0,732 |
| SatisOrt91 | 0,758 | 0,751 | 0,749 | 0,749 |
| MagazaGunOrt | 0,840 | 0,839 | 0,835 | 0,833 |

## Kronolojik bölmeler

**H = 7** (Ufuk analizi: 09.05.2015–15.05.2015)

| Katman | Eğitim | Doğrulama | Eğitim satırı | Doğrulama satırı |
|---|---|---|---|---|
| K1 | 01.01.2013–17.04.2015 | 18.04.2015–24.04.2015 | 747.520 | 6.722 |
| K2 | 01.01.2013–24.04.2015 | 25.04.2015–01.05.2015 | 754.242 | 5.626 |
| K3 | 01.01.2013–01.05.2015 | 02.05.2015–08.05.2015 | 759.868 | 6.716 |

**H = 28** (Ufuk analizi: 09.05.2015–05.06.2015)

| Katman | Eğitim | Doğrulama | Eğitim satırı | Doğrulama satırı |
|---|---|---|---|---|
| K1 | 01.01.2013–13.02.2015 | 14.02.2015–13.03.2015 | 689.365 | 26.752 |
| K2 | 01.01.2013–13.03.2015 | 14.03.2015–10.04.2015 | 716.117 | 24.683 |
| K3 | 01.01.2013–10.04.2015 | 11.04.2015–08.05.2015 | 740.800 | 25.784 |

**H = 42** (Test: 20.06.2015–31.07.2015)

| Katman | Eğitim | Doğrulama | Eğitim satırı | Doğrulama satırı |
|---|---|---|---|---|
| K1 | 01.01.2013–13.02.2015 | 14.02.2015–27.03.2015 | 689.365 | 40.175 |
| K2 | 01.01.2013–27.03.2015 | 28.03.2015–08.05.2015 | 729.540 | 37.044 |
| K3 | 01.01.2013–08.05.2015 | 09.05.2015–19.06.2015 | 766.584 | 37.472 |

**H = 84** (Ufuk analizi: 09.05.2015–31.07.2015)

| Katman | Eğitim | Doğrulama | Eğitim satırı | Doğrulama satırı |
|---|---|---|---|---|
| K1 | 01.01.2013–29.08.2014 | 30.08.2014–21.11.2014 | 552.227 | 66.012 |
| K2 | 01.01.2013–21.11.2014 | 22.11.2014–13.02.2015 | 618.239 | 71.126 |
| K3 | 01.01.2013–13.02.2015 | 14.02.2015–08.05.2015 | 689.365 | 77.219 |

## Sızıntı testi

Kesim tarihinden sonraki tüm satışlar rastgele değerlerle değiştirilip geçmiş satış öznitelikleri yeniden üretildi. Kesim + H gününe kadar hiçbir değer değişmemeli (sızıntı yok); sonrasında değişmeli (test duyarlı).

| H | Kesim | Korunan hücre | Korunan bölgede değişen | Sonrasında değişen | Sonuç |
|---|---|---|---|---|---|
| 7 | 08.05.2015 | 4.822.375 | 0 | 401.415 | geçti |
| 7 | 19.06.2015 | 5.056.525 | 0 | 184.207 | geçti |
| 7 | 31.03.2014 | 2.575.650 | 0 | 2.376.180 | geçti |
| 28 | 08.05.2015 | 4.939.450 | 0 | 290.884 | geçti |
| 28 | 19.06.2015 | 5.173.600 | 0 | 73.688 | geçti |
| 28 | 31.03.2014 | 2.692.725 | 0 | 2.265.647 | geçti |
| 42 | 08.05.2015 | 5.017.500 | 0 | 217.183 | geçti |
| 42 | 19.06.2015 | 5.251.650 | 0 | — (kesim + H veri sonunda) | geçti |
| 42 | 31.03.2014 | 2.770.775 | 0 | 2.191.951 | geçti |
| 84 | 08.05.2015 | 5.251.650 | 0 | — (kesim + H veri sonunda) | geçti |
| 84 | 19.06.2015 | 5.251.650 | 0 | — (kesim + H veri sonunda) | geçti |
| 84 | 31.03.2014 | 3.004.925 | 0 | 1.973.663 | geçti |

Toplam 12 testin 12 tanesi geçti. Testin duyarlılığı her H için en az bir kesimde doğrulandı: evet.
