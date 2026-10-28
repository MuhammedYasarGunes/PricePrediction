# EDA bulgu özeti

Sayılar `02_eda.py` tarafından veriden hesaplanmıştır. Farklar betimleyicidir, nedensellik iddiası değildir.

## Zaman içinde satış
- Açık mağaza-günlerinin ortalama satışı 6.956, medyanı 6.369.
- Aylık ortalamanın en yükseği 12.2013 (8.613), en düşüğü 01.2013 (6.241).

## Haftalık mevsimsellik
- En yüksek ortalama Paz (8.225), en düşük Cmt (5.875).
- Pazar günü mağazaların yalnızca %2,5'i açık; pazar ortalaması az sayıda mağazaya dayanır.

## Promosyon
- Promosyonlu günlerde ortalama satış 8.229, promosyonsuz günlerde 5.930 (fark %38,8).
- Mağaza-günlerinin %44,6'i promosyonlu.

## Tatiller
- Resmî tatil günlerinde açık mağaza-günü sayısı çok azdır; ortalamalar şu sayılara dayanır: Yok n=843.428 ort=6.954, Resmî tatil (a) n=694 ort=8.487, Paskalya (b) n=145 ort=9.888, Noel (c) n=71 ort=9.744.
- Okul tatili olan günlerde ortalama 7.201, olmayan günlerde 6.897 (fark %4,4).

## Yıllık mevsimsellik
- Tüm yılların ortalamasına göre en yüksek ay Ara (8.609), en düşük ay Eyl (6.547).
- 2015 yalnızca Temmuz sonuna kadar vardır; 2015 çizgisi eksik aylar içerir.

## Mağaza tipi
- Tipe göre ortalama satış: a=6.926, b=10.233, c=6.933, d=6.822.
- Bazı tip–çeşitlilik kombinasyonları çok az mağazadan oluşur; çubuğu olmayan kombinasyon veride yoktur.

## Rekabet
- Rakip mesafesi ile mağaza ortalama satışı arasındaki Spearman korelasyonu -0,03 (1112 mağaza; 'rakip yok' için doldurulan büyük değerler hariç).

## Promo2
- Ortalama satış: Promo2'ye girmeyen=7.351, Promo2 mağazası, bu gün pasif=6.597, Promo2 etkin gün=6.471.

## Hedef değişken dağılımı
- Çarpıklık: ham satış 1,59, log(1+satış) -0,11. Log dönüşümü dağılımı simetriğe yaklaştırır; modelde log hedefin kullanılma gerekçesi budur.

## Veri boşluğu
- Günlük kaydı olan mağaza sayısı en fazla 1115, en az 935.
- Eksik mağaza olan gün aralığı 01.01.2013 - 31.12.2014 (185 gün). Bu dönemde gecikmeli öznitelikler eksik kalır.
