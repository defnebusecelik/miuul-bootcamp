
#############################################
# Kural Tabanlı Sınıflandırma ile Potansiyel Müşteri Getirisi Hesaplama
#############################################

#############################################
# İş Problemi
#############################################
# Bir oyun şirketi müşterilerinin bazı özelliklerini kullanarak seviye tabanlı (level based) yeni müşteri tanımları (persona)
# oluşturmak ve bu yeni müşteri tanımlarına göre segmentler oluşturup bu segmentlere göre yeni gelebilecek müşterilerin şirkete
# ortalama ne kadar kazandırabileceğini tahmin etmek istemektedir.

# Örneğin: Türkiye’den IOS kullanıcısı olan 25 yaşındaki bir erkek kullanıcının ortalama ne kadar kazandırabileceği belirlenmek isteniyor.


#############################################
# Veri Seti Hikayesi
#############################################
# Persona.csv veri seti uluslararası bir oyun şirketinin sattığı ürünlerin fiyatlarını ve bu ürünleri satın alan kullanıcıların bazı
# demografik bilgilerini barındırmaktadır. Veri seti her satış işleminde oluşan kayıtlardan meydana gelmektedir. Bunun anlamı tablo
# tekilleştirilmemiştir. Diğer bir ifade ile belirli demografik özelliklere sahip bir kullanıcı birden fazla alışveriş yapmış olabilir.

# Price: Müşterinin harcama tutarı
# Source: Müşterinin bağlandığı cihaz türü
# Sex: Müşterinin cinsiyeti
# Country: Müşterinin ülkesi
# Age: Müşterinin yaşı

################# Uygulama Öncesi #####################

#    PRICE   SOURCE   SEX COUNTRY  AGE
# 0     39  android  male     bra   17
# 1     39  android  male     bra   17
# 2     49  android  male     bra   17
# 3     29  android  male     tur   17
# 4     49  android  male     tur   17

################# Uygulama Sonrası #####################

#       customers_level_based        PRICE SEGMENT
# 0   BRA_ANDROID_FEMALE_0_18  1139.800000       A
# 1  BRA_ANDROID_FEMALE_19_23  1070.600000       A
# 2  BRA_ANDROID_FEMALE_24_30   508.142857       A
# 3  BRA_ANDROID_FEMALE_31_40   233.166667       C
# 4  BRA_ANDROID_FEMALE_41_66   236.666667       C


#############################################
# PROJE GÖREVLERİ
#############################################

#############################################
# GÖREV 1: Aşağıdaki soruları yanıtlayınız.
#############################################
import pandas as pd

# Soru 1: persona.csv dosyasını okutunuz ve veri seti ile ilgili genel bilgileri gösteriniz.
df=pd.read_csv("persona.csv")
print(df.head())
'''
   PRICE   SOURCE   SEX COUNTRY  AGE
0     39  android  male     bra   17
1     39  android  male     bra   17
2     49  android  male     bra   17
3     29  android  male     tur   17
4     49  android  male     tur   17
'''

print(df.tail())
'''
      PRICE   SOURCE     SEX COUNTRY  AGE
4995     29  android  female     bra   31
4996     29  android  female     bra   31
4997     29  android  female     bra   31
4998     39  android  female     bra   31
4999     29  android  female     bra   31
'''

print(df.shape)   #(5000, 5)

print(df.info())
'''
<class 'pandas.core.frame.DataFrame'>
Int64Index: 5000 entries, 0 to 4999
Data columns (total 5 columns):
 #   Column   Non-Null Count  Dtype 
---  ------   --------------  ----- 
 0   PRICE    5000 non-null   int64 
 1   SOURCE   5000 non-null   object
 2   SEX      5000 non-null   object
 3   COUNTRY  5000 non-null   object
 4   AGE      5000 non-null   int64 
dtypes: int64(2), object(3)
memory usage: 234.4+ KB
'''

# Soru 2: Kaç unique SOURCE vardır? Frekansları nedir?
print(df["SOURCE"].nunique())      #2

print(df["SOURCE"].value_counts())
'''
android    2974
ios        2026
Name: SOURCE, dtype: int64
'''

# Soru 3: Kaç unique PRICE vardır?
print(df["PRICE"].nunique())      #6

# Soru 4: Hangi PRICE'dan kaçar tane satış gerçekleşmiş?
print(df["PRICE"].value_counts())
'''
29    1305
39    1260
49    1031
19     992
59     212
9      200
Name: PRICE, dtype: int64
'''

# Soru 5: Hangi ülkeden kaçar tane satış olmuş?
print(df.groupby("COUNTRY")["PRICE"].count())
'''
COUNTRY
bra    1496
can     230
deu     455
fra     303
tur     451
usa    2065
Name: PRICE, dtype: int64
'''

# Soru 6: Ülkelere göre satışlardan toplam ne kadar kazanılmış?
print(df.groupby("COUNTRY")["PRICE"].sum())
'''
COUNTRY
bra    51354
can     7730
deu    15485
fra    10177
tur    15689
usa    70225
Name: PRICE, dtype: int64
'''

# Soru 7: SOURCE türlerine göre göre satış sayıları nedir?
print(df["SOURCE"].value_counts())
'''
android    2974
ios        2026
Name: SOURCE, dtype: int64
'''

# Soru 8: Ülkelere göre PRICE ortalamaları nedir?
print(df.groupby("COUNTRY")["PRICE"].mean())
'''
COUNTRY
bra    34.327540
can    33.608696
deu    34.032967
fra    33.587459
tur    34.787140
usa    34.007264
Name: PRICE, dtype: float64
'''

# Soru 9: SOURCE'lara göre PRICE ortalamaları nedir?
print(df.groupby("SOURCE")["PRICE"].mean())
'''
SOURCE
android    34.174849
ios        34.069102
Name: PRICE, dtype: float64
'''

# Soru 10: COUNTRY-SOURCE kırılımında PRICE ortalamaları nedir?
print(df.groupby(["COUNTRY", 'SOURCE']).agg({"PRICE": "mean"}))
'''
                     PRICE
COUNTRY SOURCE            
bra     android  34.387029
        ios      34.222222
can     android  33.330709
        ios      33.951456
deu     android  33.869888
        ios      34.268817
fra     android  34.312500
        ios      32.776224
tur     android  36.229437
        ios      33.272727
usa     android  33.760357
        ios      34.371703
'''


#############################################
# GÖREV 2: COUNTRY, SOURCE, SEX, AGE kırılımında ortalama kazançlar nedir?
# #############################################

print(df.groupby(["COUNTRY", 'SOURCE', "SEX", "AGE"]).agg({"PRICE": "mean"}).head())
'''
                                PRICE
COUNTRY SOURCE  SEX    AGE           
bra     android female 15   38.714286
                       16   35.944444
                       17   35.666667
                       18   32.255814
                       19   35.206897
'''

#############################################
# GÖREV 3: Çıktıyı PRICE'a göre sıralayınız.
#############################################
# Önceki sorudaki çıktıyı daha iyi görebilmek için sort_values metodunu azalan olacak şekilde PRICE'a uygulayınız.
# Çıktıyı agg_df olarak kaydediniz.

agg_df = df.groupby(["COUNTRY", 'SOURCE', "SEX", "AGE"]).agg({"PRICE": "mean"}).sort_values("PRICE", ascending=False)
print(agg_df.head())
'''
                            PRICE
COUNTRY SOURCE  SEX    AGE       
bra     android male   46    59.0
usa     android male   36    59.0
fra     android female 24    59.0
usa     ios     male   32    54.0
deu     android female 36    49.0
'''


#############################################
# GÖREV 4: Indekste yer alan isimleri değişken ismine çeviriniz.
#############################################
# Üçüncü sorunun çıktısında yer alan PRICE dışındaki tüm değişkenler index isimleridir.
# Bu isimleri değişken isimlerine çeviriniz.

agg_df = agg_df.reset_index()
print(agg_df.head())
'''
  COUNTRY   SOURCE     SEX  AGE  PRICE
0     bra  android    male   46   59.0
1     usa  android    male   36   59.0
2     fra  android  female   24   59.0
3     usa      ios    male   32   54.0
4     deu  android  female   36   49.0
'''


#############################################
# GÖREV 5: AGE değişkenini kategorik değişkene çeviriniz ve agg_df'e ekleyiniz.
#############################################
# Age sayısal değişkenini kategorik değişkene çeviriniz.
# Aralıkları ikna edici olacağını düşündüğünüz şekilde oluşturunuz.
# Örneğin: '0_18', '19_23', '24_30', '31_40', '41_70'

bins = [0, 18, 23, 30, 40, 70]
labels = ['0_18', '19_23', '24_30', '31_40', '41_70']
agg_df["AGE_CAT"] = pd.cut(agg_df["AGE"], bins, labels=labels)
print(agg_df.head())
'''
  COUNTRY   SOURCE     SEX  AGE  PRICE AGE_CAT
0     bra  android    male   46   59.0   41_70
1     usa  android    male   36   59.0   31_40
2     fra  android  female   24   59.0   24_30
3     usa      ios    male   32   54.0   31_40
4     deu  android  female   36   49.0   31_40
'''


#############################################
# GÖREV 6: Yeni level based müşterileri tanımlayınız ve veri setine değişken olarak ekleyiniz.
#############################################
# Yeni eklenecek değişkenin adı: customers_level_based
# Bunları groupby'a alıp price ortalamalarını almak gerekmektedir.

agg_df["customers_level_based"] = [row[0].upper() + "_" + row[1].upper() + "_"
                                   + row[2].upper() + "_" + row[5].upper()
                                   for row in agg_df.values]
agg_df = agg_df[["customers_level_based", "PRICE"]]
print(agg_df.head())
'''
      customers_level_based  PRICE
0    BRA_ANDROID_MALE_41_70   59.0
1    USA_ANDROID_MALE_31_40   59.0
2  FRA_ANDROID_FEMALE_24_30   59.0
3        USA_IOS_MALE_31_40   54.0
4  DEU_ANDROID_FEMALE_31_40   49.0
'''



#############################################
# GÖREV 7: Yeni müşterileri (USA_ANDROID_MALE_0_18) segmentlere ayırınız.
#############################################
# PRICE'a göre segmentlere ayırınız,
# segmentleri "SEGMENT" isimlendirmesi ile agg_df'e ekleyiniz,
# segmentleri betimleyiniz,(mean, max, sum)

agg_df["SEGMENT"] = pd.qcut(agg_df["PRICE"], 4, labels=["D", "C", "B", "A"])
print(agg_df.head())
'''
      customers_level_based  PRICE SEGMENT
0    BRA_ANDROID_MALE_41_70   59.0       A
1    USA_ANDROID_MALE_31_40   59.0       A
2  FRA_ANDROID_FEMALE_24_30   59.0       A
3        USA_IOS_MALE_31_40   54.0       A
4  DEU_ANDROID_FEMALE_31_40   49.0       A
'''


print(agg_df.groupby("SEGMENT").agg({"PRICE": ["mean","max","sum"]}))
'''
             PRICE                        
              mean        max          sum
SEGMENT                                   
D        27.302596  31.105263  2375.325850
C        32.933339  34.000000  3128.667165
B        35.436170  37.000000  2870.329792
A        41.434736  59.000000  3521.952577
'''


#############################################
# GÖREV 8: Yeni gelen müşterileri sınıflandırınız ne kadar gelir getirebileceğini tahmin ediniz.
#############################################
# 33 yaşında ANDROID kullanan bir Türk kadını hangi segmente aittir ve ortalama ne kadar gelir kazandırması beklenir?

new_person = "TUR_ANDROID_FEMALE_31_40"
print(agg_df[agg_df["customers_level_based"] == new_person])
'''
       customers_level_based      PRICE SEGMENT
18  TUR_ANDROID_FEMALE_31_40  43.000000       A
35  TUR_ANDROID_FEMALE_31_40  40.666667       A
'''

print(agg_df[agg_df["customers_level_based"] == new_person].mean())
'''
PRICE    41.833333
dtype: float64
'''
# 35 yaşında IOS kullanan bir Fransız kadını hangi segmente ve ortalama ne kadar gelir kazandırması beklenir?

new_person2 = "FRA_IOS_FEMALE_31_40"
print(agg_df[agg_df["customers_level_based"] == new_person2])
'''
  customers_level_based      PRICE SEGMENT
208  FRA_IOS_FEMALE_31_40  33.000000       C
221  FRA_IOS_FEMALE_31_40  32.636364       C
'''

print(agg_df[agg_df["customers_level_based"] == new_person2].mean())
'''
PRICE    32.818182
'''
