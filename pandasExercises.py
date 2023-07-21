import seaborn as sns
import pandas as pd

#Görev 1: Seaborn kütüphanesi içerisinden Titanic veri setini tanımlayınız.
df=sns.load_dataset("titanic")

#Görev 2: Titanic veri setindeki kadın ve erkek yolcuların sayısını bulunuz.
print(df["sex"].value_counts())
'''male      577
female    314
Name: sex, dtype: int64'''

#Görev 3: Her bir sutuna ait unique değerlerin sayısını bulunuz.
print(df.nunique())
'''
survived         2
pclass           3
sex              2
age             88
sibsp            7
parch            7
fare           248
embarked         3
class            3
who              3
adult_male       2
deck             7
embark_town      3
alive            2
alone            2
dtype: int64'''

# Görev 4:  pclass değişkeninin unique değerlerinin sayısını bulunuz
print(len(df["pclass"].unique()))
'''
3
'''

#Görev 5: pclass ve parch değişkenlerinin unique değerlerinin sayısını bulunuz.
print(df[["pclass","parch"]].nunique())
'''
pclass    3
parch     7
dtype: int64 '''

#Görev 6: embarked değişkeninin tipini kontrol ediniz. Tipini category olarak değiştiriniz ve tekrar kontrol ediniz.
print(df["embarked"].dtypes)   #object
df["embarked"]= df["embarked"].astype("category")
print(df["embarked"].dtypes) #category

#Görev 7: embarked değeri C olanların tüm bilgelerini gösteriniz
print(df[df["embarked"] == "C"])
'''
     survived  pclass     sex   age  ...  deck  embark_town  alive  alone
1           1       1  female  38.0  ...     C    Cherbourg    yes  False
9           1       2  female  14.0  ...   NaN    Cherbourg    yes  False
19          1       3  female   NaN  ...   NaN    Cherbourg    yes   True
26          0       3    male   NaN  ...   NaN    Cherbourg     no   True
30          0       1    male  40.0  ...   NaN    Cherbourg     no   True
..        ...     ...     ...   ...  ...   ...          ...    ...    ...
866         1       2  female  27.0  ...   NaN    Cherbourg    yes  False
874         1       2  female  28.0  ...   NaN    Cherbourg    yes  False
875         1       3  female  15.0  ...   NaN    Cherbourg    yes   True
879         1       1  female  56.0  ...     C    Cherbourg    yes  False
889         1       1    male  26.0  ...     C    Cherbourg    yes   True

[168 rows x 15 columns]'''

#Görev 8: embarked değeri S olmayanların tüm bilgelerini gösteriniz
print(df[df["embarked"] != "S"])
'''
     survived  pclass     sex   age  ...  deck  embark_town  alive  alone
1           1       1  female  38.0  ...     C    Cherbourg    yes  False
5           0       3    male   NaN  ...   NaN   Queenstown     no   True
9           1       2  female  14.0  ...   NaN    Cherbourg    yes  False
16          0       3    male   2.0  ...   NaN   Queenstown     no  False
19          1       3  female   NaN  ...   NaN    Cherbourg    yes   True
..        ...     ...     ...   ...  ...   ...          ...    ...    ...
875         1       3  female  15.0  ...   NaN    Cherbourg    yes   True
879         1       1  female  56.0  ...     C    Cherbourg    yes  False
885         0       3  female  39.0  ...   NaN   Queenstown     no  False
889         1       1    male  26.0  ...     C    Cherbourg    yes   True
890         0       3    male  32.0  ...   NaN   Queenstown     no   True

[247 rows x 15 columns]'''

#Görev 9: Yaşı 30 dan küçük ve kadın olan yolcuların tüm bilgilerini gösteriniz
print(df.loc[(df["age"] < 30) & (df["sex"] == "female")])
'''
     survived  pclass     sex   age  ...  deck  embark_town  alive  alone
2           1       3  female  26.0  ...   NaN  Southampton    yes   True
8           1       3  female  27.0  ...   NaN  Southampton    yes  False
9           1       2  female  14.0  ...   NaN    Cherbourg    yes  False
10          1       3  female   4.0  ...     G  Southampton    yes  False
14          0       3  female  14.0  ...   NaN  Southampton     no   True
..        ...     ...     ...   ...  ...   ...          ...    ...    ...
874         1       2  female  28.0  ...   NaN    Cherbourg    yes  False
875         1       3  female  15.0  ...   NaN    Cherbourg    yes   True
880         1       2  female  25.0  ...   NaN  Southampton    yes  False
882         0       3  female  22.0  ...   NaN  Southampton     no   True
887         1       1  female  19.0  ...     B  Southampton    yes   True

[147 rows x 15 columns]
'''

#Görev 10: Fare'i 500'den büyük veya yaşı 70 den büyük yolcuların bilgilerini gösteriniz
print(df.loc[(df["fare"] > 500) | (df["age"] > 70 )])
'''
     survived  pclass     sex   age  ...  deck  embark_town  alive  alone
96          0       1    male  71.0  ...     A    Cherbourg     no   True
116         0       3    male  70.5  ...   NaN   Queenstown     no   True
258         1       1  female  35.0  ...   NaN    Cherbourg    yes   True
493         0       1    male  71.0  ...   NaN    Cherbourg     no   True
630         1       1    male  80.0  ...     A  Southampton    yes   True
679         1       1    male  36.0  ...     B    Cherbourg    yes  False
737         1       1    male  35.0  ...     B    Cherbourg    yes   True
851         0       3    male  74.0  ...   NaN  Southampton     no   True

[8 rows x 15 columns]
'''

#Görev 11: Her bir değişkendeki boş değerlerin toplamını bulunuz.
print(df.isnull().sum())
'''
[8 rows x 15 columns]
survived         0
pclass           0
sex              0
age            177
sibsp            0
parch            0
fare             0
embarked         2
class            0
who              0
adult_male       0
deck           688
embark_town      2
alive            0
alone            0
dtype: int64
'''

# Görev 12: who değişkenini dataframe’den çıkarınız.
df=df.drop("who",axis=1)
print(df.columns)
'''
Index(['survived', 'pclass', 'sex', 'age', 'sibsp', 'parch', 'fare',
       'embarked', 'class', 'adult_male', 'deck', 'embark_town', 'alive',
       'alone'],
      dtype='object')
'''

#Görev 13: deck değikenindeki boş değerleri deck değişkenin en çok tekrar eden değeri (mode) ile doldurunuz.
df["deck"] = df["deck"].fillna(df["deck"].mode()[0])
print(df["deck"])
'''
0      C
1      C
2      C
3      C
4      C
      ..
886    C
887    B
888    C
889    C
890    C
Name: deck, Length: 891, dtype: category
Categories (7, object): ['A', 'B', 'C', 'D', 'E', 'F', 'G']
'''

#Görev 14: age değikenindeki boş değerleri age değişkenin medyanı ile doldurunuz.
df["age"] = df["age"].fillna(df["age"].median())
print(df["age"])
'''
0      22.0
1      38.0
2      26.0
3      35.0
4      35.0
       ... 
886    27.0
887    19.0
888    28.0
889    26.0
890    32.0
Name: age, Length: 891, dtype: float64
'''

#Görev 15: survived değişkeninin pclass ve cinsiyet değişkenleri kırılımınında sum, count, mean değerlerini bulunuz
print(df.groupby(["pclass","sex"]).agg({"survived":["sum","count","mean"]}))
'''
              survived                
                   sum count      mean
pclass sex                            
1      female       91    94  0.968085
       male         45   122  0.368852
2      female       70    76  0.921053
       male         17   108  0.157407
3      female       72   144  0.500000
       male         47   347  0.135447
'''

#Görev 16: 30 yaşın altında olanlar 1, 30'a eşit ve üstünde olanlara 0 vericek bir fonksiyon yazın.
# Yazdığınız fonksiyonu kullanarak titanik veri setinde age_flag adında bir değişken oluşturunuz.
df["age_flag"] = df['age'].apply(lambda x: 1 if x < 30 else 0)
print(df["age_flag"])
'''
0      1
1      0
2      1
3      0
4      0
      ..
886    1
887    1
888    1
889    1
890    0
Name: age_flag, Length: 891, dtype: int64
'''

#Görev 17: Seaborn kütüphanesi içerisinden Tips veri setini tanımlayınız
dftips = sns.load_dataset('tips')

#Görev 18: Time değişkeninin kategorilerine (Dinner, Lunch) göre
# total_bill değerlerinin toplamını, min, max ve ortalamasını bulunuz.
print(dftips.groupby("time").agg({"total_bill":["sum","min","max","mean"]}))
'''
       total_bill                        
              sum   min    max       mean
time                                     
Lunch     1167.47  7.51  43.11  17.168676
Dinner    3660.30  3.07  50.81  20.797159
'''

#Görev 19: Günlere ve time göre total_bill değerlerinin toplamını, min, max ve ortalamasını bulunuz.
print(dftips.groupby(["day","time"]).agg({"total_bill":["sum","min","max","mean"]}))
'''
            total_bill                         
                   sum    min    max       mean
day  time                                      
Thur Lunch     1077.55   7.51  43.11  17.664754
     Dinner      18.78  18.78  18.78  18.780000
Fri  Lunch       89.92   8.58  16.27  12.845714
     Dinner     235.96   5.75  40.17  19.663333
Sat  Lunch        0.00    NaN    NaN        NaN
     Dinner    1778.40   3.07  50.81  20.441379
Sun  Lunch        0.00    NaN    NaN        NaN
     Dinner    1627.16   7.25  48.17  21.410000
'''

#Görev 20: Lunch zamanına ve kadın müşterilere ait total_bill ve tip değerlerinin
# day'e göre toplamını, min, max ve ortalamasını bulunuz.
print(dftips.loc[(dftips["time"]== "Lunch") & (dftips["sex"]=="Female")].groupby(["day"])
      .agg({"total_bill":["sum","min","max","mean"], "tip":["sum","min","max","mean"]}))
'''
     total_bill                            tip                      
            sum    min    max      mean    sum   min   max      mean
day                                                                 
Thur     516.11   8.35  43.11  16.64871  79.42  1.25  5.17  2.561935
Fri       55.76  10.09  16.27  13.94000  10.98  2.00  3.48  2.745000
Sat        0.00    NaN    NaN       NaN   0.00   NaN   NaN       NaN
Sun        0.00    NaN    NaN       NaN   0.00   NaN   NaN       NaN
'''

#Görev 21: size'i 3'ten küçük, total_bill'i 10'dan büyük olan siparişlerin ortalaması nedir
print(dftips.loc[(dftips["size"] < 3) & (dftips["total_bill"] > 10)].mean())
'''
total_bill    17.184965
tip            2.638811
size           1.993007
dtype: float64
'''

#Görev 22: total_bill_tip_sum adında yeni bir değişken oluşturunuz.
# Her bir müşterinin ödediği totalbill ve tip in toplamını versin.
dftips["total_bill_tip_sum"]= dftips["total_bill"]+dftips["tip"]
print(dftips["total_bill_tip_sum"])
'''
0      18.00
1      12.00
2      24.51
3      26.99
4      28.20
       ...  
239    34.95
240    29.18
241    24.67
242    19.57
243    21.78
Name: total_bill_tip_sum, Length: 244, dtype: float64
'''

#Görev 23: total_bill_tip_sum değişkenine göre büyükten küçüğe sıralayınız ve
# ilk 30 kişiyi yeni bir dataframe'e atayınız.
newdf=dftips.sort_values("total_bill_tip_sum", ascending=False).head(30)
print(newdf)
'''
     total_bill    tip     sex smoker   day    time  size  total_bill_tip_sum
170       50.81  10.00    Male    Yes   Sat  Dinner     3               60.81
212       48.33   9.00    Male     No   Sat  Dinner     4               57.33
59        48.27   6.73    Male     No   Sat  Dinner     4               55.00
156       48.17   5.00    Male     No   Sun  Dinner     6               53.17
182       45.35   3.50    Male    Yes   Sun  Dinner     3               48.85
197       43.11   5.00  Female    Yes  Thur   Lunch     4               48.11
23        39.42   7.58    Male     No   Sat  Dinner     4               47.00
102       44.30   2.50  Female    Yes   Sat  Dinner     3               46.80
142       41.19   5.00    Male     No  Thur   Lunch     5               46.19
95        40.17   4.73    Male    Yes   Fri  Dinner     4               44.90
184       40.55   3.00    Male    Yes   Sun  Dinner     2               43.55
112       38.07   4.00    Male     No   Sun  Dinner     3               42.07
207       38.73   3.00    Male    Yes   Sat  Dinner     4               41.73
56        38.01   3.00    Male    Yes   Sat  Dinner     4               41.01
141       34.30   6.70    Male     No  Thur   Lunch     6               41.00
238       35.83   4.67  Female     No   Sat  Dinner     3               40.50
11        35.26   5.00  Female     No   Sun  Dinner     4               40.26
52        34.81   5.20  Female     No   Sun  Dinner     4               40.01
85        34.83   5.17  Female     No  Thur   Lunch     4               40.00
47        32.40   6.00    Male     No   Sun  Dinner     4               38.40
180       34.65   3.68    Male    Yes   Sun  Dinner     4               38.33
179       34.63   3.55    Male    Yes   Sun  Dinner     2               38.18
83        32.68   5.00    Male    Yes  Thur   Lunch     2               37.68
39        31.27   5.00    Male     No   Sat  Dinner     3               36.27
167       31.71   4.50    Male     No   Sun  Dinner     4               36.21
175       32.90   3.11    Male    Yes   Sun  Dinner     2               36.01
44        30.40   5.60    Male     No   Sun  Dinner     4               36.00
173       31.85   3.18    Male    Yes   Sun  Dinner     2               35.03
116       29.93   5.07    Male     No   Sun  Dinner     4               35.00
155       29.85   5.14  Female     No   Sun  Dinner     5               34.99
'''
