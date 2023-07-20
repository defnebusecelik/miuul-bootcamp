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
