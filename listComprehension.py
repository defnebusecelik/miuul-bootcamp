# List Comprehension Exercises

# Görev 1: List Comprehension yapısı kullanarak car_crashes verisindeki numeric değişkenlerin isimlerini büyük
# harfe çeviriniz ve başına NUM ekleyiniz.

import seaborn as sns

df = sns.load_dataset("car_crashes")
print(df.columns)

df.columns=["NUM_" + col.upper() if df[col].dtype != 'O' else col.upper() for col in df.columns]
print(df.columns)

# Görev 2: List Comprehension yapısı kullanarak car_crashes verisinde isminde "no" barındırmayan
# değişkenlerin isimlerinin sonuna "FLAG" yazınız.

df.columns= [col.upper() if "no" in col else col.upper() + "_FLAG" for col in df.columns]
print(df.columns)

#Görev 3: List Comprehension yapısı kullanarak aşağıda verilen değişken isimlerinden FARKLI olan
# değişkenlerin isimlerini seçiniz ve yeni bir dataframe oluşturunuz.

og_list = ["abbrev", "no_previous"]

new_cols = [col for col in df.columns if not(col in og_list)]
new_df = df[new_cols]
print(new_df.head())
