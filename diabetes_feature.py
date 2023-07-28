################ Diyabet Feature Engineering ##################

#İş Problemi
#Özellikleri belirtildiğinde kişilerin diyabet hastası olup olmadıklarını tahmin
#edebilecek bir makine öğrenmesi modeli geliştirilmesi istenmektedir.

#Veri Seti
#Veri seti ABD'deki Ulusal Diyabet-Sindirim-Böbrek Hastalıkları Enstitüleri'nde tutulan büyük veri setinin parçasıdır. ABD'deki
#Arizona Eyaleti'nin en büyük 5. şehri olan Phoenix şehrinde yaşayan 21 yaş ve üzerinde olan Pima Indian kadınları üzerinde
#yapılan diyabet araştırması için kullanılan verilerdir.
#Hedef değişken "outcome" olarak belirtilmiş olup; 1 diyabet test sonucunun pozitif oluşunu,
# 0 ise negatif oluşunu belirtmektedir.

# Pregnancies: Hamilelik sayısı
# Glucose: Glikoz
# BloodPressure: Kan basıncı (Diastolic(Küçük Tansiyon))
# SkinThickness: Cilt Kalınlığı
# Insulin: İnsülin.
# BMI: Beden kitle indeksi.
# DiabetesPedigreeFunction: Soyumuzdaki kişilere göre diyabet olma ihtimalimizi hesaplayan bir fonksiyon.
# Age: Yaş (yıl)
# Outcome: Kişinin diyabet olup olmadığı bilgisi. Hastalığa sahip (1) ya da değil (0)


#Görev 1 : Keşifçi Veri Analizi
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score,roc_auc_score

pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_rows', 30)

#Adım 1: Genel resmi inceleyiniz.
df = pd.read_csv("diabetes.csv")
print(df.head())
'''
   Pregnancies  Glucose  BloodPressure  SkinThickness  Insulin   BMI  DiabetesPedigreeFunction  Age  Outcome
0            6      148             72             35        0  33.6                     0.627   50        1
1            1       85             66             29        0  26.6                     0.351   31        0
2            8      183             64              0        0  23.3                     0.672   32        1
3            1       89             66             23       94  28.1                     0.167   21        0
4            0      137             40             35      168  43.1                     2.288   33        1
'''
print(df.tail())
'''
     Pregnancies  Glucose  BloodPressure  SkinThickness  Insulin   BMI  DiabetesPedigreeFunction  Age  Outcome
763           10      101             76             48      180  32.9                     0.171   63        0
764            2      122             70             27        0  36.8                     0.340   27        0
765            5      121             72             23      112  26.2                     0.245   30        0
766            1      126             60              0        0  30.1                     0.349   47        1
767            1       93             70             31        0  30.4                     0.315   23        0
'''
print(df.shape)     #(768, 9)
print(df.dtypes)
'''
Pregnancies                   int64
Glucose                       int64
BloodPressure                 int64
SkinThickness                 int64
Insulin                       int64
BMI                         float64
DiabetesPedigreeFunction    float64
Age                           int64
Outcome                       int64
dtype: object'''
print(df.isnull().sum())
'''
Pregnancies                 0
Glucose                     0
BloodPressure               0
SkinThickness               0
Insulin                     0
BMI                         0
DiabetesPedigreeFunction    0
Age                         0
Outcome                     0
dtype: int64'''
print(df.info())
'''
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 768 entries, 0 to 767
Data columns (total 9 columns):
 #   Column                    Non-Null Count  Dtype  
---  ------                    --------------  -----  
 0   Pregnancies               768 non-null    int64  
 1   Glucose                   768 non-null    int64  
 2   BloodPressure             768 non-null    int64  
 3   SkinThickness             768 non-null    int64  
 4   Insulin                   768 non-null    int64  
 5   BMI                       768 non-null    float64
 6   DiabetesPedigreeFunction  768 non-null    float64
 7   Age                       768 non-null    int64  
 8   Outcome                   768 non-null    int64  
dtypes: float64(2), int64(7)
memory usage: 54.1 KB
'''
print(df.describe().T)
'''
                          count        mean         std     min       25%       50%        75%     max
Pregnancies               768.0    3.845052    3.369578   0.000   1.00000    3.0000    6.00000   17.00
Glucose                   768.0  120.894531   31.972618   0.000  99.00000  117.0000  140.25000  199.00
BloodPressure             768.0   69.105469   19.355807   0.000  62.00000   72.0000   80.00000  122.00
SkinThickness             768.0   20.536458   15.952218   0.000   0.00000   23.0000   32.00000   99.00
Insulin                   768.0   79.799479  115.244002   0.000   0.00000   30.5000  127.25000  846.00
BMI                       768.0   31.992578    7.884160   0.000  27.30000   32.0000   36.60000   67.10
DiabetesPedigreeFunction  768.0    0.471876    0.331329   0.078   0.24375    0.3725    0.62625    2.42
Age                       768.0   33.240885   11.760232  21.000  24.00000   29.0000   41.00000   81.00
Outcome                   768.0    0.348958    0.476951   0.000   0.00000    0.0000    1.00000    1.00
'''

#Adım 2: Numerik ve kategorik değişkenleri yakalayınız.
def grab_col_names(dataframe, cat_th=10, car_th=20):
    cat_cols = [col for col in dataframe.columns if dataframe[col].dtypes == "O"]
    num_but_cat = [col for col in dataframe.columns if dataframe[col].nunique() < cat_th and dataframe[col].dtypes != "O"]
    cat_but_car = [col for col in dataframe.columns if dataframe[col].nunique() > car_th and dataframe[col].dtypes == "O"]
    cat_cols = cat_cols + num_but_cat
    cat_cols = [col for col in cat_cols if col not in cat_but_car]

    num_cols = [col for col in dataframe.columns if dataframe[col].dtypes != "O"]
    num_cols = [col for col in num_cols if col not in num_but_cat]

    print(f"Observations: {dataframe.shape[0]}")
    print(f"Variables: {dataframe.shape[1]}")
    print(f'cat_cols: {len(cat_cols)}')
    print(f'num_cols: {len(num_cols)}')
    print(f'cat_but_car: {len(cat_but_car)}')
    print(f'num_but_cat: {len(num_but_cat)}')

    return cat_cols, num_cols, cat_but_car

cat_cols, num_cols, cat_but_car = grab_col_names(df)
'''
Observations: 768
Variables: 9
cat_cols: 1
num_cols: 8
cat_but_car: 0
num_but_cat: 1
'''
print(cat_cols)
'''
['Outcome']
'''
print(num_cols)
'''
['Pregnancies', 'Glucose', 'BloodPressure', 
'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']
'''
print(cat_but_car)
'''
[]
'''

#Adım 3: Numerik ve kategorik değişkenlerin analizini yapınız.
def cat_summary(dataframe, col_name, plot=False):
    print(pd.DataFrame({col_name: dataframe[col_name].value_counts(),
                        "Ratio": 100 * dataframe[col_name].value_counts() / len(dataframe)}))
    if plot:
        sns.countplot(x=dataframe[col_name], data=dataframe)
        plt.show()

for col in cat_cols:
     cat_summary(df, col)

'''
   Outcome      Ratio
0      500  65.104167
1      268  34.895833
'''

def num_summary(dataframe, numerical_col, plot=False):
    #quantiles = [0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.99]
    print(dataframe[numerical_col].describe().T)

    if plot:
        dataframe[numerical_col].hist(bins=20)
        plt.xlabel(numerical_col)
        plt.title(numerical_col)
        plt.show()

for col in num_cols:
    num_summary(df, col)

'''
count    768.000000
mean       3.845052
std        3.369578
min        0.000000
25%        1.000000
50%        3.000000
75%        6.000000
max       17.000000
Name: Pregnancies, dtype: float64
count    768.000000
mean     120.894531
std       31.972618
min        0.000000
25%       99.000000
50%      117.000000
75%      140.250000
max      199.000000
Name: Glucose, dtype: float64
count    768.000000
mean      69.105469
std       19.355807
min        0.000000
25%       62.000000
50%       72.000000
75%       80.000000
max      122.000000
Name: BloodPressure, dtype: float64
count    768.000000
mean      20.536458
std       15.952218
min        0.000000
25%        0.000000
50%       23.000000
75%       32.000000
max       99.000000
Name: SkinThickness, dtype: float64
count    768.000000
mean      79.799479
std      115.244002
min        0.000000
25%        0.000000
50%       30.500000
75%      127.250000
max      846.000000
Name: Insulin, dtype: float64
count    768.000000
mean      31.992578
std        7.884160
min        0.000000
25%       27.300000
50%       32.000000
75%       36.600000
max       67.100000
Name: BMI, dtype: float64
count    768.000000
mean       0.471876
std        0.331329
min        0.078000
25%        0.243750
50%        0.372500
75%        0.626250
max        2.420000
Name: DiabetesPedigreeFunction, dtype: float64
count    768.000000
mean      33.240885
std       11.760232
min       21.000000
25%       24.000000
50%       29.000000
75%       41.000000
max       81.000000
Name: Age, dtype: float64
'''

#Adım 4: Hedef değişken analizi yapınız.
import pandas as pd

def analyzer_target_variable(dataframe, categorical_column, target_column, numeric_columns):
    # Kategorik değişkenlere göre hedef değişkenin ortalamaları
    target_means_by_categorical = dataframe.groupby(categorical_column)[target_column].mean()

    # Hedef değişkene göre numerik değişkenlerin ortalamaları
    numeric_means_by_target = dataframe.groupby(target_column)[numeric_columns].mean()

    return target_means_by_categorical, numeric_means_by_target

target_means_by_categorical, numeric_means_by_target = analyzer_target_variable(df,
    categorical_column=cat_cols,
    target_column="Outcome",
    numeric_columns=num_cols)

print(target_means_by_categorical)
'''
Outcome
0    0.0
1    1.0
Name: Outcome, dtype: float64
 '''

print(numeric_means_by_target)
'''
         Pregnancies     Glucose  BloodPressure  SkinThickness     Insulin        BMI  DiabetesPedigreeFunction        Age
Outcome                                                                                                                   
0           3.298000  109.980000      68.184000      19.664000   68.792000  30.304200                  0.429734  31.190000
1           4.865672  141.257463      70.824627      22.164179  100.335821  35.142537                  0.550500  37.067164

'''

#Adım 5: Aykırı gözlem analizi yapınız.
def outlier_thresholds(dataframe, col_name, q1=0.25, q3=0.75):
    quartile1 = dataframe[col_name].quantile(q1)
    quartile3 = dataframe[col_name].quantile(q3)
    interquantile_range = quartile3 - quartile1
    up_limit = quartile3 + 1.5 * interquantile_range
    low_limit = quartile1 - 1.5 * interquantile_range
    return low_limit, up_limit

for col in num_cols:
    outlier_thresholds(df,col)
def check_outlier(dataframe, col_name):
    low_limit, up_limit = outlier_thresholds(dataframe, col_name)
    if dataframe[(dataframe[col_name] > up_limit) | (dataframe[col_name] < low_limit)].any(axis=None):
        return True
    else:
        return False

for col in num_cols:
    print(col,check_outlier(df,col))
'''
Pregnancies True
Glucose True
BloodPressure True
SkinThickness True
Insulin True
BMI True
DiabetesPedigreeFunction True
Age True
'''

def grab_outliers(dataframe, col_name, index=False):
    low, up = outlier_thresholds(dataframe, col_name)

    if dataframe[((dataframe[col_name] < low) | (dataframe[col_name] > up))].shape[0] > 10:
        print(dataframe[((dataframe[col_name] < low) | (dataframe[col_name] > up))].head())
    else:
        print(dataframe[((dataframe[col_name] < low) | (dataframe[col_name] > up))])

    if index:
        outlier_index = dataframe[((dataframe[col_name] < low) | (dataframe[col_name] > up))].index
        return outlier_index

for col in num_cols:
    print(col,grab_outliers(df,col))
'''
     Pregnancies  Glucose  BloodPressure  SkinThickness  Insulin   BMI  DiabetesPedigreeFunction  Age  Outcome
88            15      136             70             32      110  37.1                     0.153   43        1
159           17      163             72             41      114  40.9                     0.817   47        1
298           14      100             78             25      184  36.6                     0.412   46        1
455           14      175             62             30        0  33.6                     0.212   38        1
Pregnancies None
     Pregnancies  Glucose  BloodPressure  SkinThickness  Insulin   BMI  DiabetesPedigreeFunction  Age  Outcome
75             1        0             48             20        0  24.7                     0.140   22        0
182            1        0             74             20       23  27.7                     0.299   21        0
342            1        0             68             35        0  32.0                     0.389   22        0
349            5        0             80             32        0  41.0                     0.346   37        1
502            6        0             68             41        0  39.0                     0.727   41        1
Glucose None
    Pregnancies  Glucose  BloodPressure  SkinThickness  Insulin   BMI  DiabetesPedigreeFunction  Age  Outcome
7            10      115              0              0        0  35.3                     0.134   29        0
15            7      100              0              0        0  30.0                     0.484   32        1
18            1      103             30             38       83  43.3                     0.183   33        0
43            9      171            110             24      240  45.4                     0.721   54        1
49            7      105              0              0        0   0.0                     0.305   24        0
BloodPressure None
     Pregnancies  Glucose  BloodPressure  SkinThickness  Insulin   BMI  DiabetesPedigreeFunction  Age  Outcome
579            2      197             70             99        0  34.7                     0.575   62        1
SkinThickness None
     Pregnancies  Glucose  BloodPressure  SkinThickness  Insulin   BMI  DiabetesPedigreeFunction  Age  Outcome
8              2      197             70             45      543  30.5                     0.158   53        1
13             1      189             60             23      846  30.1                     0.398   59        1
54             7      150             66             42      342  34.7                     0.718   42        0
111            8      155             62             26      495  34.0                     0.543   46        1
139            5      105             72             29      325  36.9                     0.159   28        0
Insulin None
     Pregnancies  Glucose  BloodPressure  SkinThickness  Insulin   BMI  DiabetesPedigreeFunction  Age  Outcome
9              8      125             96              0        0   0.0                     0.232   54        1
49             7      105              0              0        0   0.0                     0.305   24        0
60             2       84              0              0        0   0.0                     0.304   21        0
81             2       74              0              0        0   0.0                     0.102   22        0
120            0      162             76             56      100  53.2                     0.759   25        1
BMI None
    Pregnancies  Glucose  BloodPressure  SkinThickness  Insulin   BMI  DiabetesPedigreeFunction  Age  Outcome
4             0      137             40             35      168  43.1                     2.288   33        1
12           10      139             80              0        0  27.1                     1.441   57        0
39            4      111             72             47      207  37.1                     1.390   56        1
45            0      180             66             39        0  42.0                     1.893   25        1
58            0      146             82              0        0  40.5                     1.781   44        0
DiabetesPedigreeFunction None
     Pregnancies  Glucose  BloodPressure  SkinThickness  Insulin   BMI  DiabetesPedigreeFunction  Age  Outcome
123            5      132             80              0        0  26.8                     0.186   69        0
363            4      146             78              0        0  38.5                     0.520   67        1
453            2      119              0              0        0  19.6                     0.832   72        0
459            9      134             74             33       60  25.9                     0.460   81        0
489            8      194             80              0        0  26.1                     0.551   67        0
537            0       57             60              0        0  21.7                     0.735   67        0
666            4      145             82             18        0  32.5                     0.235   70        1
674            8       91             82              0        0  35.6                     0.587   68        0
684            5      136             82              0        0   0.0                     0.640   69        0
Age None
 '''

def replace_with_thresholds(dataframe, variable):
    low_limit, up_limit = outlier_thresholds(dataframe, variable)
    dataframe.loc[(dataframe[variable] < low_limit), variable] = low_limit
    dataframe.loc[(dataframe[variable] > up_limit), variable] = up_limit

for col in num_cols:
    replace_with_thresholds(df,col)
for col in num_cols:
    print(col, check_outlier(df, col))
'''
Pregnancies False
Glucose False
BloodPressure False
SkinThickness False
Insulin False
BMI False
DiabetesPedigreeFunction False
Age False
'''

#Adım 6: Eksik gözlem analizi yapınız.
def missing_values_table(dataframe, na_name=False):
    na_columns = [col for col in dataframe.columns if dataframe[col].isnull().sum() > 0]

    n_miss = dataframe[na_columns].isnull().sum().sort_values(ascending=False)
    ratio = (dataframe[na_columns].isnull().sum() / dataframe.shape[0] * 100).sort_values(ascending=False)
    missing_df = pd.concat([n_miss, np.round(ratio, 2)], axis=1, keys=['n_miss', 'ratio'])
    print(missing_df, end="\n")

    if na_name:
        return na_columns

print(missing_values_table(df))
'''
Empty DataFrame
'''
print(df.isnull().values.any())    #False

#Adım 7: Korelasyon analizi yapınız.
print(df.corr())
'''
                          Pregnancies   Glucose  BloodPressure  SkinThickness   Insulin       BMI  DiabetesPedigreeFunction       Age   Outcome
Pregnancies                  1.000000  0.129459       0.141282      -0.081672 -0.073535  0.017683                 -0.033523  0.544341  0.221898
Glucose                      0.129459  1.000000       0.152590       0.057328  0.331357  0.221071                  0.137337  0.263514  0.466581
BloodPressure                0.141282  0.152590       1.000000       0.207371  0.088933  0.281805                  0.041265  0.239528  0.065068
SkinThickness               -0.081672  0.057328       0.207371       1.000000  0.436783  0.392573                  0.183928 -0.113970  0.074752
Insulin                     -0.073535  0.331357       0.088933       0.436783  1.000000  0.197859                  0.185071 -0.042163  0.130548
BMI                          0.017683  0.221071       0.281805       0.392573  0.197859  1.000000                  0.140647  0.036242  0.292695
DiabetesPedigreeFunction    -0.033523  0.137337       0.041265       0.183928  0.185071  0.140647                  1.000000  0.033561  0.173844
Age                          0.544341  0.263514       0.239528      -0.113970 -0.042163  0.036242                  0.033561  1.000000  0.238356
Outcome                      0.221898  0.466581       0.065068       0.074752  0.130548  0.292695                  0.173844  0.238356  1.000000
'''

f, ax = plt.subplots(figsize=[18, 13])
sns.heatmap(df.corr(), annot=True, fmt=".2f", ax=ax, cmap="magma")
ax.set_title("Correlation Matrix", fontsize=20)
plt.show()



#Görev 2 : Feature Engineering

#Adım 1: Eksik ve aykırı değerler için gerekli işlemleri yapınız.
zero_columns = [col for col in df.columns if (df[col].min() == 0 and col not in ["Pregnancies", "Outcome"])]
print(zero_columns)
'''
['SkinThickness', 'Insulin']
'''
for col in zero_columns:
    df[col] = np.where(df[col] == 0, np.nan, df[col])

print(df.isnull().sum())
'''
Pregnancies                   0
Glucose                       0
BloodPressure                 0
SkinThickness               227
Insulin                     374
BMI                           0
DiabetesPedigreeFunction      0
Age                           0
Outcome                       0
dtype: int64
'''
na_columns = missing_values_table(df, na_name=True)
print(na_columns)
'''
               n_miss  ratio
Insulin           374  48.70
SkinThickness     227  29.56
['SkinThickness', 'Insulin']
'''
for col in zero_columns:
    df.loc[df[col].isnull(), col] = df[col].median()

#Adım 2: Yeni değişkenler oluşturunuz.

def missing_vs_target(dataframe, target, na_columns):
    temp_df = dataframe.copy()
    for col in na_columns:
        temp_df[col + '_NA_FLAG'] = np.where(temp_df[col].isnull(), 1, 0)
    na_flags = temp_df.loc[:, temp_df.columns.str.contains("_NA_")].columns
    for col in na_flags:
        print(pd.DataFrame({"TARGET_MEAN": temp_df.groupby(col)[target].mean(),
                            "Count": temp_df.groupby(col)[target].count()}), end="\n\n\n")


missing_vs_target(df, "Outcome", na_columns)
'''
                       TARGET_MEAN  Count
SkinThickness_NA_FLAG                    
0                         0.332717    541
1                         0.387665    227


                 TARGET_MEAN  Count
Insulin_NA_FLAG                    
0                   0.329949    394
1                   0.368984    374

'''

#Adım 2: Yeni değişkenler oluşturunuz.
print(df.head())
'''
   Pregnancies  Glucose  BloodPressure  SkinThickness  Insulin   BMI  DiabetesPedigreeFunction   Age  Outcome
0          6.0    148.0             72           35.0      NaN  33.6                     0.627  50.0        1
1          1.0     85.0             66           29.0      NaN  26.6                     0.351  31.0        0
2          8.0    183.0             64            NaN      NaN  23.3                     0.672  32.0        1
3          1.0     89.0             66           23.0     94.0  28.1                     0.167  21.0        0
4          0.0    137.0             40           35.0    168.0  43.1                     1.200  33.0        1
'''
df.loc[(df["Age"] >= 21) & (df["Age"] < 50), "NEW_AGE"] = "mature"
df.loc[(df["Age"] >= 50), "NEW_AGE"] = "senior"
print(df.head())
'''
   Pregnancies  Glucose  BloodPressure  SkinThickness  Insulin   BMI  DiabetesPedigreeFunction   Age  Outcome NEW_AGE
0          6.0    148.0             72           35.0      NaN  33.6                     0.627  50.0        1  senior
1          1.0     85.0             66           29.0      NaN  26.6                     0.351  31.0        0  mature
2          8.0    183.0             64            NaN      NaN  23.3                     0.672  32.0        1  mature
3          1.0     89.0             66           23.0     94.0  28.1                     0.167  21.0        0  mature
4          0.0    137.0             40           35.0    168.0  43.1                     1.200  33.0        1  mature
'''
df['NEW_BMI'] = pd.cut(x=df['BMI'], bins=[0, 18.5, 24.9, 29.9, 100],labels=["Underweight", "Healthy", "Overweight", "Obese"])
print(df.head())
'''
   Pregnancies  Glucose  BloodPressure  SkinThickness  Insulin   BMI  DiabetesPedigreeFunction   Age  Outcome NEW_AGE     NEW_BMI
0          6.0    148.0             72           35.0      NaN  33.6                     0.627  50.0        1  senior       Obese
1          1.0     85.0             66           29.0      NaN  26.6                     0.351  31.0        0  mature  Overweight
2          8.0    183.0             64            NaN      NaN  23.3                     0.672  32.0        1  mature     Healthy
3          1.0     89.0             66           23.0     94.0  28.1                     0.167  21.0        0  mature  Overweight
4          0.0    137.0             40           35.0    168.0  43.1                     1.200  33.0        1  mature       Obese
'''

df.loc[(df["BMI"] < 18.5) & ((df["Age"] >= 21) & (df["Age"] < 50)), "NEW_AGE_BMI_NOM"] = "underweightmature"
df.loc[(df["BMI"] < 18.5) & (df["Age"] >= 50), "NEW_AGE_BMI_NOM"] = "underweightsenior"
df.loc[((df["BMI"] >= 18.5) & (df["BMI"] < 25)) & ((df["Age"] >= 21) & (df["Age"] < 50)), "NEW_AGE_BMI_NOM"] = "healthymature"
df.loc[((df["BMI"] >= 18.5) & (df["BMI"] < 25)) & (df["Age"] >= 50), "NEW_AGE_BMI_NOM"] = "healthysenior"
df.loc[((df["BMI"] >= 25) & (df["BMI"] < 30)) & ((df["Age"] >= 21) & (df["Age"] < 50)), "NEW_AGE_BMI_NOM"] = "overweightmature"
df.loc[((df["BMI"] >= 25) & (df["BMI"] < 30)) & (df["Age"] >= 50), "NEW_AGE_BMI_NOM"] = "overweightsenior"
df.loc[(df["BMI"] > 18.5) & ((df["Age"] >= 21) & (df["Age"] < 50)), "NEW_AGE_BMI_NOM"] = "obesemature"
df.loc[(df["BMI"] > 18.5) & (df["Age"] >= 50), "NEW_AGE_BMI_NOM"] = "obesesenior"
print(df.head())
'''
 Pregnancies  Glucose  BloodPressure  SkinThickness  Insulin   BMI  DiabetesPedigreeFunction   Age  Outcome NEW_AGE     NEW_BMI NEW_AGE_BMI_NOM
0          6.0    148.0             72           35.0      NaN  33.6                     0.627  50.0        1  senior       Obese     obesesenior
1          1.0     85.0             66           29.0      NaN  26.6                     0.351  31.0        0  mature  Overweight     obesemature
2          8.0    183.0             64            NaN      NaN  23.3                     0.672  32.0        1  mature     Healthy     obesemature
3          1.0     89.0             66           23.0     94.0  28.1                     0.167  21.0        0  mature  Overweight     obesemature
4          0.0    137.0             40           35.0    168.0  43.1                     1.200  33.0        1  mature       Obese     obesemature
'''
df.loc[(df["Glucose"] < 70) & ((df["Age"] >= 21) & (df["Age"] < 50)), "NEW_AGE_GLUCOSE_NOM"] = "lowmature"
df.loc[(df["Glucose"] < 70) & (df["Age"] >= 50), "NEW_AGE_GLUCOSE_NOM"] = "lowsenior"
df.loc[((df["Glucose"] >= 70) & (df["Glucose"] < 100)) & ((df["Age"] >= 21) & (df["Age"] < 50)), "NEW_AGE_GLUCOSE_NOM"] = "normalmature"
df.loc[((df["Glucose"] >= 70) & (df["Glucose"] < 100)) & (df["Age"] >= 50), "NEW_AGE_GLUCOSE_NOM"] = "normalsenior"
df.loc[((df["Glucose"] >= 100) & (df["Glucose"] <= 125)) & ((df["Age"] >= 21) & (df["Age"] < 50)), "NEW_AGE_GLUCOSE_NOM"] = "hiddenmature"
df.loc[((df["Glucose"] >= 100) & (df["Glucose"] <= 125)) & (df["Age"] >= 50), "NEW_AGE_GLUCOSE_NOM"] = "hiddensenior"
df.loc[(df["Glucose"] > 125) & ((df["Age"] >= 21) & (df["Age"] < 50)), "NEW_AGE_GLUCOSE_NOM"] = "highmature"
df.loc[(df["Glucose"] > 125) & (df["Age"] >= 50), "NEW_AGE_GLUCOSE_NOM"] = "highsenior"
print(df.head())
'''
   Pregnancies  Glucose  BloodPressure  SkinThickness  Insulin   BMI  DiabetesPedigreeFunction   Age  Outcome NEW_AGE     NEW_BMI NEW_AGE_BMI_NOM NEW_AGE_GLUCOSE_NOM
0          6.0    148.0             72           35.0    125.0  33.6                     0.627  50.0        1  senior       Obese     obesesenior          highsenior
1          1.0     85.0             66           29.0    125.0  26.6                     0.351  31.0        0  mature  Overweight     obesemature        normalmature
2          8.0    183.0             64           29.0    125.0  23.3                     0.672  32.0        1  mature     Healthy     obesemature          highmature
3          1.0     89.0             66           23.0     94.0  28.1                     0.167  21.0        0  mature  Overweight     obesemature        normalmature
4          0.0    137.0             40           35.0    168.0  43.1                     1.200  33.0        1  mature       Obese     obesemature          highmature

'''

#Adım 3: Encoding işlemlerini gerçekleştiriniz.
cat_cols, num_cols, cat_but_car = grab_col_names(df)
'''
Observations: 768
Variables: 12
cat_cols: 4
num_cols: 8
cat_but_car: 0
num_but_cat: 2
'''
def label_encoder(dataframe, binary_col):
    labelencoder = LabelEncoder()
    dataframe[binary_col] = labelencoder.fit_transform(dataframe[binary_col])
    return dataframe
binary_cols = [col for col in df.columns if df[col].dtypes == "O" and df[col].nunique() == 2]
for col in binary_cols:
    df = label_encoder(df, col)
print(df.head())
'''
   Pregnancies  Glucose  BloodPressure  SkinThickness  Insulin   BMI  DiabetesPedigreeFunction   Age  Outcome  NEW_AGE     NEW_BMI NEW_AGE_BMI_NOM NEW_AGE_GLUCOSE_NOM
0          6.0    148.0             72           35.0    125.0  33.6                     0.627  50.0        1        1       Obese     obesesenior          highsenior
1          1.0     85.0             66           29.0    125.0  26.6                     0.351  31.0        0        0  Overweight     obesemature        normalmature
2          8.0    183.0             64           29.0    125.0  23.3                     0.672  32.0        1        0     Healthy     obesemature          highmature
3          1.0     89.0             66           23.0     94.0  28.1                     0.167  21.0        0        0  Overweight     obesemature        normalmature
4          0.0    137.0             40           35.0    168.0  43.1                     1.200  33.0        1        0       Obese     obesemature          highmature
'''
print(binary_cols)     #['NEW_AGE'] senior->1 mature->0

cat_cols = [col for col in cat_cols if col not in binary_cols and col not in ["Outcome"]]
def one_hot_encoder(dataframe, categorical_cols, drop_first=False):
    dataframe = pd.get_dummies(dataframe, columns=categorical_cols, drop_first=drop_first)
    return dataframe
df = one_hot_encoder(df, cat_cols, drop_first=True)
print(df.head())
'''
   Pregnancies  Glucose  BloodPressure  SkinThickness  Insulin   BMI  DiabetesPedigreeFunction   Age  Outcome  NEW_AGE  NEW_AGE_BMI_NOM_obesesenior  NEW_AGE_BMI_NOM_underweightmature  NEW_AGE_BMI_NOM_underweightsenior  NEW_AGE_GLUCOSE_NOM_hiddensenior  NEW_AGE_GLUCOSE_NOM_highmature  NEW_AGE_GLUCOSE_NOM_highsenior  NEW_AGE_GLUCOSE_NOM_lowmature  NEW_AGE_GLUCOSE_NOM_lowsenior  NEW_AGE_GLUCOSE_NOM_normalmature  NEW_AGE_GLUCOSE_NOM_normalsenior  NEW_BMI_Healthy  NEW_BMI_Overweight  NEW_BMI_Obese
0          6.0    148.0             72           35.0    125.0  33.6                     0.627  50.0        1        1                            1                                  0                                  0                                 0                               0                               1                              0                              0                                 0                                 0                0                   0              1
1          1.0     85.0             66           29.0    125.0  26.6                     0.351  31.0        0        0                            0                                  0                                  0                                 0                               0                               0                              0                              0                                 1                                 0                0                   1              0
2          8.0    183.0             64           29.0    125.0  23.3                     0.672  32.0        1        0                            0                                  0                                  0                                 0                               1                               0                              0                              0                                 0                                 0                1                   0              0
3          1.0     89.0             66           23.0     94.0  28.1                     0.167  21.0        0        0                            0                                  0                                  0                                 0                               0                               0                              0                              0                                 1                                 0                0                   1              0
4          0.0    137.0             40           35.0    168.0  43.1                     1.200  33.0        1        0                            0                                  0                                  0                                 0                               1                               0                              0                              0                                 0                                 0                0                   0              1
 '''

#Adım 4: Numerik değişkenler için standartlaştırma yapınız.
scaler = StandardScaler()
df[num_cols] = scaler.fit_transform(df[num_cols])
print(df.head())
'''
   Pregnancies   Glucose  BloodPressure  SkinThickness   Insulin       BMI  DiabetesPedigreeFunction       Age  Outcome  NEW_AGE  NEW_AGE_BMI_NOM_obesesenior  NEW_AGE_BMI_NOM_underweightmature  NEW_AGE_BMI_NOM_underweightsenior  NEW_AGE_GLUCOSE_NOM_hiddensenior  NEW_AGE_GLUCOSE_NOM_highmature  NEW_AGE_GLUCOSE_NOM_highsenior  NEW_AGE_GLUCOSE_NOM_lowmature  NEW_AGE_GLUCOSE_NOM_lowsenior  NEW_AGE_GLUCOSE_NOM_normalmature  NEW_AGE_GLUCOSE_NOM_normalsenior  NEW_BMI_Healthy  NEW_BMI_Overweight  NEW_BMI_Obese
0     0.647150  0.861926       0.092691       0.686889 -0.156977  0.209359                  0.588927  1.445691        1        1                            1                                  0                                  0                                 0                               0                               1                              0                              0                                 0                                 0                0                   0              1
1    -0.848970 -1.159433      -0.330201      -0.009674 -0.156977 -0.784254                 -0.378101 -0.189304        0        0                            0                                  0                                  0                                 0                               0                               0                              0                              0                                 1                                 0                0                   1              0
2     1.245598  1.984903      -0.471166      -0.009674 -0.156977 -1.252672                  0.746595 -0.103252        1        0                            0                                  0                                  0                                 0                               1                               0                              0                              0                                 0                                 0                1                   0              0
3    -0.848970 -1.031093      -0.330201      -0.706238 -0.667869 -0.571337                 -1.022787 -1.049828        0        0                            0                                  0                                  0                                 0                               0                               0                              0                              0                                 1                                 0                0                   1              0
4    -1.148194  0.508990      -2.162737       0.686889  0.551680  1.557835                  2.596563 -0.017199        1        0                            0                                  0                                  0                                 0                               1                               0                              0                              0                                 0                                 0                0                   0              1
'''

#Adım 5: Model oluşturunuz

y = df["Outcome"]
X = df.drop("Outcome", axis=1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=13)
rf_model = RandomForestClassifier(random_state=46).fit(X_train, y_train)
y_pred = rf_model.predict(X_test)
print(f"Accuracy: {round(accuracy_score(y_pred, y_test), 2)}")
print(f"Recall: {round(recall_score(y_pred,y_test),3)}")
print(f"Precision: {round(precision_score(y_pred,y_test), 2)}")
print(f"F1: {round(f1_score(y_pred,y_test), 2)}")
print(f"Auc: {round(roc_auc_score(y_pred,y_test), 2)}")
'''
Accuracy: 0.76
Recall: 0.718
Precision: 0.59
F1: 0.65
Auc: 0.75
'''
def plot_importance(model, features, num=len(X), save=False):
    feature_imp = pd.DataFrame({'Value': model.feature_importances_, 'Feature': features.columns})
    print(feature_imp.sort_values("Value",ascending=False))
    plt.figure(figsize=(10, 10))
    sns.set(font_scale=1)
    sns.barplot(x="Value", y="Feature", data=feature_imp.sort_values(by="Value",
                                                                     ascending=False)[0:num])
    plt.title('Features')
    plt.tight_layout()
    plt.show(block=True)
    if save:
        plt.savefig('importance.png')

plot_importance(rf_model, X)
'''
       Value                            Feature
1   0.200011                            Glucose
7   0.125783                                Age
5   0.124035                                BMI
6   0.103365           DiabetesPedigreeFunction
0   0.082187                        Pregnancies
2   0.080579                      BloodPressure
4   0.067745                            Insulin
3   0.066302                      SkinThickness
13  0.048212     NEW_AGE_GLUCOSE_NOM_highmature
17  0.029233   NEW_AGE_GLUCOSE_NOM_normalmature
21  0.027131                      NEW_BMI_Obese
19  0.010983                    NEW_BMI_Healthy
20  0.007578                 NEW_BMI_Overweight
8   0.005791                            NEW_AGE
9   0.005663        NEW_AGE_BMI_NOM_obesesenior
14  0.005539     NEW_AGE_GLUCOSE_NOM_highsenior
12  0.004403   NEW_AGE_GLUCOSE_NOM_hiddensenior
18  0.002039   NEW_AGE_GLUCOSE_NOM_normalsenior
15  0.001861      NEW_AGE_GLUCOSE_NOM_lowmature
11  0.000952  NEW_AGE_BMI_NOM_underweightsenior
10  0.000610  NEW_AGE_BMI_NOM_underweightmature
16  0.000000      NEW_AGE_GLUCOSE_NOM_lowsenior

'''
