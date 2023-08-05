################## Makine Öğrenmesi ile Maaş Tahmini ########################

#İş Problemi
#Maaş bilgileri ve 1986 yılına ait kariyer istatistikleri paylaşılan beyzbol
#oyuncularının maaş tahminleri için bir makine öğrenmesi modeli geliştiriniz.

#Veri Seti Hikayesi
#Bu veri seti orijinal olarak Carnegie Mellon Üniversitesi'nde bulunan
# StatLib kütüphanesinden alınmıştır. Veri seti 1988 ASA Grafik Bölümü
# Poster Oturumu'nda kullanılan verilerin bir parçasıdır. Maaş verileri
# orijinal olarak Sports Illustrated, 20 Nisan 1987'den alınmıştır. 1986 ve
# kariyer istatistikleri, Collier Books, Macmillan Publishing Company,
# New York tarafından yayınlanan 1987 Beyzbol Ansiklopedisi Güncellemesinden elde edilmiştir.

#AtBat	1986-1987 sezonunda bir beyzbol sopası ile topa yapılan vuruş sayısı
#Hits 1986-1987 sezonundaki isabet sayısı
#HmRun 1986-1987 sezonundaki en değerli vuruş sayısı
#Runs	1986-1987 sezonunda takımına kazandırdığı sayı
#RBI	Bir vurucunun vuruş yaptığında koşu yaptırdığı oyuncu sayısı
#Walks	Karşı oyuncuya yaptırılan hata sayısı
#Years	Oyuncunun major liginde oynama süresi (sene)
#CAtBat	Oyuncunun kariyeri boyunca topa vurma sayısı
#Chits	Oyuncunun kariyeri boyunca yaptığı isabetli vuruş sayısı
#CHmRun	Oyuncunun kariyeri boyunca yaptığı en değerli sayısı
#CRuns	Oyuncunun kariyeri boyunca takımına kazandırdığı sayı
#CRBI	Oyuncunun kariyeri boyunca koşu yaptırdırdığı oyuncu sayısı
#CWalks	Oyuncun kariyeri boyunca karşı oyuncuya yaptırdığı hata sayısı
#League	Oyuncunun sezon sonuna kadar oynadığı ligi gösteren A ve N seviyelerine sahip bir faktör
#Division 1986 sonunda oyuncunun oynadığı pozisyonu gösteren E ve W seviyelerine sahip bir faktör
#PutOuts	Oyun içinde takım arkadaşınla yardımlaşma
#Assists	1986-1987 sezonunda oyuncunun yaptığı asist sayısı
#Errors	1986-1987 sezonundaki oyuncunun hata sayısı
#Salary	Oyuncunun 1986-1987 sezonunda aldığı maaş (bin üzerinden)
#NewLeague	1987 sezonunun başında oyuncunun ligini gösteren A ve N seviyelerine sahip bir faktör

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LinearRegression, Ridge, RidgeCV, ElasticNet, Lasso, LassoCV
from sklearn.metrics import mean_squared_error
from sklearn import preprocessing
from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import LocalOutlierFactor


pd.set_option('display.max_columns', None)
pd.set_option('display.width', 170)
pd.set_option('display.max_rows', 20)
pd.set_option('display.float_format', lambda x: '%.2f' % x)

#Import DataSet

df = pd.read_csv("hitters.csv")

print(df.head())
'''
   AtBat  Hits  HmRun  Runs  RBI  Walks  Years  CAtBat  CHits  CHmRun  CRuns  CRBI  CWalks League Division  PutOuts  Assists  Errors  Salary NewLeague
0    293    66      1    30   29     14      1     293     66       1     30    29      14      A        E      446       33      20     NaN         A
1    315    81      7    24   38     39     14    3449    835      69    321   414     375      N        W      632       43      10  475.00         N
2    479   130     18    66   72     76      3    1624    457      63    224   266     263      A        W      880       82      14  480.00         A
3    496   141     20    65   78     37     11    5628   1575     225    828   838     354      N        E      200       11       3  500.00         N
4    321    87     10    39   42     30      2     396    101      12     48    46      33      N        E      805       40       4   91.50         N
'''

#Exploratory Data Analysis

print(df.shape)     #(322, 20)

print(df.info())
'''
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 322 entries, 0 to 321
Data columns (total 20 columns):
 #   Column     Non-Null Count  Dtype  
---  ------     --------------  -----  
 0   AtBat      322 non-null    int64  
 1   Hits       322 non-null    int64  
 2   HmRun      322 non-null    int64  
 3   Runs       322 non-null    int64  
 4   RBI        322 non-null    int64  
 5   Walks      322 non-null    int64  
 6   Years      322 non-null    int64  
 7   CAtBat     322 non-null    int64  
 8   CHits      322 non-null    int64  
 9   CHmRun     322 non-null    int64  
 10  CRuns      322 non-null    int64  
 11  CRBI       322 non-null    int64  
 12  CWalks     322 non-null    int64  
 13  League     322 non-null    object 
 14  Division   322 non-null    object 
 15  PutOuts    322 non-null    int64  
 16  Assists    322 non-null    int64  
 17  Errors     322 non-null    int64  
 18  Salary     263 non-null    float64
 19  NewLeague  322 non-null    object 
dtypes: float64(1), int64(16), object(3)
memory usage: 50.4+ KB
'''
print(df.dtypes)
'''
AtBat          int64
Hits           int64
HmRun          int64
Runs           int64
RBI            int64
Walks          int64
Years          int64
CAtBat         int64
CHits          int64
CHmRun         int64
CRuns          int64
CRBI           int64
CWalks         int64
League        object
Division      object
PutOuts        int64
Assists        int64
Errors         int64
Salary       float64
NewLeague     object
dtype: object
'''
print(df.isnull().sum())
'''
AtBat         0
Hits          0
HmRun         0
Runs          0
RBI           0
Walks         0
Years         0
CAtBat        0
CHits         0
CHmRun        0
CRuns         0
CRBI          0
CWalks        0
League        0
Division      0
PutOuts       0
Assists       0
Errors        0
Salary       59
NewLeague     0
dtype: int64
'''

print(df.describe().T)
'''
         count    mean     std   min    25%     50%     75%      max
AtBat   322.00  380.93  153.40 16.00 255.25  379.50  512.00   687.00
Hits    322.00  101.02   46.45  1.00  64.00   96.00  137.00   238.00
HmRun   322.00   10.77    8.71  0.00   4.00    8.00   16.00    40.00
Runs    322.00   50.91   26.02  0.00  30.25   48.00   69.00   130.00
RBI     322.00   48.03   26.17  0.00  28.00   44.00   64.75   121.00
Walks   322.00   38.74   21.64  0.00  22.00   35.00   53.00   105.00
Years   322.00    7.44    4.93  1.00   4.00    6.00   11.00    24.00
CAtBat  322.00 2648.68 2324.21 19.00 816.75 1928.00 3924.25 14053.00
CHits   322.00  717.57  654.47  4.00 209.00  508.00 1059.25  4256.00
CHmRun  322.00   69.49   86.27  0.00  14.00   37.50   90.00   548.00
CRuns   322.00  358.80  334.11  1.00 100.25  247.00  526.25  2165.00
CRBI    322.00  330.12  333.22  0.00  88.75  220.50  426.25  1659.00
CWalks  322.00  260.24  267.06  0.00  67.25  170.50  339.25  1566.00
PutOuts 322.00  288.94  280.70  0.00 109.25  212.00  325.00  1378.00
Assists 322.00  106.91  136.85  0.00   7.00   39.50  166.00   492.00
Errors  322.00    8.04    6.37  0.00   3.00    6.00   11.00    32.00
Salary  263.00  535.93  451.12 67.50 190.00  425.00  750.00  2460.00
'''

# Analysis of Target Variable

print(df["Salary"].describe())
'''
count    263.00
mean     535.93
std      451.12
min       67.50
25%      190.00
50%      425.00
75%      750.00
max     2460.00
Name: Salary, dtype: float64
'''

sns.boxplot(df["Salary"])
plt.show()

#Determination Variables

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
Observations: 322
Variables: 20
cat_cols: 3
num_cols: 17
cat_but_car: 0
num_but_cat: 0
'''
print(cat_cols)
'''
['League', 'Division', 'NewLeague']
'''
print(num_cols)
'''
['AtBat', 'Hits', 'HmRun', 'Runs', 'RBI', 'Walks', 'Years', 'CAtBat', 'CHits', 'CHmRun', 
'CRuns', 'CRBI', 'CWalks', 'PutOuts', 'Assists', 'Errors', 'Salary']
'''

def cat_summary(dataframe, col_name, plot=False):
    print(pd.DataFrame({col_name: dataframe[col_name].value_counts(),
                        "Ratio": 100 * dataframe[col_name].value_counts() / len(dataframe)}))
    if plot:
        sns.countplot(x=dataframe[col_name], data=dataframe)
        plt.show()

for col in cat_cols:
     cat_summary(df, col)
'''
   League  Ratio
A     175  54.35
N     147  45.65
   Division  Ratio
W       165  51.24
E       157  48.76
   NewLeague  Ratio
A        176  54.66
N        146  45.34
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
count   322.00
mean    380.93
std     153.40
min      16.00
25%     255.25
50%     379.50
75%     512.00
max     687.00
Name: AtBat, dtype: float64
count   322.00
mean    101.02
std      46.45
min       1.00
25%      64.00
50%      96.00
75%     137.00
max     238.00
Name: Hits, dtype: float64
count   322.00
mean     10.77
std       8.71
min       0.00
25%       4.00
50%       8.00
75%      16.00
max      40.00
Name: HmRun, dtype: float64
count   322.00
mean     50.91
std      26.02
min       0.00
25%      30.25
50%      48.00
75%      69.00
max     130.00
Name: Runs, dtype: float64
count   322.00
mean     48.03
std      26.17
min       0.00
25%      28.00
50%      44.00
75%      64.75
max     121.00
Name: RBI, dtype: float64
count   322.00
mean     38.74
std      21.64
min       0.00
25%      22.00
50%      35.00
75%      53.00
max     105.00
Name: Walks, dtype: float64
count   322.00
mean      7.44
std       4.93
min       1.00
25%       4.00
50%       6.00
75%      11.00
max      24.00
Name: Years, dtype: float64
count     322.00
mean     2648.68
std      2324.21
min        19.00
25%       816.75
50%      1928.00
75%      3924.25
max     14053.00
Name: CAtBat, dtype: float64
count    322.00
mean     717.57
std      654.47
min        4.00
25%      209.00
50%      508.00
75%     1059.25
max     4256.00
Name: CHits, dtype: float64
count   322.00
mean     69.49
std      86.27
min       0.00
25%      14.00
50%      37.50
75%      90.00
max     548.00
Name: CHmRun, dtype: float64
count    322.00
mean     358.80
std      334.11
min        1.00
25%      100.25
50%      247.00
75%      526.25
max     2165.00
Name: CRuns, dtype: float64
count    322.00
mean     330.12
std      333.22
min        0.00
25%       88.75
50%      220.50
75%      426.25
max     1659.00
Name: CRBI, dtype: float64
count    322.00
mean     260.24
std      267.06
min        0.00
25%       67.25
50%      170.50
75%      339.25
max     1566.00
Name: CWalks, dtype: float64
count    322.00
mean     288.94
std      280.70
min        0.00
25%      109.25
50%      212.00
75%      325.00
max     1378.00
Name: PutOuts, dtype: float64
count   322.00
mean    106.91
std     136.85
min       0.00
25%       7.00
50%      39.50
75%     166.00
max     492.00
Name: Assists, dtype: float64
count   322.00
mean      8.04
std       6.37
min       0.00
25%       3.00
50%       6.00
75%      11.00
max      32.00
Name: Errors, dtype: float64
count    263.00
mean     535.93
std      451.12
min       67.50
25%      190.00
50%      425.00
75%      750.00
max     2460.00
Name: Salary, dtype: float64
'''

# Analysis of Outliers

def outlier_thresholds(dataframe, col_name, q1=0.10, q3=0.90):
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
AtBat False
Hits False
HmRun False
Runs False
RBI False
Walks False
Years False
CAtBat False
CHits True
CHmRun True
CRuns False
CRBI False
CWalks True
PutOuts False
Assists False
Errors False
Salary False
'''
print(df.isnull().values.any())   #True

print((df[df.columns] == 0).sum())
'''
AtBat         0
Hits          0
HmRun        18
Runs          1
RBI           3
Walks         2
Years         0
CAtBat        0
CHits         0
CHmRun        7
CRuns         0
CRBI          1
CWalks        1
League        0
Division      0
PutOuts      15
Assists      19
Errors       21
Salary        0
NewLeague     0
dtype: int64
'''

def replace_with_thresholds(dataframe, variable, q1=0.10, q3=0.90):
    low_limit, up_limit = outlier_thresholds(dataframe, variable, q1=0.10, q3=0.90)
    dataframe.loc[(dataframe[variable] < low_limit), variable] = low_limit
    dataframe.loc[(dataframe[variable] > up_limit), variable] = up_limit
for col in num_cols:
    print(col, check_outlier(df, col))
    if check_outlier(df, col):
        replace_with_thresholds(df, col)
'''
AtBat False
Hits False
HmRun False
Runs False
RBI False
Walks False
Years False
CAtBat False
CHits True
CHmRun True
CRuns False
CRBI False
CWalks True
PutOuts False
Assists False
Errors False
Salary False
'''

#Analysis of Missing Values

def missing_values_table(dataframe, na_name=False):
    na_columns = [col for col in dataframe.columns if dataframe[col].isnull().sum() > 0]
    n_miss = dataframe[na_columns].isnull().sum().sort_values(ascending=False)
    ratio = (dataframe[na_columns].isnull().sum() / dataframe.shape[0] * 100).sort_values(ascending=False)
    missing_df = pd.concat([n_miss, np.round(ratio, 2)], axis=1, keys=['n_miss', 'ratio'])
    print(missing_df, end="\n")
    if na_name:
        return na_columns

na_columns = missing_values_table(df, na_name=True)
'''
        n_miss  ratio
Salary      59  18.32
'''

#Analysis of Correlation

print(df.corr())
'''
         AtBat  Hits  HmRun  Runs  RBI  Walks  Years  CAtBat  CHits  CHmRun  CRuns  CRBI  CWalks  PutOuts  Assists  Errors  Salary
AtBat     1.00  0.97   0.59  0.91 0.82   0.67   0.05    0.24   0.26    0.24   0.27  0.24    0.17     0.32     0.35    0.35    0.39
Hits      0.97  1.00   0.56  0.92 0.81   0.64   0.04    0.23   0.26    0.21   0.26  0.23    0.15     0.31     0.32    0.31    0.44
HmRun     0.59  0.56   1.00  0.65 0.86   0.48   0.12    0.22   0.22    0.50   0.26  0.35    0.23     0.28    -0.11    0.04    0.34
Runs      0.91  0.92   0.65  1.00 0.80   0.73   0.00    0.19   0.21    0.23   0.25  0.21    0.18     0.28     0.22    0.24    0.42
RBI       0.82  0.81   0.86  0.80 1.00   0.62   0.15    0.29   0.31    0.45   0.32  0.39    0.25     0.34     0.11    0.19    0.45
Walks     0.67  0.64   0.48  0.73 0.62   1.00   0.14    0.28   0.28    0.33   0.34  0.31    0.42     0.30     0.15    0.13    0.44
Years     0.05  0.04   0.12  0.00 0.15   0.14   1.00    0.92   0.91    0.73   0.88  0.87    0.84    -0.00    -0.08   -0.16    0.40
CAtBat    0.24  0.23   0.22  0.19 0.29   0.28   0.92    1.00   1.00    0.80   0.98  0.95    0.91     0.06     0.00   -0.07    0.53
CHits     0.26  0.26   0.22  0.21 0.31   0.28   0.91    1.00   1.00    0.79   0.98  0.95    0.89     0.08    -0.00   -0.06    0.55
CHmRun    0.24  0.21   0.50  0.23 0.45   0.33   0.73    0.80   0.79    1.00   0.82  0.93    0.80     0.12    -0.16   -0.14    0.53
CRuns     0.27  0.26   0.26  0.25 0.32   0.34   0.88    0.98   0.98    0.82   1.00  0.94    0.93     0.06    -0.02   -0.08    0.56
CRBI      0.24  0.23   0.35  0.21 0.39   0.31   0.87    0.95   0.95    0.93   0.94  1.00    0.88     0.11    -0.08   -0.10    0.57
CWalks    0.17  0.15   0.23  0.18 0.25   0.42   0.84    0.91   0.89    0.80   0.93  0.88    1.00     0.06    -0.04   -0.12    0.49
PutOuts   0.32  0.31   0.28  0.28 0.34   0.30  -0.00    0.06   0.08    0.12   0.06  0.11    0.06     1.00    -0.03    0.11    0.30
Assists   0.35  0.32  -0.11  0.22 0.11   0.15  -0.08    0.00  -0.00   -0.16  -0.02 -0.08   -0.04    -0.03     1.00    0.71    0.03
Errors    0.35  0.31   0.04  0.24 0.19   0.13  -0.16   -0.07  -0.06   -0.14  -0.08 -0.10   -0.12     0.11     0.71    1.00   -0.01
Salary    0.39  0.44   0.34  0.42 0.45   0.44   0.40    0.53   0.55    0.53   0.56  0.57    0.49     0.30     0.03   -0.01    1.00
'''

f, ax = plt.subplots(figsize=[18, 13])
sns.heatmap(df.corr(), annot=True, fmt=".2f", ax=ax, cmap="magma")
ax.set_title("Correlation Matrix", fontsize=20)
plt.show()

#Data PreProcessing
