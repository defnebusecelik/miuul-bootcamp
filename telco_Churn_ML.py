######################## Telco Churn Feature Engineering #######################

#İş Problemi
#Şirketi terk edecek müşterileri tahmin edebilecek bir makine öğrenmesi modeli
#geliştirilmesi istenmektedir.

#Veri Seti
#Telco müşteri kaybı verileri, üçüncü çeyrekte Kaliforniya'daki 7043 müşteriye ev telefonu ve İnternet hizmetleri sağlayan hayali
#bir telekom şirketi hakkında bilgi içerir. Hangi müşterilerin hizmetlerinden ayrıldığını, kaldığını veya hizmete kaydolduğunu
#gösterir.
#CustomerId : Customer Id
#Gender : Gender
#SeniorCitizen : Whether the customer is old (1, 0)
#Partner : Whether the customer has a partner (Yes, No)
#Dependents : Whether the customer has dependents (Yes, No)
#tenure : Number of months the customer has stayed with the company
#PhoneService : Whether the customer has telephone service (Yes, No)
#MultipleLines : Whether the customer has more than one line (Yes, No, No phone service)
#InternetService : Customer's internet service provider (DSL, Fiber optic, No)
#OnlineSecurity : Whether the customer has online security (Yes, No, no Internet service)
#OnlineBackup : Whether the customer has an online backup (Yes, No, no Internet service)
#DeviceProtection : Whether the customer has device protection (Yes, No, no Internet service)
#TechSupport : Whether the customer has technical support (Yes, No, no Internet service)
#StreamingTV : Whether the customer has a TV broadcast (Yes, No, no Internet service)
#StreamingMovies : Whether the client is streaming movies (Yes, No, no Internet service)
#Contract : Customer's contract duration (Month to month, One year, Two years)
#PaperlessBilling : Whether the customer has a paperless invoice (Yes, No)
#PaymentMethod : Customer's payment method (Electronic check, Postal check, Bank transfer (automatic), Credit card (automatic))
#MonthlyCharges : Amount collected from the customer on a monthly basis
# 3TotalCharges : Total amount charged from customer
#Churn : Whether the customer is using (Yes or No)

#Görev 1 : Keşifçi Veri Analizi

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split, cross_validate, GridSearchCV
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from sklearn.svm import SVC

pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_rows', 30)

#Adım 1: Genel resmi inceleyiniz.
df = pd.read_csv("Telco-Customer-Churn.csv")
print(df.head())
'''
   customerID  gender  SeniorCitizen Partner Dependents  tenure PhoneService     MultipleLines InternetService OnlineSecurity OnlineBackup DeviceProtection TechSupport StreamingTV StreamingMovies        Contract PaperlessBilling              PaymentMethod  MonthlyCharges TotalCharges Churn
0  7590-VHVEG  Female              0     Yes         No       1           No  No phone service             DSL             No          Yes               No          No          No              No  Month-to-month              Yes           Electronic check           29.85        29.85    No
1  5575-GNVDE    Male              0      No         No      34          Yes                No             DSL            Yes           No              Yes          No          No              No        One year               No               Mailed check           56.95       1889.5    No
2  3668-QPYBK    Male              0      No         No       2          Yes                No             DSL            Yes          Yes               No          No          No              No  Month-to-month              Yes               Mailed check           53.85       108.15   Yes
3  7795-CFOCW    Male              0      No         No      45           No  No phone service             DSL            Yes           No              Yes         Yes          No              No        One year               No  Bank transfer (automatic)           42.30      1840.75    No
4  9237-HQITU  Female              0      No         No       2          Yes                No     Fiber optic             No           No               No          No          No              No  Month-to-month              Yes           Electronic check           70.70       151.65   Yes
'''
print(df.tail())
'''
      customerID  gender  SeniorCitizen Partner Dependents  tenure PhoneService     MultipleLines InternetService OnlineSecurity OnlineBackup DeviceProtection TechSupport StreamingTV StreamingMovies        Contract PaperlessBilling              PaymentMethod  MonthlyCharges TotalCharges Churn
7038  6840-RESVB    Male              0     Yes        Yes      24          Yes               Yes             DSL            Yes           No              Yes         Yes         Yes             Yes        One year              Yes               Mailed check           84.80       1990.5    No
7039  2234-XADUH  Female              0     Yes        Yes      72          Yes               Yes     Fiber optic             No          Yes              Yes          No         Yes             Yes        One year              Yes    Credit card (automatic)          103.20       7362.9    No
7040  4801-JZAZL  Female              0     Yes        Yes      11           No  No phone service             DSL            Yes           No               No          No          No              No  Month-to-month              Yes           Electronic check           29.60       346.45    No
7041  8361-LTMKD    Male              1     Yes         No       4          Yes               Yes     Fiber optic             No           No               No          No          No              No  Month-to-month              Yes               Mailed check           74.40        306.6   Yes
7042  3186-AJIEK    Male              0      No         No      66          Yes                No     Fiber optic            Yes           No              Yes         Yes         Yes             Yes        Two year              Yes  Bank transfer (automatic)          105.65       6844.5    No
'''
print(df.shape)   #(7043, 21)
print(df.dtypes)
'''
customerID           object
gender               object
SeniorCitizen         int64
Partner              object
Dependents           object
tenure                int64
PhoneService         object
MultipleLines        object
InternetService      object
OnlineSecurity       object
OnlineBackup         object
DeviceProtection     object
TechSupport          object
StreamingTV          object
StreamingMovies      object
Contract             object
PaperlessBilling     object
PaymentMethod        object
MonthlyCharges      float64
TotalCharges         object
Churn                object
dtype: object
'''
print(df.isnull().sum())
'''
customerID          0
gender              0
SeniorCitizen       0
Partner             0
Dependents          0
tenure              0
PhoneService        0
MultipleLines       0
InternetService     0
OnlineSecurity      0
OnlineBackup        0
DeviceProtection    0
TechSupport         0
StreamingTV         0
StreamingMovies     0
Contract            0
PaperlessBilling    0
PaymentMethod       0
MonthlyCharges      0
TotalCharges        0
Churn               0
dtype: int64
'''
print(df.info())
'''
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 7043 entries, 0 to 7042
Data columns (total 21 columns):
 #   Column            Non-Null Count  Dtype  
---  ------            --------------  -----  
 0   customerID        7043 non-null   object 
 1   gender            7043 non-null   object 
 2   SeniorCitizen     7043 non-null   int64  
 3   Partner           7043 non-null   object 
 4   Dependents        7043 non-null   object 
 5   tenure            7043 non-null   int64  
 6   PhoneService      7043 non-null   object 
 7   MultipleLines     7043 non-null   object 
 8   InternetService   7043 non-null   object 
 9   OnlineSecurity    7043 non-null   object 
 10  OnlineBackup      7043 non-null   object 
 11  DeviceProtection  7043 non-null   object 
 12  TechSupport       7043 non-null   object 
 13  StreamingTV       7043 non-null   object 
 14  StreamingMovies   7043 non-null   object 
 15  Contract          7043 non-null   object 
 16  PaperlessBilling  7043 non-null   object 
 17  PaymentMethod     7043 non-null   object 
 18  MonthlyCharges    7043 non-null   float64
 19  TotalCharges      7043 non-null   object 
 20  Churn             7043 non-null   object 
dtypes: float64(1), int64(2), object(18)
memory usage: 1.1+ MB
'''
print(df.describe().T)
'''
                 count       mean        std    min   25%    50%    75%     max
SeniorCitizen   7043.0   0.162147   0.368612   0.00   0.0   0.00   0.00    1.00
tenure          7043.0  32.371149  24.559481   0.00   9.0  29.00  55.00   72.00
MonthlyCharges  7043.0  64.761692  30.090047  18.25  35.5  70.35  89.85  118.75
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
Observations: 7043
Variables: 21
cat_cols: 17
num_cols: 2
cat_but_car: 2
num_but_cat: 1
'''

print(cat_cols)
'''
['gender', 'Partner', 'Dependents', 'PhoneService', 'MultipleLines',
 'InternetService', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 
 'TechSupport', 'StreamingTV', 'StreamingMovies', 'Contract', 'PaperlessBilling', 
 'PaymentMethod', 'Churn', 'SeniorCitizen']
'''
print(num_cols)
'''
['tenure', 'MonthlyCharges']
'''
print(cat_but_car)
'''
['customerID', 'TotalCharges']
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
        gender     Ratio
Male      3555  50.47565
Female    3488  49.52435
     Partner     Ratio
No      3641  51.69672
Yes     3402  48.30328
     Dependents      Ratio
No         4933  70.041176
Yes        2110  29.958824
     PhoneService      Ratio
Yes          6361  90.316626
No            682   9.683374
                  MultipleLines      Ratio
No                         3390  48.132898
Yes                        2971  42.183729
No phone service            682   9.683374
             InternetService      Ratio
Fiber optic             3096  43.958540
DSL                     2421  34.374556
No                      1526  21.666903
                     OnlineSecurity      Ratio
No                             3498  49.666335
Yes                            2019  28.666761
No internet service            1526  21.666903
                     OnlineBackup      Ratio
No                           3088  43.844952
Yes                          2429  34.488144
No internet service          1526  21.666903
                     DeviceProtection      Ratio
No                               3095  43.944342
Yes                              2422  34.388755
No internet service              1526  21.666903
                     TechSupport      Ratio
No                          3473  49.311373
Yes                         2044  29.021724
No internet service         1526  21.666903
                     StreamingTV      Ratio
No                          2810  39.897771
Yes                         2707  38.435326
No internet service         1526  21.666903
                     StreamingMovies      Ratio
No                              2785  39.542808
Yes                             2732  38.790288
No internet service             1526  21.666903
                Contract      Ratio
Month-to-month      3875  55.019168
Two year            1695  24.066449
One year            1473  20.914383
     PaperlessBilling      Ratio
Yes              4171  59.221922
No               2872  40.778078
                           PaymentMethod      Ratio
Electronic check                    2365  33.579441
Mailed check                        1612  22.887974
Bank transfer (automatic)           1544  21.922476
Credit card (automatic)             1522  21.610109
     Churn      Ratio
No    5174  73.463013
Yes   1869  26.536987
   SeniorCitizen      Ratio
0           5901  83.785319
1           1142  16.214681
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
count    7043.000000
mean       32.371149
std        24.559481
min         0.000000
25%         9.000000
50%        29.000000
75%        55.000000
max        72.000000
Name: tenure, dtype: float64
count    7043.000000
mean       64.761692
std        30.090047
min        18.250000
25%        35.500000
50%        70.350000
75%        89.850000
max       118.750000
Name: MonthlyCharges, dtype: float64
'''

#Adım 4: Hedef değişken analizi yapınız.

def target_summary_with_num(dataframe, target, numerical_col):
    print(dataframe.groupby(target).agg({numerical_col: "mean"}), end="\n\n\n")

for col in num_cols:
    target_summary_with_num(df, "Churn", col)
'''
          tenure
Churn           
No     37.569965
Yes    17.979133


       MonthlyCharges
Churn                
No          61.265124
Yes         74.441332

'''


le = LabelEncoder()
df["Churn"] = le.fit_transform(df["Churn"])
def target_summary_with_cat(dataframe, target, categorical_col):
    print(categorical_col)
    print(pd.DataFrame({"TARGET_MEAN": dataframe.groupby(categorical_col)[target].mean(),
                        "Count": dataframe[categorical_col].value_counts(),
                        "Ratio": 100 * dataframe[categorical_col].value_counts() / len(dataframe)}), end="\n\n\n")

for col in cat_cols:
    target_summary_with_cat(df, "Churn", col)
'''

gender
        TARGET_MEAN  Count     Ratio
Female     0.269209   3488  49.52435
Male       0.261603   3555  50.47565


Partner
     TARGET_MEAN  Count     Ratio
No      0.329580   3641  51.69672
Yes     0.196649   3402  48.30328


Dependents
     TARGET_MEAN  Count      Ratio
No      0.312791   4933  70.041176
Yes     0.154502   2110  29.958824


PhoneService
     TARGET_MEAN  Count      Ratio
No      0.249267    682   9.683374
Yes     0.267096   6361  90.316626


MultipleLines
                  TARGET_MEAN  Count      Ratio
No                   0.250442   3390  48.132898
No phone service     0.249267    682   9.683374
Yes                  0.286099   2971  42.183729


InternetService
             TARGET_MEAN  Count      Ratio
DSL             0.189591   2421  34.374556
Fiber optic     0.418928   3096  43.958540
No              0.074050   1526  21.666903


OnlineSecurity
                     TARGET_MEAN  Count      Ratio
No                      0.417667   3498  49.666335
No internet service     0.074050   1526  21.666903
Yes                     0.146112   2019  28.666761


OnlineBackup
                     TARGET_MEAN  Count      Ratio
No                      0.399288   3088  43.844952
No internet service     0.074050   1526  21.666903
Yes                     0.215315   2429  34.488144


DeviceProtection
                     TARGET_MEAN  Count      Ratio
No                      0.391276   3095  43.944342
No internet service     0.074050   1526  21.666903
Yes                     0.225021   2422  34.388755


TechSupport
                     TARGET_MEAN  Count      Ratio
No                      0.416355   3473  49.311373
No internet service     0.074050   1526  21.666903
Yes                     0.151663   2044  29.021724


StreamingTV
                     TARGET_MEAN  Count      Ratio
No                      0.335231   2810  39.897771
No internet service     0.074050   1526  21.666903
Yes                     0.300702   2707  38.435326


StreamingMovies
                     TARGET_MEAN  Count      Ratio
No                      0.336804   2785  39.542808
No internet service     0.074050   1526  21.666903
Yes                     0.299414   2732  38.790288


Contract
                TARGET_MEAN  Count      Ratio
Month-to-month     0.427097   3875  55.019168
One year           0.112695   1473  20.914383
Two year           0.028319   1695  24.066449


PaperlessBilling
     TARGET_MEAN  Count      Ratio
No      0.163301   2872  40.778078
Yes     0.335651   4171  59.221922


PaymentMethod
                           TARGET_MEAN  Count      Ratio
Bank transfer (automatic)     0.167098   1544  21.922476
Credit card (automatic)       0.152431   1522  21.610109
Electronic check              0.452854   2365  33.579441
Mailed check                  0.191067   1612  22.887974


Churn
   TARGET_MEAN  Count      Ratio
0          0.0   5174  73.463013
1          1.0   1869  26.536987


SeniorCitizen
   TARGET_MEAN  Count      Ratio
0     0.236062   5901  83.785319
1     0.416813   1142  16.214681

'''
df["Churn"] = le.inverse_transform(df["Churn"])

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
tenure False
MonthlyCharges False
'''

#Adım 6: Eksik gözlem analizi yapınız.

print(df.isnull().values.any())    #False
print((df[df.columns] == " ").sum())
'''
customerID           0
gender               0
SeniorCitizen        0
Partner              0
Dependents           0
tenure               0
PhoneService         0
MultipleLines        0
InternetService      0
OnlineSecurity       0
OnlineBackup         0
DeviceProtection     0
TechSupport          0
StreamingTV          0
StreamingMovies      0
Contract             0
PaperlessBilling     0
PaymentMethod        0
MonthlyCharges       0
TotalCharges        11
Churn                0
dtype: int64
'''
print((df[df.columns] == 0).sum())
'''
customerID             0
gender                 0
SeniorCitizen       5901
Partner                0
Dependents             0
tenure                11
PhoneService           0
MultipleLines          0
InternetService        0
OnlineSecurity         0
OnlineBackup           0
DeviceProtection       0
TechSupport            0
StreamingTV            0
StreamingMovies        0
Contract               0
PaperlessBilling       0
PaymentMethod          0
MonthlyCharges         0
TotalCharges           0
Churn                  0
dtype: int64
'''

#Adım 7: Korelasyon analizi yapınız.
print(df.corr())
'''
                SeniorCitizen    tenure  MonthlyCharges
SeniorCitizen        1.000000  0.016567        0.220173
tenure               0.016567  1.000000        0.247900
MonthlyCharges       0.220173  0.247900        1.000000
'''
f, ax = plt.subplots(figsize=[18, 13])
sns.heatmap(df.corr(), annot=True, fmt=".2f", ax=ax, cmap="magma")
ax.set_title("Correlation Matrix", fontsize=20)
plt.show()


#Görev 2 : Feature Engineering
#Adım 1: Eksik ve aykırı gözlemler için gerekli işlemleri yapınız.

print(df.isnull().sum())
'''
customerID          0
gender              0
SeniorCitizen       0
Partner             0
Dependents          0
tenure              0
PhoneService        0
MultipleLines       0
InternetService     0
OnlineSecurity      0
OnlineBackup        0
DeviceProtection    0
TechSupport         0
StreamingTV         0
StreamingMovies     0
Contract            0
PaperlessBilling    0
PaymentMethod       0
MonthlyCharges      0
TotalCharges        0
Churn               0
dtype: int64
'''
def missing_values_table(dataframe, na_name=False):
    na_columns = [col for col in dataframe.columns if dataframe[col].isnull().sum() > 0]
    n_miss = dataframe[na_columns].isnull().sum().sort_values(ascending=False)
    ratio = (dataframe[na_columns].isnull().sum() / dataframe.shape[0] * 100).sort_values(ascending=False)
    missing_df = pd.concat([n_miss, np.round(ratio, 2)], axis=1, keys=['n_miss', 'ratio'])
    print(missing_df, end="\n")
    if na_name:
        return na_columns

na_columns = missing_values_table(df, na_name=True)   #Empty DataFrame

print((df[df.columns] == " ").sum())
'''
customerID           0
gender               0
SeniorCitizen        0
Partner              0
Dependents           0
tenure               0
PhoneService         0
MultipleLines        0
InternetService      0
OnlineSecurity       0
OnlineBackup         0
DeviceProtection     0
TechSupport          0
StreamingTV          0
StreamingMovies      0
Contract             0
PaperlessBilling     0
PaymentMethod        0
MonthlyCharges       0
TotalCharges        11
Churn                0
dtype: int64
'''
print(df.loc[(df["TotalCharges"] == " ")].head())
'''
      customerID  gender  SeniorCitizen Partner Dependents  tenure PhoneService     MultipleLines InternetService       OnlineSecurity         OnlineBackup     DeviceProtection          TechSupport          StreamingTV      StreamingMovies  Contract PaperlessBilling              PaymentMethod  MonthlyCharges TotalCharges Churn
488   4472-LVYGI  Female              0     Yes        Yes       0           No  No phone service             DSL                  Yes                   No                  Yes                  Yes                  Yes                   No  Two year              Yes  Bank transfer (automatic)           52.55                 No
753   3115-CZMZD    Male              0      No        Yes       0          Yes                No              No  No internet service  No internet service  No internet service  No internet service  No internet service  No internet service  Two year               No               Mailed check           20.25                 No
936   5709-LVOEQ  Female              0     Yes        Yes       0          Yes                No             DSL                  Yes                  Yes                  Yes                   No                  Yes                  Yes  Two year               No               Mailed check           80.85                 No
1082  4367-NUYAO    Male              0     Yes        Yes       0          Yes               Yes              No  No internet service  No internet service  No internet service  No internet service  No internet service  No internet service  Two year               No               Mailed check           25.75                 No
1340  1371-DWPAZ  Female              0     Yes        Yes       0           No  No phone service             DSL                  Yes                  Yes                  Yes                  Yes                  Yes                   No  Two year               No    Credit card (automatic)           56.05                 No
'''
print(df.loc[df["TotalCharges"] == " "].index)
'''
Int64Index([488, 753, 936, 1082, 1340, 3331, 3826, 4380, 5218, 6670, 6754], dtype='int64')
'''
nan_index = df.loc[df["TotalCharges"] == " "].index
df.loc[nan_index, "TotalCharges"] =df.iloc[nan_index]["MonthlyCharges"].values
print(df.iloc[nan_index])
'''
      customerID  gender  SeniorCitizen Partner Dependents  tenure PhoneService     MultipleLines InternetService       OnlineSecurity         OnlineBackup     DeviceProtection          TechSupport          StreamingTV      StreamingMovies  Contract PaperlessBilling              PaymentMethod  MonthlyCharges TotalCharges Churn
488   4472-LVYGI  Female              0     Yes        Yes       0           No  No phone service             DSL                  Yes                   No                  Yes                  Yes                  Yes                   No  Two year              Yes  Bank transfer (automatic)           52.55        52.55    No
753   3115-CZMZD    Male              0      No        Yes       0          Yes                No              No  No internet service  No internet service  No internet service  No internet service  No internet service  No internet service  Two year               No               Mailed check           20.25        20.25    No
936   5709-LVOEQ  Female              0     Yes        Yes       0          Yes                No             DSL                  Yes                  Yes                  Yes                   No                  Yes                  Yes  Two year               No               Mailed check           80.85        80.85    No
1082  4367-NUYAO    Male              0     Yes        Yes       0          Yes               Yes              No  No internet service  No internet service  No internet service  No internet service  No internet service  No internet service  Two year               No               Mailed check           25.75        25.75    No
1340  1371-DWPAZ  Female              0     Yes        Yes       0           No  No phone service             DSL                  Yes                  Yes                  Yes                  Yes                  Yes                   No  Two year               No    Credit card (automatic)           56.05        56.05    No
3331  7644-OMVMY    Male              0     Yes        Yes       0          Yes                No              No  No internet service  No internet service  No internet service  No internet service  No internet service  No internet service  Two year               No               Mailed check           19.85        19.85    No
3826  3213-VVOLG    Male              0     Yes        Yes       0          Yes               Yes              No  No internet service  No internet service  No internet service  No internet service  No internet service  No internet service  Two year               No               Mailed check           25.35        25.35    No
4380  2520-SGTTA  Female              0     Yes        Yes       0          Yes                No              No  No internet service  No internet service  No internet service  No internet service  No internet service  No internet service  Two year               No               Mailed check           20.00         20.0    No
5218  2923-ARZLG    Male              0     Yes        Yes       0          Yes                No              No  No internet service  No internet service  No internet service  No internet service  No internet service  No internet service  One year              Yes               Mailed check           19.70         19.7    No
6670  4075-WKNIU  Female              0     Yes        Yes       0          Yes               Yes             DSL                   No                  Yes                  Yes                  Yes                  Yes                   No  Two year               No               Mailed check           73.35        73.35    No
6754  2775-SEFEE    Male              0      No        Yes       0          Yes               Yes             DSL                  Yes                  Yes                   No                  Yes                   No                   No  Two year              Yes  Bank transfer (automatic)           61.90         61.9    No
'''

df["TotalCharges"] = df["TotalCharges"].astype("float")
print(df.isnull().sum())
'''
customerID          0
gender              0
SeniorCitizen       0
Partner             0
Dependents          0
tenure              0
PhoneService        0
MultipleLines       0
InternetService     0
OnlineSecurity      0
OnlineBackup        0
DeviceProtection    0
TechSupport         0
StreamingTV         0
StreamingMovies     0
Contract            0
PaperlessBilling    0
PaymentMethod       0
MonthlyCharges      0
TotalCharges        0
Churn               0
dtype: int64
'''
def outlier_thresholds(dataframe, col_name, q1=0.05, q3=0.95):
    quartile1 = dataframe[col_name].quantile(q1)
    quartile3 = dataframe[col_name].quantile(q3)
    interquantile_range = quartile3 - quartile1
    up_limit = quartile3 + 1.5 * interquantile_range
    low_limit = quartile1 - 1.5 * interquantile_range
    return low_limit, up_limit

def check_outlier(dataframe, col_name):
    low_limit, up_limit = outlier_thresholds(dataframe, col_name)
    if dataframe[(dataframe[col_name] > up_limit) | (dataframe[col_name] < low_limit)].any(axis=None):
        return True
    else:
        return False

def replace_with_thresholds(dataframe, variable, q1=0.05, q3=0.95):
    low_limit, up_limit = outlier_thresholds(dataframe, variable, q1=0.05, q3=0.95)
    dataframe.loc[(dataframe[variable] < low_limit), variable] = low_limit
    dataframe.loc[(dataframe[variable] > up_limit), variable] = up_limit

num_cols.append("TotalCharges")
cat_but_car.remove("TotalCharges")

for col in num_cols:
    print(col, check_outlier(df, col))
    if check_outlier(df, col):
        replace_with_thresholds(df, col)
'''
tenure False
MonthlyCharges False
TotalCharges False
'''

#Adım 2: Yeni değişkenler oluşturunuz.

df.loc[(df["tenure"]>=0) & (df["tenure"]<=12),"NEW_TENURE_YEAR"] = "0-1 Year"
df.loc[(df["tenure"]>12) & (df["tenure"]<=24),"NEW_TENURE_YEAR"] = "1-2 Year"
df.loc[(df["tenure"]>24) & (df["tenure"]<=36),"NEW_TENURE_YEAR"] = "2-3 Year"
df.loc[(df["tenure"]>36) & (df["tenure"]<=48),"NEW_TENURE_YEAR"] = "3-4 Year"
df.loc[(df["tenure"]>48) & (df["tenure"]<=60),"NEW_TENURE_YEAR"] = "4-5 Year"
df.loc[(df["tenure"]>60) & (df["tenure"]<=72),"NEW_TENURE_YEAR"] = "5-6 Year"
print(df.head())

'''
   customerID  gender  SeniorCitizen Partner Dependents  tenure PhoneService     MultipleLines InternetService OnlineSecurity OnlineBackup DeviceProtection TechSupport StreamingTV StreamingMovies        Contract PaperlessBilling              PaymentMethod  MonthlyCharges  TotalCharges Churn NEW_TENURE_YEAR
0  7590-VHVEG  Female              0     Yes         No       1           No  No phone service             DSL             No          Yes               No          No          No              No  Month-to-month              Yes           Electronic check           29.85         29.85    No        0-1 Year
1  5575-GNVDE    Male              0      No         No      34          Yes                No             DSL            Yes           No              Yes          No          No              No        One year               No               Mailed check           56.95       1889.50    No        2-3 Year
2  3668-QPYBK    Male              0      No         No       2          Yes                No             DSL            Yes          Yes               No          No          No              No  Month-to-month              Yes               Mailed check           53.85        108.15   Yes        0-1 Year
3  7795-CFOCW    Male              0      No         No      45           No  No phone service             DSL            Yes           No              Yes         Yes          No              No        One year               No  Bank transfer (automatic)           42.30       1840.75    No        3-4 Year
4  9237-HQITU  Female              0      No         No       2          Yes                No     Fiber optic             No           No               No          No          No              No  Month-to-month              Yes           Electronic check           70.70        151.65   Yes        0-1 Year
'''

df["NEW_AVG_Charges"] = df["TotalCharges"] / (df["tenure"] + 1)
print(df.head())
'''
   customerID  gender  SeniorCitizen Partner Dependents  tenure PhoneService     MultipleLines InternetService OnlineSecurity OnlineBackup DeviceProtection TechSupport StreamingTV StreamingMovies        Contract PaperlessBilling              PaymentMethod  MonthlyCharges  TotalCharges Churn NEW_TENURE_YEAR  NEW_AVG_Charges
0  7590-VHVEG  Female              0     Yes         No       1           No  No phone service             DSL             No          Yes               No          No          No              No  Month-to-month              Yes           Electronic check           29.85         29.85    No        0-1 Year        14.925000
1  5575-GNVDE    Male              0      No         No      34          Yes                No             DSL            Yes           No              Yes          No          No              No        One year               No               Mailed check           56.95       1889.50    No        2-3 Year        53.985714
2  3668-QPYBK    Male              0      No         No       2          Yes                No             DSL            Yes          Yes               No          No          No              No  Month-to-month              Yes               Mailed check           53.85        108.15   Yes        0-1 Year        36.050000
3  7795-CFOCW    Male              0      No         No      45           No  No phone service             DSL            Yes           No              Yes         Yes          No              No        One year               No  Bank transfer (automatic)           42.30       1840.75    No        3-4 Year        40.016304
4  9237-HQITU  Female              0      No         No       2          Yes                No     Fiber optic             No           No               No          No          No              No  Month-to-month              Yes           Electronic check           70.70        151.65   Yes        0-1 Year        50.550000
'''

#Adım 3: Encoding işlemlerini gerçekleştiriniz.

cat_cols, num_cols, cat_but_car = grab_col_names(df)
'''
Observations: 7043
Variables: 23
cat_cols: 18
num_cols: 4
cat_but_car: 1
num_but_cat: 1
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
   customerID  gender  SeniorCitizen  Partner  Dependents  tenure  PhoneService     MultipleLines InternetService OnlineSecurity OnlineBackup DeviceProtection TechSupport StreamingTV StreamingMovies        Contract  PaperlessBilling              PaymentMethod  MonthlyCharges  TotalCharges  Churn NEW_TENURE_YEAR  NEW_AVG_Charges
0  7590-VHVEG       0              0        1           0       1             0  No phone service             DSL             No          Yes               No          No          No              No  Month-to-month                 1           Electronic check           29.85         29.85      0        0-1 Year        14.925000
1  5575-GNVDE       1              0        0           0      34             1                No             DSL            Yes           No              Yes          No          No              No        One year                 0               Mailed check           56.95       1889.50      0        2-3 Year        53.985714
2  3668-QPYBK       1              0        0           0       2             1                No             DSL            Yes          Yes               No          No          No              No  Month-to-month                 1               Mailed check           53.85        108.15      1        0-1 Year        36.050000
3  7795-CFOCW       1              0        0           0      45             0  No phone service             DSL            Yes           No              Yes         Yes          No              No        One year                 0  Bank transfer (automatic)           42.30       1840.75      0        3-4 Year        40.016304
4  9237-HQITU       0              0        0           0       2             1                No     Fiber optic             No           No               No          No          No              No  Month-to-month                 1           Electronic check           70.70        151.65      1        0-1 Year        50.550000
'''
print(binary_cols)
'''
['gender', 'Partner', 'Dependents', 'PhoneService', 'PaperlessBilling', 'Churn']
'''

cat_cols = [col for col in cat_cols if col not in binary_cols and col not in ["Churn"]]
def one_hot_encoder(dataframe, categorical_cols, drop_first=False):
    dataframe = pd.get_dummies(dataframe, columns=categorical_cols, drop_first=drop_first)
    return dataframe
df = one_hot_encoder(df, cat_cols, drop_first=True)
print(df.head())
'''
   customerID  gender  Partner  Dependents  tenure  PhoneService  PaperlessBilling  MonthlyCharges  TotalCharges  Churn  NEW_AVG_Charges  MultipleLines_No phone service  MultipleLines_Yes  InternetService_Fiber optic  InternetService_No  OnlineSecurity_No internet service  OnlineSecurity_Yes  OnlineBackup_No internet service  OnlineBackup_Yes  DeviceProtection_No internet service  DeviceProtection_Yes  TechSupport_No internet service  TechSupport_Yes  StreamingTV_No internet service  StreamingTV_Yes  StreamingMovies_No internet service  StreamingMovies_Yes  Contract_One year  Contract_Two year  PaymentMethod_Credit card (automatic)  PaymentMethod_Electronic check  PaymentMethod_Mailed check  NEW_TENURE_YEAR_1-2 Year  NEW_TENURE_YEAR_2-3 Year  NEW_TENURE_YEAR_3-4 Year  NEW_TENURE_YEAR_4-5 Year  NEW_TENURE_YEAR_5-6 Year  SeniorCitizen_1
0  7590-VHVEG       0        1           0       1             0                 1           29.85         29.85      0        14.925000                               1                  0                            0                   0                                   0                   0                                 0                 1                                     0                     0                                0                0                                0                0                                    0                    0                  0                  0                                      0                               1                           0                         0                         0                         0                         0                         0                0
1  5575-GNVDE       1        0           0      34             1                 0           56.95       1889.50      0        53.985714                               0                  0                            0                   0                                   0                   1                                 0                 0                                     0                     1                                0                0                                0                0                                    0                    0                  1                  0                                      0                               0                           1                         0                         1                         0                         0                         0                0
2  3668-QPYBK       1        0           0       2             1                 1           53.85        108.15      1        36.050000                               0                  0                            0                   0                                   0                   1                                 0                 1                                     0                     0                                0                0                                0                0                                    0                    0                  0                  0                                      0                               0                           1                         0                         0                         0                         0                         0                0
3  7795-CFOCW       1        0           0      45             0                 0           42.30       1840.75      0        40.016304                               1                  0                            0                   0                                   0                   1                                 0                 0                                     0                     1                                0                1                                0                0                                    0                    0                  1                  0                                      0                               0                           0                         0                         0                         1                         0                         0                0
4  9237-HQITU       0        0           0       2             1                 1           70.70        151.65      1        50.550000                               0                  0                            1                   0                                   0                   0                                 0                 0                                     0                     0                                0                0                                0                0                                    0                    0                  0                  0                                      0                               1                           0                         0                         0                         0                         0                         0                0
'''

#Adım 4: Numerik değişkenler için standartlaştırma yapınız.

print(num_cols)
'''
['tenure', 'MonthlyCharges', 'TotalCharges', 'NEW_AVG_Charges']
'''

mms = MinMaxScaler()
df[num_cols] = mms.fit_transform(df[num_cols])
print(df.head())
'''
   customerID  gender  Partner  Dependents    tenure  PhoneService  PaperlessBilling  MonthlyCharges  TotalCharges  Churn  NEW_AVG_Charges  MultipleLines_No phone service  MultipleLines_Yes  InternetService_Fiber optic  InternetService_No  OnlineSecurity_No internet service  OnlineSecurity_Yes  OnlineBackup_No internet service  OnlineBackup_Yes  DeviceProtection_No internet service  DeviceProtection_Yes  TechSupport_No internet service  TechSupport_Yes  StreamingTV_No internet service  StreamingTV_Yes  StreamingMovies_No internet service  StreamingMovies_Yes  Contract_One year  Contract_Two year  PaymentMethod_Credit card (automatic)  PaymentMethod_Electronic check  PaymentMethod_Mailed check  NEW_TENURE_YEAR_1-2 Year  NEW_TENURE_YEAR_2-3 Year  NEW_TENURE_YEAR_3-4 Year  NEW_TENURE_YEAR_4-5 Year  NEW_TENURE_YEAR_5-6 Year  SeniorCitizen_1
0  7590-VHVEG       0        1           0  0.013889             0                 1        0.115423      0.001275      0         0.052298                               1                  0                            0                   0                                   0                   0                                 0                 1                                     0                     0                                0                0                                0                0                                    0                    0                  0                  0                                      0                               1                           0                         0                         0                         0                         0                         0                0
1  5575-GNVDE       1        0           0  0.472222             1                 0        0.385075      0.215867      0         0.408086                               0                  0                            0                   0                                   0                   1                                 0                 0                                     0                     1                                0                0                                0                0                                    0                    0                  1                  0                                      0                               0                           1                         0                         1                         0                         0                         0                0
2  3668-QPYBK       1        0           0  0.027778             1                 1        0.354229      0.010310      1         0.244717                               0                  0                            0                   0                                   0                   1                                 0                 1                                     0                     0                                0                0                                0                0                                    0                    0                  0                  0                                      0                               0                           1                         0                         0                         0                         0                         0                0
3  7795-CFOCW       1        0           0  0.625000             0                 0        0.239303      0.210241      0         0.280845                               1                  0                            0                   0                                   0                   1                                 0                 0                                     0                     1                                0                1                                0                0                                    0                    0                  1                  0                                      0                               0                           0                         0                         0                         1                         0                         0                0
4  9237-HQITU       0        0           0  0.027778             1                 1        0.521891      0.015330      1         0.376792                               0                  0                            1                   0                                   0                   0                                 0                 0                                     0                     0                                0                0                                0                0                                    0                    0                  0                  0                                      0                               1                           0                         0                         0                         0                         0                         0                0

'''

#Adım 5: Model oluşturunuz.

# Adım 1: Sınıflandırma algoritmaları ile modeller kurup, accuracy skorlarını inceleyip. En iyi 4 modeli seçiniz
x = df.drop(["Churn", "customerID"], axis=1)
y = df["Churn"]

models = [('LR', LogisticRegression(random_state=12345)),
          ('KNN', KNeighborsClassifier(n_neighbors=10)),
          ('CART', DecisionTreeClassifier(random_state=12345)),
          ('RF', RandomForestClassifier(random_state=12345)),
          ('SVM', SVC(gamma='auto', random_state=12345)),
          ('XGB', XGBClassifier(random_state=12345)),
          ("LightGBM", LGBMClassifier(random_state=12345)),
          ]

for name, model in models:
    cv_results = cross_validate(model, x, y, cv=10, scoring=["accuracy", "f1", "roc_auc", "precision", "recall"])
    print(f"########## {name} ##########")
    print(f"Accuracy: {round(cv_results['test_accuracy'].mean(), 4)}")
    print(f"Auc: {round(cv_results['test_roc_auc'].mean(), 4)}")
    print(f"Recall: {round(cv_results['test_recall'].mean(), 4)}")
    print(f"Precision: {round(cv_results['test_precision'].mean(), 4)}")
    print(f"F1: {round(cv_results['test_f1'].mean(), 4)}")
'''
########## KNN ##########
Accuracy: nan
Auc: nan
Recall: nan
Precision: nan
F1: nan
########## CART ##########
Accuracy: 0.7271
Auc: 0.6547
Recall: 0.4971
Precision: 0.487
F1: 0.4916
########## RF ##########
Accuracy: 0.7945
Auc: 0.8279
Recall: 0.4912
Precision: 0.6497
F1: 0.5591
########## SVM ##########
Accuracy: 0.7992
Auc: 0.834
Recall: 0.4575
Precision: 0.6811
F1: 0.547
########## XGB ##########
Accuracy: 0.7892
Auc: 0.8261
Recall: 0.5126
Precision: 0.6257
F1: 0.5631
########## LightGBM ##########
Accuracy: 0.8005
Auc: 0.8371
Recall: 0.5345
Precision: 0.6527
F1: 0.5872'''


#RandomForestClassifier
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.30, random_state=13)
rf_model = RandomForestClassifier(random_state = 1)
rf_model.fit(x_train, y_train)
y_pred = rf_model.predict(x_test)
print(f"Accuracy: {round(accuracy_score(y_pred, y_test), 2)}")
print(f"Recall: {round(recall_score(y_pred,y_test),2)}")
print(f"Precision: {round(precision_score(y_pred,y_test), 2)}")
print(f"F1: {round(f1_score(y_pred,y_test), 2)}")
print(f"Auc: {round(roc_auc_score(y_pred,y_test), 2)}")
'''
Accuracy: 0.8
Recall: 0.63
Precision: 0.5
F1: 0.56
Auc: 0.74
'''
# Adım 2: Seçtiğiniz modeller ile hiperparametre optimizasyonu gerçekleştirin ve bulduğunuz hiparparametreler ile modeli
# tekrar kurunuz.

rf_model = RandomForestClassifier(random_state=17)

rf_params = {"max_depth": [5, 8, None],
             "max_features": [3, 5, 7, "auto"],
             "min_samples_split": [2, 5, 8, 15, 20],
             "n_estimators": [100, 200, 500]}

rf_best_grid = GridSearchCV(rf_model, rf_params, cv=5, n_jobs=-1, verbose=True).fit(x, y)

rf_best_grid.best_params_

rf_best_grid.best_score_

rf_final = rf_model.set_params(**rf_best_grid.best_params_, random_state=17).fit(x, y)


cv_results = cross_validate(rf_final, x, y, cv=10, scoring=["accuracy", "f1", "roc_auc"])
print(cv_results['test_accuracy'].mean())
print(cv_results['test_f1'].mean())
print(cv_results['test_roc_auc'].mean())

'''
0.8030661669890394
0.575706251561156
0.8413255839541997
'''

def plot_importance(model, features, num=len(x_train), save=False):
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

plot_importance(rf_model, x_train)
'''
       Value                               Feature
7   0.149885                          TotalCharges
3   0.144021                                tenure
8   0.133156                       NEW_AVG_Charges
6   0.129791                        MonthlyCharges
11  0.038200           InternetService_Fiber optic
..       ...                                   ...
23  0.005314   StreamingMovies_No internet service
17  0.004471  DeviceProtection_No internet service
4   0.004040                          PhoneService
9   0.003884        MultipleLines_No phone service
15  0.003642      OnlineBackup_No internet service
'''
