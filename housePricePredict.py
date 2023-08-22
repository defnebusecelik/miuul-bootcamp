################################# Ev Fiyat Tahmin Modeli ###############################
# İş Problemi
# Her bir eve ait özelliklerin ve ev fiyatlarının bulunduğu veriseti kullanılarak,
# farklı tipteki evlerin fiyatlarına ilişkin bir makine öğrenmesi projesi
# gerçekleştirilmek istenmektedir.

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import cross_validate, cross_val_score, GridSearchCV
from sklearn.preprocessing import LabelEncoder
from lightgbm import LGBMClassifier
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
from sklearn.linear_model import Ridge
from sklearn.linear_model import ElasticNet
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import GradientBoostingRegressor
from lightgbm import LGBMRegressor
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.width', None)
pd.set_option('display.float_format', lambda x: '%.3f' % x)

df_train=pd.read_csv("train.csv")
df_test=pd.read_csv("test.csv")

##Exploratory Data Analysis

print(df_train.head())
'''
   Id  MSSubClass MSZoning  LotFrontage  LotArea Street Alley LotShape LandContour Utilities LotConfig LandSlope Neighborhood Condition1 Condition2 BldgType HouseStyle  OverallQual  OverallCond  YearBuilt  YearRemodAdd RoofStyle RoofMatl Exterior1st Exterior2nd MasVnrType  MasVnrArea ExterQual ExterCond Foundation BsmtQual BsmtCond BsmtExposure BsmtFinType1  BsmtFinSF1 BsmtFinType2  BsmtFinSF2  BsmtUnfSF  TotalBsmtSF Heating HeatingQC CentralAir Electrical  1stFlrSF  2ndFlrSF  LowQualFinSF  GrLivArea  BsmtFullBath  BsmtHalfBath  FullBath  HalfBath  BedroomAbvGr  KitchenAbvGr KitchenQual  TotRmsAbvGrd Functional  Fireplaces FireplaceQu GarageType  GarageYrBlt GarageFinish  GarageCars  GarageArea GarageQual GarageCond PavedDrive  WoodDeckSF  OpenPorchSF  EnclosedPorch  3SsnPorch  ScreenPorch  PoolArea PoolQC Fence MiscFeature  MiscVal  MoSold  YrSold SaleType SaleCondition  SalePrice
0   1          60       RL       65.000     8450   Pave   NaN      Reg         Lvl    AllPub    Inside       Gtl      CollgCr       Norm       Norm     1Fam     2Story            7            5       2003          2003     Gable  CompShg     VinylSd     VinylSd    BrkFace     196.000        Gd        TA      PConc       Gd       TA           No          GLQ         706          Unf           0        150          856    GasA        Ex          Y      SBrkr       856       854             0       1710             1             0         2         1             3             1          Gd             8        Typ           0         NaN     Attchd     2003.000          RFn           2         548         TA         TA          Y           0           61              0          0            0         0    NaN   NaN         NaN        0       2    2008       WD        Normal     208500
1   2          20       RL       80.000     9600   Pave   NaN      Reg         Lvl    AllPub       FR2       Gtl      Veenker      Feedr       Norm     1Fam     1Story            6            8       1976          1976     Gable  CompShg     MetalSd     MetalSd       None       0.000        TA        TA     CBlock       Gd       TA           Gd          ALQ         978          Unf           0        284         1262    GasA        Ex          Y      SBrkr      1262         0             0       1262             0             1         2         0             3             1          TA             6        Typ           1          TA     Attchd     1976.000          RFn           2         460         TA         TA          Y         298            0              0          0            0         0    NaN   NaN         NaN        0       5    2007       WD        Normal     181500
2   3          60       RL       68.000    11250   Pave   NaN      IR1         Lvl    AllPub    Inside       Gtl      CollgCr       Norm       Norm     1Fam     2Story            7            5       2001          2002     Gable  CompShg     VinylSd     VinylSd    BrkFace     162.000        Gd        TA      PConc       Gd       TA           Mn          GLQ         486          Unf           0        434          920    GasA        Ex          Y      SBrkr       920       866             0       1786             1             0         2         1             3             1          Gd             6        Typ           1          TA     Attchd     2001.000          RFn           2         608         TA         TA          Y           0           42              0          0            0         0    NaN   NaN         NaN        0       9    2008       WD        Normal     223500
3   4          70       RL       60.000     9550   Pave   NaN      IR1         Lvl    AllPub    Corner       Gtl      Crawfor       Norm       Norm     1Fam     2Story            7            5       1915          1970     Gable  CompShg     Wd Sdng     Wd Shng       None       0.000        TA        TA     BrkTil       TA       Gd           No          ALQ         216          Unf           0        540          756    GasA        Gd          Y      SBrkr       961       756             0       1717             1             0         1         0             3             1          Gd             7        Typ           1          Gd     Detchd     1998.000          Unf           3         642         TA         TA          Y           0           35            272          0            0         0    NaN   NaN         NaN        0       2    2006       WD       Abnorml     140000
4   5          60       RL       84.000    14260   Pave   NaN      IR1         Lvl    AllPub       FR2       Gtl      NoRidge       Norm       Norm     1Fam     2Story            8            5       2000          2000     Gable  CompShg     VinylSd     VinylSd    BrkFace     350.000        Gd        TA      PConc       Gd       TA           Av          GLQ         655          Unf           0        490         1145    GasA        Ex          Y      SBrkr      1145      1053             0       2198             1             0         2         1             4             1          Gd             9        Typ           1          TA     Attchd     2000.000          RFn           3         836         TA         TA          Y         192           84              0          0            0         0    NaN   NaN         NaN        0      12    2008       WD        Normal     250000
 '''

print(df_test.head())
'''
     Id  MSSubClass MSZoning  LotFrontage  LotArea Street Alley LotShape LandContour Utilities LotConfig LandSlope Neighborhood Condition1 Condition2 BldgType HouseStyle  OverallQual  OverallCond  YearBuilt  YearRemodAdd RoofStyle RoofMatl Exterior1st Exterior2nd MasVnrType  MasVnrArea ExterQual ExterCond Foundation BsmtQual BsmtCond BsmtExposure BsmtFinType1  BsmtFinSF1 BsmtFinType2  BsmtFinSF2  BsmtUnfSF  TotalBsmtSF Heating HeatingQC CentralAir Electrical  1stFlrSF  2ndFlrSF  LowQualFinSF  GrLivArea  BsmtFullBath  BsmtHalfBath  FullBath  HalfBath  BedroomAbvGr  KitchenAbvGr KitchenQual  TotRmsAbvGrd Functional  Fireplaces FireplaceQu GarageType  GarageYrBlt GarageFinish  GarageCars  GarageArea GarageQual GarageCond PavedDrive  WoodDeckSF  OpenPorchSF  EnclosedPorch  3SsnPorch  ScreenPorch  PoolArea PoolQC  Fence MiscFeature  MiscVal  MoSold  YrSold SaleType SaleCondition
0  1461          20       RH       80.000    11622   Pave   NaN      Reg         Lvl    AllPub    Inside       Gtl        NAmes      Feedr       Norm     1Fam     1Story            5            6       1961          1961     Gable  CompShg     VinylSd     VinylSd       None       0.000        TA        TA     CBlock       TA       TA           No          Rec     468.000          LwQ     144.000    270.000      882.000    GasA        TA          Y      SBrkr       896         0             0        896         0.000         0.000         1         0             2             1          TA             5        Typ           0         NaN     Attchd     1961.000          Unf       1.000     730.000         TA         TA          Y         140            0              0          0          120         0    NaN  MnPrv         NaN        0       6    2010       WD        Normal
1  1462          20       RL       81.000    14267   Pave   NaN      IR1         Lvl    AllPub    Corner       Gtl        NAmes       Norm       Norm     1Fam     1Story            6            6       1958          1958       Hip  CompShg     Wd Sdng     Wd Sdng    BrkFace     108.000        TA        TA     CBlock       TA       TA           No          ALQ     923.000          Unf       0.000    406.000     1329.000    GasA        TA          Y      SBrkr      1329         0             0       1329         0.000         0.000         1         1             3             1          Gd             6        Typ           0         NaN     Attchd     1958.000          Unf       1.000     312.000         TA         TA          Y         393           36              0          0            0         0    NaN    NaN        Gar2    12500       6    2010       WD        Normal
2  1463          60       RL       74.000    13830   Pave   NaN      IR1         Lvl    AllPub    Inside       Gtl      Gilbert       Norm       Norm     1Fam     2Story            5            5       1997          1998     Gable  CompShg     VinylSd     VinylSd       None       0.000        TA        TA      PConc       Gd       TA           No          GLQ     791.000          Unf       0.000    137.000      928.000    GasA        Gd          Y      SBrkr       928       701             0       1629         0.000         0.000         2         1             3             1          TA             6        Typ           1          TA     Attchd     1997.000          Fin       2.000     482.000         TA         TA          Y         212           34              0          0            0         0    NaN  MnPrv         NaN        0       3    2010       WD        Normal
3  1464          60       RL       78.000     9978   Pave   NaN      IR1         Lvl    AllPub    Inside       Gtl      Gilbert       Norm       Norm     1Fam     2Story            6            6       1998          1998     Gable  CompShg     VinylSd     VinylSd    BrkFace      20.000        TA        TA      PConc       TA       TA           No          GLQ     602.000          Unf       0.000    324.000      926.000    GasA        Ex          Y      SBrkr       926       678             0       1604         0.000         0.000         2         1             3             1          Gd             7        Typ           1          Gd     Attchd     1998.000          Fin       2.000     470.000         TA         TA          Y         360           36              0          0            0         0    NaN    NaN         NaN        0       6    2010       WD        Normal
4  1465         120       RL       43.000     5005   Pave   NaN      IR1         HLS    AllPub    Inside       Gtl      StoneBr       Norm       Norm   TwnhsE     1Story            8            5       1992          1992     Gable  CompShg     HdBoard     HdBoard       None       0.000        Gd        TA      PConc       Gd       TA           No          ALQ     263.000          Unf       0.000   1017.000     1280.000    GasA        Ex          Y      SBrkr      1280         0             0       1280         0.000         0.000         2         0             2             1          Gd             5        Typ           0         NaN     Attchd     1992.000          RFn       2.000     506.000         TA         TA          Y           0           82              0          0          144         0    NaN    NaN         NaN        0       1    2010       WD        Normal
'''

print(df_train.shape)
#(1460, 81)
print(df_test.shape)
#(1459, 80) --> 'Saleprice'

print(df_train.info())
'''
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 1460 entries, 0 to 1459
Data columns (total 81 columns):
 #   Column         Non-Null Count  Dtype  
---  ------         --------------  -----  
 0   Id             1460 non-null   int64  
 1   MSSubClass     1460 non-null   int64  
 2   MSZoning       1460 non-null   object 
 3   LotFrontage    1201 non-null   float64
 4   LotArea        1460 non-null   int64  
 5   Street         1460 non-null   object 
 6   Alley          91 non-null     object 
 7   LotShape       1460 non-null   object 
 8   LandContour    1460 non-null   object 
 9   Utilities      1460 non-null   object 
 10  LotConfig      1460 non-null   object 
 11  LandSlope      1460 non-null   object 
 12  Neighborhood   1460 non-null   object 
 13  Condition1     1460 non-null   object 
 14  Condition2     1460 non-null   object 
 15  BldgType       1460 non-null   object 
 16  HouseStyle     1460 non-null   object 
 17  OverallQual    1460 non-null   int64  
 18  OverallCond    1460 non-null   int64  
 19  YearBuilt      1460 non-null   int64  
 20  YearRemodAdd   1460 non-null   int64  
 21  RoofStyle      1460 non-null   object 
 22  RoofMatl       1460 non-null   object 
 23  Exterior1st    1460 non-null   object 
 24  Exterior2nd    1460 non-null   object 
 25  MasVnrType     1452 non-null   object 
 26  MasVnrArea     1452 non-null   float64
 27  ExterQual      1460 non-null   object 
 28  ExterCond      1460 non-null   object 
 29  Foundation     1460 non-null   object 
 30  BsmtQual       1423 non-null   object 
 31  BsmtCond       1423 non-null   object 
 32  BsmtExposure   1422 non-null   object 
 33  BsmtFinType1   1423 non-null   object 
 34  BsmtFinSF1     1460 non-null   int64  
 35  BsmtFinType2   1422 non-null   object 
 36  BsmtFinSF2     1460 non-null   int64  
 37  BsmtUnfSF      1460 non-null   int64  
 38  TotalBsmtSF    1460 non-null   int64  
 39  Heating        1460 non-null   object 
 40  HeatingQC      1460 non-null   object 
 41  CentralAir     1460 non-null   object 
 42  Electrical     1459 non-null   object 
 43  1stFlrSF       1460 non-null   int64  
 44  2ndFlrSF       1460 non-null   int64  
 45  LowQualFinSF   1460 non-null   int64  
 46  GrLivArea      1460 non-null   int64  
 47  BsmtFullBath   1460 non-null   int64  
 48  BsmtHalfBath   1460 non-null   int64  
 49  FullBath       1460 non-null   int64  
 50  HalfBath       1460 non-null   int64  
 51  BedroomAbvGr   1460 non-null   int64  
 52  KitchenAbvGr   1460 non-null   int64  
 53  KitchenQual    1460 non-null   object 
 54  TotRmsAbvGrd   1460 non-null   int64  
 55  Functional     1460 non-null   object 
 56  Fireplaces     1460 non-null   int64  
 57  FireplaceQu    770 non-null    object 
 58  GarageType     1379 non-null   object 
 59  GarageYrBlt    1379 non-null   float64
 60  GarageFinish   1379 non-null   object 
 61  GarageCars     1460 non-null   int64  
 62  GarageArea     1460 non-null   int64  
 63  GarageQual     1379 non-null   object 
 64  GarageCond     1379 non-null   object 
 65  PavedDrive     1460 non-null   object 
 66  WoodDeckSF     1460 non-null   int64  
 67  OpenPorchSF    1460 non-null   int64  
 68  EnclosedPorch  1460 non-null   int64  
 69  3SsnPorch      1460 non-null   int64  
 70  ScreenPorch    1460 non-null   int64  
 71  PoolArea       1460 non-null   int64  
 72  PoolQC         7 non-null      object 
 73  Fence          281 non-null    object 
 74  MiscFeature    54 non-null     object 
 75  MiscVal        1460 non-null   int64  
 76  MoSold         1460 non-null   int64  
 77  YrSold         1460 non-null   int64  
 78  SaleType       1460 non-null   object 
 79  SaleCondition  1460 non-null   object 
 80  SalePrice      1460 non-null   int64  
dtypes: float64(3), int64(35), object(43)
memory usage: 924.0+ KB
None
'''

print(df_test.info())
'''
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 1459 entries, 0 to 1458
Data columns (total 80 columns):
 #   Column         Non-Null Count  Dtype  
---  ------         --------------  -----  
 0   Id             1459 non-null   int64  
 1   MSSubClass     1459 non-null   int64  
 2   MSZoning       1455 non-null   object 
 3   LotFrontage    1232 non-null   float64
 4   LotArea        1459 non-null   int64  
 5   Street         1459 non-null   object 
 6   Alley          107 non-null    object 
 7   LotShape       1459 non-null   object 
 8   LandContour    1459 non-null   object 
 9   Utilities      1457 non-null   object 
 10  LotConfig      1459 non-null   object 
 11  LandSlope      1459 non-null   object 
 12  Neighborhood   1459 non-null   object 
 13  Condition1     1459 non-null   object 
 14  Condition2     1459 non-null   object 
 15  BldgType       1459 non-null   object 
 16  HouseStyle     1459 non-null   object 
 17  OverallQual    1459 non-null   int64  
 18  OverallCond    1459 non-null   int64  
 19  YearBuilt      1459 non-null   int64  
 20  YearRemodAdd   1459 non-null   int64  
 21  RoofStyle      1459 non-null   object 
 22  RoofMatl       1459 non-null   object 
 23  Exterior1st    1458 non-null   object 
 24  Exterior2nd    1458 non-null   object 
 25  MasVnrType     1443 non-null   object 
 26  MasVnrArea     1444 non-null   float64
 27  ExterQual      1459 non-null   object 
 28  ExterCond      1459 non-null   object 
 29  Foundation     1459 non-null   object 
 30  BsmtQual       1415 non-null   object 
 31  BsmtCond       1414 non-null   object 
 32  BsmtExposure   1415 non-null   object 
 33  BsmtFinType1   1417 non-null   object 
 34  BsmtFinSF1     1458 non-null   float64
 35  BsmtFinType2   1417 non-null   object 
 36  BsmtFinSF2     1458 non-null   float64
 37  BsmtUnfSF      1458 non-null   float64
 38  TotalBsmtSF    1458 non-null   float64
 39  Heating        1459 non-null   object 
 40  HeatingQC      1459 non-null   object 
 41  CentralAir     1459 non-null   object 
 42  Electrical     1459 non-null   object 
 43  1stFlrSF       1459 non-null   int64  
 44  2ndFlrSF       1459 non-null   int64  
 45  LowQualFinSF   1459 non-null   int64  
 46  GrLivArea      1459 non-null   int64  
 47  BsmtFullBath   1457 non-null   float64
 48  BsmtHalfBath   1457 non-null   float64
 49  FullBath       1459 non-null   int64  
 50  HalfBath       1459 non-null   int64  
 51  BedroomAbvGr   1459 non-null   int64  
 52  KitchenAbvGr   1459 non-null   int64  
 53  KitchenQual    1458 non-null   object 
 54  TotRmsAbvGrd   1459 non-null   int64  
 55  Functional     1457 non-null   object 
 56  Fireplaces     1459 non-null   int64  
 57  FireplaceQu    729 non-null    object 
 58  GarageType     1383 non-null   object 
 59  GarageYrBlt    1381 non-null   float64
 60  GarageFinish   1381 non-null   object 
 61  GarageCars     1458 non-null   float64
 62  GarageArea     1458 non-null   float64
 63  GarageQual     1381 non-null   object 
 64  GarageCond     1381 non-null   object 
 65  PavedDrive     1459 non-null   object 
 66  WoodDeckSF     1459 non-null   int64  
 67  OpenPorchSF    1459 non-null   int64  
 68  EnclosedPorch  1459 non-null   int64  
 69  3SsnPorch      1459 non-null   int64  
 70  ScreenPorch    1459 non-null   int64  
 71  PoolArea       1459 non-null   int64  
 72  PoolQC         3 non-null      object 
 73  Fence          290 non-null    object 
 74  MiscFeature    51 non-null     object 
 75  MiscVal        1459 non-null   int64  
 76  MoSold         1459 non-null   int64  
 77  YrSold         1459 non-null   int64  
 78  SaleType       1458 non-null   object 
 79  SaleCondition  1459 non-null   object 
dtypes: float64(11), int64(26), object(43)
memory usage: 912.0+ KB
None
'''

print(df_train.isna().sum())
'''
Id                  0
MSSubClass          0
MSZoning            0
LotFrontage       259
LotArea             0
Street              0
Alley            1369
LotShape            0
LandContour         0
Utilities           0
LotConfig           0
LandSlope           0
Neighborhood        0
Condition1          0
Condition2          0
BldgType            0
HouseStyle          0
OverallQual         0
OverallCond         0
YearBuilt           0
YearRemodAdd        0
RoofStyle           0
RoofMatl            0
Exterior1st         0
Exterior2nd         0
MasVnrType          8
MasVnrArea          8
ExterQual           0
ExterCond           0
Foundation          0
BsmtQual           37
BsmtCond           37
BsmtExposure       38
BsmtFinType1       37
BsmtFinSF1          0
BsmtFinType2       38
BsmtFinSF2          0
BsmtUnfSF           0
TotalBsmtSF         0
Heating             0
HeatingQC           0
CentralAir          0
Electrical          1
1stFlrSF            0
2ndFlrSF            0
LowQualFinSF        0
GrLivArea           0
BsmtFullBath        0
BsmtHalfBath        0
FullBath            0
HalfBath            0
BedroomAbvGr        0
KitchenAbvGr        0
KitchenQual         0
TotRmsAbvGrd        0
Functional          0
Fireplaces          0
FireplaceQu       690
GarageType         81
GarageYrBlt        81
GarageFinish       81
GarageCars          0
GarageArea          0
GarageQual         81
GarageCond         81
PavedDrive          0
WoodDeckSF          0
OpenPorchSF         0
EnclosedPorch       0
3SsnPorch           0
ScreenPorch         0
PoolArea            0
PoolQC           1453
Fence            1179
MiscFeature      1406
MiscVal             0
MoSold              0
YrSold              0
SaleType            0
SaleCondition       0
SalePrice           0
dtype: int64
'''

print(df_test.isna().sum())
'''
Id                  0
MSSubClass          0
MSZoning            4
LotFrontage       227
LotArea             0
Street              0
Alley            1352
LotShape            0
LandContour         0
Utilities           2
LotConfig           0
LandSlope           0
Neighborhood        0
Condition1          0
Condition2          0
BldgType            0
HouseStyle          0
OverallQual         0
OverallCond         0
YearBuilt           0
YearRemodAdd        0
RoofStyle           0
RoofMatl            0
Exterior1st         1
Exterior2nd         1
MasVnrType         16
MasVnrArea         15
ExterQual           0
ExterCond           0
Foundation          0
BsmtQual           44
BsmtCond           45
BsmtExposure       44
BsmtFinType1       42
BsmtFinSF1          1
BsmtFinType2       42
BsmtFinSF2          1
BsmtUnfSF           1
TotalBsmtSF         1
Heating             0
HeatingQC           0
CentralAir          0
Electrical          0
1stFlrSF            0
2ndFlrSF            0
LowQualFinSF        0
GrLivArea           0
BsmtFullBath        2
BsmtHalfBath        2
FullBath            0
HalfBath            0
BedroomAbvGr        0
KitchenAbvGr        0
KitchenQual         1
TotRmsAbvGrd        0
Functional          2
Fireplaces          0
FireplaceQu       730
GarageType         76
GarageYrBlt        78
GarageFinish       78
GarageCars          1
GarageArea          1
GarageQual         78
GarageCond         78
PavedDrive          0
WoodDeckSF          0
OpenPorchSF         0
EnclosedPorch       0
3SsnPorch           0
ScreenPorch         0
PoolArea            0
PoolQC           1456
Fence            1169
MiscFeature      1408
MiscVal             0
MoSold              0
YrSold              0
SaleType            1
SaleCondition       0
dtype: int64
'''

print(df_train.describe().T)
'''
                 count       mean       std       min        25%        50%        75%        max
Id            1460.000    730.500   421.610     1.000    365.750    730.500   1095.250   1460.000
MSSubClass    1460.000     56.897    42.301    20.000     20.000     50.000     70.000    190.000
LotFrontage   1201.000     70.050    24.285    21.000     59.000     69.000     80.000    313.000
LotArea       1460.000  10516.828  9981.265  1300.000   7553.500   9478.500  11601.500 215245.000
OverallQual   1460.000      6.099     1.383     1.000      5.000      6.000      7.000     10.000
OverallCond   1460.000      5.575     1.113     1.000      5.000      5.000      6.000      9.000
YearBuilt     1460.000   1971.268    30.203  1872.000   1954.000   1973.000   2000.000   2010.000
YearRemodAdd  1460.000   1984.866    20.645  1950.000   1967.000   1994.000   2004.000   2010.000
MasVnrArea    1452.000    103.685   181.066     0.000      0.000      0.000    166.000   1600.000
BsmtFinSF1    1460.000    443.640   456.098     0.000      0.000    383.500    712.250   5644.000
BsmtFinSF2    1460.000     46.549   161.319     0.000      0.000      0.000      0.000   1474.000
BsmtUnfSF     1460.000    567.240   441.867     0.000    223.000    477.500    808.000   2336.000
TotalBsmtSF   1460.000   1057.429   438.705     0.000    795.750    991.500   1298.250   6110.000
1stFlrSF      1460.000   1162.627   386.588   334.000    882.000   1087.000   1391.250   4692.000
2ndFlrSF      1460.000    346.992   436.528     0.000      0.000      0.000    728.000   2065.000
LowQualFinSF  1460.000      5.845    48.623     0.000      0.000      0.000      0.000    572.000
GrLivArea     1460.000   1515.464   525.480   334.000   1129.500   1464.000   1776.750   5642.000
BsmtFullBath  1460.000      0.425     0.519     0.000      0.000      0.000      1.000      3.000
BsmtHalfBath  1460.000      0.058     0.239     0.000      0.000      0.000      0.000      2.000
FullBath      1460.000      1.565     0.551     0.000      1.000      2.000      2.000      3.000
HalfBath      1460.000      0.383     0.503     0.000      0.000      0.000      1.000      2.000
BedroomAbvGr  1460.000      2.866     0.816     0.000      2.000      3.000      3.000      8.000
KitchenAbvGr  1460.000      1.047     0.220     0.000      1.000      1.000      1.000      3.000
TotRmsAbvGrd  1460.000      6.518     1.625     2.000      5.000      6.000      7.000     14.000
Fireplaces    1460.000      0.613     0.645     0.000      0.000      1.000      1.000      3.000
GarageYrBlt   1379.000   1978.506    24.690  1900.000   1961.000   1980.000   2002.000   2010.000
GarageCars    1460.000      1.767     0.747     0.000      1.000      2.000      2.000      4.000
GarageArea    1460.000    472.980   213.805     0.000    334.500    480.000    576.000   1418.000
WoodDeckSF    1460.000     94.245   125.339     0.000      0.000      0.000    168.000    857.000
OpenPorchSF   1460.000     46.660    66.256     0.000      0.000     25.000     68.000    547.000
EnclosedPorch 1460.000     21.954    61.119     0.000      0.000      0.000      0.000    552.000
3SsnPorch     1460.000      3.410    29.317     0.000      0.000      0.000      0.000    508.000
ScreenPorch   1460.000     15.061    55.757     0.000      0.000      0.000      0.000    480.000
PoolArea      1460.000      2.759    40.177     0.000      0.000      0.000      0.000    738.000
MiscVal       1460.000     43.489   496.123     0.000      0.000      0.000      0.000  15500.000
MoSold        1460.000      6.322     2.704     1.000      5.000      6.000      8.000     12.000
YrSold        1460.000   2007.816     1.328  2006.000   2007.000   2008.000   2009.000   2010.000
SalePrice     1460.000 180921.196 79442.503 34900.000 129975.000 163000.000 214000.000 755000.000
 '''

print(df_test.describe().T)
'''
                 count     mean      std      min      25%      50%       75%       max
Id            1459.000 2190.000  421.321 1461.000 1825.500 2190.000  2554.500  2919.000
MSSubClass    1459.000   57.378   42.747   20.000   20.000   50.000    70.000   190.000
LotFrontage   1232.000   68.580   22.377   21.000   58.000   67.000    80.000   200.000
LotArea       1459.000 9819.161 4955.517 1470.000 7391.000 9399.000 11517.500 56600.000
OverallQual   1459.000    6.079    1.437    1.000    5.000    6.000     7.000    10.000
OverallCond   1459.000    5.554    1.114    1.000    5.000    5.000     6.000     9.000
YearBuilt     1459.000 1971.358   30.390 1879.000 1953.000 1973.000  2001.000  2010.000
YearRemodAdd  1459.000 1983.663   21.130 1950.000 1963.000 1992.000  2004.000  2010.000
MasVnrArea    1444.000  100.709  177.626    0.000    0.000    0.000   164.000  1290.000
BsmtFinSF1    1458.000  439.204  455.268    0.000    0.000  350.500   753.500  4010.000
BsmtFinSF2    1458.000   52.619  176.754    0.000    0.000    0.000     0.000  1526.000
BsmtUnfSF     1458.000  554.295  437.260    0.000  219.250  460.000   797.750  2140.000
TotalBsmtSF   1458.000 1046.118  442.899    0.000  784.000  988.000  1305.000  5095.000
1stFlrSF      1459.000 1156.535  398.166  407.000  873.500 1079.000  1382.500  5095.000
2ndFlrSF      1459.000  325.968  420.610    0.000    0.000    0.000   676.000  1862.000
LowQualFinSF  1459.000    3.544   44.043    0.000    0.000    0.000     0.000  1064.000
GrLivArea     1459.000 1486.046  485.566  407.000 1117.500 1432.000  1721.000  5095.000
BsmtFullBath  1457.000    0.434    0.531    0.000    0.000    0.000     1.000     3.000
BsmtHalfBath  1457.000    0.065    0.252    0.000    0.000    0.000     0.000     2.000
FullBath      1459.000    1.571    0.555    0.000    1.000    2.000     2.000     4.000
HalfBath      1459.000    0.378    0.503    0.000    0.000    0.000     1.000     2.000
BedroomAbvGr  1459.000    2.854    0.830    0.000    2.000    3.000     3.000     6.000
KitchenAbvGr  1459.000    1.042    0.208    0.000    1.000    1.000     1.000     2.000
TotRmsAbvGrd  1459.000    6.385    1.509    3.000    5.000    6.000     7.000    15.000
Fireplaces    1459.000    0.581    0.647    0.000    0.000    0.000     1.000     4.000
GarageYrBlt   1381.000 1977.721   26.431 1895.000 1959.000 1979.000  2002.000  2207.000
GarageCars    1458.000    1.766    0.776    0.000    1.000    2.000     2.000     5.000
GarageArea    1458.000  472.769  217.049    0.000  318.000  480.000   576.000  1488.000
WoodDeckSF    1459.000   93.175  127.745    0.000    0.000    0.000   168.000  1424.000
OpenPorchSF   1459.000   48.314   68.883    0.000    0.000   28.000    72.000   742.000
EnclosedPorch 1459.000   24.243   67.228    0.000    0.000    0.000     0.000  1012.000
3SsnPorch     1459.000    1.794   20.208    0.000    0.000    0.000     0.000   360.000
ScreenPorch   1459.000   17.064   56.610    0.000    0.000    0.000     0.000   576.000
PoolArea      1459.000    1.744   30.492    0.000    0.000    0.000     0.000   800.000
MiscVal       1459.000   58.168  630.807    0.000    0.000    0.000     0.000 17000.000
MoSold        1459.000    6.104    2.722    1.000    4.000    6.000     8.000    12.000
YrSold        1459.000 2007.770    1.302 2006.000 2007.000 2008.000  2009.000  2010.000
'''

# Analysis of Target Variable

print(df_train['SalePrice'].describe())
'''
count     1460.000
mean    180921.196
std      79442.503
min      34900.000
25%     129975.000
50%     163000.000
75%     214000.000
max     755000.000
Name: SalePrice, dtype: float64
'''

sns.boxplot(df_train['SalePrice'])
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

cat_cols, num_cols, cat_but_car = grab_col_names(df_train)
'''
Observations: 1460
Variables: 81
cat_cols: 53
num_cols: 27
cat_but_car: 1
num_but_cat: 11
'''

print(cat_cols)
#['MSZoning', 'Street', 'Alley', 'LotShape', 'LandContour', 'Utilities', 'LotConfig', 'LandSlope', 'Condition1', 'Condition2', 'BldgType', 'HouseStyle', 'RoofStyle', 'RoofMatl', 'Exterior1st', 'Exterior2nd', 'MasVnrType', 'ExterQual', 'ExterCond', 'Foundation', 'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2', 'Heating', 'HeatingQC', 'CentralAir', 'Electrical', 'KitchenQual', 'Functional', 'FireplaceQu', 'GarageType', 'GarageFinish', 'GarageQual', 'GarageCond', 'PavedDrive', 'PoolQC', 'Fence', 'MiscFeature', 'SaleType', 'SaleCondition', 'OverallCond', 'BsmtFullBath', 'BsmtHalfBath', 'FullBath', 'HalfBath', 'BedroomAbvGr', 'KitchenAbvGr', 'Fireplaces', 'GarageCars', 'PoolArea', 'YrSold']
print(num_cols)
#['Id', 'MSSubClass', 'LotFrontage', 'LotArea', 'OverallQual', 'YearBuilt', 'YearRemodAdd', 'MasVnrArea', 'BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF', '1stFlrSF', '2ndFlrSF', 'LowQualFinSF', 'GrLivArea', 'TotRmsAbvGrd', 'GarageYrBlt', 'GarageArea', 'WoodDeckSF', 'OpenPorchSF', 'EnclosedPorch', '3SsnPorch', 'ScreenPorch', 'MiscVal', 'MoSold', 'SalePrice']
print(cat_but_car)
#['Neighborhood']

#Cat Summary
def cat_summary(dataframe, col_name, plot=False):
    print(pd.DataFrame({col_name: dataframe[col_name].value_counts(),
                        "Ratio": 100 * dataframe[col_name].value_counts() / len(dataframe)}))
    if plot:
        sns.countplot(x=dataframe[col_name], data=dataframe)
        plt.show()

for col in cat_cols:
     cat_summary(df_train, col)

'''
         MSZoning  Ratio
RL           1151 78.836
RM            218 14.932
FV             65  4.452
RH             16  1.096
C (all)        10  0.685
      Street  Ratio
Pave    1454 99.589
Grvl       6  0.411
      Alley  Ratio
Grvl     50  3.425
Pave     41  2.808
     LotShape  Ratio
Reg       925 63.356
IR1       484 33.151
IR2        41  2.808
IR3        10  0.685
     LandContour  Ratio
Lvl         1311 89.795
Bnk           63  4.315
HLS           50  3.425
Low           36  2.466
        Utilities  Ratio
AllPub       1459 99.932
NoSeWa          1  0.068
         LotConfig  Ratio
Inside        1052 72.055
Corner         263 18.014
CulDSac         94  6.438
FR2             47  3.219
FR3              4  0.274
     LandSlope  Ratio
Gtl       1382 94.658
Mod         65  4.452
Sev         13  0.890
        Condition1  Ratio
Norm          1260 86.301
Feedr           81  5.548
Artery          48  3.288
RRAn            26  1.781
PosN            19  1.301
RRAe            11  0.753
PosA             8  0.548
RRNn             5  0.342
RRNe             2  0.137
        Condition2  Ratio
Norm          1445 98.973
Feedr            6  0.411
Artery           2  0.137
RRNn             2  0.137
PosN             2  0.137
PosA             1  0.068
RRAn             1  0.068
RRAe             1  0.068
        BldgType  Ratio
1Fam        1220 83.562
TwnhsE       114  7.808
Duplex        52  3.562
Twnhs         43  2.945
2fmCon        31  2.123
        HouseStyle  Ratio
1Story         726 49.726
2Story         445 30.479
1.5Fin         154 10.548
SLvl            65  4.452
SFoyer          37  2.534
1.5Unf          14  0.959
2.5Unf          11  0.753
2.5Fin           8  0.548
         RoofStyle  Ratio
Gable         1141 78.151
Hip            286 19.589
Flat            13  0.890
Gambrel         11  0.753
Mansard          7  0.479
Shed             2  0.137
         RoofMatl  Ratio
CompShg      1434 98.219
Tar&Grv        11  0.753
WdShngl         6  0.411
WdShake         5  0.342
Metal           1  0.068
Membran         1  0.068
Roll            1  0.068
ClyTile         1  0.068
         Exterior1st  Ratio
VinylSd          515 35.274
HdBoard          222 15.205
MetalSd          220 15.068
Wd Sdng          206 14.110
Plywood          108  7.397
CemntBd           61  4.178
BrkFace           50  3.425
WdShing           26  1.781
Stucco            25  1.712
AsbShng           20  1.370
BrkComm            2  0.137
Stone              2  0.137
AsphShn            1  0.068
ImStucc            1  0.068
CBlock             1  0.068
         Exterior2nd  Ratio
VinylSd          504 34.521
MetalSd          214 14.658
HdBoard          207 14.178
Wd Sdng          197 13.493
Plywood          142  9.726
CmentBd           60  4.110
Wd Shng           38  2.603
Stucco            26  1.781
BrkFace           25  1.712
AsbShng           20  1.370
ImStucc           10  0.685
Brk Cmn            7  0.479
Stone              5  0.342
AsphShn            3  0.205
Other              1  0.068
CBlock             1  0.068
         MasVnrType  Ratio
None            864 59.178
BrkFace         445 30.479
Stone           128  8.767
BrkCmn           15  1.027
    ExterQual  Ratio
TA        906 62.055
Gd        488 33.425
Ex         52  3.562
Fa         14  0.959
    ExterCond  Ratio
TA       1282 87.808
Gd        146 10.000
Fa         28  1.918
Ex          3  0.205
Po          1  0.068
        Foundation  Ratio
PConc          647 44.315
CBlock         634 43.425
BrkTil         146 10.000
Slab            24  1.644
Stone            6  0.411
Wood             3  0.205
    BsmtQual  Ratio
TA       649 44.452
Gd       618 42.329
Ex       121  8.288
Fa        35  2.397
    BsmtCond  Ratio
TA      1311 89.795
Gd        65  4.452
Fa        45  3.082
Po         2  0.137
    BsmtExposure  Ratio
No           953 65.274
Av           221 15.137
Gd           134  9.178
Mn           114  7.808
     BsmtFinType1  Ratio
Unf           430 29.452
GLQ           418 28.630
ALQ           220 15.068
BLQ           148 10.137
Rec           133  9.110
LwQ            74  5.068
     BsmtFinType2  Ratio
Unf          1256 86.027
Rec            54  3.699
LwQ            46  3.151
BLQ            33  2.260
ALQ            19  1.301
GLQ            14  0.959
       Heating  Ratio
GasA      1428 97.808
GasW        18  1.233
Grav         7  0.479
Wall         4  0.274
OthW         2  0.137
Floor        1  0.068
    HeatingQC  Ratio
Ex        741 50.753
TA        428 29.315
Gd        241 16.507
Fa         49  3.356
Po          1  0.068
   CentralAir  Ratio
Y        1365 93.493
N          95  6.507
       Electrical  Ratio
SBrkr        1334 91.370
FuseA          94  6.438
FuseF          27  1.849
FuseP           3  0.205
Mix             1  0.068
    KitchenQual  Ratio
TA          735 50.342
Gd          586 40.137
Ex          100  6.849
Fa           39  2.671
      Functional  Ratio
Typ         1360 93.151
Min2          34  2.329
Min1          31  2.123
Mod           15  1.027
Maj1          14  0.959
Maj2           5  0.342
Sev            1  0.068
    FireplaceQu  Ratio
Gd          380 26.027
TA          313 21.438
Fa           33  2.260
Ex           24  1.644
Po           20  1.370
         GarageType  Ratio
Attchd          870 59.589
Detchd          387 26.507
BuiltIn          88  6.027
Basment          19  1.301
CarPort           9  0.616
2Types            6  0.411
     GarageFinish  Ratio
Unf           605 41.438
RFn           422 28.904
Fin           352 24.110
    GarageQual  Ratio
TA        1311 89.795
Fa          48  3.288
Gd          14  0.959
Ex           3  0.205
Po           3  0.205
    GarageCond  Ratio
TA        1326 90.822
Fa          35  2.397
Gd           9  0.616
Po           7  0.479
Ex           2  0.137
   PavedDrive  Ratio
Y        1340 91.781
N          90  6.164
P          30  2.055
    PoolQC  Ratio
Gd       3  0.205
Ex       2  0.137
Fa       2  0.137
       Fence  Ratio
MnPrv    157 10.753
GdPrv     59  4.041
GdWo      54  3.699
MnWw      11  0.753
      MiscFeature  Ratio
Shed           49  3.356
Gar2            2  0.137
Othr            2  0.137
TenC            1  0.068
       SaleType  Ratio
WD         1267 86.781
New         122  8.356
COD          43  2.945
ConLD         9  0.616
ConLI         5  0.342
ConLw         5  0.342
CWD           4  0.274
Oth           3  0.205
Con           2  0.137
         SaleCondition  Ratio
Normal            1198 82.055
Partial            125  8.562
Abnorml            101  6.918
Family              20  1.370
Alloca              12  0.822
AdjLand              4  0.274
   OverallCond  Ratio
5          821 56.233
6          252 17.260
7          205 14.041
8           72  4.932
4           57  3.904
3           25  1.712
9           22  1.507
2            5  0.342
1            1  0.068
   BsmtFullBath  Ratio
0           856 58.630
1           588 40.274
2            15  1.027
3             1  0.068
   BsmtHalfBath  Ratio
0          1378 94.384
1            80  5.479
2             2  0.137
   FullBath  Ratio
2       768 52.603
1       650 44.521
3        33  2.260
0         9  0.616
   HalfBath  Ratio
0       913 62.534
1       535 36.644
2        12  0.822
   BedroomAbvGr  Ratio
3           804 55.068
2           358 24.521
4           213 14.589
1            50  3.425
5            21  1.438
6             7  0.479
0             6  0.411
8             1  0.068
   KitchenAbvGr  Ratio
1          1392 95.342
2            65  4.452
3             2  0.137
0             1  0.068
   Fireplaces  Ratio
0         690 47.260
1         650 44.521
2         115  7.877
3           5  0.342
   GarageCars  Ratio
2         824 56.438
1         369 25.274
3         181 12.397
0          81  5.548
4           5  0.342
     PoolArea  Ratio
0        1453 99.521
512         1  0.068
648         1  0.068
576         1  0.068
555         1  0.068
480         1  0.068
519         1  0.068
738         1  0.068
      YrSold  Ratio
2009     338 23.151
2007     329 22.534
2006     314 21.507
2008     304 20.822
2010     175 11.986
'''

#Num Summary

def num_summary(dataframe, numerical_col, plot=False):
    #quantiles = [0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.99]
    print(dataframe[numerical_col].describe().T)

    if plot:
        dataframe[numerical_col].hist(bins=20)
        plt.xlabel(numerical_col)
        plt.title(numerical_col)
        plt.show()

for col in num_cols:
    num_summary(df_train, col)

'''
count   1460.000
mean     730.500
std      421.610
min        1.000
25%      365.750
50%      730.500
75%     1095.250
max     1460.000
Name: Id, dtype: float64
count   1460.000
mean      56.897
std       42.301
min       20.000
25%       20.000
50%       50.000
75%       70.000
max      190.000
Name: MSSubClass, dtype: float64
count   1201.000
mean      70.050
std       24.285
min       21.000
25%       59.000
50%       69.000
75%       80.000
max      313.000
Name: LotFrontage, dtype: float64
count     1460.000
mean     10516.828
std       9981.265
min       1300.000
25%       7553.500
50%       9478.500
75%      11601.500
max     215245.000
Name: LotArea, dtype: float64
count   1460.000
mean       6.099
std        1.383
min        1.000
25%        5.000
50%        6.000
75%        7.000
max       10.000
Name: OverallQual, dtype: float64
count   1460.000
mean    1971.268
std       30.203
min     1872.000
25%     1954.000
50%     1973.000
75%     2000.000
max     2010.000
Name: YearBuilt, dtype: float64
count   1460.000
mean    1984.866
std       20.645
min     1950.000
25%     1967.000
50%     1994.000
75%     2004.000
max     2010.000
Name: YearRemodAdd, dtype: float64
count   1452.000
mean     103.685
std      181.066
min        0.000
25%        0.000
50%        0.000
75%      166.000
max     1600.000
Name: MasVnrArea, dtype: float64
count   1460.000
mean     443.640
std      456.098
min        0.000
25%        0.000
50%      383.500
75%      712.250
max     5644.000
Name: BsmtFinSF1, dtype: float64
count   1460.000
mean      46.549
std      161.319
min        0.000
25%        0.000
50%        0.000
75%        0.000
max     1474.000
Name: BsmtFinSF2, dtype: float64
count   1460.000
mean     567.240
std      441.867
min        0.000
25%      223.000
50%      477.500
75%      808.000
max     2336.000
Name: BsmtUnfSF, dtype: float64
count   1460.000
mean    1057.429
std      438.705
min        0.000
25%      795.750
50%      991.500
75%     1298.250
max     6110.000
Name: TotalBsmtSF, dtype: float64
count   1460.000
mean    1162.627
std      386.588
min      334.000
25%      882.000
50%     1087.000
75%     1391.250
max     4692.000
Name: 1stFlrSF, dtype: float64
count   1460.000
mean     346.992
std      436.528
min        0.000
25%        0.000
50%        0.000
75%      728.000
max     2065.000
Name: 2ndFlrSF, dtype: float64
count   1460.000
mean       5.845
std       48.623
min        0.000
25%        0.000
50%        0.000
75%        0.000
max      572.000
Name: LowQualFinSF, dtype: float64
count   1460.000
mean    1515.464
std      525.480
min      334.000
25%     1129.500
50%     1464.000
75%     1776.750
max     5642.000
Name: GrLivArea, dtype: float64
count   1460.000
mean       6.518
std        1.625
min        2.000
25%        5.000
50%        6.000
75%        7.000
max       14.000
Name: TotRmsAbvGrd, dtype: float64
count   1379.000
mean    1978.506
std       24.690
min     1900.000
25%     1961.000
50%     1980.000
75%     2002.000
max     2010.000
Name: GarageYrBlt, dtype: float64
count   1460.000
mean     472.980
std      213.805
min        0.000
25%      334.500
50%      480.000
75%      576.000
max     1418.000
Name: GarageArea, dtype: float64
count   1460.000
mean      94.245
std      125.339
min        0.000
25%        0.000
50%        0.000
75%      168.000
max      857.000
Name: WoodDeckSF, dtype: float64
count   1460.000
mean      46.660
std       66.256
min        0.000
25%        0.000
50%       25.000
75%       68.000
max      547.000
Name: OpenPorchSF, dtype: float64
count   1460.000
mean      21.954
std       61.119
min        0.000
25%        0.000
50%        0.000
75%        0.000
max      552.000
Name: EnclosedPorch, dtype: float64
count   1460.000
mean       3.410
std       29.317
min        0.000
25%        0.000
50%        0.000
75%        0.000
max      508.000
Name: 3SsnPorch, dtype: float64
count   1460.000
mean      15.061
std       55.757
min        0.000
25%        0.000
50%        0.000
75%        0.000
max      480.000
Name: ScreenPorch, dtype: float64
count    1460.000
mean       43.489
std       496.123
min         0.000
25%         0.000
50%         0.000
75%         0.000
max     15500.000
Name: MiscVal, dtype: float64
count   1460.000
mean       6.322
std        2.704
min        1.000
25%        5.000
50%        6.000
75%        8.000
max       12.000
Name: MoSold, dtype: float64
count     1460.000
mean    180921.196
std      79442.503
min      34900.000
25%     129975.000
50%     163000.000
75%     214000.000
max     755000.000
Name: SalePrice, dtype: float64

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
    outlier_thresholds(df_train,col)
def check_outlier(dataframe, col_name):
    low_limit, up_limit = outlier_thresholds(dataframe, col_name)
    if dataframe[(dataframe[col_name] > up_limit) | (dataframe[col_name] < low_limit)].any(axis=None):
        return True
    else:
        return False

for col in num_cols:
    print(col,check_outlier(df_train,col))
'''
Id False
MSSubClass False
LotFrontage True
LotArea True
OverallQual False
YearBuilt False
YearRemodAdd False
MasVnrArea True
BsmtFinSF1 True
BsmtFinSF2 True
BsmtUnfSF False
TotalBsmtSF True
1stFlrSF True
2ndFlrSF False
LowQualFinSF True
GrLivArea True
TotRmsAbvGrd False
GarageYrBlt False
GarageArea False
WoodDeckSF True
OpenPorchSF True
EnclosedPorch True
3SsnPorch True
ScreenPorch True
MiscVal True
MoSold False
SalePrice True
'''

def replace_with_thresholds(dataframe, variable):
    low_limit, up_limit = outlier_thresholds(dataframe, variable)
    dataframe.loc[(dataframe[variable] < low_limit), variable] = low_limit
    dataframe.loc[(dataframe[variable] > up_limit), variable] = up_limit


for col in num_cols:
    if col != "SalePrice":
        replace_with_thresholds(df_train,col)

for col in num_cols:
    print(col,check_outlier(df_train,col))
'''
Id False
MSSubClass False
LotFrontage False
LotArea False
OverallQual False
YearBuilt False
YearRemodAdd False
MasVnrArea False
BsmtFinSF1 False
BsmtFinSF2 False
BsmtUnfSF False
TotalBsmtSF False
1stFlrSF False
2ndFlrSF False
LowQualFinSF False
GrLivArea False
TotRmsAbvGrd False
GarageYrBlt False
GarageArea False
WoodDeckSF False
OpenPorchSF False
EnclosedPorch False
3SsnPorch False
ScreenPorch False
MiscVal False
MoSold False
SalePrice True
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

na_columns = missing_values_table(df_train, na_name=True)
'''
              n_miss  ratio
PoolQC          1453 99.520
MiscFeature     1406 96.300
Alley           1369 93.770
Fence           1179 80.750
FireplaceQu      690 47.260
LotFrontage      259 17.740
GarageType        81  5.550
GarageYrBlt       81  5.550
GarageFinish      81  5.550
GarageQual        81  5.550
GarageCond        81  5.550
BsmtExposure      38  2.600
BsmtFinType2      38  2.600
BsmtFinType1      37  2.530
BsmtCond          37  2.530
BsmtQual          37  2.530
MasVnrArea         8  0.550
MasVnrType         8  0.550
Electrical         1  0.070
'''

no_cols = ["Alley","BsmtQual","BsmtCond","BsmtExposure","BsmtFinType1","BsmtFinType2","FireplaceQu",
           "GarageType","GarageFinish","GarageQual","GarageCond","PoolQC","Fence","MiscFeature"]
for col in no_cols:
    df_train[col].fillna("No", inplace=True)
na_columns = missing_values_table(df_train, na_name=True)
'''
             n_miss  ratio
LotFrontage     259 17.740
GarageYrBlt      81  5.550
MasVnrType        8  0.550
MasVnrArea        8  0.550
Electrical        1  0.070
'''
def quick_missing_imp(data, num_method="median", cat_length=20, target="SalePrice"):
    variables_with_na = [col for col in data.columns if data[col].isnull().sum() > 0]  # Eksik değere sahip olan değişkenler listelenir

    temp_target = data[target]

    print("# BEFORE")
    print(data[variables_with_na].isnull().sum(), "\n\n")  # Uygulama öncesi değişkenlerin eksik değerlerinin sayısı

    # değişken object ve sınıf sayısı cat_lengthe eşit veya altındaysa boş değerleri mode ile doldur
    data = data.apply(lambda x: x.fillna(x.mode()[0]) if (x.dtype == "O" and len(x.unique()) <= cat_length) else x, axis=0)

    # num_method mean ise tipi object olmayan değişkenlerin boş değerleri ortalama ile dolduruluyor
    if num_method == "mean":
        data = data.apply(lambda x: x.fillna(x.mean()) if x.dtype != "O" else x, axis=0)
    # num_method median ise tipi object olmayan değişkenlerin boş değerleri ortalama ile dolduruluyor
    elif num_method == "median":
        data = data.apply(lambda x: x.fillna(x.median()) if x.dtype != "O" else x, axis=0)

    data[target] = temp_target

    print("# AFTER \n Imputation method is 'MODE' for categorical variables!")
    print(" Imputation method is '" + num_method.upper() + "' for numeric variables! \n")
    print(data[variables_with_na].isnull().sum(), "\n\n")

    return data

df_train = quick_missing_imp(df_train, num_method="median", cat_length=17)
'''
# BEFORE
LotFrontage    259
MasVnrType       8
MasVnrArea       8
Electrical       1
GarageYrBlt     81
dtype: int64 


# AFTER 
 Imputation method is 'MODE' for categorical variables!
 Imputation method is 'MEDIAN' for numeric variables! 

LotFrontage    0
MasVnrType     0
MasVnrArea     0
Electrical     0
GarageYrBlt    0
dtype: int64 
'''

#Analysis of Correlation

print(df_train.corr())
'''
                  Id  MSSubClass  LotFrontage  LotArea  OverallQual  OverallCond  YearBuilt  YearRemodAdd  MasVnrArea  BsmtFinSF1  BsmtFinSF2  BsmtUnfSF  TotalBsmtSF  1stFlrSF  2ndFlrSF  LowQualFinSF  GrLivArea  BsmtFullBath  BsmtHalfBath  FullBath  HalfBath  BedroomAbvGr  KitchenAbvGr  TotRmsAbvGrd  Fireplaces  GarageYrBlt  GarageCars  GarageArea  WoodDeckSF  OpenPorchSF  EnclosedPorch  3SsnPorch  ScreenPorch  PoolArea  MiscVal  MoSold  YrSold  SalePrice
Id             1.000       0.011       -0.019   -0.003       -0.028        0.013     -0.013        -0.022      -0.049      -0.011      -0.003     -0.008       -0.022     0.007     0.006           NaN      0.005         0.002        -0.020     0.006     0.007         0.038         0.003         0.027      -0.020       -0.000       0.017       0.018      -0.029       -0.002          0.006        NaN          NaN     0.057      NaN   0.021   0.001     -0.022
MSSubClass     0.011       1.000       -0.380   -0.323        0.033       -0.059      0.028         0.041       0.022      -0.073      -0.063     -0.141       -0.249    -0.258     0.308           NaN      0.076         0.003        -0.002     0.132     0.177        -0.023         0.282         0.040      -0.046        0.081      -0.040      -0.099      -0.013       -0.005         -0.015        NaN          NaN     0.008      NaN  -0.014  -0.021     -0.084
LotFrontage   -0.019      -0.380        1.000    0.518        0.237       -0.061      0.122         0.079       0.181       0.161       0.041      0.139        0.332     0.386     0.075           NaN      0.347         0.078        -0.005     0.187     0.049         0.248        -0.003         0.322       0.231        0.066       0.287       0.327       0.082        0.138         -0.000        NaN          NaN     0.140      NaN   0.012   0.006      0.358
LotArea       -0.003      -0.323        0.518    1.000        0.195       -0.017      0.044         0.041       0.157       0.237       0.079      0.058        0.345     0.439     0.120           NaN      0.423         0.144         0.049     0.196     0.083         0.246        -0.015         0.355       0.351       -0.007       0.284       0.327       0.218        0.161         -0.013        NaN          NaN     0.117      NaN  -0.005  -0.036      0.405
OverallQual   -0.028       0.033        0.237    0.195        1.000       -0.092      0.572         0.551       0.418       0.236      -0.102      0.308        0.546     0.478     0.295           NaN      0.596         0.111        -0.040     0.551     0.273         0.102        -0.184         0.427       0.397        0.514       0.601       0.562       0.239        0.320         -0.123        NaN          NaN     0.065      NaN   0.071  -0.027      0.791
OverallCond    0.013      -0.059       -0.061   -0.017       -0.092        1.000     -0.376         0.074      -0.130      -0.046       0.060     -0.137       -0.176    -0.146     0.029           NaN     -0.080        -0.055         0.118    -0.194    -0.061         0.013        -0.087        -0.058      -0.024       -0.306      -0.186      -0.152      -0.002       -0.040          0.075        NaN          NaN    -0.002      NaN  -0.004   0.044     -0.078
YearBuilt     -0.013       0.028        0.122    0.044        0.572       -0.376      1.000         0.593       0.323       0.253      -0.068      0.149        0.402     0.285     0.010           NaN      0.199         0.188        -0.038     0.468     0.243        -0.071        -0.175         0.096       0.148        0.777       0.538       0.479       0.227        0.210         -0.389        NaN          NaN     0.005      NaN   0.012  -0.014      0.523
YearRemodAdd  -0.022       0.041        0.079    0.041        0.551        0.074      0.593         1.000       0.183       0.128      -0.104      0.181        0.298     0.242     0.140           NaN      0.290         0.119        -0.012     0.439     0.183        -0.041        -0.150         0.192       0.113        0.616       0.421       0.372       0.209        0.243         -0.201        NaN          NaN     0.006      NaN   0.021   0.036      0.507
MasVnrArea    -0.049       0.022        0.181    0.157        0.418       -0.130      0.323         0.183       1.000       0.258      -0.071      0.120        0.367     0.348     0.165           NaN      0.385         0.089         0.027     0.275     0.202         0.104        -0.038         0.287       0.252        0.253       0.374       0.385       0.166        0.142         -0.114        NaN          NaN     0.014      NaN  -0.004  -0.001      0.473
BsmtFinSF1    -0.011      -0.073        0.161    0.237        0.236       -0.046      0.253         0.128       0.258       1.000      -0.024     -0.513        0.484     0.414    -0.149           NaN      0.166         0.660         0.071     0.057    -0.001        -0.112        -0.083         0.030       0.253        0.149       0.231       0.288       0.206        0.102         -0.110        NaN          NaN     0.091      NaN  -0.007   0.014      0.403
BsmtFinSF2    -0.003      -0.063        0.041    0.079       -0.102        0.060     -0.068        -0.104      -0.071      -0.024       1.000     -0.232        0.076     0.069    -0.106           NaN     -0.037         0.142         0.112    -0.100    -0.049         0.008        -0.034        -0.044       0.029       -0.105      -0.058      -0.025       0.078       -0.039          0.050        NaN          NaN     0.070      NaN  -0.022   0.031     -0.051
BsmtUnfSF     -0.008      -0.141        0.139    0.058        0.308       -0.137      0.149         0.181       0.120      -0.513      -0.232      1.000        0.433     0.325     0.004           NaN      0.246        -0.423        -0.096     0.289    -0.041         0.167         0.030         0.251       0.052        0.186       0.214       0.183      -0.003        0.131         -0.003        NaN          NaN    -0.035      NaN   0.035  -0.041      0.214
TotalBsmtSF   -0.022      -0.249        0.332    0.345        0.546       -0.176      0.402         0.298       0.367       0.484       0.076      0.433        1.000     0.810    -0.189           NaN      0.427         0.305         0.001     0.333    -0.057         0.052        -0.071         0.280       0.336        0.316       0.451       0.485       0.238        0.243         -0.105        NaN          NaN     0.072      NaN   0.023  -0.016      0.641
1stFlrSF       0.007      -0.258        0.386    0.439        0.478       -0.146      0.285         0.242       0.348       0.414       0.069      0.325        0.810     1.000    -0.212           NaN      0.549         0.241         0.003     0.386    -0.126         0.129         0.070         0.408       0.409        0.226       0.448       0.488       0.238        0.207         -0.072        NaN          NaN     0.100      NaN   0.038  -0.014      0.620
2ndFlrSF       0.006       0.308        0.075    0.120        0.295        0.029      0.010         0.140       0.165      -0.149      -0.106      0.004       -0.189    -0.212     1.000           NaN      0.695        -0.169        -0.024     0.421     0.610         0.503         0.059         0.616       0.195        0.068       0.184       0.138       0.093        0.209          0.052        NaN          NaN     0.081      NaN   0.035  -0.029      0.319
LowQualFinSF     NaN         NaN          NaN      NaN          NaN          NaN        NaN           NaN         NaN         NaN         NaN        NaN          NaN       NaN       NaN           NaN        NaN           NaN           NaN       NaN       NaN           NaN           NaN           NaN         NaN          NaN         NaN         NaN         NaN          NaN            NaN        NaN          NaN       NaN      NaN     NaN     NaN        NaN
GrLivArea      0.005       0.076        0.347    0.423        0.596       -0.080      0.199         0.290       0.385       0.166      -0.037      0.246        0.427     0.549     0.695           NaN      1.000         0.028        -0.020     0.638     0.421         0.532         0.103         0.833       0.462        0.221       0.475       0.467       0.250        0.328         -0.003        NaN          NaN     0.140      NaN   0.055  -0.037      0.719
BsmtFullBath   0.002       0.003        0.078    0.144        0.111       -0.055      0.188         0.119       0.089       0.660       0.142     -0.423        0.305     0.241    -0.169           NaN      0.028         1.000        -0.148    -0.065    -0.031        -0.151        -0.042        -0.053       0.138        0.119       0.132       0.179       0.173        0.074         -0.053        NaN          NaN     0.068      NaN  -0.025   0.067      0.227
BsmtHalfBath  -0.020      -0.002       -0.005    0.049       -0.040        0.118     -0.038        -0.012       0.027       0.071       0.112     -0.096        0.001     0.003    -0.024           NaN     -0.020        -0.148         1.000    -0.055    -0.012         0.047        -0.038        -0.024       0.029       -0.075      -0.021      -0.025       0.041       -0.024         -0.021        NaN          NaN     0.020      NaN   0.033  -0.047     -0.017
FullBath       0.006       0.132        0.187    0.196        0.551       -0.194      0.468         0.439       0.275       0.057      -0.100      0.289        0.333     0.386     0.421           NaN      0.638        -0.065        -0.055     1.000     0.136         0.363         0.133         0.555       0.244        0.467       0.470       0.406       0.193        0.267         -0.121        NaN          NaN     0.050      NaN   0.056  -0.020      0.561
HalfBath       0.007       0.177        0.049    0.083        0.273       -0.061      0.243         0.183       0.202      -0.001      -0.049     -0.041       -0.057    -0.126     0.610           NaN      0.421        -0.031        -0.012     0.136     1.000         0.227        -0.068         0.343       0.204        0.190       0.219       0.164       0.107        0.208         -0.097        NaN          NaN     0.022      NaN  -0.009  -0.010      0.284
BedroomAbvGr   0.038      -0.023        0.248    0.246        0.102        0.013     -0.071        -0.041       0.104      -0.112       0.008      0.167        0.052     0.129     0.503           NaN      0.532        -0.151         0.047     0.363     0.227         1.000         0.199         0.677       0.108       -0.060       0.086       0.065       0.050        0.093          0.036        NaN          NaN     0.071      NaN   0.047  -0.036      0.168
KitchenAbvGr   0.003       0.282       -0.003   -0.015       -0.184       -0.087     -0.175        -0.150      -0.038      -0.083      -0.034      0.030       -0.071     0.070     0.059           NaN      0.103        -0.042        -0.038     0.133    -0.068         0.199         1.000         0.256      -0.124       -0.105      -0.051      -0.064      -0.091       -0.072          0.040        NaN          NaN    -0.015      NaN   0.027   0.032     -0.136
TotRmsAbvGrd   0.027       0.040        0.322    0.355        0.427       -0.058      0.096         0.192       0.287       0.030      -0.044      0.251        0.280     0.408     0.616           NaN      0.833        -0.053        -0.024     0.555     0.343         0.677         0.256         1.000       0.326        0.140       0.362       0.338       0.168        0.241         -0.001        NaN          NaN     0.084      NaN   0.037  -0.035      0.534
Fireplaces    -0.020      -0.046        0.231    0.351        0.397       -0.024      0.148         0.113       0.252       0.253       0.029      0.052        0.336     0.409     0.195           NaN      0.462         0.138         0.029     0.244     0.204         0.108        -0.124         0.326       1.000        0.043       0.301       0.269       0.201        0.172         -0.029        NaN          NaN     0.095      NaN   0.046  -0.024      0.467
GarageYrBlt   -0.000       0.081        0.066   -0.007        0.514       -0.306      0.777         0.616       0.253       0.149      -0.105      0.186        0.316     0.226     0.068           NaN      0.221         0.119        -0.075     0.467     0.190        -0.060        -0.105         0.140       0.043        1.000       0.474       0.469       0.222        0.234         -0.283        NaN          NaN    -0.015      NaN   0.005  -0.001      0.467
GarageCars     0.017      -0.040        0.287    0.284        0.601       -0.186      0.538         0.421       0.374       0.231      -0.058      0.214        0.451     0.448     0.184           NaN      0.475         0.132        -0.021     0.470     0.219         0.086        -0.051         0.362       0.301        0.474       1.000       0.882       0.227        0.226         -0.160        NaN          NaN     0.021      NaN   0.041  -0.039      0.640
GarageArea     0.018      -0.099        0.327    0.327        0.562       -0.152      0.479         0.372       0.385       0.288      -0.025      0.183        0.485     0.488     0.138           NaN      0.467         0.179        -0.025     0.406     0.164         0.065        -0.064         0.338       0.269        0.469       0.882       1.000       0.225        0.252         -0.132        NaN          NaN     0.061      NaN   0.028  -0.027      0.623
WoodDeckSF    -0.029      -0.013        0.082    0.218        0.239       -0.002      0.227         0.209       0.166       0.206       0.078     -0.003        0.238     0.238     0.093           NaN      0.250         0.173         0.041     0.193     0.107         0.050        -0.091         0.168       0.201        0.222       0.227       0.225       1.000        0.065         -0.129        NaN          NaN     0.074      NaN   0.019   0.024      0.325
OpenPorchSF   -0.002      -0.005        0.138    0.161        0.320       -0.040      0.210         0.243       0.142       0.102      -0.039      0.131        0.243     0.207     0.209           NaN      0.328         0.074        -0.024     0.267     0.208         0.093        -0.072         0.241       0.172        0.234       0.226       0.252       0.065        1.000         -0.098        NaN          NaN     0.064      NaN   0.069  -0.061      0.329
EnclosedPorch  0.006      -0.015       -0.000   -0.013       -0.123        0.075     -0.389        -0.201      -0.114      -0.110       0.050     -0.003       -0.105    -0.072     0.052           NaN     -0.003        -0.053        -0.021    -0.121    -0.097         0.036         0.040        -0.001      -0.029       -0.283      -0.160      -0.132      -0.129       -0.098          1.000        NaN          NaN     0.016      NaN  -0.025  -0.006     -0.137
3SsnPorch        NaN         NaN          NaN      NaN          NaN          NaN        NaN           NaN         NaN         NaN         NaN        NaN          NaN       NaN       NaN           NaN        NaN           NaN           NaN       NaN       NaN           NaN           NaN           NaN         NaN          NaN         NaN         NaN         NaN          NaN            NaN        NaN          NaN       NaN      NaN     NaN     NaN        NaN
ScreenPorch      NaN         NaN          NaN      NaN          NaN          NaN        NaN           NaN         NaN         NaN         NaN        NaN          NaN       NaN       NaN           NaN        NaN           NaN           NaN       NaN       NaN           NaN           NaN           NaN         NaN          NaN         NaN         NaN         NaN          NaN            NaN        NaN          NaN       NaN      NaN     NaN     NaN        NaN
PoolArea       0.057       0.008        0.140    0.117        0.065       -0.002      0.005         0.006       0.014       0.091       0.070     -0.035        0.072     0.100     0.081           NaN      0.140         0.068         0.020     0.050     0.022         0.071        -0.015         0.084       0.095       -0.015       0.021       0.061       0.074        0.064          0.016        NaN          NaN     1.000      NaN  -0.034  -0.060      0.092
MiscVal          NaN         NaN          NaN      NaN          NaN          NaN        NaN           NaN         NaN         NaN         NaN        NaN          NaN       NaN       NaN           NaN        NaN           NaN           NaN       NaN       NaN           NaN           NaN           NaN         NaN          NaN         NaN         NaN         NaN          NaN            NaN        NaN          NaN       NaN      NaN     NaN     NaN        NaN
MoSold         0.021      -0.014        0.012   -0.005        0.071       -0.004      0.012         0.021      -0.004      -0.007      -0.022      0.035        0.023     0.038     0.035           NaN      0.055        -0.025         0.033     0.056    -0.009         0.047         0.027         0.037       0.046        0.005       0.041       0.028       0.019        0.069         -0.025        NaN          NaN    -0.034      NaN   1.000  -0.146      0.046
YrSold         0.001      -0.021        0.006   -0.036       -0.027        0.044     -0.014         0.036      -0.001       0.014       0.031     -0.041       -0.016    -0.014    -0.029           NaN     -0.037         0.067        -0.047    -0.020    -0.010        -0.036         0.032        -0.035      -0.024       -0.001      -0.039      -0.027       0.024       -0.061         -0.006        NaN          NaN    -0.060      NaN  -0.146   1.000     -0.029
SalePrice     -0.022      -0.084        0.358    0.405        0.791       -0.078      0.523         0.507       0.473       0.403      -0.051      0.214        0.641     0.620     0.319           NaN      0.719         0.227        -0.017     0.561     0.284         0.168        -0.136         0.534       0.467        0.467       0.640       0.623       0.325        0.329         -0.137        NaN          NaN     0.092      NaN   0.046  -0.029      1.000
'''

f, ax = plt.subplots(figsize=[18, 13])
sns.heatmap(df_train.corr(), annot=True, fmt=".2f", ax=ax, cmap="magma")
ax.set_title("Correlation Matrix", fontsize=20)
plt.show()

#Feature Engineering

cat_cols,num_cols,cat_but_car=grab_col_names(df_train)
'''
Observations: 1460
Variables: 81
cat_cols: 57
num_cols: 23
cat_but_car: 1
num_but_cat: 15
'''

def label_encoder(dataframe, binary_col):
    labelencoder = LabelEncoder()
    dataframe[binary_col] = labelencoder.fit_transform(dataframe[binary_col])
    return dataframe
binary_cols = [col for col in df_train.columns if df_train[col].dtypes == "O" and df_train[col].nunique() == 2]
for col in binary_cols:
    df = label_encoder(df_train, col)
print(df_train.head())
'''
     Id  MSSubClass MSZoning  LotFrontage   LotArea  Street Alley LotShape LandContour  Utilities LotConfig LandSlope Neighborhood Condition1 Condition2 BldgType HouseStyle  OverallQual  OverallCond  YearBuilt  YearRemodAdd RoofStyle RoofMatl Exterior1st Exterior2nd MasVnrType  MasVnrArea ExterQual ExterCond Foundation BsmtQual BsmtCond BsmtExposure BsmtFinType1  BsmtFinSF1 BsmtFinType2  BsmtFinSF2  BsmtUnfSF  TotalBsmtSF Heating HeatingQC  CentralAir Electrical  1stFlrSF  2ndFlrSF  LowQualFinSF  GrLivArea  BsmtFullBath  BsmtHalfBath  FullBath  HalfBath  BedroomAbvGr  KitchenAbvGr KitchenQual  TotRmsAbvGrd Functional  Fireplaces FireplaceQu GarageType  GarageYrBlt GarageFinish  GarageCars  GarageArea GarageQual GarageCond PavedDrive  WoodDeckSF  OpenPorchSF  EnclosedPorch  3SsnPorch  ScreenPorch  PoolArea PoolQC Fence MiscFeature  MiscVal  MoSold  YrSold SaleType SaleCondition  SalePrice
0 1.000          60       RL       65.000  8450.000       1    No      Reg         Lvl          0    Inside       Gtl                                                                                        Norm       Norm     1Fam     2Story        7.000            5   2003.000          2003     Gable  CompShg     VinylSd     VinylSd    BrkFace     196.000        Gd        TA      PConc       Gd       TA           No          GLQ     706.000          Unf       0.000    150.000      856.000    GasA        Ex           1      SBrkr   856.000   854.000             0   1710.000             1             0         2         1             3             1          Gd             8        Typ           0          No     Attchd     2003.000          RFn           2     548.000         TA         TA          Y           0           61              0          0            0         0     No    No          No        0   2.000    2008       WD        Normal     208500
1 2.000          20       RL       80.000  9600.000       1    No      Reg         Lvl          0       FR2       Gtl      Veenker      Feedr       Norm     1Fam     1Story        6.000            8   1976.000          1976     Gable  CompShg     MetalSd     MetalSd       None       0.000        TA        TA     CBlock       Gd       TA           Gd          ALQ     978.000          Unf       0.000    284.000     1262.000    GasA        Ex           1      SBrkr  1262.000     0.000             0   1262.000             0             1         2         0             3             1          TA             6        Typ           1          TA     Attchd     1976.000          RFn           2     460.000         TA         TA          Y         298            0              0          0            0         0     No    No          No        0   5.000    2007       WD        Normal     181500
2 3.000          60       RL       68.000 11250.000       1    No      IR1         Lvl          0    Inside       Gtl      CollgCr       Norm       Norm     1Fam     2Story        7.000            5   2001.000          2002     Gable  CompShg     VinylSd     VinylSd    BrkFace     162.000        Gd        TA      PConc       Gd       TA           Mn          GLQ     486.000          Unf       0.000    434.000      920.000    GasA        Ex           1      SBrkr   920.000   866.000             0   1786.000             1             0         2         1             3             1          Gd             6        Typ           1          TA     Attchd     2001.000          RFn           2     608.000         TA         TA          Y           0           42              0          0            0         0     No    No          No        0   9.000    2008       WD        Normal     223500
3 4.000          70       RL       60.000  9550.000       1    No      IR1         Lvl          0    Corner       Gtl      Crawfor       Norm       Norm     1Fam     2Story        7.000            5   1915.000          1970     Gable  CompShg     Wd Sdng     Wd Shng       None       0.000        TA        TA     BrkTil       TA       Gd           No          ALQ     216.000          Unf       0.000    540.000      756.000    GasA        Gd           1      SBrkr   961.000   756.000             0   1717.000             1             0         1         0             3             1          Gd             7        Typ           1          Gd     Detchd     1998.000          Unf           3     642.000         TA         TA          Y           0           35            272          0            0         0     No    No          No        0   2.000    2006       WD       Abnorml     140000
4 5.000          60       RL       84.000 14260.000       1    No      IR1         Lvl          0       FR2       Gtl      NoRidge       Norm       Norm     1Fam     2Story        8.000            5   2000.000          2000     Gable  CompShg     VinylSd     VinylSd    BrkFace     350.000        Gd        TA      PConc       Gd       TA           Av          GLQ     655.000          Unf       0.000    490.000     1145.000    GasA        Ex           1      SBrkr  1145.000  1053.000             0   2198.000             1             0         2         1             4             1          Gd             9        Typ           1          TA     Attchd     2000.000          RFn           3     836.000         TA         TA          Y         192           84              0          0            0         0     No    No          No        0  12.000    2008       WD        Normal     250000
'''
print(binary_cols)           #['Street', 'Utilities', 'CentralAir']

cat_cols = [col for col in cat_cols if col not in binary_cols]
def one_hot_encoder(dataframe, categorical_cols, drop_first=False):
    dataframe = pd.get_dummies(dataframe, columns=categorical_cols, drop_first=drop_first)
    return dataframe
df_train = one_hot_encoder(df_train, cat_cols, drop_first=True)
df_train = one_hot_encoder(df_train, cat_but_car, drop_first=True)
print(df_train.head())
'''
     Id  MSSubClass  LotFrontage   LotArea  Street  Utilities Neighborhood  OverallQual  YearBuilt  YearRemodAdd  MasVnrArea  BsmtFinSF1  BsmtFinSF2  BsmtUnfSF  TotalBsmtSF  CentralAir  1stFlrSF  2ndFlrSF  GrLivArea  TotRmsAbvGrd  GarageYrBlt  GarageArea  WoodDeckSF  OpenPorchSF  EnclosedPorch  MoSold  SalePrice  MSZoning_FV  MSZoning_RH  MSZoning_RL  MSZoning_RM  Alley_No  Alley_Pave  LotShape_IR2  LotShape_IR3  LotShape_Reg  LandContour_HLS  LandContour_Low  LandContour_Lvl  LotConfig_CulDSac  LotConfig_FR2  LotConfig_FR3  LotConfig_Inside  LandSlope_Mod  LandSlope_Sev  Condition1_Feedr  Condition1_Norm  Condition1_PosA  Condition1_PosN  Condition1_RRAe  Condition1_RRAn  Condition1_RRNe  Condition1_RRNn  Condition2_Feedr  Condition2_Norm  Condition2_PosA  Condition2_PosN  Condition2_RRAe  Condition2_RRAn  Condition2_RRNn  BldgType_2fmCon  BldgType_Duplex  BldgType_Twnhs  BldgType_TwnhsE  HouseStyle_1.5Unf  HouseStyle_1Story  HouseStyle_2.5Fin  HouseStyle_2.5Unf  HouseStyle_2Story  HouseStyle_SFoyer  HouseStyle_SLvl  RoofStyle_Gable  RoofStyle_Gambrel  RoofStyle_Hip  RoofStyle_Mansard  RoofStyle_Shed  RoofMatl_CompShg  RoofMatl_Membran  RoofMatl_Metal  RoofMatl_Roll  RoofMatl_Tar&Grv  RoofMatl_WdShake  RoofMatl_WdShngl  Exterior1st_AsphShn  Exterior1st_BrkComm  Exterior1st_BrkFace  Exterior1st_CBlock  Exterior1st_CemntBd  Exterior1st_HdBoard  Exterior1st_ImStucc  Exterior1st_MetalSd  Exterior1st_Plywood  Exterior1st_Stone  Exterior1st_Stucco  Exterior1st_VinylSd  Exterior1st_Wd Sdng  Exterior1st_WdShing  Exterior2nd_AsphShn  Exterior2nd_Brk Cmn  Exterior2nd_BrkFace  Exterior2nd_CBlock  Exterior2nd_CmentBd  Exterior2nd_HdBoard  Exterior2nd_ImStucc  Exterior2nd_MetalSd  Exterior2nd_Other  Exterior2nd_Plywood  Exterior2nd_Stone  Exterior2nd_Stucco  Exterior2nd_VinylSd  Exterior2nd_Wd Sdng  Exterior2nd_Wd Shng  MasVnrType_BrkFace  MasVnrType_None  MasVnrType_Stone  ExterQual_Fa  ExterQual_Gd  ExterQual_TA  ExterCond_Fa  ExterCond_Gd  ExterCond_Po  ExterCond_TA  Foundation_CBlock  Foundation_PConc  Foundation_Slab  Foundation_Stone  Foundation_Wood  BsmtQual_Fa  BsmtQual_Gd  BsmtQual_No  BsmtQual_TA  BsmtCond_Gd  BsmtCond_No  BsmtCond_Po  BsmtCond_TA  BsmtExposure_Gd  BsmtExposure_Mn  BsmtExposure_No  BsmtFinType1_BLQ  BsmtFinType1_GLQ  BsmtFinType1_LwQ  BsmtFinType1_No  BsmtFinType1_Rec  BsmtFinType1_Unf  BsmtFinType2_BLQ  BsmtFinType2_GLQ  BsmtFinType2_LwQ  BsmtFinType2_No  BsmtFinType2_Rec  BsmtFinType2_Unf  Heating_GasA  Heating_GasW  Heating_Grav  Heating_OthW  Heating_Wall  HeatingQC_Fa  HeatingQC_Gd  HeatingQC_Po  HeatingQC_TA  Electrical_FuseF  Electrical_FuseP  Electrical_Mix  Electrical_SBrkr  KitchenQual_Fa  KitchenQual_Gd  KitchenQual_TA  Functional_Maj2  Functional_Min1  Functional_Min2  Functional_Mod  Functional_Sev  Functional_Typ  FireplaceQu_Fa  FireplaceQu_Gd  FireplaceQu_No  FireplaceQu_Po  FireplaceQu_TA  GarageType_Attchd  GarageType_Basment  GarageType_BuiltIn  GarageType_CarPort  GarageType_Detchd  GarageType_No  GarageFinish_No  GarageFinish_RFn  GarageFinish_Unf  GarageQual_Fa  GarageQual_Gd  GarageQual_No  GarageQual_Po  GarageQual_TA  GarageCond_Fa  GarageCond_Gd  GarageCond_No  GarageCond_Po  GarageCond_TA  PavedDrive_P  PavedDrive_Y  PoolQC_Fa  PoolQC_Gd  PoolQC_No  Fence_GdWo  Fence_MnPrv  Fence_MnWw  Fence_No  MiscFeature_No  MiscFeature_Othr  MiscFeature_Shed  MiscFeature_TenC  SaleType_CWD  SaleType_Con  SaleType_ConLD  SaleType_ConLI  SaleType_ConLw  SaleType_New  SaleType_Oth  SaleType_WD  SaleCondition_AdjLand  SaleCondition_Alloca  SaleCondition_Family  SaleCondition_Normal  SaleCondition_Partial  OverallCond_2  OverallCond_3  OverallCond_4  OverallCond_5  OverallCond_6  OverallCond_7  OverallCond_8  OverallCond_9  BsmtFullBath_1  BsmtFullBath_2  BsmtFullBath_3  BsmtHalfBath_1  BsmtHalfBath_2  FullBath_1  FullBath_2  FullBath_3  HalfBath_1  HalfBath_2  BedroomAbvGr_1  BedroomAbvGr_2  BedroomAbvGr_3  BedroomAbvGr_4  BedroomAbvGr_5  BedroomAbvGr_6  BedroomAbvGr_8  KitchenAbvGr_1  KitchenAbvGr_2  KitchenAbvGr_3  Fireplaces_1  Fireplaces_2  Fireplaces_3  GarageCars_1  GarageCars_2  GarageCars_3  GarageCars_4  PoolArea_480  PoolArea_512  PoolArea_519  PoolArea_555  PoolArea_576  PoolArea_648  PoolArea_738  YrSold_2007  YrSold_2008  YrSold_2009  YrSold_2010
0 1.000          60       65.000  8450.000       1          0      CollgCr        7.000   2003.000          2003     196.000     706.000       0.000    150.000      856.000           1   856.000   854.000   1710.000             8     2003.000     548.000           0           61              0   2.000     208500            0            0            1            0         1           0             0             0             1                0                0                1                  0              0              0                 1              0              0                 0                1                0                0                0                0                0                0                 0                1                0                0                0                0                0                0                0               0                0                  0                  0                  0                  0                  1                  0                0                1                  0              0                  0               0                 1                 0               0              0                 0                 0                 0                    0                    0                    0                   0                    0                    0                    0                    0                    0                  0                   0                    1                    0                    0                    0                    0                    0                   0                    0                    0                    0                    0                  0                    0                  0                   0                    1                    0                    0                   1                0                 0             0             1             0             0             0             0             1                  0                 1                0                 0                0            0            1            0            0            0            0            0            1                0                0                1                 0                 1                 0                0                 0                 0                 0                 0                 0                0                 0                 1             1             0             0             0             0             0             0             0             0                 0                 0               0                 1               0               1               0                0                0                0               0               0               1               0               0               1               0               0                  1                   0                   0                   0                  0              0                0                 1                 0              0              0              0              0              1              0              0              0              0              1             0             1          0          0          1           0            0           0         1               1                 0                 0                 0             0             0               0               0               0             0             0            1                      0                     0                     0                     1                      0              0              0              0              1              0              0              0              0               1               0               0               0               0           0           1           0           1           0               0               0               1               0               0               0               0               1               0               0             0             0             0             0             1             0             0             0             0             0             0             0             0             0            0            1            0            0
1 2.000          20       80.000  9600.000       1          0      Veenker        6.000   1976.000          1976       0.000     978.000       0.000    284.000     1262.000           1  1262.000     0.000   1262.000             6     1976.000     460.000         298            0              0   5.000     181500            0            0            1            0         1           0             0             0             1                0                0                1                  0              1              0                 0              0              0                 1                0                0                0                0                0                0                0                 0                1                0                0                0                0                0                0                0               0                0                  0                  1                  0                  0                  0                  0                0                1                  0              0                  0               0                 1                 0               0              0                 0                 0                 0                    0                    0                    0                   0                    0                    0                    0                    1                    0                  0                   0                    0                    0                    0                    0                    0                    0                   0                    0                    0                    0                    1                  0                    0                  0                   0                    0                    0                    0                   0                1                 0             0             0             1             0             0             0             1                  1                 0                0                 0                0            0            1            0            0            0            0            0            1                1                0                0                 0                 0                 0                0                 0                 0                 0                 0                 0                0                 0                 1             1             0             0             0             0             0             0             0             0                 0                 0               0                 1               0               0               1                0                0                0               0               0               1               0               0               0               0               1                  1                   0                   0                   0                  0              0                0                 1                 0              0              0              0              0              1              0              0              0              0              1             0             1          0          0          1           0            0           0         1               1                 0                 0                 0             0             0               0               0               0             0             0            1                      0                     0                     0                     1                      0              0              0              0              0              0              0              1              0               0               0               0               1               0           0           1           0           0           0               0               0               1               0               0               0               0               1               0               0             1             0             0             0             1             0             0             0             0             0             0             0             0             0            1            0            0            0
2 3.000          60       68.000 11250.000       1          0      CollgCr        7.000   2001.000          2002     162.000     486.000       0.000    434.000      920.000           1   920.000   866.000   1786.000             6     2001.000     608.000           0           42              0   9.000     223500            0            0            1            0         1           0             0             0             0                0                0                1                  0              0              0                 1              0              0                 0                1                0                0                0                0                0                0                 0                1                0                0                0                0                0                0                0               0                0                  0                  0                  0                  0                  1                  0                0                1                  0              0                  0               0                 1                 0               0              0                 0                 0                 0                    0                    0                    0                   0                    0                    0                    0                    0                    0                  0                   0                    1                    0                    0                    0                    0                    0                   0                    0                    0                    0                    0                  0                    0                  0                   0                    1                    0                    0                   1                0                 0             0             1             0             0             0             0             1                  0                 1                0                 0                0            0            1            0            0            0            0            0            1                0                1                0                 0                 1                 0                0                 0                 0                 0                 0                 0                0                 0                 1             1             0             0             0             0             0             0             0             0                 0                 0               0                 1               0               1               0                0                0                0               0               0               1               0               0               0               0               1                  1                   0                   0                   0                  0              0                0                 1                 0              0              0              0              0              1              0              0              0              0              1             0             1          0          0          1           0            0           0         1               1                 0                 0                 0             0             0               0               0               0             0             0            1                      0                     0                     0                     1                      0              0              0              0              1              0              0              0              0               1               0               0               0               0           0           1           0           1           0               0               0               1               0               0               0               0               1               0               0             1             0             0             0             1             0             0             0             0             0             0             0             0             0            0            1            0            0
3 4.000          70       60.000  9550.000       1          0      Crawfor        7.000   1915.000          1970       0.000     216.000       0.000    540.000      756.000           1   961.000   756.000   1717.000             7     1998.000     642.000           0           35            272   2.000     140000            0            0            1            0         1           0             0             0             0                0                0                1                  0              0              0                 0              0              0                 0                1                0                0                0                0                0                0                 0                1                0                0                0                0                0                0                0               0                0                  0                  0                  0                  0                  1                  0                0                1                  0              0                  0               0                 1                 0               0              0                 0                 0                 0                    0                    0                    0                   0                    0                    0                    0                    0                    0                  0                   0                    0                    1                    0                    0                    0                    0                   0                    0                    0                    0                    0                  0                    0                  0                   0                    0                    0                    1                   0                1                 0             0             0             1             0             0             0             1                  0                 0                0                 0                0            0            0            0            1            1            0            0            0                0                0                1                 0                 0                 0                0                 0                 0                 0                 0                 0                0                 0                 1             1             0             0             0             0             0             1             0             0                 0                 0               0                 1               0               1               0                0                0                0               0               0               1               0               1               0               0               0                  0                   0                   0                   0                  1              0                0                 0                 1              0              0              0              0              1              0              0              0              0              1             0             1          0          0          1           0            0           0         1               1                 0                 0                 0             0             0               0               0               0             0             0            1                      0                     0                     0                     0                      0              0              0              0              1              0              0              0              0               1               0               0               0               0           1           0           0           0           0               0               0               1               0               0               0               0               1               0               0             1             0             0             0             0             1             0             0             0             0             0             0             0             0            0            0            0            0
4 5.000          60       84.000 14260.000       1          0      NoRidge        8.000   2000.000          2000     350.000     655.000       0.000    490.000     1145.000           1  1145.000  1053.000   2198.000             9     2000.000     836.000         192           84              0  12.000     250000            0            0            1            0         1           0             0             0             0                0                0                1                  0              1              0                 0              0              0                 0                1                0                0                0                0                0                0                 0                1                0                0                0                0                0                0                0               0                0                  0                  0                  0                  0                  1                  0                0                1                  0              0                  0               0                 1                 0               0              0                 0                 0                 0                    0                    0                    0                   0                    0                    0                    0                    0                    0                  0                   0                    1                    0                    0                    0                    0                    0                   0                    0                    0                    0                    0                  0                    0                  0                   0                    1                    0                    0                   1                0                 0             0             1             0             0             0             0             1                  0                 1                0                 0                0            0            1            0            0            0            0            0            1                0                0                0                 0                 1                 0                0                 0                 0                 0                 0                 0                0                 0                 1             1             0             0             0             0             0             0             0             0                 0                 0               0                 1               0               1               0                0                0                0               0               0               1               0               0               0               0               1                  1                   0                   0                   0                  0              0                0                 1                 0              0              0              0              0              1              0              0              0              0              1             0             1          0          0          1           0            0           0         1               1                 0                 0                 0             0             0               0               0               0             0             0            1                      0                     0                     0                     1                      0              0              0              0              1              0              0              0              0               1               0               0               0               0           0           1           0           1           0               0               0               0               1               0               0               0               1               0               0             1             0             0             0             0             1             0             0             0             0             0             0             0             0            0            1            0            0
'''""

#Modelling
y = np.log1p(df_train['SalePrice'])
X = df_train.drop(["Id", "SalePrice"], axis=1)

classifiers = [('LR', LinearRegression()),
                   ('Lasso', Lasso()),
                   ("Ridge", Ridge()),
                   ("ElasticNet", ElasticNet()),
                   ("KNeighborsRegressor", KNeighborsRegressor()),
                   ('DecisionTreeRegressor', DecisionTreeRegressor()),
                   ('CART', DecisionTreeRegressor()),
                   ('RF', RandomForestRegressor()),
                   ('GBR', GradientBoostingRegressor()),
                   ('XGBoost', XGBRegressor(use_label_encoder=False, eval_metric='logloss')),
                   ('LightGBR', LGBMRegressor())
                   ]
for name, regressor in classifiers:
    rmse = np.mean(np.sqrt(-cross_val_score(regressor, X, y, cv=5, scoring="neg_mean_squared_error")))
    print(f"RMSE: {round(rmse, 4)} ({name}) ")

'''
RMSE: 0.1516 (LR) 
RMSE: 0.1824 (Lasso) 
RMSE: 0.1381 (Ridge) 
RMSE: 0.1763 (ElasticNet) 
RMSE: 0.2201 (KNeighborsRegressor) 
RMSE: 0.2083 (DecisionTreeRegressor) 
RMSE: 0.2073 (CART) 
RMSE: 0.1462 (RF) 
RMSE: 0.1312 (GBR) 
RMSE: 0.1444 (XGBoost) 
RMSE: 0.1345 (LightGBR) 
'''

lgbm_model = LGBMRegressor(random_state=13)

rmse = np.mean(np.sqrt(-cross_val_score(lgbm_model, X, y, cv=5, scoring="neg_mean_squared_error")))
print(rmse)         #0.13453001272370504

lgbm_params = {"learning_rate": [0.01, 0.1],
               "n_estimators": [500, 1500]  }

lgbm_gs_best = GridSearchCV(lgbm_model,
                            lgbm_params,
                            cv=3,
                            n_jobs=-1,
                            verbose=True).fit(X,y)

final_model = lgbm_model.set_params(**lgbm_gs_best.best_params_).fit(X, y)

rmse = np.mean(np.sqrt(-cross_val_score(final_model, X, y, cv=5, scoring="neg_mean_squared_error")))
print(rmse)
'''
Fitting 3 folds for each of 4 candidates, totalling 12 fits
0.13353591049173888'''
