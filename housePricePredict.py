################################# Ev Fiyat Tahmin Modeli ###############################
# İş Problemi
# Her bir eve ait özelliklerin ve ev fiyatlarının bulunduğu veriseti kullanılarak,
# farklı tipteki evlerin fiyatlarına ilişkin bir makine öğrenmesi projesi
# gerçekleştirilmek istenmektedir.

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

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
