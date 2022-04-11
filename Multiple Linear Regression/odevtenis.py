#1.kutuphaneler
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sbn

#tenis oynanır yada oynanmaz
veriler = pd.read_csv("odev_tenis.txt")

temp = veriler.iloc[:,1:3].values
#test

from sklearn import preprocessing

veriler2 = veriler.apply(preprocessing.LabelEncoder().fit_transform)# bütün verileri otomatik encod eder yani 0 1

#havanın label encode olmasını istemiyorduk bu yüzden ayırdık ve onehotencoding uyguladık
hava = veriler2.iloc[:,:1]

ohe = preprocessing.OneHotEncoder()
hava = ohe.fit_transform(hava).toarray()
print(hava)

#numpy dizileri dataframe dönüşümleri
havadurumu = pd.DataFrame(data=hava, index = range(14), columns=['sunny','overcast','rainy'])
sonveriler = pd.concat([havadurumu,veriler.iloc[:,1:3]],axis=1)
sonveriler = pd.concat([veriler2.iloc[:,-2:],sonveriler],axis=1)

#verilerin test için bölünmesi
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(sonveriler.iloc[:,:-1],sonveriler.iloc[:,-1:],test_size=0.33,random_state=0)


from sklearn.linear_model import LinearRegression
regressor = LinearRegression()

regressor.fit(x_train,y_train)

y_pred = regressor.predict(x_test)



#GERİ ELEME (Backward elimination)
import statsmodels.api as sm

X = np.append(arr = np.ones((14,1)).astype(int), values=sonveriler.iloc[:,:-1], axis=1) #axis = 1 kolon olarak eklemesini sağlar

X_l = sonveriler.iloc[:,[0,1,2,3,4,5]].values
X_l = np.array(X_l,dtype=float)
model = sm.OLS(sonveriler.iloc[:,-1:],X_l).fit()
print(model.summary()) #Rapor
#rapordaki P>|t| değerlerine bakıyoruz ve en yüksek p değerine sahip olanı elicez çünkü modeli kötü etkiliyor

sonveriler = sonveriler.iloc[:,1:]

X = np.append(arr = np.ones((14,1)).astype(int), values=sonveriler.iloc[:,:-1], axis=1) #axis = 1 kolon olarak eklemesini sağlar

X_l = sonveriler.iloc[:,[0,1,2,3,4]].values
X_l = np.array(X_l,dtype=float)
model = sm.OLS(sonveriler.iloc[:,-1:],X_l).fit()
print(model.summary()) #Rapor

x_train = x_train.iloc[:,1:]
x_test = x_test.iloc[:,1:]

regressor.fit(x_train,y_train)

y_pred = regressor.predict(x_test)
