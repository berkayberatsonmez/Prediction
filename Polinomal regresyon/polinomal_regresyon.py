#1.kutuphaneler
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#veri yükleme
veriler = pd.read_csv("maaslar.txt")

#data frame dilimleme (slice)
x = veriler.iloc[:,1:2] 
y = veriler.iloc[:,2:]

#NumPY dizi (array) dönüşümü
X = x.values
Y = y.values

#linear regression
#doğrsual model oluşturma
from sklearn.linear_model import LinearRegression

lin_reg = LinearRegression()
lin_reg.fit(X,Y)

#polinomal regresyonu
#doğrusal olmayan (nonlinear model)
#2.dereceden polinom
from sklearn.preprocessing import PolynomialFeatures

poly_reg = PolynomialFeatures(degree=2) #2.dereceden

x_poly = poly_reg.fit_transform(X)

lin_reg2 = LinearRegression()
lin_reg2.fit(x_poly,y)


#4. dereceden polinom
poly_reg3 = PolynomialFeatures(degree=4) 

x_poly3 = poly_reg3.fit_transform(X)

lin_reg3 = LinearRegression()
lin_reg3.fit(x_poly3,y)

#görselleştirme
plt.scatter(X,Y, color='red')
plt.plot(x,lin_reg.predict(X),color = 'blue')
plt.show()

plt.scatter(X,Y, color='red' )
plt.plot(x,lin_reg2.predict(poly_reg.fit_transform(X)),color='blue') #aralarında fark predict etmeden önce fit transform yapıyoruz
plt.show()

plt.scatter(X,Y, color='red' )
plt.plot(x,lin_reg3.predict(poly_reg3.fit_transform(X)),color='blue') #aralarında fark predict etmeden önce fit transform yapıyoruz
plt.show()

#tahminler
print(lin_reg.predict([[11]]))
print(lin_reg.predict([[6.6]]))

print(lin_reg2.predict(poly_reg.fit_transform([[6.6]])))
print(lin_reg2.predict(poly_reg.fit_transform([[11]])))