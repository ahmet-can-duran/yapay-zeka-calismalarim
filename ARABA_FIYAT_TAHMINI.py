import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
#veriyi olusturma
veri = {
    'Yil': [2015, 2018, 2020, 2012, 2022],
    'KM': [120000, 50000, 30000, 180000, 10000],
    'Motor_Gucu': [110, 140, 150, 90, 160],
    'Vites': ['Manuel', 'Otomatik', 'Otomatik', 'Manuel', 'Otomatik'],
    'Fiyat': [600, 950, 1100, 450, 1300]
}
# tabloya çevirme
df=pd.DataFrame(veri)
# vites ozelligini sayıya donusturdum
df["Vites"]=df["Vites"].map({"Manuel":0,"Otomatik":1})
#tablo son hali
print(df)
# Girdi ve ciktilari ayarladik
X=df[["Yil","KM","Motor_Gucu","Vites"]]
y=df["Fiyat"]
#veriyi blecez egitim ve test olarak
#test_size 0.2 demek test verisinin yuzde 20si demek
X_train, X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)
#modeli kurma
model=LinearRegression()
#modeli egit
model.fit(X_train, y_train)
#tahmin yap
tahmin=model.predict(X_test)

# muteri bilgilerine gre tahmin yapalim 
yeni_araba=[[2025,12000,145,1]]
yeni_tahmin=model.predict(yeni_araba)
print("BEYEFENDI/HANIMEFENDI ARACINIZIN FIYATI:",(int(yeni_tahmin[0]))," BIN TL'DIR")
