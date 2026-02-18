import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
# bankadaki veriler
veri={'Gelir':[50000,20000,80000,30000,100000,70000,15000,60000],
      'Borc':[0,100000,500000,0,20000,10000,50000,0],
      'Karar':[1,0,0,0,1,1,0,1]
      }
# tablya cevime
df=pd.DataFrame(veri)
#girdi ve ciktiyi belirle
X=df[['Gelir','Borc']] 
y=df['Karar']
#veriyi bol 
X_train,X_test,y_train,y_test=train_test_split(X, y,test_size=0.25,random_state=42)
#modeli kur
model=LogisticRegression()
#modeli egit
model.fit(X_train, y_train)
yeni_musteri=[[9000,8000]]
#karar verme
sonuc=model.predict(yeni_musteri)
if(sonuc[0]==1):
    print("kredi onaylandi.")
else:
   print("kredi reddedildi..") 
   