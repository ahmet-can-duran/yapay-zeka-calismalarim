import seaborn as sns
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
import numpy as np
#titanic veri setini indirdik
df=sns.load_dataset("titanic")
#veri setinin ilk 5satirini aldık
print(df.head())
#hangi sutunda ne kadar eksik var
print(df.isnull().sum())
#deck sutunundaki veri kaybi 588 bu cok yuksek bir veri kaybi bu yuzden tamir edilmez bunun tamamini silecez
df.drop("deck",axis=1,inplace=True)
#ago sutununda veri kaybi 177 onarilabilir bir seviyede bunu onarmanin yolu diger yolcularin yas ortalamasini bunlara vermek bu yaklasik olarak dogru yapacaktir
ortalama_yas=df["age"].mean()
#buldugum ortalama yas degerini veri setinde bos olan yerlere yerlestirdim
df["age"].fillna(ortalama_yas,inplace=True)
print("TEMİZLİK SONRASİ")
print(df.isnull().sum())
#su an veri setinde cinsiyet (sex) kismi male ve female bunlari 0 ve 1 donusturecez
df["sex"]=df["sex"].map({"male":0,"female":1})
print(df)
#x ve y degerlerini belirle
X=df[['pclass', 'sex', 'age', 'sibsp', 'fare']]
y=df["survived"]
#veriyi bol
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)
#model kurma
model=DecisionTreeClassifier()
#modeli egitim
model.fit(X_train, y_train)
#teat etme
tahminler=model.predict(X_test)
#basariyi olc
basari=accuracy_score(y_test,tahminler)
print("modelin basarisi:",basari*100)#%75 cikti
# Rose'un Bilgileri: [1. Sınıf, Kadın(1), 22 Yaş, 1 Eş/Kardeş, 150$]
rose = np.array([[1, 1, 22, 1, 150]])
# Jack'in Bilgileri: [3. Sınıf, Erkek(0), 20 Yaş, 0 Eş/Kardeş, 5$]
jack = np.array([[3, 0, 20, 0, 5]])
# Tahmin Et Bakalım!
rose_sonuc = model.predict(rose)
jack_sonuc = model.predict(jack)
print("\n--- FİLMİN SONU ---")
print(f"Rose'un Kaderi: {'KURTULDU' if rose_sonuc[0] == 1 else 'ÖLDÜ'}")
print(f"Jack'in Kaderi: {'KURTULDU' if jack_sonuc[0] == 1 else 'ÖLDÜ'}")
