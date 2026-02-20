import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.tree import plot_tree
import matplotlib.pyplot as plt
#hastalarin verilerini aldik
hasta_verileri={
    "yas":[45,23,65,35,70,28,55,40],
    "tansiyon":[140,110,160,120,150,115,145,125],
    "kolestrol":[240,180,280,190,300,175,260,200],
    "nabiz":[85,70,90,75,95,72,88,78],
    "risk":[1,0,1,0,1,0,1,0]}#1:hasta 0 saglikli
df=pd.DataFrame(hasta_verileri)
#girdiler ve ciktilari belirtecez
X=df[["yas","tansiyon","kolestrol","nabiz"]]
y=df["risk"]
#hastalari bol
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.25,random_state=42)
#modeli olusturma
model=DecisionTreeClassifier()
#modeli egit
model.fit(X_train, y_train)
#deneme asamasi
yeni_hasta=[[20,198,267,112]]
teshis=model.predict(yeni_hasta)
if (teshis==1):
    print("acil mudahale edilmeli")
else:
    print("korkacak bir sey yok receteyi yazip gonder..")
tahminler=model.predict(X_test)
basari=accuracy_score(y_test,tahminler)
print("modelimizin basari orani:",basari*100)
# Çizim alanının boyutunu belirliyoruz (Geniş olsun ki rahat görelim)
plt.figure(figsize=(12, 8))

# Karar ağacını çizdiriyoruz
plot_tree(model, 
          feature_names=['Yas', 'Tansiyon', 'Kolesterol', 'Nabiz'], 
          class_names=['Saglikli', 'Riskli'], 
          filled=True, 
          rounded=True,
          fontsize=12)

# Resmi ekrana bas
plt.show()   
    
    
