import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression 
#1. Modeli çağır
# 2. Veriyi hazırla(geçmiş veriler)
# X (GİRDİ) Antrenman saatleri(Makine bunu 2 boyutlu ister bu yüzden köşeli parantezler iç içe olur)
X_antrenman=np.array([[1],[2],[3],[4],[5]])
#Y (ÇIKTI) Atılan goller( bu tek boyutlu olabilir)
Y_goller=np.array([1,2,3,4,5])
# 1 saat çalışırsan 1 , 5 saat çalışırsan 5 gol atarsın basit bir lineer ilişki kurduk
#3 Modeli oluşturma
model=LinearRegression()
# 4 Modeli eğitme
model.fit(X_antrenman,Y_goller)
# 5 Tahmin etme
#SORU: 10 saat çalışırsa kaç gol atar
yeni_veri=np.array([[10]])
#Sorunun cevabı
tahmin=model.predict(yeni_veri)
print("10 saat çalışırsa tahmini gol sayisi:",tahmin[0])
#6 Görselleştirme
plt.scatter(X_antrenman,Y_goller,colorizer="red",label="gerçek veriler")
plt.plot(X_antrenman, model.predict(X_antrenman), color='blue', label='Tahmin Çizgisi')
plt.xlabel("Antrenman Saati")
plt.ylabel("Gol Sayısı")
plt.legend()
plt.show()
