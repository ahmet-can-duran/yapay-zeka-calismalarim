import pickle


tam_adres = "C:/Users/Ahmet Can/Desktop/PYTHON KUTUPHANELERİ/kanser_asistani.pkl"
with open(tam_adres, 'rb') as dosya:
    doktor_asistan = pickle.load(dosya)

print("🧠 Asistan uyandı, hafızası yerinde ve göreve hazır!\n")

yeni_hasta = [[17.99, 10.38, 122.8, 1001.0, 0.1184, 0.2776, 0.3001, 0.1471, 0.2419, 0.07871, 
               1.095, 0.9053, 8.589, 153.4, 0.006399, 0.04904, 0.05373, 0.01587, 0.03003, 0.006193, 
               25.38, 17.33, 184.6, 2019.0, 0.1622, 0.6656, 0.7119, 0.2654, 0.4601, 0.1189]]

teshis = doktor_asistan.predict(yeni_hasta)

if teshis[0] == 0:
    print(" TEŞHİS: Kötü Huylu Tümör (Acil Onkolojiye Sevk Edilmeli!)")
else:
    print(" TEŞHİS: İyi Huylu Tümör (Risk Yok)")