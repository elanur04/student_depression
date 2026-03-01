#Öğrenci Depresyon Analizi ve Tahminleme
Bu proje, öğrencilerin ruh sağlığını etkileyen demografik, akademik ve yaşam tarzı faktörlerini incelemek ve makine öğrenmesi modelleri kullanarak depresyon durumunu tahmin etmek amacıyla geliştirilmiştir. Veri seti üzerinde kapsamlı veri ön işleme, görselleştirme ve sınıflandırma analizleri uygulanmıştır.

 #Proje Özeti
Öğrenci depresyonu, akademik başarıyı ve yaşam kalitesini doğrudan etkileyen kritik bir sorundur. 
Bu çalışmada:
Veri Temizleme: Eksik verilerin (Financial Stress vb.) mod/medyan ile doldurulması ve tutarsız veri girişlerinin temizlenmesi.
Özellik Mühendisliği: Uyku süresi ve çalışma saatleri arasındaki dengeyi ölçen Work_Sleep_Balance gibi yeni değişkenlerin türetilmesi.
Analiz: Depresyon ile akademik baskı, not ortalaması (CGPA) ve aile geçmişi arasındaki ilişkilerin istatistiksel ve görsel analizi.
Modelleme: Random Forest, KNN, Lojistik Regresyon ve SVM gibi modellerle yüksek doğruluklu tahminleme.

#Veri Seti Özellikleri
Veri seti 27,901 kayıt ve 18 değişkenden oluşmaktadır:
Hedef Değişken: Depression (0: Yok, 1: Var)
Akademik: CGPA, Akademik Baskı, Çalışma Memnuniyeti, Mezuniyet Derecesi.
Yaşam Tarzı: Uyku Süresi, Beslenme Alışkanlıkları, Günlük Çalışma Saatleri.
Kişisel: Ailede Ruhsal Hastalık Geçmişi, İntihar Düşüncesi, Finansal Stres.

#Uygulanan Adımlar

Veri Ön İşleme: - Degree sütunu; Lisans, Lisansüstü ve Doktora olarak kategorize edildi.
Kategorik veriler LabelEncoder ve get_dummies ile sayısal forma çevirildi.
Financial Stress ve Sleep Duration verilerindeki gürültüler temizlendi.

Keşifçi Veri Analizi (EDA):
Cinsiyet bazlı depresyon oranları incelendi.
Akademik baskının (Academic Pressure) depresyon üzerindeki doğrudan etkisi boxplot grafikeriyle doğrulandı.
Finansal stres ve intihar düşüncesi değişkenlerinin depresyonla korelasyonu görselleştirildi.

Makine Öğrenmesi:
Veri seti Eğitim ve Test olarak ayrıldı.
Modellerin performansı classification_report ve accuracy_score ile değerlendirildi.

#Öne Çıkan Bulgular
Akademik Baskı: Depresyon tanısı alan öğrencilerde akademik baskı seviyesinin belirgin şekilde daha yüksek olduğu gözlemlenmiştir.
Uyku Dengesi: Az uyku ve uzun çalışma saatlerinin (Work_Sleep_Balance) depresyon riskini artırdığı saptanmıştır.
Aile Geçmişi: Ailesinde ruhsal hastalık geçmişi olan öğrencilerin depresyona daha eğilimli olduğu verilerle desteklenmiştir.

#Kurulum
Projeyi yerel ortamınızda çalıştırmak için:

# Depoyu klonlayın
git clone https://github.com/elanur04/student_depression.git

# Gerekli kütüphaneleri yükleyin
pip install pandas numpy scikit-learn matplotlib seaborn
Proje Dosyası İçeriği
student_depression.ipynb: Tüm analiz ve modelleme kodlarını içeren Jupyter Notebook.

student_depression_dataset.csv: Analizde kullanılan ham veri seti.
