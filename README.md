# Öğrenci Depresyon Analizi ve Tahminleme

Öğrencilerin ruh sağlığını etkileyen demografik, akademik ve yaşam tarzı faktörlerini
inceleyen ve makine öğrenmesiyle depresyon durumunu tahmin eden bir veri bilimi projesi.
Veri ön işleme, keşifçi veri analizi, dört sınıflandırma modelinin karşılaştırılması ve
en iyi modeli kullanan bir **Streamlit** arayüzü içerir.

## Sonuçlar

Aşağıdaki değerler notebook içinde kaydedilmiş deney çıktılarıdır; bu düzenlemede modeller yeniden eğitilmedi. Test seti: 5.581 kayıt (verinin %20'si), `random_state=42`.

| Model | Accuracy | F1 (macro) | Not |
|---|---|---|---|
| **SVM (RBF)** | **0.84** | **0.83** | Uygulamada kullanılan model (train 0.859 / test 0.836) |
| Lojistik Regresyon | 0.83 | 0.83 | `class_weight='balanced'` |
| Random Forest | 0.83 | 0.83 | ROC AUC **0.91** |
| KNN (k=5) | 0.81 | 0.81 | |

SVM eğitim/test doğruluğu farkı yaklaşık 2,3 puandır. Tek bir bölünmedeki bu fark, modelin genellenebilirliğini veya aşırı öğrenme bulunmadığını tek başına kanıtlamaz.

## Veri Seti

27.901 kayıt ve 18 değişken.

- **Hedef:** `Depression` (0: yok, 1: var)
- **Akademik:** CGPA, akademik baskı, çalışma memnuniyeti, öğrenim derecesi
- **Yaşam tarzı:** uyku süresi, beslenme alışkanlıkları, günlük çalışma saatleri
- **Kişisel:** ailede ruhsal hastalık geçmişi, finansal stres ve diğer değişkenler

## Uygulanan Adımlar

**Veri ön işleme**
- Eksik değerler (ör. Financial Stress) mod/medyan ile dolduruldu, tutarsız girişler temizlendi.
- `Degree` sütunu lisans, lisansüstü ve doktora olarak gruplandı.
- Kategorik değişkenler `LabelEncoder` ve `get_dummies` ile sayısallaştırıldı.

**Özellik mühendisliği**
- Uyku süresi ile çalışma saatleri arasındaki dengeyi ölçen `Work_Sleep_Balance` türetildi.

**Keşifçi veri analizi**
- Cinsiyete göre depresyon oranları, akademik baskının etkisi (boxplot) ve finansal
  stres ile depresyon arasındaki ilişki görselleştirildi.

**Modelleme**
- Veri eğitim/test olarak ayrıldı, `StandardScaler` ile ölçeklendi.
- Modeller `classification_report`, karışıklık matrisi ve ROC AUC ile değerlendirildi.

## Öne Çıkan Bulgular

- **Akademik baskı:** Depresyon görülen öğrencilerde akademik baskı belirgin şekilde daha yüksek.
- **Uyku dengesi:** Az uyku ve uzun çalışma saatleri depresyon riskiyle ilişkili.
- **Aile geçmişi:** Ailesinde ruhsal hastalık geçmişi olan öğrencilerde oran daha yüksek.

## Kurulum ve Çalıştırma

```bash
git clone https://github.com/elanur04/student_depression.git
cd student_depression
pip install -r requirements.txt

# Tahmin arayüzü (eğitilmiş SVM modeli: model_svm.pkl + scaler.pkl)
streamlit run main.py
```

Analiz ve modelleme adımlarının tamamı `student_depression.ipynb` dosyasındadır
(notebook için ek olarak `matplotlib` ve `seaborn` gerekir).

## Dosyalar

| Dosya | İçerik |
|---|---|
| `student_depression.ipynb` | Veri ön işleme, EDA ve model karşılaştırması |
| `main.py` | Streamlit tahmin arayüzü |
| `model_svm.pkl`, `scaler.pkl` | Eğitilmiş SVM modeli ve ölçekleyici |
| `student_depression_dataset.csv` | Ham veri seti |

> Bu proje eğitim amaçlıdır; klinik tanı aracı değildir.

## Çıkarım akışı ve test

Arayüzdeki kategoriler `inference.py` içinde notebook ile aynı kodlara ve aynı 12 sütun sırasına dönüştürülür. Girdi önce `scaler.transform`, ardından `model.predict` işleminden geçer. Önceki arayüzde model yerine kullanılan elle yazılmış puanlama kaldırılmıştır.

Python 3.11 veya 3.12 ile ayrı bir sanal ortam kullanın. Kaydedilmiş iki model dosyası scikit-learn **1.6.1** ile üretildiği için bu sürüm sabitlenmiştir. Yalnızca güvendiğiniz model dosyalarını yükleyin; farklı scikit-learn sürümleri arasında uyumluluk garanti edilmez ([resmî belge](https://scikit-learn.org/stable/model_persistence.html)).

```bash
python -m unittest discover -p "test_*.py"
```

Testler özellik sırasını, kategori dönüşümlerini, gerçek model çıkarımını, başka çalışma dizininden model yüklemeyi ve Streamlit form gönderimini kontrol eder.

## Sınırlamalar ve sonraki adımlar

- Eğitim amaçlı bir sınıflandırma demosudur; klinik doğrulaması yoktur.
- Notebook'ta bazı eksik değer doldurma işlemleri eğitim/test ayrımından önce yapılır. Bu, test bilgisinin ön işlemeye sızmasına yol açabilir. Sonraki deneyde bu adımlar yalnızca eğitim verisine uyan bir Pipeline içine alınmalıdır.
- `Work_Sleep_Balance` araştırılmış, ancak son modelin girdilerinden çıkarılmıştır.
- Veri setinin özgün kaynak bağlantısı ve kullanım lisansı bu depoda belgelenmemiştir; dış kullanım öncesinde doğrulanmalıdır.
- Yeni bir veri setiyle bağımsız doğrulama ve alt gruplar bazında hata analizi yapılmamıştır.
