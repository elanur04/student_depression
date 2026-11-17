import streamlit as st
from inference import DIET, EDUCATION, FAMILY, GENDER, SLEEP, YES_NO, load_artifacts, predict_label

st.set_page_config(page_title="Öğrenci Verileri · SVM Demo", layout="centered")
st.title("Öğrenci verileri üzerinde SVM sınıflandırması")
st.write("Veri ön işleme, ölçekleme ve eğitilmiş bir modelle çıkarım adımlarını gösteren eğitim projesi.")
st.info("Bu demo klinik tanı veya kişisel risk değerlendirmesi yapmaz. Model çıktısı sağlık kararı vermek için kullanılmamalıdır.")
st.caption("Notebook: 27.901 kayıt · 12 model girdisi · StandardScaler + SVM (RBF)")

@st.cache_resource
def get_artifacts():
    return load_artifacts()

try:
    model, scaler = get_artifacts()
except (FileNotFoundError, ValueError, ImportError):
    st.error("Model yüklenemedi. Model dosyalarını ve requirements.txt ile kurulan paket sürümlerini kontrol edin.")
    st.stop()

with st.form("sample"):
    st.subheader("Örnek gözlem")
    st.caption("Eğitim verisinin desteklediği kategoriler gösterilir. Gerçek kişisel veri girmek gerekmez.")
    left, right = st.columns(2)
    with left:
        age = st.number_input("Yaş", min_value=18, max_value=59, value=25)
        academic_pressure = st.slider("Akademik baskı", 0, 5, 3)
        cgpa = st.number_input("CGPA (0–10)", min_value=0.0, max_value=10.0, value=7.0, step=0.1)
        study_satisfaction = st.slider("Öğrenim memnuniyeti", 0, 5, 3)
        diet = st.selectbox("Beslenme alışkanlığı", list(DIET))
        thoughts = st.selectbox("İntihar düşüncesi geçmişi (veri setindeki özellik)", list(YES_NO))
    with right:
        hours = st.number_input("Günlük çalışma süresi (saat)", min_value=0, max_value=12, value=8)
        financial_stress = st.slider("Finansal stres", 1, 5, 3)
        family = st.selectbox("Ailede ruhsal hastalık geçmişi", list(FAMILY))
        gender = st.selectbox("Cinsiyet (veri setindeki kategoriler)", list(GENDER))
        sleep = st.selectbox("Uyku süresi", list(SLEEP))
        education = st.selectbox("Eğitim düzeyi", list(EDUCATION), index=3)
    submitted = st.form_submit_button("Model çıktısını göster")

if submitted:
    sample = dict(age=age, academic_pressure=academic_pressure, cgpa=cgpa,
                  study_satisfaction=study_satisfaction, diet=diet, thoughts=thoughts,
                  hours=hours, financial_stress=financial_stress, family=family,
                  gender=gender, sleep=sleep, education=education)
    label = predict_label(sample, model, scaler)
    st.write(f"**Tahmin edilen veri seti etiketi: {label}**")
    st.caption("Etiketler veri setindeki Depression sütununa aittir (0 / 1); klinik değerlendirme değildir.")

with st.expander("Yöntem ve sınırlar"):
    st.write("Girdiler notebook ile aynı sırada ve kategorik kodlarla hazırlanır, kaydedilmiş ölçekleyiciden geçirilir ve model.predict ile sınıflandırılır.")
    st.write("Notebook'ta kaydedilen test doğruluğu yaklaşık %84'tür. Bu sonuç tek bir veri bölünmesine aittir; farklı topluluklara genellenebilirliği göstermez. Ön işleme ve bağımsız doğrulama sınırları README'de açıklanır.")
    st.markdown("[Kod ve analiz notebook'u](https://github.com/elanur04/student_depression)")
