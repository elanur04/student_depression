import streamlit as st
import numpy as np
import joblib

# Model ve scaler yükleme
model = joblib.load("model_svm.pkl")
scaler = joblib.load("scaler.pkl")

# Haritalama fonksiyonları
education_levels_map = {'Doctorate': 0, 'Other': 1, 'Postgraduate': 2, 'Undergraduate': 3}

def map_education_level(level_str):
    return education_levels_map.get(level_str, 1)

def map_dietary_habits(val):
    return {"Healthy": 1, "Moderate": 0, "Unhealthy": -1}.get(val, 0)

def map_suicidal_thoughts(val):
    return 1 if val == "Yes" else 0

def map_family_history(val):
    return 1 if val == "Yes" else 0

def map_gender(val):
    return 1 if val == "Male" else 0

def map_sleep_duration(val):
    return {
        'Less than 5 hours': 0,
        '5-6 hours': 1,
        '7-8 hours': 2,
        'More than 8 hours': 3,
        'Others': 2
    }.get(val, 2)

def preprocess_input(data):
    return np.array([[
        data['Age'],
        data['Academic Pressure'],
        data['CGPA'],
        data['Study Satisfaction'],
        map_dietary_habits(data['Dietary Habits']),
        map_suicidal_thoughts(data['Have you ever had suicidal thoughts ?']),
        data['Work/Study Hours'],
        data['Financial Stress'],
        map_family_history(data['Family History of Mental Illness']),
        map_gender(data['Gender_Male']),
        map_sleep_duration(data['Sleep_Duration_UIQ']),
        map_education_level(data['Education_Level'])
    ]])

# Türkçe seçenekler ve karşılıkları
dietary_mapping = {"Sağlıklı": "Healthy", "Orta": "Moderate", "Sağlıksız": "Unhealthy"}
suicidal_mapping = {"Hayır": "No", "Evet": "Yes"}
family_history_mapping = {"Hayır": "No", "Evet": "Yes"}
gender_mapping = {"Kadın": "Female", "Erkek": "Male"}
sleep_duration_mapping = {
    "5 saatten az": 'Less than 5 hours',
    "5-6 saat": '5-6 hours',
    "7-8 saat": '7-8 hours',
    "8 saatten fazla": 'More than 8 hours',
    "Diğer": 'Others'
}
education_level_options_tr = {
    'Lisans': 'Undergraduate',
    'Yüksek Lisans': 'Postgraduate',
    'Doktora': 'Doctorate',
    'Diğer': 'Other'
}

# Sayfa düzeni ve stil
st.set_page_config(page_title="Depresyon Tahmin Uygulaması", layout="wide")

# Arka plan rengini değiştirme
st.markdown(
    """
    <style>
        body {
            background-color: #a0a0a0;
        }
        .stApp {
            background-color: #a0a0a0;
        }
    </style>
    """,
    unsafe_allow_html=True
)

with st.container():
    st.markdown(
        """
        <div style="background-color:#f0f0f0; padding:20px; border-radius:15px; border-left:5px solid #6c63ff; margin-bottom:30px; font-family:'Segoe UI', sans-serif;">
            <h2 style="text-align:center; color:#333333;">💯Veri Seti Hakkında💯</h2>
            <p style="color:#444444; font-size:16px; text-align:center;">
                Bu uygulama, <strong>27.901 üniversite öğrencisinden</strong> toplanan kapsamlı bir veri setine dayanmaktadır. 
                Yaş, cinsiyet, akademik ve iş stresi, uyku düzeni, beslenme alışkanlıkları gibi pek çok faktör değerlendirilerek depresyon riski analiz edilmektedir. 
                Amaç; depresyonun nedenlerini daha iyi anlayarak <strong>erken teşhis</strong> ve <strong>müdahale</strong> süreçlerine katkı sağlamaktır.
            </p>
            <h2 style="text-align:center; color:#333333;">Kendi Durumunuzu Öğrenin 🤙</h2>
            <p style="color:#444444; font-size:16px; text-align:center;">
                Uygulama sayesinde siz de kendi bilgilerinizi girerek depresyonda olup olmadığınızı <strong>kolayca tahmin edebilirsiniz</strong>. 
                Böylece ruh sağlığınızı daha bilinçli takip edebilir ve gerekirse <strong>erken önlemler</strong> alabilirsiniz.
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )



with st.container():
    st.markdown(
        """
        <div style="background:#f0f0f0; padding:15px; border-radius:12px; color:#82716e; max-width:700px; margin:auto; font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;">
        <h3 style="text-align:center; font-weight:700; margin-bottom:10px; color:#333333">Değişkenler ve Açıklamaları</h3>
        <ul style="line-height:1.6; font-size:16px; padding-left:20px;">
            <li style="color:#333333"><b>Yaş:</b> 15-100 arası sayısal değer</li>
            <li style="color:#333333"><b>Akademik Baskı:</b> 1 (düşük) - 5 (yüksek) arası stres seviyesi</li>
            <li style="color:#333333"><b>CGPA:</b> Akademik not ortalaması (0-10 arası)</li>
            <li style="color:#333333"><b>Çalışma İsteği:</b> 1-5 arası çalışma tatmini</li>
            <li style="color:#333333"><b>Beslenme Alışkanlığı:</b> Sağlıklı, Orta veya Kötü</li>
            <li style="color:#333333"><b>İntihar Düşüncesi:</b> Evet veya Hayır</li>
            <li style="color:#333333"><b>Çalışma/İş Saatleri:</b> Günlük çalışma süresi (0-24 saat)</li>
            <li style="color:#333333"><b>Maddi Baskı:</b> 1 (düşük) - 5 (yüksek) arası finansal stres</li>
            <li style="color:#333333"><b>Ailede Ruhsal Hastalık:</b> Var veya Yok</li>
            <li style="color:#333333"><b>Cinsiyet:</b> Erkek, Kadın veya Diğer</li>
            <li style="color:#333333"><b>Uyku Süresi:</b> 5 saatten az, 5-7 saat, 7-9 saat, 9 saatten fazla</li>
            <li style="color:#333333"><b>Eğitim Düzeyi:</b> Lise'den Doktora'ya kadar</li>
        </ul>
        </div>
        """,
        unsafe_allow_html=True
    )


col1, col2, col3 = st.columns([1, 3, 1])

with col2:
    age = st.number_input("Yaş", min_value=15, max_value=100, value=25)
    academic_pressure = st.slider("Akademik Baskı (1-5)", 1, 5, 3)
    cgpa = st.number_input("CGPA (Ortalama)", min_value=0.0, max_value=10.0, value=7.0, step=0.01)
    study_satisfaction = st.slider("Çalışma İsteği (1-5)", 1, 5, 3)
    dietary_options_tr = ["Sağlıklı", "Orta", "Kötü"]
    suicidal_options_tr = ["Hayır", "Evet"]
    family_history_options_tr = ["Yok", "Var"]
    gender_options_tr = ["Erkek", "Kadın", "Diğer"]
    sleep_duration_options_tr = ["5 saatten az", "5-6 saat", "7-8 saat", "8 saatten fazla"]
    education_level_options_tr = {
        "Lise": 1,
        "Ön Lisans": 2,
        "Lisans": 3,
        "Yüksek Lisans": 4,
        "Doktora": 5
    }

    dietary_habits_tr = st.selectbox("Beslenme Alışkanlığı", dietary_options_tr)
    suicidal_thoughts_tr = st.selectbox("İntihar Düşüncesi Geçmişi", suicidal_options_tr)
    work_study_hours = st.number_input("Çalışma/İş Saatleri (Günlük)", min_value=0, max_value=24, value=8)
    financial_stress = st.slider("Maddi Baskı (1-5)", 1, 5, 3)
    family_history_tr = st.selectbox("Ailede Ruhsal Hastalık Geçmişi", family_history_options_tr)
    gender_tr = st.selectbox("Cinsiyet", gender_options_tr)
    sleep_duration_tr = st.selectbox("Uyku Süresi", sleep_duration_options_tr)
    education_level_tr = st.selectbox("Eğitim Düzeyi", list(education_level_options_tr.keys()))

    if st.button("Tahmin Et"):
        risk_score = 0
        risk_score += (academic_pressure - 3) * 2
        risk_score += (5 - study_satisfaction) * 1.5
        risk_score += (work_study_hours - 8) * 0.5
        risk_score += 5 if suicidal_thoughts_tr == "Evet" else 0
        risk_score += 3 if family_history_tr == "Var" else 0
        risk_score += 2 if financial_stress > 3 else 0

        if risk_score > 5:
            st.error(
                """
                **Depresyonda olma ihtimaliniz yüksek.**  
                Öneriler:  
                - Günlük 30 dakika yürüyüş yapın  
                - Düzenli uyku saatlerine dikkat edin  
                - Bir uzmana danışmayı ihmal etmeyin  
                \n[İzlemeniz için komik video önerisi 🎬](https://www.youtube.com/watch?v=adB9BjBMV8I)
                """
            )
        else:
            st.success(
                """
                🤙🤙
                **Depresyonda olma ihtimaliniz düşük!**  
                Hayatı pek umursamadığınız için sizi tebrik ederiz! 😄  
                Pozitif kalmaya devam edin! 🌟
                """
            )

