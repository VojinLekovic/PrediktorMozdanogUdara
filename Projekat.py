import streamlit as st
import pandas as pd
import numpy as np  
from sklearn.preprocessing import LabelEncoder, StandardScaler
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import joblib
import os

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report, roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns

podaci = pd.read_csv("final_modified_healthcare-dataset-stroke-data.csv") #


podaci.replace('?', np.nan, inplace=True) #
podaci['bmi'] = pd.to_numeric(podaci['bmi'], errors='coerce') #
podaci['bmi'].fillna(podaci['bmi'].median(), inplace=True) #


enkoderi = {} #
kategorijske_vrijednosti = ['gender', 'ever_married', 'work_type', 'Residence_type', 'smoking_status'] #
for kolona in kategorijske_vrijednosti: #
    le = LabelEncoder() #
    podaci[kolona] = le.fit_transform(podaci[kolona].astype(str)) #
    enkoderi[kolona] = le #


if not os.path.exists('enkoderi'): #
    os.makedirs('enkoderi') #
for kolona, le in enkoderi.items(): #
    joblib.dump(le, f'enkoderi/{kolona}_enkoder.pkl') #

# Definisanje karakteristika i ciljne varijable
X = podaci.drop(['id', 'stroke'], axis=1) #
y = podaci['stroke'] #



smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X, y)


# Split je lijep grad
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)

# Treniranje modela
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
model = RandomForestClassifier(random_state=42)
model.fit(X_train_scaled, y_train)


joblib.dump(scaler, 'scaler.pkl')
joblib.dump(model, 'model.pkl')


st.title("Predviđanje moždanog udara")

enkoderi = {}
for kolona in kategorijske_vrijednosti:
    enkoderi[kolona] = joblib.load(f'enkoderi/{kolona}_enkoder.pkl')

# Input
gender = st.selectbox("Pol", enkoderi['gender'].classes_)
age = st.number_input("Godine", min_value=0, max_value=120, value=25)
hypertension = st.selectbox("Hipertenzija", [0, 1])
heart_disease = st.selectbox("Srčano Oboljenje", [0, 1])
ever_married = st.selectbox("Brak", enkoderi['ever_married'].classes_)
work_type = st.selectbox("Posao", enkoderi['work_type'].classes_)
Residence_type = st.selectbox("Mjesto Prebivališta", enkoderi['Residence_type'].classes_)
avg_glucose_level = st.number_input("Šećer", min_value=0.0, value=100.0)
bmi = st.number_input("BMI", min_value=0.0, value=25.0)
smoking_status = st.selectbox("Pušač", enkoderi['smoking_status'].classes_)


input_data = {
    'gender': enkoderi['gender'].transform([gender])[0],
    'age': age,
    'hypertension': hypertension,
    'heart_disease': heart_disease,
    'ever_married': enkoderi['ever_married'].transform([ever_married])[0],
    'work_type': enkoderi['work_type'].transform([work_type])[0],
    'Residence_type': enkoderi['Residence_type'].transform([Residence_type])[0],
    'avg_glucose_level': avg_glucose_level,
    'bmi': bmi,
    'smoking_status': enkoderi['smoking_status'].transform([smoking_status])[0]
}

korisnikov_df = pd.DataFrame(input_data, index=[0])


scaler = joblib.load('scaler.pkl')
model = joblib.load('model.pkl')



korisnikov_df_skaliran = scaler.transform(korisnikov_df)

# Predikcija
predikcija = model.predict(korisnikov_df_skaliran)
predikcija_proba = model.predict_proba(korisnikov_df_skaliran)
if predikcija[0] == 1:
    st.error(f"Procjena modela jeste da postoji šansa od {predikcija_proba[0][1]*100:.2f}% da Vam se desi moždani udar.")
else:
    st.success(f"Procjena modela jeste da postoji šansa od {predikcija_proba[0][0]*100:.2f}% da Vam se ne desi moždani udar.")

###

st.header("Evaluacija Modela")

y_pred = model.predict(X_test_scaled)
st.metric("Tačnost", f"{accuracy_score(y_test, y_pred)*100:.2f}%")
st.metric("Preciznost", f"{precision_score(y_test, y_pred)*100:.2f}%")
st.metric("Odziv (Recall)", f"{recall_score(y_test, y_pred)*100:.2f}%")
st.metric("F1 Score", f"{f1_score(y_test, y_pred)*100:.2f}%")
st.metric("AUC ROC", f"{roc_auc_score(y_test, model.predict_proba(X_test_scaled)[:,1]):.2f}")

# CM
st.subheader("Matrica Konfuzije")
cm = confusion_matrix(y_test, y_pred)
fig, ax = plt.subplots()
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
ax.set_xlabel('Predviđeno')
ax.set_ylabel('Stvarno')
st.pyplot(fig)

# FI
st.subheader("Značaj Karakteristika")
st.bar_chart(pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False))

# Izvjestaj
st.subheader("Klasifikacioni Izvještaj")
st.text(classification_report(y_test, y_pred))
