import streamlit as st
import pickle
import pandas as pd
import matplotlib.pyplot as plt

#функции:
#Расчет индекса массы тела:
def add_BMI(h: int, w: int):
    return round((w / ((h/100)**2)),0)
#Перевод Да и Нет в 1 и 0:
def bool(z):
    if z == 'Yes':
       return 1
    else:
       return 0
#Перевод пола в 2 и 1:
def bool_1(k):
    if k=='Female':
       return 2
    else:
       return 1
#расчет разницы АД:
def add_Pressure_Difference(x: int, y: int):
    return (x - y)
#функция вызова модели:
def load_model():
        with open("./model_finish","rb") as fid:
            return pickle.load(fid)


def load_scaler():
    with open("./scaler",
              "rb") as fid_1:
        return pickle.load(fid_1)

# название и картинка
col1, col2 = st.columns(2)
from PIL import Image

image = Image.open('./dva-serdca.jpg')
col1.image(image)

col2.title(':green[Heart Diseases Prediction]')
col2.write('To get the probability of cardiovascular diseases occurrence, answer all the questions and click "Press for Prediction"')

# Основные параметры:
col01, col02, col03 = st.columns(3)
# Пол, возраст,
col01.subheader(':green[Your Gender:]')
gender = col01.radio("", ('Female', 'Male'), horizontal=True,key='gender')
gender_model = bool_1(gender)

col03.subheader(':green[Your Age:]')
age_year = col03.slider('', 18, 100, key='age')
age_model = age_year*365

# активность, алкоголь, курение:
col11, col12, col13 = st.columns(3)
col11.subheader(':green[Are You Active?]')
active = col11.radio("", ('Yes', 'No'), horizontal=True,key='active')
active_model = bool(active)
col12.subheader(':green[Drink Alcohol?]')
alco = col12.radio("",('Yes', 'No'), horizontal=True, key='alco')
alco_model = bool(alco)
col13.subheader(':green[Do You Smoke?]')
smoke = col13.radio("",('Yes', 'No'), horizontal=True, key='smoke')
smoke_model = bool(smoke)
# Давление
col11.subheader(':green[Arterial Pressure]')
sap = col11.slider('Systolic Arterial Pressure', 50, 280, key='sap')
dap = col11.slider('Diastolic Arterial Pressure', 40, 120, key='dap')

Pressure_Difference = add_Pressure_Difference(sap, dap)
col11.metric(":blue[Pressure Difference:  ]", value= (Pressure_Difference))
#индекс массы тела:
col13.subheader(':green[Body Mass Index]')
h = col13.slider('Your Height, cm', 100, 210,key='h')
w = col13.slider('Your Weight, kg', 35, 300, key='w')
bmi = add_BMI(h, w)
#col31.write(':blue[BMI:  ] ' + repr(bmi))
col13.metric(":blue[BMI:  ]", value= (bmi))
#холестерин и глюкоза
col11.subheader(':green[Cholesterol Levels:]')
chol = col11.selectbox("",(1,2,3), key='cholesterol')

col13.subheader(':green[Glucose Level:]')
gluc = col13.selectbox("",(1,2,3), key ='glucose')


#from sklearn.preprocessing import StandardScaler
#import numpy as np
#Вводимые пользователем данные
input_data = {
    'age': age_model,
    'gender': gender_model,
    'height': h,
    'weight': w,
    'ap_hi': sap,
    'ap_lo': dap,
    'cholesterol': chol,
    'gluc': gluc,
    'smoke': smoke_model,
    'alco': alco_model,
    'active': active_model,
    'BMI': bmi,
    'hi_lo': Pressure_Difference
}
data_test = pd.DataFrame(input_data, index=[0])
st.subheader(':green[Input Data]')
st.dataframe(data_test)
numeric = ['age','height','ap_lo','cholesterol','gluc','BMI','hi_lo']
data_test.drop(columns= ['ap_hi','weight'], inplace=True)
feature_test = data_test
#st.dataframe(feature_test)

scaler = load_scaler()
feature_test[numeric] = scaler.transform(feature_test[numeric])


#вывод предсказаний
col21, col22, col23 = st.columns(3)
if col22.button('Press for Prediction'):
    model = load_model()
    prediction = model.predict_proba(feature_test)[:,1]
    prediction = prediction.round(2)
    st.subheader(':blue[Heart Diseases Prediction]')
    st.metric("", value=(prediction))

    st.subheader(':green[Importance Features]')
    importances = model.feature_importances_


    feature_list = list(feature_test.columns)
    feature_results = pd.DataFrame({'feature': feature_list, 'importance': importances})
    #feature_results = feature_results.sort_values('importance', ascending=False).reset_index(drop=True)
    fig = plt.figure(figsize=(10, 5))
    plt.barh(feature_list, importances)
    plt.title('Important Features')
    plt.xlabel('Importance')
    plt.ylabel('Features')
    plt.tight_layout()
    st.pyplot(fig)
    #st.write(feature_results)

