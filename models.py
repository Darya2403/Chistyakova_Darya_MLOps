import numpy as np
import joblib
from sklearn.preprocessing import LabelEncoder
import pandas as pd

# Загрузка модели
model = joblib.load('training_models/random_forest_model.pkl')

# Загрузка LabelEncoder для категориальных признаков
cat_le_dict = joblib.load('training_models/category_label_encoder.pkl')
print("Загруженный словарь энкодеров:")
print(cat_le_dict)

target_le = joblib.load('training_models/target_label_encoder.pkl')

def predict_obesity(data):
    data_new = data
    # Преобразуем категориальные признаки
    label_cols = ['CALC', 'CAEC', 'MTRANS']
    for col in label_cols:
        encoder = cat_le_dict.get(col)
        data_new[col] = encoder.transform([str(data_new[col])])[0]

    # Преобразуем данные в pandas DataFrame, чтобы передать с именами признаков
    features = pd.DataFrame([{
        'Age': data_new['Age'],
        'Gender': data_new['Gender'],
        'Height': data_new['Height'],
        'Weight': data_new['Weight'],
        'CALC': data_new['CALC'],
        'FAVC': data_new['FAVC'],
        'FCVC': data_new['FCVC'],
        'NCP': data_new['NCP'],
        'SCC': data_new['SCC'],
        'SMOKE': data_new['SMOKE'],
        'CH2O': data_new['CH2O'],
        'family_history_with_overweight': data_new['FHWO'],
        'FAF': data_new['FAF'],
        'TUE': data_new['TUE'],
        'CAEC': data_new['CAEC'],
        'MTRANS': data_new['MTRANS']
    }])

    # Предсказание
    prediction = model.predict(features)
    predicted_class = target_le.inverse_transform([prediction])[0]  # Преобразуем в исходную метку

    return predicted_class