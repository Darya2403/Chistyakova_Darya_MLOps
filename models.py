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
    # Преобразуем категориальные признаки
    label_cols = ['CALC', 'CAEC', 'MTRANS']
    for col in label_cols:
        encoder = cat_le_dict.get(col)
        data[col] = encoder.transform([str(data[col])])[0]

    # Преобразуем данные в pandas DataFrame, чтобы передать с именами признаков
    features = pd.DataFrame([{
        'Age': data['Age'],
        'Gender': data['Gender'],
        'Height': data['Height'],
        'Weight': data['Weight'],
        'CALC': data['CALC'],
        'FAVC': data['FAVC'],
        'FCVC': data['FCVC'],
        'NCP': data['NCP'],
        'SCC': data['SCC'],
        'SMOKE': data['SMOKE'],
        'CH2O': data['CH2O'],
        'family_history_with_overweight': data['FHWO'],
        'FAF': data['FAF'],
        'TUE': data['TUE'],
        'CAEC': data['CAEC'],
        'MTRANS': data['MTRANS']
    }])

    # Предсказание
    prediction = model.predict(features)
    predicted_class = target_le.inverse_transform([prediction])[0]  # Преобразуем в исходную метку

    return predicted_class