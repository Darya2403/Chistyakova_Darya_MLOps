import numpy as np
import joblib
from sklearn.preprocessing import LabelEncoder
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

    # Преобразуем данные в нужный формат для модели
    features = np.array([
        data['Gender'], data['Age'], data['Height'], data['Weight'],
        data['FCVC'], data['NCP'], data['CH2O'], data['FAF'], data['TUE'], data['SCC'],
        data['CALC'], data['FAVC'], data['SCC'], data['FHWO'], data['CAEC'], data['MTRANS']
    ]).reshape(1, -1)

    # Предсказание
    prediction = model.predict(features)
    predicted_class = target_le.inverse_transform([prediction])[0]  # Преобразуем в исходную метку

    return predicted_class