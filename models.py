# Запуск сервера Uvicorn с автоматической перезагрузкой при изменении кода
#uvicorn models:app --port 8001 --reload

import numpy as np
import joblib
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import json
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import logging
import uuid
from db_models import RequestLog
from database import db

# Настройка логгера
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Загрузка модели
model = joblib.load('training_models/random_forest_model.pkl')

# Загрузка LabelEncoder для категориальных признаков
cat_le_dict = joblib.load('training_models/category_label_encoder.pkl')
# print("Загруженный словарь энкодеров:")
# print(cat_le_dict)

target_le = joblib.load('training_models/target_label_encoder.pkl')

# Загрузка результатов энкодирования из текстового файла
# with open('training_models/encoder_results.json', 'r') as f:
#     encoder_results = json.load(f)

# Создание словарей для преобразования категориальных признаков и целевой переменной
# cat_le_dict = encoder_results['cat_le_dict']
# target_le_dict = encoder_results['target_le']

# Функция для загрузки структуры дерева из файла
# def load_tree_structure(filename):
#     tree_structure = {}
#     with open(filename, 'r') as f:
#         lines = f.readlines()
#         tree_structure["node_count"] = int(lines[0].split(": ")[1])
#         tree_structure["n_features"] = int(lines[1].split(": ")[1])
#         tree_structure["n_classes"] = int(lines[2].split(": ")[1].strip('[]\n'))
#         tree_structure["feature_names"] = lines[3].split(": ")[1].strip().split(", ")
#         tree_structure["class_names"] = lines[4].split(": ")[1].strip().split(", ")
#         tree_structure["nodes"] = []
#
#         node_index = lines.index("Nodes:\n") + 1
#         while node_index < len(lines):
#             if lines[node_index].strip() == "" or lines[node_index].startswith("Node"):
#                 node_index += 1
#                 continue
#             node = {
#                 "feature": lines[node_index].split(": ")[1].strip(),
#                 "threshold": float(lines[node_index + 1].split(": ")[1].strip()), #пороговое значение, по которому происходит разделение
#                 "impurity": float(lines[node_index + 2].split(": ")[1].strip()), #мера неоднородности (импульсивности) узла
#                 "n_node_samples": int(lines[node_index + 3].split(": ")[1].strip()), #количество образцов (примеров)
#                 "weighted_n_node_samples": float(lines[node_index + 4].split(": ")[1].strip()), #взвешенное количество образцов
#                 "left_child": int(lines[node_index + 5].split(": ")[1].strip()), #индекс левого дочернего узла
#                 "right_child": int(lines[node_index + 6].split(": ")[1].strip()), #индекс правого дочернего узла
#                 "value": eval(lines[node_index + 7].split(": ")[1].strip()) #значения, которые представляют распределение классов
#             }
#
#             tree_structure["nodes"].append(node)
#             node_index += 9
#
#     return tree_structure

# Функция для предсказания с использованием загруженной структуры дерева
# def predict_obesity(data, tree_structure):
#     node = tree_structure["nodes"][0]
#     while node["feature"] != 'Leaf':
#         feature = node["feature"]
#         threshold = node["threshold"]
#         if data[feature] <= threshold:
#             node = tree_structure["nodes"][node["left_child"]]
#         else:
#             node = tree_structure["nodes"][node["right_child"]]
#     return node["value"].index(max(node["value"]))

# Определение схемы данных для запроса
class PredictionRequest(BaseModel):
    Gender: int
    Age: int
    Height: float
    Weight: float
    FHWO: int
    FAVC: int
    FCVC: int
    NCP: float
    CAEC: str
    SMOKE: int
    CH2O: float
    SCC: int
    FAF: float
    TUE: int
    CALC: str
    MTRANS: str

class FeedbackRequest(BaseModel):
    prediction_id: str
    correct_answer: str

# Создание экземпляра FastAPI
app = FastAPI()

@app.post("/predict")
async def predict(request: PredictionRequest):
    data = request.dict()
    try:
        prediction = predict_obesity_wrapper(data)
        prediction_id = str(uuid.uuid4())
        await log_prediction(prediction_id, data, prediction)
        return {"prediction": prediction, "prediction_id": prediction_id}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/feedback")
async def feedback(request: FeedbackRequest):
    try:
        await log_feedback(request.prediction_id, request.correct_answer)
        return {"message": "Feedback logged successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

def predict_obesity_wrapper(data):
    # Преобразуем категориальные признаки
    label_cols = ['CALC', 'CAEC', 'MTRANS']
    for col in label_cols:
        encoder = cat_le_dict.get(col)
        data[col] = encoder.transform([str(data[col])])[0]
        # le = LabelEncoder()
        # le.classes_ = np.array(cat_le_dict[col])
        # data[col] = le.transform([data[col]])[0]

    # Загрузка структуры дерева из файла
    # tree_structure = load_tree_structure('training_models/tree_structure.txt')
    #
    # Предсказание
    # prediction = predict_obesity(data, tree_structure)
    # predicted_class = target_le_dict[prediction]  # Преобразуем в исходную метку
    #
    # return predicted_class

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
    # predicted_class = predict_obesity_wrapper(data)
    prediction = model.predict(features)
    #prediction = predict_obesity(data, tree_structure)
    predicted_class = target_le.inverse_transform([prediction])[0]  # Преобразуем в исходную метку

    return predicted_class

async def log_prediction(prediction_id, data, prediction):
    log_entry = RequestLog(
        method="POST",
        url="/predict",
        data=data,
        prediction=prediction,
        prediction_id=prediction_id
    )
    result = await db.request_logs.insert_one(log_entry.to_dict())  # Вставка записи в MongoDB
    logger.info(f"Logged prediction: {log_entry.to_dict()} - Inserted ID: {result.inserted_id}")

async def log_feedback(prediction_id, correct_answer):
    # Обновление записи в MongoDB
    result = await db.request_logs.update_one(
        {"prediction_id": prediction_id},
        {"$set": {"feedback": correct_answer}}
    )
    if result.modified_count > 0:
        logger.info(f"Logged feedback for prediction_id: {prediction_id} - Correct answer: {correct_answer}")
    else:
        logger.warning(f"Failed to log feedback for prediction_id: {prediction_id}")
