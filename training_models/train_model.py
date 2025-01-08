import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
import joblib
import pandas as pd
import os
from sklearn.preprocessing import LabelEncoder

# Настройка MLflow
mlflow.set_tracking_uri("http://localhost:5000")

# Создание данных
data = pd.read_csv('ObesityDataSet_raw_and_data_sinthetic.csv')

# Преобразуем категориальные признаки в числовые
# Признак 'Gender' преобразуем вручную (Female -> 0, Male -> 1)
data['Gender'] = data['Gender'].apply(lambda x: 0 if x == 'Female' else 1)

# Признаки типа "yes" и "no" преобразуем так же вручную в 1 и 0
binary_cols = ['SMOKE', 'family_history_with_overweight', 'SCC', 'FAVC']
for col in binary_cols:
    data[col] = data[col].apply(lambda x: 1 if x == 'yes' else 0)

numerical_cols = ['FCVC', 'NCP', 'CH2O', 'FAF', 'TUE']

# Установим границы для столбцов с ошибочными максимальными значениями
max_values = {
    'FCVC': 3.0,
    'NCP': 4.0,
    'CH2O': 3.0,
    'FAF': 3.0,
    'TUE': 2.0
}
# Идентифицируем строки с ошибочными значениями
excluded_data = pd.DataFrame()

for col in numerical_cols:
    temp_data = data[data[col] > max_values[col]]
    excluded_data = pd.concat([excluded_data, temp_data])

# Удаляем выбросы
data = data[~data.index.isin(excluded_data.index)]
print(data[numerical_cols].min())
print(data[numerical_cols].max())

# Применение LabelEncoder к категориальным признакам
cat_le_dict = {}
label_cols = ['CALC', 'CAEC', 'MTRANS']
for col in label_cols:
    cat_le_dict[col] = LabelEncoder()
    data[col] = cat_le_dict[col].fit_transform(data[col])

# Применение LabelEncoder к целевой переменной
target_le = LabelEncoder()
data['NObeyesdad'] = target_le.fit_transform(data['NObeyesdad'])


# Разделение данных на признаки и целевую переменную
X = data.drop('NObeyesdad', axis=1)
y = data['NObeyesdad']

# Разделение данных на обучающую и тестовую выборки
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Убедимся, что y_train и y_test являются одномерными массивами
y_train = y_train.ravel()
y_test = y_test.ravel()

# Обучение модели RandomForest
model = RandomForestClassifier(random_state=45)
model.fit(X_train, y_train)

# Прогнозирование результатов
y_pred = model.predict(X_test)

# Оценка модели
print("Classification Report:")
print(classification_report(y_test, y_pred))
print("Accuracy Score:")
print(accuracy_score(y_test, y_pred))
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# Логирование модели и артефактов в MLflow
with mlflow.start_run() as run:
    # модель
    mlflow.sklearn.log_model(model, "model")
    run_id = run.info.run_id

    # метрики
    mlflow.log_metric("accuracy", accuracy_score(y_test, y_pred))

    # параметры
    mlflow.log_param("n_estimators", model.n_estimators)
    mlflow.log_param("random_state", model.random_state)

    # дополнительные артефакты
    artifact_path = '/mlflow/artifacts'
    if not os.path.exists(artifact_path):
        os.makedirs(artifact_path)
    joblib.dump(model, os.path.join(artifact_path, 'random_forest_model.pkl'))
    mlflow.log_artifact(os.path.join(artifact_path, 'random_forest_model.pkl'))

    # Сохранение LabelEncoder
    joblib.dump(cat_le_dict, os.path.join(artifact_path, 'category_label_encoder.pkl'))
    mlflow.log_artifact(os.path.join(artifact_path, 'category_label_encoder.pkl'))
    joblib.dump(target_le, os.path.join(artifact_path, 'target_label_encoder.pkl'))
    mlflow.log_artifact(os.path.join(artifact_path, 'target_label_encoder.pkl'))

    # версия модели
    client = MlflowClient()
    model_uri = f"runs:/{run_id}/model"
    model_name = "example_model"

    # существует ли модель
    try:
        client.get_registered_model(model_name)
        print(f"Model '{model_name}' already exists. Updating version...")
    except mlflow.exceptions.RestException as e:
        if "RESOURCE_DOES_NOT_EXIST" in e.message:
            print(f"Model '{model_name}' does not exist. Creating new model...")
            client.create_registered_model(model_name)
        else:
            raise e

    # новая версия модели
    result = client.create_model_version(model_name, model_uri, run_id)
    print(f"Model version created: {result}")

    # переводим модель в стадию production
    client.transition_model_version_stage(
        name=model_name,
        version=result.version,
        stage="production",
        archive_existing_versions=True
    )
    print(f"Model version {result.version} transitioned to production stage.")

# Проверка создания версии модели
try:
    model_versions = client.get_latest_versions(model_name, stages=["production"])
    if model_versions:
        model_version = model_versions[0]
        print(f"Latest model version in production: {model_version}")
    else:
        print(f"No model versions found in production for model '{model_name}'")
except mlflow.exceptions.RestException as e:
    print(f"Error fetching model version: {e}")
