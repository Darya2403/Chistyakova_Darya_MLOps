import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
import joblib
import pandas as pd
import os

# Настройка MLflow
mlflow.set_tracking_uri("http://localhost:5000")

# Создание данных
data = pd.read_csv('output.csv')

# Разделение данных на признаки и целевую переменную
X = data.drop('NObeyesdad', axis=1)
y = data['NObeyesdad']

# Разделение данных на обучающую и тестовую выборки
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Убедимся, что y_train и y_test являются одномерными массивами
y_train = y_train.ravel()
y_test = y_test.ravel()

# Обучение модели RandomForest
model = RandomForestClassifier(random_state=42)
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
    # Залогируйте модель
    mlflow.sklearn.log_model(model, "model")
    run_id = run.info.run_id

    # Залогируйте метрики
    mlflow.log_metric("accuracy", accuracy_score(y_test, y_pred))

    # Залогируйте параметры
    mlflow.log_param("n_estimators", model.n_estimators)
    mlflow.log_param("random_state", model.random_state)

    # Логирование дополнительных артефактов
    artifact_path = '/mlflow/artifacts/training_models'
    if not os.path.exists(artifact_path):
        os.makedirs(artifact_path)
    joblib.dump(model, os.path.join(artifact_path, 'random_forest_model.pkl'))
    mlflow.log_artifact(os.path.join(artifact_path, 'random_forest_model.pkl'))

    # Создайте или обновите версию модели
    client = MlflowClient()
    model_uri = f"runs:/{run_id}/model"
    model_name = "example_model"

    # Проверьте, существует ли модель
    try:
        client.get_registered_model(model_name)
        print(f"Model '{model_name}' already exists. Updating version...")
    except mlflow.exceptions.RestException as e:
        if "RESOURCE_DOES_NOT_EXIST" in e.message:
            print(f"Model '{model_name}' does not exist. Creating new model...")
            client.create_registered_model(model_name)
        else:
            raise e

    # Создайте новую версию модели
    result = client.create_model_version(model_name, model_uri, "None")
    print(f"Model version created: {result}")

    # Переведите модель в стадию production
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

# Загрузите модель из MLflow
model_uri = f"models:/{model_name}/production"
try:
    loaded_model = mlflow.pyfunc.load_model(model_uri)
    print("Model loaded successfully.")
except mlflow.exceptions.MlflowException as e:
    print(f"Error loading model: {e}")

# Проверьте загруженную модель
if 'loaded_model' in locals():
    test_predictions = loaded_model.predict(X_test)
    print(f"Test predictions: {test_predictions}")
else:
    print("Model loading failed. Cannot make predictions.")
