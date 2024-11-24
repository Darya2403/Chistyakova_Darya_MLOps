from pymongo import MongoClient

# Доступ к MongDB
DATABASE_URL = "mongodb://localhost:27017/"
DATABASE_NAME = "Chistyakova_Darya_MLOps"

client = MongoClient(DATABASE_URL)
db = client[DATABASE_NAME]