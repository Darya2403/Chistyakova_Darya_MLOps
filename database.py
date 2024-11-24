from pymongo import MongoClient

DATABASE_URL = "mongodb://localhost:27017/"
DATABASE_NAME = "Chistyakova_Darya_MLOps"

client = MongoClient(DATABASE_URL)
db = client[DATABASE_NAME]