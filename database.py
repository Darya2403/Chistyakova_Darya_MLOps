from motor.motor_asyncio import AsyncIOMotorClient

# Доступ к MongDB
DATABASE_URL = "mongodb://mongo:27017/"
DATABASE_NAME = "Chistyakova_Darya_MLOps"

client = AsyncIOMotorClient(DATABASE_URL)
db = client[DATABASE_NAME]