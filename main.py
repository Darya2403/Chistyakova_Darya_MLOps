# Установка необходимых библиотек
#pip install fastapi uvicorn
#pip install jinja2
#pip install aiofiles
#pip install python-multipart
#pip install pymongo

# Запуск сервера Uvicorn с автоматической перезагрузкой при изменении кода
#uvicorn main:app --reload


from fastapi import FastAPI, Request, Form
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
from database import db
from db_models import RequestLog
from models import predict_obesity
import logging


# Настройка логгера
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()  # Создание экземпляра FastAPI

templates = Jinja2Templates(directory="templates")  # Настройка шаблонов Jinja2

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    data = {}  # Инициализация пустого словаря для данных
    # Запись лога в MongoDB
    log_entry = RequestLog(
        method=request.method,  # Метод запроса
        url=str(request.url),  # URL запроса
        ip=str(request.client.host),  # IP-адрес клиента
        data=data  # Данные запроса
    )
    result = db.request_logs.insert_one(log_entry.to_dict())  # Вставка записи в MongoDB
    logger.info(f"Logged GET request to MongoDB: {log_entry.to_dict()} - Inserted ID: {result.inserted_id}")

    return templates.TemplateResponse("index.html", {"request": request, "data": data})

@app.post("/predict", response_class=HTMLResponse)
async def predict(
    request: Request,
    Gender: int = Form(None),  # Пол
    Age: int = Form(None),  # Возраст
    Height: float = Form(...),  # Рост
    Weight: float = Form(...),  # Вес
    FHWO: int = Form(...),  # Страдает ли кто-то из членов семьи избыточным весом
    FAVC: int = Form(...),  # Часто ли вы едите высококалорийную пищу
    FCVC: int = Form(None),  # Как часто вы едите овощи во время еды
    NCP: float = Form(None),  # Сколько основных приемов пищи у вас в день
    CAEC: str = Form(...),  # Едите ли вы между основными приемами пищи
    SMOKE: int = Form(...),  # Вы курите
    CH2O: float = Form(None),  # Сколько литров воды вы пьете ежедневно
    SCC: int = Form(...),  # Следите ли вы за потреблением калорий
    FAF: float = Form(None),  # Как часто вы занимаете физической активностью
    TUE: int = Form(None),  # Как много времени вы проводите за технологическими устройствами
    CALC: str = Form(...),  # Как часто вы употребляете алкоголь
    MTRANS: str = Form(...)  # Каким транспортом вы обычно пользуетесь
):
    data = {
        "Gender": Gender,
        "Age": Age,
        "Height": Height,
        "Weight": Weight,
        "FHWO": FHWO,
        "FAVC": FAVC,
        "FCVC": FCVC,
        "NCP": NCP,
        "CAEC": CAEC,
        "SMOKE": SMOKE,
        "CH2O": CH2O,
        "SCC": SCC,
        "FAF": FAF,
        "TUE": TUE,
        "CALC": CALC,
        "MTRANS": MTRANS
    }

    # Запись лога в MongoDB
    log_entry = RequestLog(
        method=request.method,  # Метод запроса
        url=str(request.url),  # URL запроса
        ip=str(request.client.host),  # IP-адрес клиента
        data=data  # Данные запроса
    )

    result = db.request_logs.insert_one(log_entry.to_dict())  # Вставка записи в MongoDB
    logger.info(f"Logged POST request to MongoDB: {log_entry.to_dict()} - Inserted ID: {result.inserted_id}")

    # Здесь предполагается, что dummy_model возвращает какой-то прогноз
    prediction = predict_obesity(data)
    return templates.TemplateResponse("index.html", {"request": request, "data": data, "prediction": prediction})
