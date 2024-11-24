#pip install fastapi uvicorn
#pip install jinja2
#pip install aiofiles
#pip install python-multipart
#pip install pymongo

#uvicorn main:app --reload


from fastapi import FastAPI, Request, Form
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
from database import db
from db_models import RequestLog
from models import dummy_model
import logging
from functools import wraps
import asyncio

# Настройка логгера
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

app = FastAPI()

templates = Jinja2Templates(directory="templates")

@app.on_event("startup")
async def startup_event():
    # Принудительная запись в базу данных при старте приложения
    test_log_entry = RequestLog(method="GET", url="/start")
    result = db.request_logs.insert_one(test_log_entry.to_dict())
    logger.info(f"Inserted test log entry: {test_log_entry.to_dict()}, Result: {result.inserted_id}")

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    logger.info("Receive")
    data = {}
    # Запись в MongoDB прямо в методе
    log_entry = RequestLog(
        method=request.method,
        url=str(request.url),
        data = data
    )
    result = db.request_logs.insert_one(log_entry.to_dict())
    logger.info(f"Logged GET request to MongoDB: {log_entry.to_dict()} - Inserted ID: {result.inserted_id}")

    return templates.TemplateResponse("index.html", {"request": request, "data": data})

@app.post("/predict", response_class=HTMLResponse)
async def predict(
    request: Request,
    Gender: str = Form(...),
    Age: int = Form(None),
    Height: float = Form(...),
    Weight: float = Form(...),
    FHWO: int = Form(...),
    FAVC: int = Form(...),
    FCVC: int = Form(None),
    NCP: float = Form(None),
    CAEC: str = Form(...),
    SMOKE: int = Form(...),
    CH2O: float = Form(None),
    SCC: int = Form(...),
    FAF: float = Form(None),
    TUE: int = Form(None),
    CALC: str = Form(...),
    MTRANS: str = Form(...)
):
    logger.info("Receive POST!!!!!!")
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

    # Запись в MongoDB прямо в методе
    log_entry = RequestLog(
        method=request.method,
        url=str(request.url),
        data = data
    )

    result = db.request_logs.insert_one(log_entry.to_dict())
    logger.info(f"Logged POST request to MongoDB: {log_entry.to_dict()} - Inserted ID: {result.inserted_id}")

    # Здесь предполагается, что dummy_model возвращает какую-то строку или прогноз
    prediction = dummy_model(data)
    return templates.TemplateResponse("index.html", {"request": request, "data": data, "prediction": prediction})