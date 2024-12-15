from datetime import datetime
from typing import Optional

class RequestLog:
    class RequestLog:
        def __init__(self, method: str, url: str, data: Optional[dict] = None,
                     prediction: Optional[str] = None, prediction_id: Optional[str] = None,
                     feedback: Optional[str] = None):
            # Инициализация объекта RequestLog
            self.method = method
            self.url = url
            self.timestamp = datetime.utcnow()
            self.data = data if data else {}
            self.prediction = prediction
            self.prediction_id = prediction_id
            self.feedback = feedback

    def to_dict(self):
        # Преобразование объекта RequestLog в словарь.
        return {
            "method": self.method,
            "url": self.url,
            "timestamp": self.timestamp,
            "data": self.data,
            "prediction": self.prediction,
            "prediction_id": self.prediction_id,
            "feedback": self.feedback
        }