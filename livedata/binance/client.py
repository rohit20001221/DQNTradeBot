from redis import Redis
from livedata.binance.entities import LiveData
import json


class BinanceClient:
    def __init__(self):
        self.db = Redis(decode_responses=True)
        
    def get(self, symbol) -> LiveData:
        return LiveData(json.loads(self.db.get(symbol)))