import datetime
from urllib import response
from fastapi import FastAPI, HTTPException
import joblib
from pydantic import BaseModel

from model import BASE_DIR, TODAY, predict, convert
import pandas as pd
from pathlib import Path


app = FastAPI()


#pydantic models
class StockIn(BaseModel):
    ticker: str

class StockOut(BaseModel):
    forecast: dict

#routes

@app.get("/ping")
def pong():
    return {"ping": "pong!"}

@app.post("/predict", response_model=StockOut, status_code=200)
def get_prediction(payload: StockIn):
    ticker = payload.ticker

    prediction_list = predict(ticker)

    if not prediction_list:
        raise HTTPException(status_code=400, detail="Ticker not found")

    response_object = {"ticker": ticker, "forecast": convert(prediction_list)}
    return response_object

def predict(ticker="MSFT", days = 7):
    model_file = Path(BASE_DIR).joinpath(f"{ticker}.joblib")
    if not model_file.exists():
        return False

    model = joblib.load(model_file)

    future = TODAY + datetime.timedelta(days=days)

    dates = pd.date_range(start="2020-01-01", end=future.strftime("%m/%d/%Y"),)
    df = pd.DataFrame({"ds": dates})

    forecast = model.predict(df)

    # model.plot(forecast).savefig(f"{ticker}_plot.png")
    # model.plot_components(forecast).savefig(f"{ticker}_plot_components.png")

    return forecast.tail(days).to_dict("records")

