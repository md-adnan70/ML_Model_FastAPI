from fastapi import FastAPI
from schema import InputSchema, OutputSchema
from predict import make_prediction, make_batch_predictions
from typing import List

app =FastAPI()

@app.get("/")
def index():
    return {'message':'Welcome to the ML model prediction'}

@app.post('/predict', response_model=OutputSchema)
def predict(user_input: InputSchema):
    prediction = make_prediction(user_input.model_dump())
    return OutputSchema(predicted_price=prediction)  #model_dump() converts the json object to dict


@app.post('/batch_prediction', response_model=List[OutputSchema])
def batch_predict(user_input: List[InputSchema]):
    predictions = make_batch_predictions([x.model_dump() for x in user_input])
    return [OutputSchema(predicted_price=round(prediction,2)) for prediction in predictions]
