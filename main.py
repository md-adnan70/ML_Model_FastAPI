from fastapi import FastAPI
from schema import InputSchema, OutputSchema
from predict import make_prediction

app =FastAPI()

@app.get("/")
def index():
    return {'message':'Welcome to the ML model prediction'}

@app.post('/predict', response_model=OutputSchema)
def predict(user_input: InputSchema):
    return OutputSchema(predicted_price=make_prediction(user_input.model_dump()))  #model_dump() converts the json object to dict

