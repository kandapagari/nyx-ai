import os
from json import dumps

import numpy as np
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from tensorflow.keras.models import load_model
from uvicorn import run

from tensorflow_utils import class_predictions, load_image, model_predict

app = FastAPI()
model_dir = "food-vision-model.h5"
model = load_model(model_dir)

origins = ["*"]
methods = ["*"]
headers = ["*"]

app.add_middleware(CORSMiddleware,
                   allow_origins=origins,
                   allow_credentials=True,
                   allow_methods=methods,
                   allow_headers=headers)


@app.get("/")
async def root():
    return {"message": "Welcome to the NYX-AI service API!"}


@app.post("/net/image/prediction/")
async def get_net_image_prediction(image_link: str = "",
                                   image_data: str = None):

    if image_link == "" and image_data is None:
        return {"message": "No image link provided"}
    if image_link != "" and image_data is not None:
        return {"message": "Please provide either an image link or image data"}
    image = load_image(image_link, image_data)
    score = model_predict(image, model)

    class_prediction = class_predictions[np.argmax(score)]      
    model_score = round(np.max(score) * 100, 2)
    model_score = dumps(model_score.tolist())

    return {
        "model-prediction": class_prediction,
        "model-prediction-confidence-score": model_score
    }


if __name__ == "__main__":
    port = int(os.environ.get('PORT', 5000))
    run(app, host="0.0.0.0", port=port)
