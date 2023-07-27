from fastapi import FastAPI, UploadFile, File, Depends, HTTPException, Header
from typing import Union, List
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from typing import List, Optional
from pydantic import BaseModel
from PIL import Image
import tempfile
import prediction
import geo_location_file
from io import BytesIO
import insect_prediction_no
import requests
import os
import base64

app = FastAPI(title="Plant and Insect Predictor API", description="This API is used to predict the type of plant or insect present in an uploaded image")

def api_key_header(x_api_key: str = Header(...)):
    if x_api_key != "12345678910":
        raise HTTPException(status_code=400, detail="Invalid API Key")
    return x_api_key

class Prediction(BaseModel):
    name: str
    probability: float
    gbif_data: Optional[dict]
    german_name: Optional[str]
    image_url: Optional[str]

def encode_image(image_path):
    with open(image_path, 'rb') as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

@app.on_event("startup")
def load_model():
    global flower_model
    global insect_model
    global flower_class
    global insect_class
    flower_model, insect_model = prediction.model()
    insect_class, flower_class = prediction.label()

@app.get("/images/{folder}/{filename}")
async def read_image(folder: str, filename: str):
    plant_path = os.path.join("/Users/methunraj/Desktop/ML/plant_photo", folder, filename)
    insect_path = os.path.join("/Users/methunraj/Desktop/ML/Insect", folder, filename)
    
    if os.path.isfile(plant_path):
        return FileResponse(plant_path)
    elif os.path.isfile(insect_path):
        return FileResponse(insect_path)
    else:
        raise HTTPException(status_code=404, detail="File not found")
    
example_response = {
    "lat_lon": "(None, None)",
    "flower_predictions": [
        {
            "name": "Crepis capillaris (L.) Wallr",
            "probability": 0.39339977502822876,
            "gbif_data": {
                "scientific_name": "Crepis capillaris (L.) Wallr.",
                "rank": "SPECIES"
            },
            "german_name": None,
            "image_url": "/images/Crepis capillaris (L.) Wallr/photo_510.jpg"
        },
        # More flowers...
    ],
    "insect_predictions": [
        {
            "name": "Wild Bees",
            "probability": 0.42566871643066406,
            "gbif_data": None,
            "german_name": None,
            "image_url": None
        },
        # More insects...
    ],
    "Number_of_Insects": 5
}


@app.post('/predict',
          responses={
              200: {
                  "model": Prediction,
                  "description": "Successful Response",
                  "content": {
                      "application/json": {
                          "example": example_response
                      }
                  }
              }
          },
          dependencies=[Depends(api_key_header)])

async def predict(file: UploadFile = File(...)):
    contents = await file.read()
    image = Image.open(BytesIO(contents))
    lat, lon = geo_location_file.get_geolocation(image)

    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as temp:
        image_path = temp.name
        image.save(image_path)

    insect_model_no = insect_prediction_no.model()
    insect_results = insect_model_no.predict(image_path, confidence=45, overlap=40)

    flower_indices, flower_predictions = prediction.process_image(image, flower_model, 224)

    flower_preds = [Prediction(name=flower_class.inverse_transform([idx])[0], 
                               probability=float(pred),
                               gbif_data=None,
                               german_name=None,
                               image_url=None) for idx, pred in zip(flower_indices, flower_predictions)]

    for pred in flower_preds:
        response = requests.get(f"https://api.gbif.org/v1/species/match?name={pred.name}")
        if response.content:
            data = response.json()
            pred.gbif_data = {"scientific_name": data.get('scientificName'), "rank": data.get('rank')}
            species_key = data.get('usageKey')

            response = requests.get(f"https://api.gbif.org/v1/species/{species_key}/vernacularNames")
            if response.content:
                try:
                    vernacular_data = response.json()
                except Exception:
                    continue

                for name in vernacular_data.get('results', []):
                    if name.get('language') == 'deu':
                        pred.german_name = name.get('vernacularName')
                        break

        image_files = os.listdir(os.path.join("/Users/methunraj/Desktop/ML/plant_photo", pred.name))
        if image_files:
            pred.image_url = encode_image(os.path.join("/Users/methunraj/Desktop/ML/plant_photo", pred.name, image_files[0]))

    if len(insect_results) > 0:
        num_insects = len(insect_results)
        insect_indices, insect_predictions = prediction.process_image(image, insect_model, 224)

        insect_preds = [Prediction(name=insect_class.inverse_transform([idx])[0], 
                                   probability=float(pred),
                                   gbif_data=None,
                                   german_name=None,
                                   image_url=None) for idx, pred in zip(insect_indices, insect_predictions)]

        for pred in insect_preds:
            response = requests.get(f"https://api.gbif.org/v1/species/match?name={pred.name}")
            if response.content:
                data = response.json()
                pred.gbif_data = {"scientific_name": data.get('scientificName'), "rank": data.get('rank')}
                species_key = data.get('usageKey')

                response = requests.get(f"https://api.gbif.org/v1/species/{species_key}/vernacularNames")
                if response.content:
                    try:
                        vernacular_data = response.json()
                    except Exception:
                        continue

                    for name in vernacular_data.get('results', []):
                        if name.get('language') == 'deu':
                            pred.german_name = name.get('vernacularName')
                            break
                            
            image_files = os.listdir(os.path.join("/Users/methunraj/Desktop/ML/Insect", pred.name))
            if image_files:
                pred.image_url = encode_image(os.path.join("/Users/methunraj/Desktop/ML/Insect", pred.name, image_files[0]))

    else:
        insect_preds = "No Insect Found in the Image"
        num_insects = 0

    return {"lat,lon": f"{lat,lon}", "flower_predictions": flower_preds, "insect_predictions": insect_preds, "Number of Insects": num_insects}
