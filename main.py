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

app = FastAPI(title="Plant and Insect Predictor API", description="This API is used to predict the type of plant or insect present in an uploaded image")


def api_key_header(x_api_key: str = Header(...)):
    """
    This function checks if the provided API Key in the request header is valid.
    
    - **x_api_key**: API key to be verified.
    """
    if x_api_key != "12345678910":
        raise HTTPException(status_code=400, detail="Invalid API Key")
    return x_api_key

class Prediction(BaseModel):
    """
    A model class representing a single prediction result.
    
    - **name**: The name of the predicted species
    - **probability**: The probability of the prediction
    - **gbif_data**: The GBIF data of the predicted species (optional)
    - **german_name**: The German name of the species (optional)
    - **image_url**: The URL of the image of the species (optional)
    """
    name: str
    probability: float
    gbif_data: Optional[dict]
    german_name: Optional[str]
    image_url: Optional[str]
    

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


@app.get("/images/{folder}/{filename}")
async def read_image(folder: str, filename: str):
    """
    Function to return an image file stored in the specified folder and filename.
    It first tries to locate the file in the plant photos folder, and if it doesn't find it, 
    it looks for the file in the insect photos folder.
    """
    plant_path = os.path.join("/Users/methunraj/Desktop/ML/plant_photo", folder, filename)
    insect_path = os.path.join("/Users/methunraj/Desktop/ML/Insect", folder, filename)
    
    if os.path.isfile(plant_path):
        return FileResponse(plant_path)
    elif os.path.isfile(insect_path):
        return FileResponse(insect_path)
    else:
        raise HTTPException(status_code=404, detail="File not found")

# @app.post('/predict',dependencies=[Depends(api_key_header)])
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


async def predict(file: Optional[UploadFile] = File(None), image_url: Optional[str] = None):

    """
    Endpoint to predict the type of plant or insect in an uploaded image.
    The function first reads the image and retrieves its geolocation.
    It then uses two models to predict if the image contains a flower or an insect.
    Additionally, it retrieves the scientific name and the German name of the species if available 
    and adds the URL of an image of the species to the response.
    """

    # Read the contents of the uploaded file and open the image
    
    # Check if an image URL is provided
    if image_url:
        response = requests.get(image_url)
        response.raise_for_status()  # Raise an exception for HTTP errors
        contents = response.content
    else:
        # Read the contents of the uploaded file
        contents = await file.read()

    image = Image.open(BytesIO(contents))

    # Retrieve the geolocation of the image
    lat, lon = geo_location_file.get_geolocation(image)

    # Save the image temporarily
    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as temp:
        image_path = temp.name
        image.save(image_path)

    # Load the insect prediction model and perform a prediction
    insect_model_no = insect_prediction_no.model()
    insect_results = insect_model_no.predict(image_path, confidence=45, overlap=40)

    # Load the flower and insect models and their corresponding classes
    flower_model, insect_model = prediction.model()
    insect_class, flower_class = prediction.label()

    # Process the image using the flower model and retrieve predictions
    flower_indices, flower_predictions = prediction.process_image(image, flower_model, 224)

    # Convert the flower predictions to Prediction objects
    flower_preds = [Prediction(name=flower_class.inverse_transform([idx])[0], 
                               probability=float(pred),
                               gbif_data=None,
                               german_name=None,
                               image_url=None) for idx, pred in zip(flower_indices, flower_predictions)]  # Initialize image_url as None

    # Iterate over the flower predictions
    for pred in flower_preds:
        # Retrieve GBIF data and German name for the predicted species
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

        # Add image URL to the prediction
        image_files = os.listdir(os.path.join("/Users/methunraj/Desktop/ML/plant_photo", pred.name))  # Use predicted name as folder name
        if image_files:
            pred.image_url = f"/images/{pred.name}/{image_files[0]}"  # Include predicted name in URL

    # Check if the image contains insects
    if len(insect_results) > 0:
        # Retrieve the number of insects and perform predictions
        num_insects = len(insect_results)
        insect_indices, insect_predictions = prediction.process_image(image, insect_model, 224)

        # Convert the insect predictions to Prediction objects
        insect_preds = [Prediction(name=insect_class.inverse_transform([idx])[0], 
                                   probability=float(pred),
                                   gbif_data=None,
                                   german_name=None,
                                   image_url=None) for idx, pred in zip(insect_indices, insect_predictions)]  # Initialize image_url as None

        # Iterate over the insect predictions
        for pred in insect_preds:
            # Retrieve GBIF data and German name for the predicted species
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
                            
            # Add image URL to the prediction
            image_files = os.listdir(os.path.join("/Users/methunraj/Desktop/ML/Insect", pred.name))  # Use predicted name as folder name
            if image_files:
                pred.image_url = f"/images/{pred.name}/{image_files[0]}"  # Include predicted name in URL

    else:
        insect_preds = "No Insect Found in the Image"
        num_insects = 0

    # Return the predictions along with the geolocation and the number of insects
    return {"lat,lon": f"{lat,lon}", "flower_predictions": flower_preds, "insect_predictions": insect_preds, "Number of Insects": num_insects}
