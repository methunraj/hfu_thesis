import joblib
from keras.models import load_model
import numpy as np
from sklearn.preprocessing import LabelEncoder

# Load your models and label encoders here
flower_model = load_model('plant_model.hdf5')
flower_encoder = joblib.load('plant_label.joblib')
insect_model = load_model('insect_model.hdf5')
insect_encoder = joblib.load('Insect_label.joblib')
insect_class = LabelEncoder()
flower_class = LabelEncoder()

def model():
    return flower_model,insect_model
    

def label():
    insect_class.fit(insect_encoder)
    flower_class.fit(flower_encoder)
    return insect_class,flower_class
    

# Function to process your image and predict
def process_image(image, model, input_size):
    
    img = image.resize((input_size, input_size))
    img = np.array(img) / 255.0
    img = np.expand_dims(img, axis=0)
    predictions = model.predict(img)
    top_5_indices = np.argsort(predictions[0])[-5:][::-1]  # get top 5 predictions
    return top_5_indices, predictions[0][top_5_indices],
