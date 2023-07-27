from roboflow import Roboflow

rf = Roboflow(api_key="qkX8ydyBEmS8KQAVIHHL")

def model():
    
    project = rf.workspace().project("insectdetector")  
    model = project.version(5).model
    
    return model