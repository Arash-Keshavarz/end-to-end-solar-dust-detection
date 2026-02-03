from flask import Flask, request, jsonify, render_template
import os
from flask_cors import CORS, cross_origin
from solar_dust_detection.utils.common import decodeBase64ToImage
from solar_dust_detection.pipeline.prediction import PredictionPipeline


# Define environment variables
os.putenv('LANG', 'en_US.UTF-8')
os.putenv('LC_ALL', 'en_US.UTF-8')

app = Flask(__name__)
CORS(app) 

class ClientApp:
    def __init__(self):
        self.filename = "inputImage.jpg"
        self.classifier = PredictionPipeline(self.filename)

#---------------------------------------------------------------
@app.route("/", methods=['GET'])
def home():
    return render_template('index.html')

#---------------------------------------------------------------
@app.route("/train", methods=['GET','POST'])
def trainRoute():
    os.system("python main.py")
    return "Training done successfully!"
#---------------------------------------------------------------
@app.route("/predict", methods=['POST'])
def predictRoute():
    image = request.json['image']
    decodeBase64ToImage(image, clApp.filename)
    result = clApp.classifier.predict()
    return jsonify(result)
#---------------------------------------------------------------


if __name__ == "__main__":
    clApp = ClientApp()
    # Use 0.0.0.0 for Docker/AWS, or 127.0.0.1 for local
    app.run(host='0.0.0.0', port=8080)