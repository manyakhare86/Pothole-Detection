from flask import Flask, request
from flask_cors import CORS
from PIL import Image
import numpy as np


app = Flask(__name__)
CORS(app)

@app.route("/pothole", methods=["POST"])
def predict():
    img = request.files['img']
    img = Image.open(img)
    # rgb to bgr
    imgarr = np.array(img)[:,:,::-1]
    print(imgarr.shape)
    return "hi"


app.run(debug=True)
import sys, os, distutils.core
from flask import Flask, jsonify, request
import cv2
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog

app = Flask(__name__)

# Load your trained model and config
cfg = get_cfg()
cfg.merge_from_file("config1.yaml")  # Update with the correct path to your config file
cfg.MODEL.WEIGHTS = "path/to/your/model_final.pth"  # Update with the correct path to your model weights
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # Adjust the threshold as needed
predictor = DefaultPredictor(cfg)
metadata = MetadataCatalog.get(cfg.DATASETS.TEST[0])

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get the image file from the request
        image_file = request.files['image']
        image = cv2.imdecode(np.frombuffer(image_file.read(), np.uint8), cv2.IMREAD_COLOR)

        # Make prediction
        outputs = predictor(image)

        # Visualize the predictions (optional)
        v = Visualizer(image[:, :, ::-1], metadata=metadata, scale=0.8)
        v = v.draw_instance_predictions(outputs["instances"].to("cpu"))
        visualization = v.get_image()[:, :, ::-1]

        # Convert predictions to JSON
        predictions = {
            "predictions": outputs["instances"].to("cpu").get_fields()["pred_boxes"].tensor.numpy().tolist(),
            "visualization": visualization.tolist(),
        }

        return jsonify(predictions)

    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == '__main__':
    app.run(debug=True)
