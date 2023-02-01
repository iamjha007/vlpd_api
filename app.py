#!/usr/bin/env python3

from vlpddetect import main
import os
from flask_ngrok import run_with_ngrok
from flask import Flask, request
from werkzeug.utils import secure_filename
import uuid
import easyocr

UPLOAD_FOLDER = "./uploads"
ALLOWED_EXTENSIONS = ["png", "jpg", "jpeg"]
MAX_CONTENT_LENGTH = 16 * 1024 * 1024
app = Flask(__name__)
run_with_ngrok(app)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["MAX_CONTENT_LENGTH"] = MAX_CONTENT_LENGTH

def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route("/vlpd_api", methods=["POST"])
def upload_image():
    if request.method == "POST":

        if "image" not in request.files:
            return "no image part was sent to the API"

        image = request.files["image"]

        if image.filename == "":
            return "no image was sent to the API"

        if image and allowed_file(image.filename):

            filename = str(uuid.uuid4()) + secure_filename(image.filename)
            path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
            image.save(path)
            
            np_str =main(img_path=f"{path}")
          
            # print(np_str)
            return np_str
            
if __name__ == "__main__":
    app.run()
 
    
