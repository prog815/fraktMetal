from flask import Flask, render_template, request, redirect, url_for
from PIL import Image
import tensorflow
import numpy as np

model = tensorflow.keras.models.load_model("model_keras.h5")

app = Flask(__name__, static_url_path='/static')

@app.route("/", methods=['GET','POST'])
def index():
    if request.method == 'POST':
        file = request.files['file']
        if file:
            img = Image.open(file)
            img.save('static/' + file.filename)
            img = img.rotate(90)
            img.save('static/pred_' + file.filename)
            return render_template("index.html",file=file.filename)
    return render_template("index.html")

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000,debug=True)