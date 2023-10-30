from flask import Flask, render_template, request, redirect, url_for
from PIL import Image
import tensorflow
import numpy as np


model = tensorflow.keras.models.load_model("model_keras.h5")

app = Flask(__name__, static_url_path='/static')


@app.route("/", methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        file = request.files['file']
        if file:
            img = Image.open(file)

            img = img.resize((256, 256), Image.LANCZOS)
            img.save('static/' + file.filename)
            inp_numpy = np.array(img)[None]
            segmentation_output = model.predict(inp_numpy)[0].argmax(-1)

            arr = segmentation_output

            rgb_arr = np.zeros((128, 128, 3), dtype=np.uint8)

            dicRGB = {2: (255, 0, 0), 1: (0, 255, 0), 0: (0, 0, 255)}

            for i in range(128):
                for j in range(128):
                    rgb_arr[i, j] = dicRGB[arr[i, j]]

            res_image = Image.fromarray(rgb_arr, 'RGB')

            res_image.save('static/pred_' + file.filename)
            return render_template("index.html", file=file.filename, colorsdict = colors_count(rgb_arr))
    return render_template("index.html")


def colors_count(rgb_arr):
    w,h,d = rgb_arr.shape
    total_pix_cnt = w*h

    red_cnt = np.sum(np.all(rgb_arr == [255, 0, 0],axis=2))
    green_cnt = np.sum(np.all(rgb_arr == [0, 255, 0],axis=2))
    blue_cnt = np.sum(np.all(rgb_arr == [0, 0, 255],axis=2))

    percent_red = round((red_cnt/total_pix_cnt)*100,2)
    percent_green = round((green_cnt / total_pix_cnt) * 100, 2)
    percent_blue = round((blue_cnt / total_pix_cnt) * 100, 2)

    return {'red': percent_red,
            'green': percent_green,
            'blue': percent_blue,
            'total': total_pix_cnt}


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=True)
