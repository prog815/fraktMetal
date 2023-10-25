from flask import Flask, render_template, request, redirect, url_for
from PIL import Image

app = Flask(__name__)

@app.route("/", methods=['GET','POST'])
def index():
    if request.method == 'POST':
        file = request.files['file']
        if file:
            img = Image.open(file)
            img = img.rotate(90)
            img.save(file.filename)
            return redirect(url_for('index'))
    return render_template("index.html")

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000,debug=True)