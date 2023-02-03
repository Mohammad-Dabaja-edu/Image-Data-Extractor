import shutil

import time

import flask
import os

from werkzeug.utils import secure_filename

from OCR import OCR
from image_parser import ImageParser

app = flask.Flask(__name__)
app.config[
    'UPLOAD_FOLDER'] = 'C:\\Users\\admin\\PycharmProjects\\WebDevM2\\Stage\\IDCardAttributeDetector\\static'


@app.route('/', methods=['GET'])
def index():
    return flask.render_template("home.html")


@app.route('/parse', methods=['GET', 'POST'])
def parse():
    if flask.request.method == 'GET':
        return flask.render_template("parserForm.html")
    else:
        image = flask.request.files['image']
        filename = secure_filename(image.filename)
        new_dir = filename.split(".")[0]

        # create dir to save image
        parent_dir = os.path.join(app.config['UPLOAD_FOLDER'], new_dir)
        if os.path.exists(parent_dir):
            shutil.rmtree(parent_dir)
        os.mkdir(parent_dir)

        # save image
        filepath = os.path.join(parent_dir, filename)
        image.save(filepath)

        # create dir for parses
        parses_dir = os.path.join(parent_dir, "parses")
        os.mkdir(parses_dir)

        # object detection
        #parser = ImageParser()
        start_time = time.time()
        parser.parse(filename, parses_dir)
        end_time = time.time()
        elapsed_time = end_time - start_time
        print('parse! Took {} seconds'.format(elapsed_time))


        # ocr
        ocr = OCR()
        parses = ocr.getText(parses_dir)


        return  flask.render_template("parserResult.html", parses=parses, img=filename)


if __name__ == '__main__':
    parser = ImageParser()
    # run app in debug mode
    app.run(debug=True, host="0.0.0.0", port=3133)
