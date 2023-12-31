import os
from flask import Flask, flash, request, redirect, make_response, render_template, send_from_directory
from flask_cors import CORS
from werkzeug.utils import secure_filename
from utils.anonymizer import anonymize

APP_ROOT = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER = os.path.join(APP_ROOT, 'CVs/uploads')
DOWNLOAD_FOLDER = os.path.join(APP_ROOT, 'CVs/downloads')
ALLOWED_EXTENSIONS = {'txt', 'pdf'}

app = Flask(__name__)
CORS(app)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['DOWNLOAD_FOLDER'] = DOWNLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1000 * 1000  # Limit file size to 16 MB


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/uploads/<string:name>')
def download_file(name):

    return send_from_directory(app.config["DOWNLOAD_FOLDER"], name, as_attachment=False)


@app.route("/", methods=['GET','POST'])
def home():
      if request.method == 'GET':
        return render_template('index.html')
      elif request.method == 'POST':
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            ofilename = 'anonymized_cv.txt'
            ofilepath = os.path.join(app.config['DOWNLOAD_FOLDER'], ofilename)
            with open(ofilepath, 'w', encoding='utf-8') as f:
                f.write(anonymize(filepath))
            return send_from_directory(app.config["DOWNLOAD_FOLDER"], ofilename, as_attachment=True)
        else:
            flash('Invalid file type')
            return redirect(request.url)

if __name__ == "__main__":
    app.run(debug=False, host="0.0.0.0")