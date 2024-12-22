import os
import uuid
from upscale_model import upscale
from flask import Flask, request, render_template, redirect, url_for, flash, send_from_directory

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

app = Flask(__name__)
app.secret_key = 'ayam_geprek'
app.config['UPLOAD_FOLDER'] = './static/input'

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    file = request.files['file']

    if file and allowed_file(file.filename):
        filename = file.filename

        unique_id = str(uuid.uuid4())
        filename = file.filename.rsplit('.', 1)[0].lower()
        extension = file.filename.rsplit('.', 1)[1].lower()
        new_filename = f'{unique_id}_{filename}.{extension}'

        filepath = os.path.join(app.config['UPLOAD_FOLDER'], new_filename)
        file.save(filepath)
        upscale(filepath)
        normal_path = os.path.join('./static/input', new_filename)
        upscaled_path = os.path.join('./static/output', f'upscaled_{new_filename}')
        return render_template('show.html', normal_img_path=normal_path, upscaled_img_path=upscaled_path, upscaled_name=f'upscaled_{new_filename}')
    else:
        flash('File type not allowed')
        return render_template('error.html')

@app.route('/<filename>')
def download_file(filename):
    return send_from_directory('./static/output', filename, as_attachment=True)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=80)