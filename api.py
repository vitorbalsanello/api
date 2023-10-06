
import os
import sys
sys.path.append('c:/users/admin/appdata/local/packages/pythonsoftwarefoundation.python.3.11_qbz5n2kfra8p0/localcache/local-packages/python311/site-packages')
from flask_cors import CORS
from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename
import pickle
import cv2
import numpy as np
import pickle
from tensorflow.keras.preprocessing import image

with open('C:/Users/admin/Desktop/classificacao_de_carne_smv.pkl', 'rb') as f:
        model = pickle.load(f)
UPLOAD_FOLDER = '/Users/admin/Desktop/upload'
ALLOWED_EXTENSIONS = set(['jpg'])

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

app = Flask(__name__)
CORS(app)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route('/classificacao', methods=['POST'])
def upload_media():
        if 'file' not in request.files:
            return jsonify({'error': 'Media não fornecida'}), 400
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'Nem um arquivo selecionado'}), 400
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))

        with open('C:/Users/admin/Desktop/classificacao_de_carne.pkl', 'rb') as f:
            model = pickle.load(f)

        img = cv2.imread(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (100, 100))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = img_array / 255.0

        resultado = model.predict(img_array)
        classe_predita = np.argmax(resultado)

        if classe_predita == 1:
            return jsonify({'Resultado da Carne': 'Carne estragada!'})
        elif classe_predita == 0:
            return jsonify({'Resultado da Carne': 'Carne Fresca!'})
    
        return jsonify({'error': 'Erro ao processar a imagem'}), 400

if __name__ == '__main__':
    app.run(debug=True, port=5000)