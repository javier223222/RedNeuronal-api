from flask import Flask, request, jsonify
from flask_cors import CORS
from keras import models
import cv2
import numpy as np

app = Flask(__name__)
CORS(app, origins='*')

model = models.load_model('src/model/document-model.h5')
classes=["Acta-De-Nacimiento","Carnet-De-Seguro-Medico","Cartilla-Militar","Credencial-Universitaria","Curp","Ine",
         "Licencia-De-Conducir","Pasaporte","Rfc","Tarjeta-De-Credito","Tarjeta-De-Membresia","Tarjeta-De-Metro","Visa"]
formattedNames ={
    'Acta-De-Nacimiento': 'Acta De Nacimiento',
    'Carnet-De-Seguro-Medico': 'Carnet De Seguro Medico',
    'Cartilla-Militar': 'Cartilla Militar',
    'Credencial-Universitaria': 'Credencial Universitaria',
    'Curp': 'Curp',
    'Ine': 'Ine',
    'Licencia-De-Conducir': 'Licencia De Conducir',
    'Pasaporte': 'Pasaporte',
    'Rfc': 'Rfc',
    'Tarjeta-De-Credito': 'Tarjeta De Credito',
    "Tarjeta-De-Membresia":"Tarjeta-De-Membresia",
    "Tarjeta-De-Metro":"Tarjeta De Metro",
    "Visa":"Visa"

}

@app.route('/predictdocument', methods=['POST'])
def predict():
    image_data = request.files['image']
    img = cv2.imdecode(np.fromstring(image_data.read(), np.uint8), cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (64, 64))
    img = np.array(img).reshape(-1, 64, 64, 1)
    prediction = model.predict(img)
    predicted_class = classes[np.argmax(prediction)]
    return jsonify({'document': formattedNames[predicted_class]})


if __name__ == '__main__':
    app.run(host='localhost', port=5000, debug=True)