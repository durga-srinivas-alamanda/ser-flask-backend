from flask import Flask, request, jsonify
import numpy as np
import tensorflow as tf
import os
from utils import extract_mfcc_from_path
from werkzeug.utils import secure_filename

app = Flask(__name__)
os.makedirs('static', exist_ok=True)

model = tf.keras.models.load_model('serVold_0.h5')

emotion_labels = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'pleasant_surprise', 'sad']

@app.route('/')
def home():
    return 'Speech Emotion Recognition API is Running ✅'

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400

    file = request.files['file']
    filename = secure_filename(file.filename)
    filepath = os.path.join('static', filename)
    file.save(filepath)

    try:
        mfcc = extract_mfcc_from_path(filepath)
        mfcc = np.expand_dims(mfcc, axis=0)
        prediction = model.predict(mfcc)
        predicted_label = emotion_labels[np.argmax(prediction)]

        return jsonify({
            'prediction': predicted_label,
            'confidence': round(np.max(prediction) * 100, 2)
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)