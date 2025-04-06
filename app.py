import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # suppress TensorFlow INFO logs

from flask import Flask, request, jsonify
import numpy as np
import tensorflow as tf
from utils import extract_mfcc_from_path
from werkzeug.utils import secure_filename

from transformers import Wav2Vec2ForCTC, Wav2Vec2Tokenizer
import torch
import librosa


import os

app = Flask(__name__)
os.makedirs('static', exist_ok=True)

model = tf.keras.models.load_model('serVold_0.h5')

# Load Wav2Vec2 model and tokenizer
tokenizer = Wav2Vec2Tokenizer.from_pretrained("facebook/wav2vec2-base-960h")
s2t_model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-base-960h")


emotion_labels = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'pleasant_surprise', 'sad']

def speech_to_text(filepath):
    audio, sr = librosa.load(filepath, sr=16000)  # Wav2Vec expects 16kHz
    input_values = tokenizer(audio, return_tensors="pt", padding="longest").input_values
    with torch.no_grad():
        logits = s2t_model(input_values).logits
    predicted_ids = torch.argmax(logits, dim=-1)
    transcription = tokenizer.batch_decode(predicted_ids)[0]
    return transcription


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
        # Emotion prediction
        mfcc = extract_mfcc_from_path(filepath)
        mfcc = np.expand_dims(mfcc, axis=0)
        prediction = model.predict(mfcc)
        predicted_label = emotion_labels[np.argmax(prediction)]
        confidence = round(np.max(prediction) * 100, 2)

        # Speech-to-text prediction
        transcription = speech_to_text(filepath)

        return jsonify({
            'prediction': predicted_label,
            'confidence': confidence,
            'transcription': transcription
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    app.run(debug=True)