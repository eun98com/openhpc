import numpy as np
import tensorflow as tf
from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model

# TensorFlow GPU 설정
physical_devices = tf.config.list_physical_devices('GPU')
if physical_devices:
    try:
        for device in physical_devices:
            tf.config.experimental.set_memory_growth(device, True)
    except RuntimeError as e:
        print(e)

# Flask 앱 초기화
app = Flask(__name__)

# 모델 로드
model = load_model('/home/jovyan/Notebooks/fashionmnist_model.h5')

@app.route('/')
def home():
    return "Fashion MNIST 모델 API를 통한 포트포워딩 테스트"

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        features = np.array(data['features'])
        features = features.reshape(1, 28, 28, 1)  # 모델 입력 형식에 맞게 reshape
        prediction = model.predict(features)
        predicted_class = np.argmax(prediction, axis=1)[0]
        return jsonify(prediction=int(predicted_class))
    except Exception as e:
        return jsonify({"error": str(e)}), 400

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)

