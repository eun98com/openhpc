import requests
import json
import numpy as np
from tensorflow.keras.datasets import fashion_mnist

# Fashion MNIST 데이터 로드
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

# 테스트 이미지 데이터 (첫 번째 테스트 이미지 사용)
test_image = test_images[0]

# 28x28 이미지를 1차원 배열로 변환
test_image_flatten = test_image.flatten().tolist()

# JSON 데이터 생성
data = {
    "features": test_image_flatten
}

# API 요청
response = requests.post('http://127.0.0.1:5000/predict', json=data)

# 결과 출력
print(response.json())
print("Fashion MNIST 모델 API 요청 확인")
