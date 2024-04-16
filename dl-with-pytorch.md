# 머신러닝과 딥러닝
## 머신러닝
- 인간이 특성 추출, 컴퓨터가 예측 및 분류
- 알고리즘
  - 지도학습
    - 분류
      - KNN
      - SVM
      - 결정트리
      - 로지스틱 회귀
    - 회귀
      - 선형 회귀
  - 비지도학습
    - 군집
      - K-means
      - DBSCAN
    - 차원축소
      - 주성분분석(PCA)
  - 강화학습
      - 마르코프 결정 프로세스

## 딥러닝
- 컴퓨터가 특성 추출, 예측 및 분류
- 알고리즘
  - 지도학습
    - 이미지 데이터
      - CNN
      - AlexNet
      - ResNet
    - 시계열 데이터
      - RNN
      - LSTM
  - 비지도학습
    - 군집
      - 가우시안 혼합 모델
      - 자기 조직화 지도
    - 차원축소
      - 오토인코더
      - 주성분분석(PCA)
  - 전이학습
      - BERT
      - MobileNetV2
  - 강화학습
      - 마르코프 결정 프로세스
- 성능 평가 지표
  - True Positive : 예측 1, 실제 1
  - True Negative : 예측 0, 실제 0
  - False Positive : 예측 1, 실제 0
  - False Negative : 예측 0, 실제 1
  - 정확도(accuracy) : 전체 데이터 중 정답을 맞힌 비율. (True Positive + True Negative) / 전체 데이터
  - 재현율(recall) : 실제 값이 1인 데이터 중 1이라고 예측한 비율. True Positive / (True Positive + False Negative)
  - 정밀도(precision) : 1이라고 예측한 데이터 중 실제 값이 1인 비율. True Positive / (True Positive + False Positive)
  - F1-스코어 : 재현율과 정밀도의 조화평균값. 재현율과 정밀도는 trade-off 관계이므로 따로 보았을때 판단이 어렵다. 이 둘의 지표를 복합적으로 나타내기 위해 F1 스코어를 사용

# Pytorch
## 텐서란?
- 벡터 : 단일 데이터타입의 1차원 배열, axis 0
- 행렬 : 단일 데이터타입의 2차원 배열, axis 1
- 텐서 : 단일 데이터타입의 3차원 배열, axis 2

## 텐서와 스토리지
- 텐서를 메모리에 저장하기 위해서는 1차원으로 변환해야 하는데, 이때 1차원으로 변환된 배열을 스토리지(storage)라고 한다.
- 1차원으로 변환하면서 손실되는 shape에 대한 정보를 offset과 stride로 저장한다.
  - offset : 텐서의 첫번째 요소가 스토리지에 저장된 인덱스
  - stride : 동일 차원의 다음 요소에 접근하기 위해 증가해야할 인덱스
  - ex
    - shape=(2, 3) => stride=(3, 1)
    - shape=(3, 2) => stride=(2, 1)

## torchvision
- 파이토치에서 제공하는 데이터셋을 모아둔 패키지

## Model
- 모델 : 최종적으로 구성할 네트워크로, 한개 이상의 모듈로 구성되어있다.
- 모듈 : 한개 이상의 계층으로 구성된 집합

## 모델 파라미터
- 손실함수 : 학습시 출력값과 실제값의 오차를 측정하는 함수
  - 평균제곱오차(Mean Squared Error, MSE) : 회귀모델에서 사용
  - (Binary) Cross-Entropy Loss : 분류모델에서 사용. label이 one-hot encoding 형식일때만 사용 가능
- 옵티마이저 : 데이터와 손실함수를 바탕으로 모델의 업데이트 방법을 결정하는 알고리즘
- 학습률 스케줄러 : 학습의 최적화를 위해 학습률을 지속적으로 감소시키는 알고리즘
- 지표 : 훈련과 테스트단계를 모니터링할 수 있는 파라미터

## 모델 학습
- 학습이란 임의의 함수에 대해 출력과 실제값의 오차를 최소화 하는 상수값을 찾는 과정이다.

  |딥러닝 학습 절차|Pytorch 학습 절차|
  |---|---|
  |모델과 파라미터 정의|파라미터 정의 및 모델 생성|
  |전방향 학습(입력 -> 출력)|mode(input)|
  ||기울기 초기화 : optimizer.zero_grad()|
  |오차 계산|loss = loss_func(output, target)|
  |역전파(기울기 계산)|loss.backward()|
  |기울기 업데이트|optimizer.step()|

