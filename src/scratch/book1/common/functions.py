"""
Common functions for deep learning
활성화 함수, 오차 함수, 경사하강 등의 공통 함수들을 모아둔 모듈
"""

import numpy as np


# ============================================================================
# 활성화 함수 (Activation Functions)
# ============================================================================

def step_function(x):
    """
    계단 함수 (Step Function)
    
    Args:
        x: 입력 배열
        
    Returns:
        x > 0인 경우 1, 그 외 0
    """
    return np.array(x > 0, dtype=np.int8)


def sigmoid(x):
    """
    시그모이드 함수 (Sigmoid Function)
    
    Args:
        x: 입력 배열
        
    Returns:
        sigmoid(x) = 1 / (1 + exp(-x))
    """
    return 1 / (1 + np.exp(-np.clip(x, -500, 500)))  # 오버플로우 방지


def relu(x):
    """
    ReLU 함수 (Rectified Linear Unit)
    
    Args:
        x: 입력 배열
        
    Returns:
        max(0, x)
    """
    return np.maximum(0, x)


def identity_function(x):
    """
    항등 함수 (Identity Function)
    
    Args:
        x: 입력 배열
        
    Returns:
        입력값 그대로 반환
    """
    return x


def softmax(x):
    """
    소프트맥스 함수 (Softmax Function)
    출력을 확률로 해석할 수 있도록 만들어줌
    출력층의 뉴런 수 = 클래스 수
    출력층의 output 총합 = 1
    
    Args:
        x: 입력 배열 (1차원 또는 2차원)
        
    Returns:
        softmax(x) - 각 원소의 합이 1이 되는 확률 분포
    """
    if x.ndim == 2:
        # 배치 처리
        c = np.max(x, axis=1, keepdims=True)
        exp_x = np.exp(x - c)  # 오버플로우 방지를 위해 최댓값을 빼줌
        sum_exp_x = np.sum(exp_x, axis=1, keepdims=True)
        return exp_x / sum_exp_x
    else:
        # 단일 샘플
        c = np.max(x)
        exp_x = np.exp(x - c)  # 오버플로우 방지를 위해 최댓값을 빼줌
        sum_exp_x = np.sum(exp_x)
        return exp_x / sum_exp_x


# ============================================================================
# 오차 함수 (Loss Functions)
# ============================================================================

def mean_squared_error(y, t):
    """
    평균 제곱 오차 (Mean Squared Error, MSE)
    
    Args:
        y: 예측값
        t: 정답값
        
    Returns:
        MSE = 0.5 * sum((y - t)^2)
    """
    return 0.5 * np.sum((y - t) ** 2)


def cross_entropy_error(y, t):
    """
    교차 엔트로피 오차 (Cross Entropy Error)
    레이블 형식(정수)을 받아서 처리
    
    Args:
        y: 예측값 (softmax 출력, 확률 분포)
        t: 정답 레이블 (정수 배열 또는 원핫 인코딩)
        
    Returns:
        Cross Entropy Error
    """
    if y.ndim == 1:
        t = t.reshape(1, t.size)
        y = y.reshape(1, y.size)
    
    batch_size = y.shape[0]
    
    # 원핫 인코딩인지 레이블인지 자동 판별
    if t.ndim == 1:
        # 레이블 형식 (정수)
        return -np.sum(np.log(y[np.arange(batch_size), t] + 1e-7)) / batch_size
    else:
        # 원핫 인코딩 형식
        return -np.sum(t * np.log(y + 1e-7)) / batch_size


def cross_entropy_error_onehot(y, t):
    """
    교차 엔트로피 오차 (원핫 인코딩 버전)
    
    Args:
        y: 예측값 (softmax 출력, 확률 분포)
        t: 정답값 (원핫 인코딩)
        
    Returns:
        Cross Entropy Error
    """
    if y.ndim == 1:
        t = t.reshape(1, t.size)
        y = y.reshape(1, y.size)
    
    batch_size = y.shape[0]
    return -np.sum(t * np.log(y + 1e-7)) / batch_size


def cross_entropy_error_label(y, t):
    """
    교차 엔트로피 오차 (레이블 버전)
    
    Args:
        y: 예측값 (softmax 출력, 확률 분포)
        t: 정답 레이블 (정수 배열)
        
    Returns:
        Cross Entropy Error
    """
    if y.ndim == 1:
        t = t.reshape(1, t.size)
        y = y.reshape(1, y.size)
    
    batch_size = y.shape[0]
    # 정답 레이블이 1차원 배열인 경우
    # np.arange(batch_size) = [0, 1, 2, 3, 4]
    # t = [2, 7, 0, 9, 4]
    # y[np.arange(batch_size), t] = [y[0, 2], y[1, 7], y[2, 0], y[3, 9], y[4, 4]]
    return -np.sum(np.log(y[np.arange(batch_size), t] + 1e-7)) / batch_size


# ============================================================================
# 수치 미분 (Numerical Differentiation)
# ============================================================================

def numerical_diff(f, x):
    """
    수치 미분 (중앙 차분 사용)
    
    Args:
        f: 미분할 함수
        x: 미분할 점
        
    Returns:
        f'(x) ≈ (f(x+h) - f(x-h)) / (2*h)
    """
    h = 1e-4
    return (f(x + h) - f(x - h)) / (2 * h)


def numerical_gradient(f, x):
    """
    수치 기울기 (Numerical Gradient)
    다변수 함수의 각 변수에 대한 편미분을 계산
    
    Args:
        f: 미분할 함수 (다변수 함수)
        x: 미분할 점 (배열)
        
    Returns:
        각 변수에 대한 편미분으로 구성된 기울기 벡터
    """
    h = 1e-4
    grad = np.zeros_like(x) # x와 같은 shape

    for idx in range(x.size):
        multi_idx = np.unravel_index(idx, x.shape)
        tmp_val = x[multi_idx]
        # f(x+h) 계산
        x[multi_idx] = tmp_val + h
        fxh1 = f(x)

        # f(x-h) 계산
        x[multi_idx] = tmp_val - h
        fxh2 = f(x)

        # 중앙 차분(수치 미분)
        grad[multi_idx] = (fxh1 - fxh2) / (2*h)
        x[multi_idx] = tmp_val # 값 복원
    
    return grad


# ============================================================================
# 경사 하강법 (Gradient Descent)
# ============================================================================

def gradient_descent(f, init_x, lr=0.01, step_num=100):
    """
    경사 하강법 (Gradient Descent)
    
    Args:
        f: 최적화하려는 함수
        init_x: 초기 위치
        lr: 학습률 (learning rate)
        step_num: 반복 횟수
        
    Returns:
        최적화된 x 값
    """
    x = init_x.copy()
    
    for i in range(step_num):
        grad = numerical_gradient(f, x)
        x -= lr * grad
    
    return x


def gradient_descent_with_history(f, init_x, lr=0.01, step_num=100):
    """
    경사 하강법 (갱신 과정 저장 버전)
    
    Args:
        f: 최적화하려는 함수
        init_x: 초기 위치
        lr: 학습률 (learning rate)
        step_num: 반복 횟수
        
    Returns:
        (최적화된 x 값, 갱신 과정 히스토리)
    """
    
    x = init_x.copy()
    history = [x.copy()]  # 초기 위치 저장
    
    for i in range(step_num):
        grad = numerical_gradient(f, x)
        x -= lr * grad
        history.append(x.copy())  # 각 단계의 x 저장
    
    return x, np.array(history)

