# venv
python 에 내장된 가상환경 관리 도구  

## 1. 가상환경 관리
```bash
# 1. 가상환경 생성
python -m venv myenv

# 2. 활성화
source myenv/bin/activate

# 3. 라이브러리 설치
pip install requests

# 4. 비활성화
deactivate
```

## 2. 패키지 관리
```bash
# 개별 패키지 설치
pip install [PACKAGE]

# 설치된 패키지 확인
pip list

# 현재 환경에 설치된 패키지 기록
pip freeze > requirements.txt

# requirements.txt로 부터 패키지 설치
pip install -r requirements.txt
```

## 3. 주요 파일
- requirements.txt: 패키지 의존성

## 4. 로컬 환경 구성
### 4.1. 신규 프로젝트
```bash
python -m venv .venv
source .venv/bin/activate

pip install pandas
pip freeze > requirements.txt
```

### 4.2. 기존 프로젝트
```bash
python -m venv .venv
source .venv/bin/activate

pip install -r requirements.txt
```
