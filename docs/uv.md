# uv
Rust 기반으로 만들어진 python 가상환경 및 패키지 관리 도구  

## 1. 파이썬 버전 관리
```bash
uv python install <VERSION>
uv python list
uv use <VERSION>
```

## 2. 프로젝트 생성
```bash
# 현제 위치를 루트로 프로젝트 초기화
uv init

# 패키지 형태의 프로젝트 초기화
uv init --package
```
> init 명령어는 가상환경을 자동으로 생성하지 않는다.  
> 프로젝트 초기화 이후 `uv add`, `uv run`, `uv sync` 등 패키지 관리 명령어 실행 시 가상환경 생성

## 3. 가상환경 관리
```bash
# PATH에 가상환경 생성
uv venv [PATH]

# 특정 python VERSION으로 PATH에 가상환경 생성
uv venv -p <VERSION> [PATH]
```

## 4. 패키지 관리
```bash
# 패키지 추가 (가상환경 자동 생성(없을시) + pyproject.toml 업데이트)
uv add [PACKAGE]

# 개발 의존성 추가
uv add --dev [PACKAGE]

# 설치된 패키지 확인
uv tree # 의존성 트리
uv pip list # default 가상환경
uv pip list --python .venv/bin/python # 가상환경 지정

# 의존성 잠금파일 업데이트
# 의존성 업데이트, 충돌확인, CI/CD에서 lock파일 검증할 때 사용
# - pyproject.toml의 dependencies로 부터 의존성 읽기
# - uv.lock에 잠금된 버전 기록
# - 가상환경 자동 생성하지 않음
# - 패키지 설치 하지 않음
uv lock

# 의존성 동기화
# - pyproject.toml의 dependencies로 부터 의존성 읽기
# - uv.lock이 outdated인 경우, 생성/업데이트
# - uv.lock에 명시된 정확한 버전으로 패키지 사용
# - 가상환경 자동 생성(없는 경우)
# - 불필요한 패키지 제거 (clean install)
uv sync
```

## 5. 주요 파일
- .python-version: 가상환경에서 사용할 python 버전
- pyproject.toml: 프로젝트 메타데이터 + 패키지 의존성
- uv.lock: 패키지 버전 Lock

## 6. 로컬 환경 구성
### 6.1. 신규 프로젝트
```bash
uv init

uv add pandas # 자동으로 .venv 생성

uv run python main.py
```

### 6.2. 기존 프로젝트
```bash
uv sync

uv run python main.py
```
