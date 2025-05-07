# paper2 시스템 개요 및 작동원리

## 1. 시스템 개요
- 멀티모달(이미지+뉴스) 기반 트레이딩 + 시장 레짐 동적제어 앙상블 시스템
- 딥러닝, 강화학습, 시장상황 감지, 앙상블까지 통합

## 2. 아키텍처
- 데이터 수집 → 특징 추출(이미지+뉴스) → 융합 → 시장 레짐 감지 → 앙상블 → 예측/백테스팅
- 주요 모듈: ensemble_controller, market_state_detector, dynamic_weight_adjuster 등

## 3. 사용법
1. 패키지 설치: `pip install -r requirements.txt`
2. 실행: `python run_paper2_ensemble.py`

## 4. 폴더 구조
- data/: 데이터셋
- results/: 실험 결과
- docs/: 문서

## 5. 참고
- 자세한 코드 설명은 각 파이썬 파일의 docstring 참고
- 데이터/결과는 .gitignore로 업로드 제외됨 