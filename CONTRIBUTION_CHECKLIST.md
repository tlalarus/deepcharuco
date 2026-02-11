# DeepCharuco Contribution Checklist

> 사용법: 완료한 항목은 `- [x]`로 바꾸고, 날짜/메모를 함께 남기세요.

## Goal Overview
- [x] 프로젝트 바로 구동 가능한 컨테이너 빌드용 `Dockerfile` 완성
- [x] OpenCV 구버전/신버전 호환 전략 수립 및 스모크 검증
- [ ] 완료 기준(Definition of Done) 합의

## 1) Environment Setup (Docker Only)
- [ ] Dev Container 열기 (`Reopen in Container`)
- [x] COCO 데이터 마운트 확인 (`/data/dataset/coco`)
- [x] `src/config.yaml` 경로가 COCO 경로로 설정되었는지 확인
- [x] 기본 스모크 테스트 실행: `cd src && python inference.py`

## 2) Codebase Understanding
- [ ] 모델 구조 파악 (`src/models/net.py`, `src/models/refinenet.py`)
- [ ] 데이터 파이프라인 파악 (`src/data.py`, `src/data_refinenet.py`)
- [ ] 설정/실행 경로 파악 (`src/configs.py`, `src/inference.py`)

## 3) First Contribution
- [ ] 작업 이슈 선정 (bugfix/docs/refactor 중 1개)
- [ ] 브랜치 생성 (`feat/...` or `fix/...`)
- [ ] 변경 구현 및 Docker 내 재현 테스트
- [ ] PR 생성 (변경 내용, 검증 방법, 결과 포함)

## 4) Validation & Quality
- [ ] 성능 영향 있으면 `cd src && python benchmark.py` 실행
- [ ] 결과/메트릭 또는 스크린샷 정리
- [ ] 불필요한 산출물(로그/데이터/대용량 파일) 제외 확인

## Progress Log
- 2026-02-11: [ ] 착수 / [ ] 진행 / [x] 완료 — Docker baseline 빌드 + inference/train 스모크 확인(OpenCV 4.6.0 / 4.11.0)
- YYYY-MM-DD: [ ] 착수 / [ ] 진행 / [ ] 완료 — 메모:

## Notes
- 현재 저장소 정책: 디버깅/테스트는 로컬 호스트가 아니라 Docker 컨테이너에서만 수행.
- “아래 항목” 상세 목표를 주면 이 체크리스트를 바로 맞춤형으로 업데이트 가능.
