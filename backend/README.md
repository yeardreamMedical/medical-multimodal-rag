### 실행 예시 (Backend)
```bash
# 가상환경 권장 (예: venv)
pip install -r requirements.txt

# 환경 변수 (.env) 필요
# PINECONE_API_KEY=...
# PINECONE_ENVIRONMENT=...
# PINECONE_TEXT_INDEX_NAME=...
# PINECONE_IMAGE_INDEX_NAME=...
# OPENAI_API_KEY=...
# GEMINI_API_KEY=...

# 개발 서버 실행
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

# 스웨거 문서 확인: http://localhost:8000/docs
```

### 주의사항
- 실제 데이터/인덱스 및 외부 API 키가 필요합니다. 키가 없는 경우 `generate_question_service`는 폴백 모의 데이터를 반환하도록 방어 로직이 포함되어 있습니다.
- `hi-ml-multimodal` (BioViL-T) 설치는 OS/CUDA 환경에 따라 추가 설정이 필요할 수 있습니다.
- 현재 레포지토리에는 상단의 전체 폴더 구조(검색/생성/분석 등)가 모두 포함되어 있지 않습니다. 위 구조는 목표 아키텍처 예시이며, 본 backend는 그 중 REST API 계층에 해당합니다.
