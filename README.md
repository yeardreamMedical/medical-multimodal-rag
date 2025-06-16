# 🏥 Medical Multimodal RAG System

한국 의사 국가고시를 위한 이미지 RAG를 포함한 멀티모달 질의응답 및 문제 생성 시스템 개발

## 🎯 프로젝트 목표

- **KorMedMCQA 성능**: 65% 이상 정답 정확도
- **문제 생성 품질**: 80% 이상 품질 검증
- **실용성**: 의료진 교육 및 국가고시 준비 도구

## 프로젝트 가설
1.  양질의 의학 텍스트 데이터를 RAG(Retrieval Augmented Generation) 방식으로 LLM에 제공하면, 기존 LLM 단독 모델보다 KorMedMCQA와 같은 표준화된 의학 문제를 더 정확하게 해결할 수 있을 것이다.
2. 잘 정제된 의학 지식 기반으로 RAG를 수행하면, 환각(Hallucination)을 최소화하고 신뢰도 높은 의학 문제와 해설을 생성할 수 있을 것이다.
3. 의미적으로 잘 정돈된 양질의 의학 이미지 데이터베이스를 구축하면, 생성형 AI가 의학 문제를 만들 때 문맥에 가장 적절한 이미지를 효과적으로 검색하고 첨부할 수 있을 것이다.

## 프로젝트의 중요성 및 차별점
1. 국내 최초 이미지 포함 의학 문제 RAG: 기존 한국형 의학 문제 데이터셋(예: 국시)에 이미지가 통합되지 않은 한계를 극복하고, 흉부 X-ray 이미지를 포함한 멀티모달 RAG 파이프라인을 구축하여 실제 임상 환경과 유사한 문제 해결 능력 평가 및 학습 지원.
2. KorMedMCQA 기반 RAG 성능 첫 검증 시도: 아직 RAG 방법론을 적용하여 KorMedMCQA 테스트를 수행하고 그 성능을 공개적으로 검증한 사례가 부족한 상황에서, 본 프로젝트는 그 가능성을 탐색하고 구체적인 성능 지표를 제시.
3. 의료 교육 및 임상 의사결정 지원 시스템의 혁신적 발전에 기여.

## 📅 개발 단계 (6단계)

| Phase | 설명 | 상태 | 목표일 |
|-------|------|------|--------|
| **I** | 데이터 수집 (KorMedMCQA, NIH Chest X-rays) | ✅ 완료 | - |
| **II** | 텍스트/이미지 전처리 및 임베딩 | ✅ 완료 | - |
| **III** | RAG DB 구축 (Pinecone) | ✅ 완료 | 6/15 |
| **IV** | 멀티모달 검색 시스템 | 🚧 **진행중** | 6/17 |
| **V** | 컨텍스트 생성 & 프롬프트 엔지니어링 | ⏳ 대기 | 6/19 |
| **VI** | LLM 통합 & 문제 생성 (MVP) | ⏳ 대기 | 6/20 |

## 🚀 빠른 시작

### 1. 설치

```bash
git clone https://github.com/yourusername/medical-multimodal-rag.git
cd medical-multimodal-rag

pip install -r requirements.txt
cp .env.example .env
# .env 파일에 API 키 설정
```

### 2. 데이터 준비

```bash
# 데이터 다운로드
python scripts/download_data.py

# 전처리 실행
python scripts/run_preprocessing.py

# DB 구축
python scripts/build_database.py
```

### 3. 검색 시스템 테스트 (현재 Phase IV)

```bash
# 검색 테스트
python search/search_engine.py

# 정확도 평가
python search/tests/accuracy_test.py
```

## 📊 데이터소스

- **KorMedMCQA**: 한국 의사 국가고시 문제
- **의학지식 데이터**: 필수의료, AI Hub 데이터
- **NIH Chest X-rays**: 880장 (ROI 포함)

## 🔍 현재 상황 (Phase IV)

### 완성된 기능
- ✅ 텍스트-이미지 하이브리드 검색
- ✅ 쿼리 확장 및 질병명 매칭 (85% 정확도)
- ✅ BioViL-T 기반 X-ray 이미지 검색

### 진행중
- 🚧 검색 결과 랭킹 최적화
- 🚧 텍스트-이미지 연관 매핑 강화

## 📚 사용 예시

```python
from search.search_engine import SearchEngine

# 검색 엔진 초기화
searcher = SearchEngine()

# 텍스트 검색
result = searcher.search_text("폐렴 진단")
print(f"관련 이미지: {len(result['images'])}개")

# 이미지 검색
result = searcher.search_image("chest_xray.jpg")
print(f"예측 진단: {result['diagnosis']}")
```

## 🧪 평가

```bash
# KorMedMCQA 평가
python evaluation/kormedmcqa_eval.py

# 검색 성능 평가
python evaluation/search_performance_eval.py
```

## 📁 주요 디렉토리

- `data/`: 원본 및 전처리된 데이터
- `preprocessing/`: 텍스트/이미지 전처리
- `database/`: RAG DB 관리
- `search/`: 멀티모달 검색 (현재 작업)
- `generation/`: LLM 문제 생성 (구현 예정)
- `evaluation/`: 성능 평가

## 🎯 다음 단계

1. **Phase IV 완성**: 검색 랭킹 최적화
2. **Phase V 시작**: 프롬프트 엔지니어링
3. **Phase VI MVP**: GPT-4o 연동 및 문제 생성

## 🤝 기여

프로젝트에 기여하시려면:

1. 이슈 생성 또는 기존 이슈 확인
2. 브랜치 생성: `git checkout -b feature/your-feature`
3. 커밋: `git commit -m 'Add feature'`
4. 푸시: `git push origin feature/your-feature`
5. Pull Request 생성

## 📄 라이센스

MIT License - 의료 교육 목적 자유 사용