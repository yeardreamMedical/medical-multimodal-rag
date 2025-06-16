# 🏥 Medical Multimodal RAG System
한국 의사 국가고시를 위한 이미지 RAG를 포함한 멀티모달 질의응답 및 문제 생성 시스템 개발

## 🎯 프로젝트 목표
- **KorMedMCQA 성능**: 65% 이상 정답 정확도
- **문제 생성 품질**: 80% 이상 품질 검증
- **실용성**: 의료진 교육 및 국가고시 준비 도구

## 프로젝트 가설
1. 양질의 의학 텍스트 데이터를 RAG(Retrieval Augmented Generation) 방식으로 LLM에 제공하면, 기존 LLM 단독 모델보다 KorMedMCQA와 같은 표준화된 의학 문제를 더 정확하게 해결할 수 있을 것이다.
2. 잘 정제된 의학 지식 기반으로 RAG를 수행하면, 환각(Hallucination)을 최소화하고 신뢰도 높은 의학 문제와 해설을 생성할 수 있을 것이다.
3. 의미적으로 잘 정돈된 양질의 의학 이미지 데이터베이스를 구축하면, 생성형 AI가 의학 문제를 만들 때 문맥에 가장 적절한 이미지를 효과적으로 검색하고 첨부할 수 있을 것이다.

## 프로젝트의 중요성 및 차별점
1. **국내 최초 이미지 포함 의학 문제 RAG**: 기존 한국형 의학 문제 데이터셋(예: 국시)에 이미지가 통합되지 않은 한계를 극복하고, 흉부 X-ray 이미지를 포함한 멀티모달 RAG 파이프라인을 구축하여 실제 임상 환경과 유사한 문제 해결 능력 평가 및 학습 지원.
2. **KorMedMCQA 기반 RAG 성능 첫 검증 시도**: 아직 RAG 방법론을 적용하여 KorMedMCQA 테스트를 수행하고 그 성능을 공개적으로 검증한 사례가 부족한 상황에서, 본 프로젝트는 그 가능성을 탐색하고 구체적인 성능 지표를 제시.
3. **의료 교육 및 임상 의사결정 지원 시스템의 혁신적 발전**에 기여.

## 📅 개발 단계 (6단계)
| Phase | 설명 | 상태 | 목표일 | 달성 성과 |
|-------|------|------|--------|-----------|
| **I** | 데이터 수집 (KorMedMCQA, NIH Chest X-rays) | ✅ 완료 | - | 880장 X-ray, 의학 텍스트 DB |
| **II** | 텍스트/이미지 전처리 및 임베딩 | ✅ 완료 | - | OpenAI 임베딩, BioViL-T 모델 |
| **III** | RAG DB 구축 (Pinecone) | ✅ 완료 | 6/15 | 벡터 DB 인덱싱 완료 |
| **IV** | 멀티모달 검색 시스템 | ✅ **완료** | 6/17 | **85% 정확도, 100% 테스트 통과** |
| **V** | 컨텍스트 생성 & 프롬프트 엔지니어링 | 🚧 **진행중** | 6/19 | - |
| **VI** | LLM 통합 & 문제 생성 (MVP) | ⏳ 대기 | 6/20 | - |

## 🚀 빠른 시작

### 1. 설치
```bash
git clone https://github.com/yeardreamMedical/medical-multimodal-rag
cd medical-multimodal-rag
pip install -r requirements.txt
cp .env.example .env
# .env 파일에 API 키 설정 (PINECONE_API_KEY, OPENAI_API_KEY)
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

### 3. 멀티모달 검색 시스템 사용 (Phase IV ✅)
```bash
# 검색 엔진 테스트
python search/search_engine.py test "폐렴 진단"

# 전체 정확도 평가
python search/search_engine.py eval

# 시스템 정보 확인
python search/search_engine.py info
```

## 📊 데이터소스
- **KorMedMCQA**: 한국 의사 국가고시 문제
- **의학지식 데이터**: 필수의료, AI Hub 데이터
- **NIH Chest X-rays**: 880장 (8개 주요 질병, ROI 포함)
  - Pneumonia (23), Effusion (51), Pneumothorax (12), Atelectasis (31)
  - Infiltrate (44), Mass (22), Cardiomegaly (11), Nodule (3)

## 🎉 Phase IV 완성 

### ✅ **핵심 기능 구현 완료**
- **멀티모달 통합 검색**: 텍스트-이미지 동시 검색
- **한국어-영어 바이링궐**: 실시간 용어 매핑
- **의료 특화 NLP**: 제외 패턴, 전문 용어 가중치
- **고도화된 랭킹**: 6가지 융합 알고리즘 (RRF, Bayesian, MMR 등)

### 📈 **성능 지표**
- 🎯 **질병 매칭 정확도**: **85%** (기존 25% → 340% 향상)
- ⏱️ **평균 응답 시간**: **2-3초** (실용적 수준)
- 🧪 **테스트 통과율**: **100%** (22/22 모든 테스트)
- 🔍 **지원 질병**: **8개** 주요 흉부 질환
- 🌐 **언어 지원**: 한국어 + 영어 양방향

### 🛠️ **기술 스택**
- **벡터 DB**: Pinecone (텍스트 + 이미지 임베딩)
- **텍스트 임베딩**: OpenAI text-embedding-3-small
- **이미지 임베딩**: Microsoft BioViL-T (의료 특화)
- **검색 융합**: Hybrid Ensemble (6가지 알고리즘)

## 📚 사용 예시

### 기본 검색
```python
from search import SearchEngine

# 검색 엔진 초기화
engine = SearchEngine()

# 텍스트 검색 (한국어/영어 지원)
result = engine.search_text("폐렴 진단")
print(f"진단: {result['korean_diagnosis']}")  # "폐렴 (Pneumonia)"
print(f"신뢰도: {result['confidence']}")      # "high"
print(f"관련 이미지: {len(result.get('images', []))}개")

# 이미지 검색
result = engine.search_image("chest_xray.jpg")
print(f"예측 진단: {result['diagnosis']}")

# 시스템 정보
info = engine.get_system_info()
print(f"텍스트 벡터: {info['text_index']['total_vectors']:,}개")
print(f"이미지 벡터: {info['image_index']['total_vectors']:,}개")
```

### 고급 사용법
```python
from search import SearchTester, create_result_ranker

# 정확도 테스트
tester = SearchTester(engine)
accuracy = tester.test_accuracy()  # 85%+ 예상
print(f"질병 매칭 정확도: {accuracy}%")

# 성능 테스트
performance = tester.test_performance()
print(f"평균 응답 시간: {performance['average_time']:.2f}초")

# 랭킹 방법 변경
engine.ranker = create_result_ranker("bayesian_fusion")
result = engine.search_text("기흉 치료")
```

## 🧪 평가 및 테스트

### 자동화된 테스트
```bash
# Phase IV 전체 테스트 실행
cd medical-multimodal-rag
python -m pytest search/tests/test_search.py -v

# 성능 벤치마크
python search/search_engine.py eval

# 검색 정확도 상세 분석
python evaluation/search_accuracy_analysis.py
```

### 수동 테스트
```bash
# 인터랙티브 모드
python search/search_engine.py

# 특정 쿼리 테스트
python search/search_engine.py test "심장비대 소견"
```

## 📁 프로젝트 구조

```
medical-multimodal-rag/
├── data/                          # 원본 및 전처리된 데이터
├── preprocessing/                 # 텍스트/이미지 전처리
├── database/                      # RAG DB 관리 (Pinecone)
├── search/                        # ✅ 멀티모달 검색 (Phase IV 완료)
│   ├── search_engine.py          # 🎯 통합 검색 엔진
│   ├── result_ranker.py           # 🔄 고급 랭킹 시스템
│   ├── hybrid_searcher.py         # 🔍 다중 검색 전략
│   ├── query_processor.py         # 📝 쿼리 확장
│   ├── __init__.py               # 🔧 모듈 인터페이스
│   └── tests/
│       └── test_search.py        # ✅ 100% 테스트 통과
├── context/                       # 🚧 컨텍스트 생성 (Phase V 진행중)
├── generation/                    # ⏳ LLM 문제 생성 (Phase VI 대기)
├── evaluation/                    # 성능 평가
└── api/                          # REST API (배포용)
```

## 🎯 Phase V 진행 계획

### 현재 진행중: 컨텍스트 생성 & 프롬프트 엔지니어링
- **목표**: 검색된 멀티모달 결과를 LLM이 활용할 수 있는 고품질 컨텍스트로 변환
- **구현 예정**:
  - `context/context_builder.py`: 멀티모달 컨텍스트 융합
  - `context/prompt_engineer.py`: 의료 문제 생성 프롬프트 최적화
  - `context/medical_knowledge_integration.py`: 의학 지식 통합

### Phase VI 준비: LLM 통합
- **GPT-4o 연동**: 검색 컨텍스트 → 고품질 의료 문제 생성
- **KorMedMCQA 평가**: 65% 정확도 목표
- **MVP 배포**: 실용적인 의료 교육 도구

## 🔍 성능 모니터링

### 실시간 대시보드
```python
# 시스템 상태 확인
from search import SearchEngine
engine = SearchEngine()
status = engine.get_system_info()

print(f"시스템 상태: {status['system_status']}")
print(f"검색 정확도: 85%+")
print(f"응답 속도: 2-3초")
print(f"가용 질병: {status['total_diseases']}개")
```

### 지속적 개선
- 🔄 **실시간 피드백**: 사용자 검색 결과 품질 추적
- 📊 **성능 메트릭**: 정확도, 응답시간, 사용자 만족도
- 🎯 **모델 업데이트**: 새로운 의학 지식 및 이미지 데이터 추가

## 🤝 기여


### 개발 가이드라인
- **Phase IV**: 검색 시스템 최적화 및 새로운 랭킹 알고리즘
- **Phase V**: 컨텍스트 생성 및 프롬프트 엔지니어링
- **Phase VI**: LLM 통합 및 문제 생성

---

### 🏆 최근 업데이트 (Phase IV 완료)

**2024.06.17**: ✅ **Phase IV 100% 완료**
- 멀티모달 검색 시스템 완성
- 85% 질병 매칭 정확도 달성
- 22개 테스트 모두 통과
- Phase V 컨텍스트 생성 단계 시작

**다음 마일스톤**: Phase V 완료 목표 (2024.06.19)