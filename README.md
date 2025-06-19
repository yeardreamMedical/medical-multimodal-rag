# 🏥 Medical Multimodal RAG System
한국 의사 국가고시를 위한 이미지 RAG를 포함한 멀티모달 질의응답 및 문제 생성 시스템 개발

## 🎯 프로젝트 목표
- **KorMedMCQA 성능**: 80% 이상 정답 정확도
- **문제 생성 품질**: KorMedMCQA dataset과 80% 이상 유사도 품질 검증
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

## 🚀 주요 특징

### 🖼️ **실제 X-ray 이미지 포함 질환** (ChestX-ray14 데이터셋)
- **Pneumonia (폐렴)** - 23개 영상 보유
- **Effusion (흉수)** - 51개 영상 보유  
- **Mass (종괴)** - 22개 영상 보유
- **Nodule (결절)** - 3개 영상 보유
- **Pneumothorax (기흉)** - 12개 영상 보유
- **Atelectasis (무기폐)** - 31개 영상 보유
- **Infiltrate (침윤/경화)** - 44개 영상 보유
- **Cardiomegaly (심장비대)** - 11개 영상 보유

### 📝 **텍스트 전용 질환** (모든 의료 분야)
- **내과**: 당뇨병, 고혈압, 신장질환, 간질환, 내분비 등
- **외과**: 열상, 골절, 충수염, 탈장, 화상 등  
- **소아과**: 소아 재활, 성장 발달, 예방접종 등
- **기타**: 응급의학, 정형외과, 피부과, 안과 등

## 🔄 워크플로우

```
사용자 쿼리 입력
    ↓
LLM 기반 쿼리 확장 (동적 의료 키워드 추가)
    ↓
벡터DB에서 관련 의학 지식 검색 (AIHUB 데이터)
    ↓
질병 매칭 점수 계산 및 텍스트 전용 모드 판단
    ↓
LLM이 의사국가고시 스타일 문제 생성
    ↓
LLM이 문제 내용 분석 후 적절한 이미지 타입 선택
    ↓
선택된 타입으로 실제 X-ray 이미지 검색 (필요시)
    ↓
문제 + 이미지 통합 출력 (터미널에서 시각화)
```

## 🛠️ 기술 스택
- **🤖 AI 모델**: Gemini 1.5 Pro (문제 생성 + 이미지 선택 + 쿼리 확장)
- **🔍 벡터DB**: Pinecone (의학 지식 검색)
- **🖼️ 이미지 임베딩**: BioViL-T (의료 이미지 검색)
- **🎨 UI**: Rich Console + matplotlib (터미널 기반 시각화)

## 📊 데이터소스
- **텍스트 데이터**: AIHUB 전문 의학지식 데이터 + 필수의료 의학지식 데이터, KorMedMCQA (한국 의사 국가고시 문제)
- **이미지 데이터**: NIH Chest X-rays (Kaggle) - bbox 포함 880장 선별
  - 원본: https://www.kaggle.com/datasets/nih-chest-xrays/data/data
  - 사용: ROI(Region of Interest) bbox가 있는 고품질 이미지만 선별


## 🚀 빠른 시작

### 1. 설치
```bash
git clone https://github.com/your-repo/medical-multimodal-rag
cd medical-multimodal-rag
pip install -r requirements.txt
```

### 2. 환경 설정
```bash
cp .env.example .env
# .env 파일에 API 키 설정:
# PINECONE_API_KEY=your_pinecone_key
# OPENAI_API_KEY=your_openai_key  
# GEMINI_API_KEY=your_gemini_key
```

### 3. 실행

#### 🖼️ X-ray 이미지와 함께 문제 생성 (8개 흉부 질병)
```bash
python generation/main.py "폐렴"              # 폐렴 X-ray + 문제 출력
python generation/main.py "기흉"              # 기흉 X-ray + 문제 출력
python generation/main.py "폐암"              # 종괴 X-ray + 문제 출력 (LLM이 Mass 선택)
```

#### 📝 텍스트 전용 문제 생성 (비흉부 질병)
```bash
python generation/main.py "열상 환자"         # 외과 질환 (이미지 없음)
python generation/main.py "당뇨병 관리"       # 내과 질환 (이미지 없음)
python generation/main.py "소아 재활"         # 재활의학 (이미지 없음)
```

#### 🔧 시스템 확인
```bash
python generation/main.py check-path          # 프로젝트 경로 확인
python generation/main.py --help              # 도움말
```




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

## 📁 프로젝트 구조

```
medical-multimodal-rag/
├── data/                          # 원본 및 전처리된 데이터
├── preprocessing/                 # 텍스트/이미지 전처리
├── database/                      # RAG DB 관리 (Pinecone)
├── search/                        # ✅ 멀티모달 검색 (Phase IV 완료)
│   ├── search_engine.py                     # 벡터DB 검색 + LLM 쿼리 확장
│   ├── __init__.py               
│   └── tests/
│       └── test_search.py        # ✅ 100% 테스트 통과
├── 📁 generation/                            # 🎯 핵심 문제 생성 시스템
│   ├── main.py                              # 🚀 메인 실행 파일
│   ├── dynamic_question_generator.py        # 🤖 동적 생성 엔진
│   └── __init__.py                          # 📋 패키지 정보
├── 📁 analysis/                              # 📊 데이터 분석 도구
├── evaluation/                    # 성능 평가
└── README.md
└── api/                          # REST API (배포용)
```

## 💻 사용 예시

### 기본 사용법 (Python)
```python
from generation import DynamicQuestionGenerator

# 생성기 초기화
generator = DynamicQuestionGenerator()

# 흉부 질병 - 이미지 포함 문제 생성
result = generator.generate_question_from_query("폐렴")
print(f"생성된 문제: {result['generated_question']['question']}")
print(f"선택된 이미지: {result['image_selection']['korean_name']}")

# 비흉부 질병 - 텍스트 전용 문제 생성
result = generator.generate_question_from_query("당뇨병")
print(f"이미지 선택: {result['image_selection']['selected_image_type']}")  # "None"
```

### 다중 문제 생성
```python
# 여러 주제로 배치 생성
queries = ["폐렴", "기흉", "당뇨병", "열상"]
results = generator.generate_multiple_dynamic(queries, questions_per_query=2)

# 품질 평가
evaluation = generator.evaluate_dynamic_quality(results)
print(f"고신뢰도 비율: {evaluation['summary']['high_confidence_rate']}%")
print(f"이미지 지원: {evaluation['summary']['image_supported_rate']}%")
```

## 🎯 성능 특징

### 📈 **품질 지표**
- **질병 매칭 정확도**: 흉부 질환 95%+ (8개 라벨)
- **실제 X-ray 표시**: 흉부 질병 문제시 실제 의료 영상 자동 표시
- **평균 응답 시간**: 5-10초 (검색 + 생성 + 이미지 선택 + 표시)
- **텍스트 전용 모드**: 비흉부 주제 자동 감지 및 처리

### 🔄 **동적 적응성**
- **확장성**: 새로운 의료 분야 벡터DB 추가시 자동 지원
- **유연성**: 사용자 쿼리에 따라 이미지 포함/제외 자동 결정
- **정확성**: LLM이 문제 내용과 이미지의 의학적 관련성 검증

## 🧪 시스템 특징

### 💡 **스마트 이미지 선택**
```
"폐암" 쿼리 → LLM이 문제 생성 → 문제 분석 → Mass(종괴) 이미지 선택 ✅
"열상" 쿼리 → LLM이 문제 생성 → 문제 분석 → None(텍스트 전용) 선택 ✅
```

### ⚡ **효율적 처리**
- **조기 감지**: 질병 매칭 실패시 즉시 텍스트 전용 모드
- **리소스 절약**: 불필요한 이미지 검색 생략
- **품질 보장**: 모든 단계에서 LLM 기반 검증

## 🎓 교육적 가치

### 👨‍⚕️ **의료진 교육**
- 실제 X-ray 영상과 함께 학습
- 다양한 의료 분야 문제 연습
- 표준화된 국가고시 형식

### 🏥 **임상 적용**
- 실제 의료 환경과 유사한 문제
- 최신 의학 지식 반영
- 객관적 평가 가능
