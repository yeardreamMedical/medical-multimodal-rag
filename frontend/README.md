# Medicube - AI 의료 문제 생성 웹앱

AI 기반 동적 의료 문제 생성 시스템의 웹 애플리케이션 버전입니다.

## 🎯 주요 기능

### 1. 메인 페이지 - 문제 주제 입력
- 직관적인 UI로 의료 주제 입력
- 추천 키워드 제공 (폐렴, 기흉, 심근경색 등)
- 실시간 문제 생성 요청

### 2. 문제 풀이 페이지 - 동적 생성 및 상호작용
- **로딩 단계 시각화**: AI 문제 생성 과정을 단계별로 표시
- **관련 이미지 표시**: LLM이 선택한 의료 이미지와 함께 문제 제시
- **5지선다 객관식**: 실제 의사국가고시 형식의 문제
- **단계적 학습**: 문제 풀이 → 정답 확인 순서

### 3. 결과 확인 페이지 - 정답 및 상세 해설
- **정답 표시**: 선택한 답과 정답 비교
- **상세 해설**: AI가 생성한 근거 있는 해설
- **AI 분석 정보**: 주제, 난이도, 임상적 중요도 등 메타 정보

## 🛠 기술 스택

### Frontend
- **Vanilla JavaScript**: 가벼운 SPA 구현
- **Modern CSS**: CSS Grid, Flexbox, Custom Properties
- **Responsive Design**: 모바일 친화적 디자인

### Backend (Mock)
- **Express.js**: RESTful API 서버
- **CORS**: 크로스 오리진 요청 처리
- **Static File Serving**: 프론트엔드 파일 제공

### Development
- **Vite**: 빠른 개발 서버 및 빌드 도구
- **ES Modules**: 모던 JavaScript 모듈 시스템

## 🚀 실행 방법

### 개발 환경 실행
```bash
# 의존성 설치
npm install

# 개발 서버 시작 (프론트엔드)
npm run dev

# 백엔드 서버 시작 (별도 터미널)
npm run server
```

### 프로덕션 빌드
```bash
# 빌드
npm run build

# 빌드 결과 미리보기
npm run preview
```

## 📁 프로젝트 구조

```
medicube/
├── src/
│   ├── js/
│   │   └── main.js          # 메인 애플리케이션 로직
│   └── styles/
│       └── main.css         # 전체 스타일시트
├── server/
│   └── index.js             # Express 백엔드 서버
├── backend/                 # 기존 Python 백엔드 (참고용)
│   ├── search_engine.py
│   ├── dynamic_question_generator.py
│   ├── main.py
│   └── student_main.py
├── index.html               # 메인 HTML 파일
├── package.json
├── vite.config.js
└── README.md
```

## 🎨 디자인 시스템

### 색상 팔레트
- **Primary**: Blue 계열 (의료/신뢰성)
- **Secondary**: Gray 계열 (텍스트/배경)
- **Success**: Green (정답/성공)
- **Warning**: Orange (주의)
- **Error**: Red (오류/오답)

### 타이포그래피
- **Font Family**: Inter (가독성 최적화)
- **Font Weights**: 300, 400, 500, 600, 700
- **Line Height**: 본문 150%, 제목 120%

### 간격 시스템
- **8px 기반**: 일관된 간격 체계
- **Responsive**: 모바일/태블릿/데스크톱 대응

## 🔄 사용자 플로우

1. **메인 페이지 접속**
   - 서비스 소개 및 기능 안내
   - 의료 주제 입력 또는 추천 키워드 선택

2. **문제 생성 요청**
   - 로딩 애니메이션과 진행 단계 표시
   - AI가 벡터DB 검색 → 문제 생성 → 이미지 매칭

3. **문제 풀이**
   - 관련 이미지와 함께 문제 표시
   - 5지선다 중 하나 선택
   - "정답 및 해설 보기" 버튼 활성화

4. **결과 확인**
   - 정답/오답 표시
   - 상세 해설 및 AI 분석 정보 제공
   - "다른 문제 풀기" 버튼으로 재시작

## 🔮 향후 개발 계획

### Phase 1: 백엔드 연동
- Python 백엔드와 실제 API 연동
- 벡터DB 검색 및 LLM 문제 생성 연결

### Phase 2: 기능 확장
- 사용자 계정 및 학습 기록
- 문제 난이도 조절
- 주제별 문제 모음

### Phase 3: 고도화
- 실시간 피드백 시스템
- 학습 분석 대시보드
- 모바일 앱 버전

## 📝 API 명세

### POST /api/generate-question
문제 생성 요청

**Request:**
```json
{
  "topic": "폐렴"
}
```

**Response:**
```json
{
  "question": "문제 본문...",
  "options": ["보기1", "보기2", "보기3", "보기4", "보기5"],
  "answer": 1,
  "explanation": "해설 내용...",
  "topic_analysis": {
    "estimated_topic": "세균성 폐렴",
    "difficulty_level": "중급",
    "clinical_relevance": "high"
  },
  "image_selection": {
    "selected_image_type": "Pneumonia",
    "korean_name": "폐렴",
    "reason": "선택 이유..."
  },
  "selected_images": [...]
}
```

## 🤝 기여 방법

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📄 라이선스

이 프로젝트는 MIT 라이선스 하에 배포됩니다.