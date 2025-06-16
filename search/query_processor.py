# search/query_processor.py
# 쿼리 처리 및 분석 전용 모듈

import re
import spacy
from typing import List, Dict, Any, Optional, Set
from dataclasses import dataclass
from enum import Enum


class QueryType(Enum):
    """쿼리 타입 분류"""
    DISEASE_SPECIFIC = "disease_specific"      # 특정 질병명 포함
    SYMPTOM_BASED = "symptom_based"           # 증상 기반
    DIAGNOSTIC_PROCEDURE = "diagnostic_procedure"  # 진단 절차
    TREATMENT_FOCUSED = "treatment_focused"    # 치료 중심
    GENERAL_MEDICAL = "general_medical"       # 일반 의료
    UNKNOWN = "unknown"                       # 분류 불가


class QueryLanguage(Enum):
    """쿼리 언어 분류"""
    KOREAN = "korean"
    ENGLISH = "english"
    MIXED = "mixed"


@dataclass
class QueryAnalysis:
    """쿼리 분석 결과"""
    original_query: str
    query_type: QueryType
    language: QueryLanguage
    medical_entities: List[str]
    key_terms: List[str]
    intent_confidence: float
    complexity_score: float


class MedicalQueryProcessor:
    """의료 쿼리 처리 및 분석 클래스"""
    
    def __init__(self):
        """쿼리 프로세서 초기화"""
        self.medical_keywords = self._load_medical_keywords()
        self.disease_synonyms = self._load_disease_synonyms()
        self.symptom_keywords = self._load_symptom_keywords()
        self.procedure_keywords = self._load_procedure_keywords()
        
        # NLP 모델 (선택사항 - spacy가 있다면)
        self.nlp = None
        try:
            import spacy
            # 한국어 모델이 있다면 로드
            # self.nlp = spacy.load("ko_core_news_sm")
        except:
            pass
    
    def _load_medical_keywords(self) -> Dict[str, List[str]]:
        """의료 관련 키워드 로드"""
        return {
            "anatomy": [
                "lung", "heart", "chest", "thorax", "pleura", "diaphragm",
                "폐", "심장", "흉부", "흉곽", "늑막", "횡격막"
            ],
            "imaging": [
                "xray", "x-ray", "ct", "mri", "ultrasound", "radiography",
                "엑스레이", "씨티", "자기공명영상", "초음파", "방사선"
            ],
            "modifiers": [
                "acute", "chronic", "severe", "mild", "bilateral", "unilateral",
                "급성", "만성", "중증", "경증", "양측", "일측"
            ]
        }
    
    def _load_disease_synonyms(self) -> Dict[str, List[str]]:
        """질병 동의어 매핑"""
        return {
            "Pneumonia": [
                "pneumonia", "lung infection", "pulmonary infection",
                "폐렴", "폐감염", "폐염증", "세균성폐렴", "바이러스폐렴"
            ],
            "Pneumothorax": [
                "pneumothorax", "collapsed lung", "tension pneumothorax",
                "기흉", "허탈폐", "긴장성기흉", "폐허탈"
            ],
            "Effusion": [
                "pleural effusion", "chest fluid", "pleural fluid",
                "흉수", "늑막삼출", "흉막삼출", "가슴물"
            ],
            "Atelectasis": [
                "atelectasis", "lung collapse", "partial collapse",
                "무기폐", "폐허탈", "부분허탈"
            ],
            "Infiltrate": [
                "infiltrate", "consolidation", "lung opacity",
                "침윤", "경화", "폐침윤", "폐경화", "음영"
            ],
            "Mass": [
                "mass", "tumor", "lesion", "nodule",
                "종괴", "종양", "병변", "결절", "덩어리"
            ],
            "Cardiomegaly": [
                "cardiomegaly", "enlarged heart", "cardiac enlargement",
                "심장비대", "심장확대", "심비대"
            ]
        }
    
    def _load_symptom_keywords(self) -> List[str]:
        """증상 관련 키워드"""
        return [
            "fever", "cough", "dyspnea", "chest pain", "fatigue",
            "발열", "기침", "호흡곤란", "흉통", "피로",
            "shortness of breath", "breathing difficulty",
            "숨가쁨", "호흡어려움"
        ]
    
    def _load_procedure_keywords(self) -> List[str]:
        """진단/치료 절차 키워드"""
        return [
            "diagnosis", "treatment", "therapy", "surgery", "procedure",
            "진단", "치료", "수술", "시술", "검사",
            "biopsy", "thoracentesis", "intubation",
            "생검", "흉수천자", "기관삽관"
        ]
    
    def analyze_query(self, query: str) -> QueryAnalysis:
        """쿼리 종합 분석"""
        # 1. 언어 감지
        language = self._detect_language(query)
        
        # 2. 의료 개체 추출
        medical_entities = self._extract_medical_entities(query)
        
        # 3. 핵심 용어 추출
        key_terms = self._extract_key_terms(query)
        
        # 4. 쿼리 타입 분류
        query_type = self._classify_query_type(query, medical_entities, key_terms)
        
        # 5. 의도 신뢰도 계산
        intent_confidence = self._calculate_intent_confidence(query, query_type, medical_entities)
        
        # 6. 복잡도 점수 계산
        complexity_score = self._calculate_complexity_score(query, medical_entities, key_terms)
        
        return QueryAnalysis(
            original_query=query,
            query_type=query_type,
            language=language,
            medical_entities=medical_entities,
            key_terms=key_terms,
            intent_confidence=intent_confidence,
            complexity_score=complexity_score
        )
    
    def _detect_language(self, query: str) -> QueryLanguage:
        """언어 감지"""
        korean_chars = len(re.findall(r'[가-힣]', query))
        english_chars = len(re.findall(r'[a-zA-Z]', query))
        
        if korean_chars > 0 and english_chars > 0:
            return QueryLanguage.MIXED
        elif korean_chars > 0:
            return QueryLanguage.KOREAN
        elif english_chars > 0:
            return QueryLanguage.ENGLISH
        else:
            return QueryLanguage.KOREAN  # 기본값
    
    def _extract_medical_entities(self, query: str) -> List[str]:
        """의료 개체 추출"""
        entities = []
        query_lower = query.lower()
        
        # 질병명 매칭
        for disease, synonyms in self.disease_synonyms.items():
            for synonym in synonyms:
                if synonym.lower() in query_lower:
                    entities.append(disease)
                    break
        
        # 해부학적 구조 매칭
        for anatomy_term in self.medical_keywords["anatomy"]:
            if anatomy_term.lower() in query_lower:
                entities.append(f"anatomy:{anatomy_term}")
        
        # 영상 관련 용어 매칭
        for imaging_term in self.medical_keywords["imaging"]:
            if imaging_term.lower() in query_lower:
                entities.append(f"imaging:{imaging_term}")
        
        return list(set(entities))  # 중복 제거
    
    def _extract_key_terms(self, query: str) -> List[str]:
        """핵심 용어 추출"""
        # 간단한 키워드 추출 (추후 NLP 모델로 개선 가능)
        terms = []
        query_lower = query.lower()
        
        # 의료 키워드 추출
        all_medical_terms = []
        for category in self.medical_keywords.values():
            all_medical_terms.extend(category)
        
        for term in all_medical_terms:
            if term.lower() in query_lower:
                terms.append(term)
        
        # 증상 키워드 추출
        for symptom in self.symptom_keywords:
            if symptom.lower() in query_lower:
                terms.append(symptom)
        
        # 절차 키워드 추출
        for procedure in self.procedure_keywords:
            if procedure.lower() in query_lower:
                terms.append(procedure)
        
        return list(set(terms))
    
    def _classify_query_type(self, query: str, entities: List[str], key_terms: List[str]) -> QueryType:
        """쿼리 타입 분류"""
        query_lower = query.lower()
        
        # 질병명이 명시적으로 포함된 경우
        disease_entities = [e for e in entities if not e.startswith(("anatomy:", "imaging:"))]
        if disease_entities:
            return QueryType.DISEASE_SPECIFIC
        
        # 증상 기반 쿼리
        if any(symptom.lower() in query_lower for symptom in self.symptom_keywords):
            return QueryType.SYMPTOM_BASED
        
        # 진단 절차 관련
        diagnostic_terms = ["diagnosis", "진단", "검사", "확인", "소견"]
        if any(term in query_lower for term in diagnostic_terms):
            return QueryType.DIAGNOSTIC_PROCEDURE
        
        # 치료 관련
        treatment_terms = ["treatment", "치료", "수술", "시술", "요법"]
        if any(term in query_lower for term in treatment_terms):
            return QueryType.TREATMENT_FOCUSED
        
        # 의료 관련 용어가 있으면 일반 의료
        if key_terms or entities:
            return QueryType.GENERAL_MEDICAL
        
        return QueryType.UNKNOWN
    
    def _calculate_intent_confidence(self, query: str, query_type: QueryType, entities: List[str]) -> float:
        """의도 신뢰도 계산"""
        confidence = 0.5  # 기본값
        
        # 질병명이 명확한 경우 높은 신뢰도
        disease_entities = [e for e in entities if not e.startswith(("anatomy:", "imaging:"))]
        if disease_entities:
            confidence += 0.3
        
        # 쿼리 타입별 가중치
        type_weights = {
            QueryType.DISEASE_SPECIFIC: 0.2,
            QueryType.DIAGNOSTIC_PROCEDURE: 0.15,
            QueryType.SYMPTOM_BASED: 0.1,
            QueryType.TREATMENT_FOCUSED: 0.1,
            QueryType.GENERAL_MEDICAL: 0.05,
            QueryType.UNKNOWN: -0.2
        }
        
        confidence += type_weights.get(query_type, 0)
        
        # 쿼리 길이 고려 (너무 짧거나 길면 신뢰도 감소)
        query_length = len(query.split())
        if 2 <= query_length <= 8:
            confidence += 0.1
        elif query_length < 2 or query_length > 15:
            confidence -= 0.1
        
        return max(0.0, min(1.0, confidence))
    
    def _calculate_complexity_score(self, query: str, entities: List[str], key_terms: List[str]) -> float:
        """쿼리 복잡도 점수 계산"""
        score = 0.0
        
        # 기본 복잡도
        word_count = len(query.split())
        score += word_count * 0.1
        
        # 의료 개체 수
        score += len(entities) * 0.2
        
        # 핵심 용어 수
        score += len(key_terms) * 0.15
        
        # 특수 문자나 수치 포함 시 복잡도 증가
        if re.search(r'[0-9%]', query):
            score += 0.2
        
        # 복합 질병이나 다중 조건
        if len([e for e in entities if not e.startswith(("anatomy:", "imaging:"))]) > 1:
            score += 0.3
        
        return min(1.0, score)
    
    def expand_query_contextual(self, query: str, analysis: QueryAnalysis) -> str:
        """분석 결과를 바탕으로 한 컨텍스트 기반 쿼리 확장"""
        expanded_parts = [query]
        
        # 질병별 전문 확장
        for entity in analysis.medical_entities:
            if not entity.startswith(("anatomy:", "imaging:")):
                disease_expansions = self._get_disease_specific_expansion(entity)
                expanded_parts.extend(disease_expansions)
        
        # 언어별 확장
        if analysis.language == QueryLanguage.KOREAN:
            expanded_parts.extend(self._get_korean_medical_terms())
        elif analysis.language == QueryLanguage.ENGLISH:
            expanded_parts.extend(self._get_english_medical_terms())
        
        # 쿼리 타입별 확장
        type_expansions = self._get_type_specific_expansion(analysis.query_type)
        expanded_parts.extend(type_expansions)
        
        return " ".join(expanded_parts)
    
    def _get_disease_specific_expansion(self, disease: str) -> List[str]:
        """질병별 전문 확장"""
        expansions = {
            "Pneumonia": ["bacterial infection", "antibiotic treatment", "fever cough"],
            "Pneumothorax": ["emergency treatment", "chest tube", "air leak"],
            "Effusion": ["thoracentesis", "pleural space", "fluid drainage"],
            "Atelectasis": ["volume loss", "postoperative", "respiratory therapy"],
            "Infiltrate": ["consolidation", "opacity", "inflammation"],
            "Mass": ["malignancy", "biopsy", "staging"],
            "Cardiomegaly": ["heart failure", "echocardiography", "cardiac evaluation"]
        }
        return expansions.get(disease, [])
    
    def _get_korean_medical_terms(self) -> List[str]:
        """한국어 의료 용어 확장"""
        return ["의료", "진료", "환자", "병원", "임상", "치료"]
    
    def _get_english_medical_terms(self) -> List[str]:
        """영어 의료 용어 확장"""
        return ["medical", "clinical", "patient", "hospital", "healthcare", "treatment"]
    
    def _get_type_specific_expansion(self, query_type: QueryType) -> List[str]:
        """쿼리 타입별 확장"""
        expansions = {
            QueryType.DISEASE_SPECIFIC: ["pathology", "etiology", "prognosis"],
            QueryType.SYMPTOM_BASED: ["clinical presentation", "signs symptoms"],
            QueryType.DIAGNOSTIC_PROCEDURE: ["imaging", "laboratory", "assessment"],
            QueryType.TREATMENT_FOCUSED: ["management", "therapy", "intervention"],
            QueryType.GENERAL_MEDICAL: ["healthcare", "medicine"]
        }
        return expansions.get(query_type, [])
    
    def generate_search_variants(self, query: str) -> List[str]:
        """검색 변형 생성"""
        analysis = self.analyze_query(query)
        variants = [query]
        
        # 1. 원본 쿼리
        variants.append(query)
        
        # 2. 컨텍스트 확장된 쿼리
        expanded = self.expand_query_contextual(query, analysis)
        variants.append(expanded)
        
        # 3. 핵심 용어만 추출한 쿼리
        if analysis.key_terms:
            key_terms_query = " ".join(analysis.key_terms)
            variants.append(key_terms_query)
        
        # 4. 질병명만 추출한 쿼리
        disease_entities = [e for e in analysis.medical_entities 
                           if not e.startswith(("anatomy:", "imaging:"))]
        if disease_entities:
            disease_query = " ".join(disease_entities)
            variants.append(disease_query)
        
        # 5. 동의어 확장 쿼리
        synonym_variants = self._generate_synonym_variants(query)
        variants.extend(synonym_variants)
        
        # 중복 제거 및 반환
        return list(dict.fromkeys(variants))  # 순서 유지하며 중복 제거
    
    def _generate_synonym_variants(self, query: str) -> List[str]:
        """동의어 변형 생성"""
        variants = []
        
        for disease, synonyms in self.disease_synonyms.items():
            for synonym in synonyms:
                if synonym.lower() in query.lower():
                    # 다른 동의어로 치환한 변형 생성
                    for other_synonym in synonyms:
                        if other_synonym != synonym:
                            variant = query.lower().replace(synonym.lower(), other_synonym.lower())
                            variants.append(variant)
                    break
        
        return variants[:3]  # 최대 3개만 반환
    
    def get_query_difficulty(self, query: str) -> str:
        """쿼리 난이도 평가"""
        analysis = self.analyze_query(query)
        
        if analysis.complexity_score < 0.3:
            return "easy"
        elif analysis.complexity_score < 0.6:
            return "medium"
        else:
            return "hard"
    
    def suggest_query_improvements(self, query: str) -> List[str]:
        """쿼리 개선 제안"""
        analysis = self.analyze_query(query)
        suggestions = []
        
        # 신뢰도가 낮은 경우
        if analysis.intent_confidence < 0.5:
            suggestions.append("더 구체적인 의료 용어를 사용해보세요")
        
        # 질병명이 없는 경우
        disease_entities = [e for e in analysis.medical_entities 
                           if not e.startswith(("anatomy:", "imaging:"))]
        if not disease_entities and analysis.query_type != QueryType.DISEASE_SPECIFIC:
            suggestions.append("특정 질병명을 포함하면 더 정확한 결과를 얻을 수 있습니다")
        
        # 너무 짧은 경우
        if len(query.split()) < 2:
            suggestions.append("더 자세한 설명을 추가해보세요")
        
        # 너무 긴 경우
        if len(query.split()) > 15:
            suggestions.append("핵심 키워드로 간단히 줄여보세요")
        
        return suggestions


# 편의 함수들
def analyze_query(query: str) -> QueryAnalysis:
    """쿼리 분석 편의 함수"""
    processor = MedicalQueryProcessor()
    return processor.analyze_query(query)

def expand_query(query: str) -> str:
    """쿼리 확장 편의 함수"""
    processor = MedicalQueryProcessor()
    analysis = processor.analyze_query(query)
    return processor.expand_query_contextual(query, analysis)

def generate_search_variants(query: str) -> List[str]:
    """검색 변형 생성 편의 함수"""
    processor = MedicalQueryProcessor()
    return processor.generate_search_variants(query)


# 테스트 코드
if __name__ == "__main__":
    print("🔍 의료 쿼리 프로세서 테스트")
    print("=" * 50)
    
    processor = MedicalQueryProcessor()
    
    test_queries = [
        "pneumonia diagnosis",
        "폐렴 진단",
        "pleural effusion treatment", 
        "흉수 천자술",
        "chest x-ray pneumothorax",
        "심장비대 소견"
    ]
    
    for query in test_queries:
        print(f"\n🔍 쿼리: '{query}'")
        
        # 분석
        analysis = processor.analyze_query(query)
        print(f"  타입: {analysis.query_type.value}")
        print(f"  언어: {analysis.language.value}")
        print(f"  의료 개체: {analysis.medical_entities}")
        print(f"  신뢰도: {analysis.intent_confidence:.2f}")
        print(f"  복잡도: {analysis.complexity_score:.2f}")
        
        # 확장
        expanded = processor.expand_query_contextual(query, analysis)
        print(f"  확장: '{expanded[:100]}...'")
        
        # 변형
        variants = processor.generate_search_variants(query)
        print(f"  변형 수: {len(variants)}")
        
        # 개선 제안
        suggestions = processor.suggest_query_improvements(query)
        if suggestions:
            print(f"  제안: {suggestions[0]}")
    
    print("\n✅ 테스트 완료!")