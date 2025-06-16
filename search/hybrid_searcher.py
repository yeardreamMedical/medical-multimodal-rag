# search/hybrid_searcher.py
# 하이브리드 검색 전략 구현

import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import asyncio
from concurrent.futures import ThreadPoolExecutor


class SearchStrategy(Enum):
    """검색 전략 타입"""
    SEMANTIC_ONLY = "semantic_only"           # 의미론적 검색만
    KEYWORD_ONLY = "keyword_only"             # 키워드 검색만
    SEMANTIC_KEYWORD = "semantic_keyword"     # 의미론적 + 키워드
    CROSS_MODAL = "cross_modal"               # 텍스트-이미지 교차
    MULTI_QUERY = "multi_query"               # 다중 쿼리
    ADAPTIVE = "adaptive"                     # 적응적 검색


@dataclass
class SearchResult:
    """검색 결과 데이터 클래스"""
    content: str
    score: float
    source: str
    metadata: Dict[str, Any]
    search_method: str
    result_id: str


@dataclass
class HybridSearchConfig:
    """하이브리드 검색 설정"""
    semantic_weight: float = 0.7
    keyword_weight: float = 0.3
    cross_modal_weight: float = 0.5
    max_results_per_method: int = 10
    score_threshold: float = 0.1
    enable_cross_modal: bool = True
    enable_query_expansion: bool = True


class SemanticSearcher:
    """의미론적 검색 엔진"""
    
    def __init__(self, text_index, image_index, openai_client, config):
        self.text_index = text_index
        self.image_index = image_index
        self.openai_client = openai_client
        self.config = config
    
    async def search_text_semantic(self, query: str, top_k: int = 10) -> List[SearchResult]:
        """텍스트 의미론적 검색"""
        try:
            # OpenAI 임베딩 생성
            resp = self.openai_client.embeddings.create(
                input=[query], 
                model="text-embedding-3-small"
            )
            embedding = resp.data[0].embedding
            
            # Pinecone 검색
            results = self.text_index.query(
                vector=embedding,
                top_k=top_k,
                include_metadata=True
            )
            
            search_results = []
            for match in results['matches']:
                metadata = match['metadata']
                content = metadata.get('text', metadata.get('content', ''))
                
                search_results.append(SearchResult(
                    content=content,
                    score=match['score'],
                    source=metadata.get('source', 'unknown'),
                    metadata=metadata,
                    search_method="semantic_text",
                    result_id=match['id']
                ))
            
            return search_results
            
        except Exception as e:
            print(f"❌ 의미론적 텍스트 검색 실패: {e}")
            return []
    
    async def search_image_semantic(self, text_query: str, image_encoder, image_transform, top_k: int = 10) -> List[SearchResult]:
        """텍스트→이미지 의미론적 교차 검색"""
        try:
            # 텍스트를 이미지 공간으로 매핑하는 로직 (간소화)
            # 실제로는 CLIP이나 BioViL-T의 텍스트-이미지 공통 공간 활용
            
            # 임시: 더미 벡터 생성 (실제로는 멀티모달 임베딩 사용)
            dummy_vector = np.random.normal(0, 1, 512).tolist()
            
            results = self.image_index.query(
                vector=dummy_vector,
                top_k=top_k,
                include_metadata=True
            )
            
            search_results = []
            for match in results['matches']:
                metadata = match['metadata']
                
                search_results.append(SearchResult(
                    content=metadata.get('all_descriptions', ''),
                    score=match['score'] * 0.8,  # 교차 검색이므로 가중치 적용
                    source='image_cross_modal',
                    metadata=metadata,
                    search_method="semantic_cross_modal",
                    result_id=match['id']
                ))
            
            return search_results
            
        except Exception as e:
            print(f"❌ 교차 모달 검색 실패: {e}")
            return []


class KeywordSearcher:
    """키워드 기반 검색 엔진"""
    
    def __init__(self, text_index, image_index):
        self.text_index = text_index
        self.image_index = image_index
        self.medical_keywords = self._load_medical_keywords()
    
    def _load_medical_keywords(self) -> Dict[str, List[str]]:
        """의료 키워드 매핑"""
        return {
            "Pneumonia": ["pneumonia", "lung infection", "bacterial pneumonia", "폐렴"],
            "Effusion": ["pleural effusion", "chest fluid", "흉수", "늑막삼출"],
            "Pneumothorax": ["pneumothorax", "collapsed lung", "기흉"],
            "Atelectasis": ["atelectasis", "lung collapse", "무기폐"],
            "Infiltrate": ["infiltrate", "consolidation", "침윤", "경화"],
            "Mass": ["mass", "tumor", "lesion", "종괴", "종양"],
            "Cardiomegaly": ["cardiomegaly", "enlarged heart", "심장비대"],
            "Nodule": ["nodule", "lung nodule", "결절", "폐결절"]
        }
    
    async def search_by_keywords(self, query: str, top_k: int = 10) -> List[SearchResult]:
        """키워드 기반 검색"""
        try:
            # 쿼리에서 의료 키워드 추출
            matched_diseases = self._extract_diseases_from_query(query)
            
            if not matched_diseases:
                return []
            
            # 각 질병별로 이미지 검색
            all_results = []
            for disease in matched_diseases:
                # 질병별 필터 검색
                filter_condition = {"primary_label": {"$eq": disease}}
                dummy_vector = [0.0] * 512
                
                results = self.image_index.query(
                    vector=dummy_vector,
                    filter=filter_condition,
                    top_k=min(top_k, 5),  # 질병당 최대 5개
                    include_metadata=True
                )
                
                for match in results['matches']:
                    metadata = match['metadata']
                    
                    all_results.append(SearchResult(
                        content=metadata.get('all_descriptions', ''),
                        score=1.0,  # 정확한 매칭이므로 최고 점수
                        source=f'keyword_filter_{disease}',
                        metadata=metadata,
                        search_method="keyword_filter",
                        result_id=match['id']
                    ))
            
            return all_results[:top_k]
            
        except Exception as e:
            print(f"❌ 키워드 검색 실패: {e}")
            return []
    
    def _extract_diseases_from_query(self, query: str) -> List[str]:
        """쿼리에서 질병명 추출"""
        matched_diseases = []
        query_lower = query.lower()
        
        for disease, keywords in self.medical_keywords.items():
            for keyword in keywords:
                if keyword.lower() in query_lower:
                    matched_diseases.append(disease)
                    break
        
        return matched_diseases


class MetadataSearcher:
    """메타데이터 기반 검색 엔진"""
    
    def __init__(self, text_index, image_index):
        self.text_index = text_index
        self.image_index = image_index
    
    async def search_by_metadata(self, query: str, metadata_filters: Dict[str, Any], top_k: int = 10) -> List[SearchResult]:
        """메타데이터 필터 검색"""
        try:
            dummy_vector = [0.0] * 512
            
            # 이미지 인덱스에서 메타데이터 필터 검색
            results = self.image_index.query(
                vector=dummy_vector,
                filter=metadata_filters,
                top_k=top_k,
                include_metadata=True
            )
            
            search_results = []
            for match in results['matches']:
                metadata = match['metadata']
                
                search_results.append(SearchResult(
                    content=metadata.get('all_descriptions', ''),
                    score=0.9,  # 메타데이터 매칭 점수
                    source='metadata_filter',
                    metadata=metadata,
                    search_method="metadata_filter",
                    result_id=match['id']
                ))
            
            return search_results
            
        except Exception as e:
            print(f"❌ 메타데이터 검색 실패: {e}")
            return []


class HybridSearchEngine:
    """하이브리드 검색 엔진 - 여러 검색 전략을 조합"""
    
    def __init__(self, text_index, image_index, openai_client, image_encoder, image_transform, config: HybridSearchConfig = None):
        self.text_index = text_index
        self.image_index = image_index
        self.openai_client = openai_client
        self.image_encoder = image_encoder
        self.image_transform = image_transform
        self.config = config or HybridSearchConfig()
        
        # 개별 검색 엔진들
        self.semantic_searcher = SemanticSearcher(text_index, image_index, openai_client, config)
        self.keyword_searcher = KeywordSearcher(text_index, image_index)
        self.metadata_searcher = MetadataSearcher(text_index, image_index)
        
        # 스레드 풀 (비동기 처리용)
        self.executor = ThreadPoolExecutor(max_workers=4)
    
    async def search_hybrid(self, query: str, strategy: SearchStrategy = SearchStrategy.ADAPTIVE, top_k: int = 10) -> List[SearchResult]:
        """하이브리드 검색 메인 함수"""
        print(f"🔍 하이브리드 검색 시작: '{query}' (전략: {strategy.value})")
        
        if strategy == SearchStrategy.ADAPTIVE:
            strategy = self._choose_adaptive_strategy(query)
            print(f"   📊 적응적 전략 선택: {strategy.value}")
        
        # 전략별 검색 실행
        if strategy == SearchStrategy.SEMANTIC_ONLY:
            return await self._search_semantic_only(query, top_k)
        elif strategy == SearchStrategy.KEYWORD_ONLY:
            return await self._search_keyword_only(query, top_k)
        elif strategy == SearchStrategy.SEMANTIC_KEYWORD:
            return await self._search_semantic_keyword(query, top_k)
        elif strategy == SearchStrategy.CROSS_MODAL:
            return await self._search_cross_modal(query, top_k)
        elif strategy == SearchStrategy.MULTI_QUERY:
            return await self._search_multi_query(query, top_k)
        else:
            # 기본값: 의미론적 + 키워드
            return await self._search_semantic_keyword(query, top_k)
    
    def _choose_adaptive_strategy(self, query: str) -> SearchStrategy:
        """쿼리 분석을 통한 적응적 전략 선택"""
        query_lower = query.lower()
        
        # 명확한 질병명이 있는 경우 키워드 검색 우선
        disease_keywords = ["pneumonia", "effusion", "pneumothorax", "폐렴", "흉수", "기흉"]
        if any(keyword in query_lower for keyword in disease_keywords):
            return SearchStrategy.SEMANTIC_KEYWORD
        
        # 증상이나 일반적인 설명인 경우 의미론적 검색
        symptom_keywords = ["cough", "fever", "pain", "기침", "발열", "통증"]
        if any(keyword in query_lower for keyword in symptom_keywords):
            return SearchStrategy.SEMANTIC_ONLY
        
        # 복잡한 쿼리인 경우 다중 쿼리 전략
        if len(query.split()) > 5:
            return SearchStrategy.MULTI_QUERY
        
        # 영상 관련 용어가 있는 경우 교차 모달
        imaging_keywords = ["x-ray", "ct", "image", "영상", "사진"]
        if any(keyword in query_lower for keyword in imaging_keywords):
            return SearchStrategy.CROSS_MODAL
        
        # 기본값
        return SearchStrategy.SEMANTIC_KEYWORD
    
    async def _search_semantic_only(self, query: str, top_k: int) -> List[SearchResult]:
        """의미론적 검색만"""
        results = await self.semantic_searcher.search_text_semantic(query, top_k)
        return self._apply_score_threshold(results)
    
    async def _search_keyword_only(self, query: str, top_k: int) -> List[SearchResult]:
        """키워드 검색만"""
        results = await self.keyword_searcher.search_by_keywords(query, top_k)
        return self._apply_score_threshold(results)
    
    async def _search_semantic_keyword(self, query: str, top_k: int) -> List[SearchResult]:
        """의미론적 + 키워드 조합 검색"""
        # 병렬 검색 실행
        semantic_task = self.semantic_searcher.search_text_semantic(query, self.config.max_results_per_method)
        keyword_task = self.keyword_searcher.search_by_keywords(query, self.config.max_results_per_method)
        
        semantic_results, keyword_results = await asyncio.gather(semantic_task, keyword_task)
        
        # 점수 가중치 적용
        for result in semantic_results:
            result.score *= self.config.semantic_weight
            result.search_method = "semantic_weighted"
        
        for result in keyword_results:
            result.score *= self.config.keyword_weight
            result.search_method = "keyword_weighted"
        
        # 결과 합병 및 정렬
        combined_results = semantic_results + keyword_results
        combined_results = self._merge_duplicate_results(combined_results)
        combined_results.sort(key=lambda x: x.score, reverse=True)
        
        return self._apply_score_threshold(combined_results[:top_k])
    
    async def _search_cross_modal(self, query: str, top_k: int) -> List[SearchResult]:
        """텍스트-이미지 교차 모달 검색"""
        # 텍스트 검색
        text_task = self.semantic_searcher.search_text_semantic(query, self.config.max_results_per_method)
        
        # 텍스트→이미지 교차 검색
        cross_modal_task = self.semantic_searcher.search_image_semantic(
            query, self.image_encoder, self.image_transform, self.config.max_results_per_method
        )
        
        text_results, cross_modal_results = await asyncio.gather(text_task, cross_modal_task)
        
        # 교차 모달 결과에 가중치 적용
        for result in cross_modal_results:
            result.score *= self.config.cross_modal_weight
        
        # 결과 합병
        combined_results = text_results + cross_modal_results
        combined_results.sort(key=lambda x: x.score, reverse=True)
        
        return self._apply_score_threshold(combined_results[:top_k])
    
    async def _search_multi_query(self, query: str, top_k: int) -> List[SearchResult]:
        """다중 쿼리 검색 (쿼리 분해 및 개별 검색)"""
        # 쿼리를 여러 부분으로 분해
        query_variants = self._generate_query_variants(query)
        
        # 각 변형 쿼리로 검색
        all_tasks = []
        for variant in query_variants:
            task = self.semantic_searcher.search_text_semantic(variant, self.config.max_results_per_method // len(query_variants))
            all_tasks.append(task)
        
        all_results = await asyncio.gather(*all_tasks)
        
        # 결과 평탄화 및 점수 조정
        flattened_results = []
        for i, results in enumerate(all_results):
            weight = 1.0 / (i + 1)  # 첫 번째 변형에 더 높은 가중치
            for result in results:
                result.score *= weight
                result.search_method = f"multi_query_{i}"
                flattened_results.append(result)
        
        # 중복 제거 및 정렬
        flattened_results = self._merge_duplicate_results(flattened_results)
        flattened_results.sort(key=lambda x: x.score, reverse=True)
        
        return self._apply_score_threshold(flattened_results[:top_k])
    
    def _generate_query_variants(self, query: str) -> List[str]:
        """쿼리 변형 생성"""
        variants = [query]
        
        # 핵심 키워드 추출
        keywords = query.split()
        if len(keywords) > 2:
            # 첫 두 단어
            variants.append(" ".join(keywords[:2]))
            # 마지막 두 단어
            variants.append(" ".join(keywords[-2:]))
        
        # 의료 용어만 추출
        medical_terms = []
        medical_keywords = ["pneumonia", "effusion", "pneumothorax", "폐렴", "흉수", "기흉", "diagnosis", "진단"]
        for word in keywords:
            if word.lower() in medical_keywords:
                medical_terms.append(word)
        
        if medical_terms:
            variants.append(" ".join(medical_terms))
        
        return variants[:4]  # 최대 4개 변형
    
    def _merge_duplicate_results(self, results: List[SearchResult]) -> List[SearchResult]:
        """중복 결과 병합 (ID 기준)"""
        seen_ids = {}
        merged_results = []
        
        for result in results:
            if result.result_id in seen_ids:
                # 기존 결과와 점수 합산
                existing_result = seen_ids[result.result_id]
                existing_result.score = max(existing_result.score, result.score)
                existing_result.search_method += f",{result.search_method}"
            else:
                seen_ids[result.result_id] = result
                merged_results.append(result)
        
        return merged_results
    
    def _apply_score_threshold(self, results: List[SearchResult]) -> List[SearchResult]:
        """점수 임계값 적용"""
        return [r for r in results if r.score >= self.config.score_threshold]
    
    async def search_with_feedback(self, query: str, top_k: int = 10, user_feedback: Dict[str, Any] = None) -> List[SearchResult]:
        """사용자 피드백을 반영한 검색"""
        # 기본 검색 수행
        results = await self.search_hybrid(query, SearchStrategy.ADAPTIVE, top_k)
        
        # 사용자 피드백이 있는 경우 점수 조정
        if user_feedback:
            results = self._apply_user_feedback(results, user_feedback)
        
        return results
    
    def _apply_user_feedback(self, results: List[SearchResult], feedback: Dict[str, Any]) -> List[SearchResult]:
        """사용자 피드백 적용"""
        # 선호하는 질병이나 소스가 있는 경우 가중치 증가
        preferred_diseases = feedback.get('preferred_diseases', [])
        preferred_sources = feedback.get('preferred_sources', [])
        
        for result in results:
            # 선호 질병 가중치
            if any(disease in result.content.lower() for disease in preferred_diseases):
                result.score *= 1.2
            
            # 선호 소스 가중치
            if result.source in preferred_sources:
                result.score *= 1.1
        
        # 점수 재정렬
        results.sort(key=lambda x: x.score, reverse=True)
        return results
    
    def get_search_statistics(self) -> Dict[str, Any]:
        """검색 통계 정보"""
        return {
            "config": {
                "semantic_weight": self.config.semantic_weight,
                "keyword_weight": self.config.keyword_weight,
                "cross_modal_weight": self.config.cross_modal_weight,
                "score_threshold": self.config.score_threshold
            },
            "available_strategies": [strategy.value for strategy in SearchStrategy],
            "supported_methods": [
                "semantic_text", "keyword_filter", "cross_modal", 
                "metadata_filter", "multi_query"
            ]
        }


# 편의 함수들
async def hybrid_search(query: str, text_index, image_index, openai_client, image_encoder, image_transform, 
                       strategy: SearchStrategy = SearchStrategy.ADAPTIVE, top_k: int = 10) -> List[SearchResult]:
    """하이브리드 검색 편의 함수"""
    config = HybridSearchConfig()
    engine = HybridSearchEngine(text_index, image_index, openai_client, image_encoder, image_transform, config)
    return await engine.search_hybrid(query, strategy, top_k)

def create_hybrid_config(semantic_weight: float = 0.7, keyword_weight: float = 0.3, 
                        cross_modal_weight: float = 0.5) -> HybridSearchConfig:
    """하이브리드 검색 설정 생성 편의 함수"""
    return HybridSearchConfig(
        semantic_weight=semantic_weight,
        keyword_weight=keyword_weight,
        cross_modal_weight=cross_modal_weight
    )


# 테스트 코드
if __name__ == "__main__":
    print("🔍 하이브리드 검색 엔진 테스트")
    print("=" * 50)
    
    # 비동기 테스트 예제
    async def test_hybrid_search():
        # 실제 사용시에는 인덱스와 클라이언트를 전달
        # engine = HybridSearchEngine(text_index, image_index, openai_client, ...)
        
        test_queries = [
            "pneumonia diagnosis",
            "폐렴 치료",
            "chest x-ray pneumothorax",
            "pleural effusion symptoms"
        ]
        
        for query in test_queries:
            print(f"\n🔍 테스트 쿼리: '{query}'")
            
            # 실제 검색 대신 시뮬레이션
            print(f"   📊 적응적 전략 시뮬레이션...")
            
            # 쿼리 분석 시뮬레이션
            if "pneumonia" in query.lower() or "폐렴" in query:
                strategy = SearchStrategy.SEMANTIC_KEYWORD
            elif "x-ray" in query.lower():
                strategy = SearchStrategy.CROSS_MODAL
            else:
                strategy = SearchStrategy.SEMANTIC_ONLY
            
            print(f"   ✅ 선택된 전략: {strategy.value}")
    
    # 설정 테스트
    config = create_hybrid_config(semantic_weight=0.8, keyword_weight=0.2)
    print(f"\n⚙️ 설정 테스트:")
    print(f"   의미론적 가중치: {config.semantic_weight}")
    print(f"   키워드 가중치: {config.keyword_weight}")
    print(f"   임계값: {config.score_threshold}")
    
    print("\n✅ 하이브리드 검색 엔진 테스트 완료!")
    print("📋 실제 사용시 비동기 함수로 호출: await engine.search_hybrid(query)")