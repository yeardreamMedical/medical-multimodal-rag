"""
Result ranking and fusion system for multimodal medical search.

이 모듈은 다양한 검색 전략에서 나온 결과들을 융합하고 랭킹하는 기능을 제공합니다.
베이지안 융합, 상호 순위 융합, 다양성 고려 랭킹 등의 고급 기법을 포함합니다.
"""

import logging
from typing import List, Dict, Any, Optional, Tuple, Set
import numpy as np
from dataclasses import dataclass, field
from collections import defaultdict, Counter
import math
from enum import Enum

# 로깅 설정
logger = logging.getLogger(__name__)

class RankingMethod(Enum):
    """랭킹 방법 열거형"""
    SCORE_BASED = "score_based"              # 점수 기반 단순 랭킹
    RECIPROCAL_RANK = "reciprocal_rank"      # 상호 순위 융합 (RRF)
    BAYESIAN_FUSION = "bayesian_fusion"      # 베이지안 융합
    WEIGHTED_FUSION = "weighted_fusion"      # 가중치 기반 융합
    DIVERSITY_AWARE = "diversity_aware"      # 다양성 고려 랭킹
    HYBRID_ENSEMBLE = "hybrid_ensemble"      # 하이브리드 앙상블

@dataclass
class RankingFeatures:
    """랭킹에 사용되는 특성들"""
    # 기본 검색 점수들
    semantic_score: float = 0.0              # 시맨틱 유사도 점수
    keyword_score: float = 0.0               # 키워드 매칭 점수
    metadata_score: float = 0.0              # 메타데이터 매칭 점수
    disease_match_score: float = 0.0         # 질병명 직접 매칭 점수
    
    # 추가 특성들
    text_length: int = 0                     # 텍스트 길이
    image_confidence: float = 0.0            # 이미지 분류 신뢰도
    recency_score: float = 0.0               # 최신성 점수
    authority_score: float = 0.0             # 권위성 점수
    
    # 다양성 관련
    category: str = ""                       # 카테고리 (질병명, 검사법 등)
    modality: str = ""                       # 모달리티 (text, image, mixed)
    
    # 메타 정보
    source_strategy: str = ""                # 검색 전략 출처
    confidence: float = 0.0                  # 전체 신뢰도
    
    def to_vector(self) -> np.ndarray:
        """특성을 벡터로 변환"""
        return np.array([
            self.semantic_score,
            self.keyword_score, 
            self.metadata_score,
            self.disease_match_score,
            self.image_confidence,
            self.recency_score,
            self.authority_score,
            self.confidence
        ])

@dataclass 
class RankedResult:
    """랭킹된 검색 결과"""
    content: Dict[str, Any]                  # 원본 검색 결과
    final_score: float                       # 최종 융합 점수
    features: RankingFeatures                # 랭킹 특성들
    rank: int = 0                           # 최종 순위
    explanation: str = ""                    # 랭킹 설명
    source_ranks: Dict[str, int] = field(default_factory=dict)  # 각 전략별 순위

class FeatureExtractor:
    """검색 결과에서 랭킹 특성을 추출하는 클래스"""
    
    def __init__(self):
        self.disease_keywords = {
            'pneumonia': ['pneumonia', '폐렴', 'lung infection'],
            'effusion': ['effusion', '흉수', 'pleural fluid'],
            'pneumothorax': ['pneumothorax', '기흉', 'collapsed lung'],
            'atelectasis': ['atelectasis', '무기폐', 'lung collapse'],
            'infiltrate': ['infiltrate', '침윤', 'lung infiltration'],
            'mass': ['mass', '종괴', 'lung mass', 'tumor'],
            'cardiomegaly': ['cardiomegaly', '심비대', 'enlarged heart'],
            'nodule': ['nodule', '결절', 'lung nodule']
        }
        
    def extract_features(self, result: Dict[str, Any], strategy: str) -> RankingFeatures:
        """검색 결과에서 특성을 추출"""
        features = RankingFeatures()
        
        # 기본 점수들 추출
        features.semantic_score = result.get('score', 0.0)
        features.source_strategy = strategy
        
        # 메타데이터에서 추가 정보 추출
        metadata = result.get('metadata', {})
        
        # 질병 매칭 점수 계산
        features.disease_match_score = self._calculate_disease_match_score(result)
        
        # 키워드 점수 계산  
        features.keyword_score = self._calculate_keyword_score(result)
        
        # 메타데이터 점수
        features.metadata_score = metadata.get('relevance_score', 0.0)
        
        # 이미지 관련 특성
        if 'image_confidence' in metadata:
            features.image_confidence = metadata['image_confidence']
            
        # 텍스트 길이
        text_content = result.get('text', '') or result.get('content', '')
        features.text_length = len(text_content)
        
        # 카테고리 및 모달리티 결정
        features.category = self._determine_category(result)
        features.modality = self._determine_modality(result)
        
        # 전체 신뢰도 계산
        features.confidence = self._calculate_confidence(features)
        
        return features
    
    def _calculate_disease_match_score(self, result: Dict[str, Any]) -> float:
        """질병명 직접 매칭 점수 계산"""
        text_content = (result.get('text', '') + ' ' + 
                       result.get('content', '') + ' ' +
                       str(result.get('metadata', {}))).lower()
        
        max_score = 0.0
        for disease, keywords in self.disease_keywords.items():
            for keyword in keywords:
                if keyword.lower() in text_content:
                    # 키워드 길이에 비례한 점수 (긴 키워드일수록 높은 점수)
                    score = len(keyword) / 10.0
                    max_score = max(max_score, score)
                    
        return min(max_score, 1.0)  # 최대 1.0으로 제한
    
    def _calculate_keyword_score(self, result: Dict[str, Any]) -> float:
        """키워드 매칭 점수 계산"""
        # 이미 계산된 점수가 있다면 사용
        if 'keyword_score' in result:
            return result['keyword_score']
            
        # 간단한 키워드 점수 계산 (실제로는 더 복잡한 로직 필요)
        text_content = result.get('text', '') or result.get('content', '')
        if not text_content:
            return 0.0
            
        # 의료 용어 밀도 계산
        medical_terms = ['diagnosis', 'treatment', 'symptoms', 'patient', 
                        'clinical', 'medical', '진단', '치료', '증상', '환자']
        
        term_count = sum(1 for term in medical_terms if term.lower() in text_content.lower())
        return min(term_count / len(medical_terms), 1.0)
    
    def _determine_category(self, result: Dict[str, Any]) -> str:
        """결과의 카테고리 결정"""
        metadata = result.get('metadata', {})
        
        # 메타데이터에 카테고리가 있다면 사용
        if 'category' in metadata:
            return metadata['category']
            
        # 텍스트 내용으로부터 카테고리 추론
        text_content = (result.get('text', '') + ' ' + 
                       result.get('content', '')).lower()
        
        for disease in self.disease_keywords.keys():
            if any(keyword.lower() in text_content for keyword in self.disease_keywords[disease]):
                return disease
                
        return 'general'
    
    def _determine_modality(self, result: Dict[str, Any]) -> str:
        """결과의 모달리티 결정"""
        has_image = 'image_path' in result or 'image_url' in result
        has_text = bool(result.get('text') or result.get('content'))
        
        if has_image and has_text:
            return 'mixed'
        elif has_image:
            return 'image'
        elif has_text:
            return 'text'
        else:
            return 'unknown'
    
    def _calculate_confidence(self, features: RankingFeatures) -> float:
        """전체 신뢰도 계산"""
        # 여러 점수의 가중 평균
        scores = [
            features.semantic_score * 0.3,
            features.keyword_score * 0.2,
            features.disease_match_score * 0.3,
            features.metadata_score * 0.1,
            features.image_confidence * 0.1
        ]
        
        return sum(scores)

class ResultRanker:
    """검색 결과 랭킹 및 융합 클래스"""
    
    def __init__(self, default_method: RankingMethod = RankingMethod.HYBRID_ENSEMBLE):
        self.default_method = default_method
        self.feature_extractor = FeatureExtractor()
        
        # 각 전략별 가중치 (성능에 따라 조정)
        self.strategy_weights = {
            'semantic': 0.3,
            'keyword': 0.2, 
            'metadata': 0.15,
            'cross_modal': 0.25,
            'adaptive': 0.1
        }
        
        # 베이지안 융합 파라미터
        self.bayesian_alpha = 1.0  # 디리클레 분포 파라미터
        
        logger.info(f"ResultRanker initialized with method: {default_method}")
    
    def rank_results(
        self, 
        search_results: Dict[str, List[Dict[str, Any]]], 
        method: Optional[RankingMethod] = None,
        top_k: int = 10
    ) -> List[RankedResult]:
        """
        여러 검색 전략의 결과를 융합하고 랭킹
        
        Args:
            search_results: {strategy_name: [results]} 형태의 검색 결과
            method: 사용할 랭킹 방법
            top_k: 반환할 상위 결과 개수
            
        Returns:
            랭킹된 결과 리스트
        """
        if not search_results:
            logger.warning("No search results to rank")
            return []
            
        method = method or self.default_method
        logger.info(f"Ranking results using method: {method}, strategies: {list(search_results.keys())}")
        
        # 각 전략별 결과를 특성과 함께 준비
        all_results = self._prepare_results(search_results)
        
        if not all_results:
            logger.warning("No valid results after preparation")
            return []
        
        # 선택된 방법에 따라 랭킹 수행
        if method == RankingMethod.SCORE_BASED:
            ranked_results = self._score_based_ranking(all_results)
        elif method == RankingMethod.RECIPROCAL_RANK:
            ranked_results = self._reciprocal_rank_fusion(search_results, all_results)
        elif method == RankingMethod.BAYESIAN_FUSION:
            ranked_results = self._bayesian_fusion(search_results, all_results)
        elif method == RankingMethod.WEIGHTED_FUSION:
            ranked_results = self._weighted_fusion(search_results, all_results)
        elif method == RankingMethod.DIVERSITY_AWARE:
            ranked_results = self._diversity_aware_ranking(all_results)
        elif method == RankingMethod.HYBRID_ENSEMBLE:
            ranked_results = self._hybrid_ensemble_ranking(search_results, all_results)
        else:
            logger.warning(f"Unknown ranking method: {method}, using score-based fallback")
            ranked_results = self._score_based_ranking(all_results)
        
        # 상위 k개만 반환
        final_results = ranked_results[:top_k]
        
        # 최종 순위 부여
        for i, result in enumerate(final_results):
            result.rank = i + 1
            
        logger.info(f"Ranked {len(final_results)} results using {method}")
        return final_results
    
    def _prepare_results(self, search_results: Dict[str, List[Dict[str, Any]]]) -> List[Tuple[Dict[str, Any], RankingFeatures, str]]:
        """검색 결과를 랭킹을 위해 준비"""
        all_results = []
        
        for strategy, results in search_results.items():
            for result in results:
                if not result:  # 빈 결과 건너뛰기
                    continue
                    
                try:
                    features = self.feature_extractor.extract_features(result, strategy)
                    all_results.append((result, features, strategy))
                except Exception as e:
                    logger.warning(f"Failed to extract features from result in {strategy}: {e}")
                    continue
        
        logger.debug(f"Prepared {len(all_results)} results from {len(search_results)} strategies")
        return all_results
    
    def _score_based_ranking(self, all_results: List[Tuple[Dict[str, Any], RankingFeatures, str]]) -> List[RankedResult]:
        """점수 기반 단순 랭킹"""
        ranked_results = []
        
        for result, features, strategy in all_results:
            final_score = features.confidence
            
            ranked_result = RankedResult(
                content=result,
                final_score=final_score,
                features=features,
                explanation=f"Score-based ranking from {strategy}"
            )
            ranked_results.append(ranked_result)
        
        # 점수순으로 정렬
        ranked_results.sort(key=lambda x: x.final_score, reverse=True)
        return ranked_results
    
    def _reciprocal_rank_fusion(
        self, 
        search_results: Dict[str, List[Dict[str, Any]]], 
        all_results: List[Tuple[Dict[str, Any], RankingFeatures, str]]
    ) -> List[RankedResult]:
        """상호 순위 융합 (Reciprocal Rank Fusion)"""
        
        # 각 전략별로 결과의 순위를 계산
        strategy_rankings = {}
        for strategy, results in search_results.items():
            # 점수순으로 정렬하여 순위 부여
            sorted_results = sorted(results, key=lambda x: x.get('score', 0), reverse=True)
            strategy_rankings[strategy] = {
                id(result): rank + 1 for rank, result in enumerate(sorted_results)
            }
        
        # RRF 점수 계산 (k=60은 일반적인 값)
        k = 60
        result_scores = defaultdict(float)
        result_objects = {}
        
        for result, features, strategy in all_results:
            result_id = id(result)
            result_objects[result_id] = (result, features, strategy)
            
            # 해당 전략에서의 순위
            rank = strategy_rankings.get(strategy, {}).get(result_id, len(search_results[strategy]) + 1)
            
            # RRF 점수: 1 / (k + rank)
            rrf_score = 1.0 / (k + rank)
            result_scores[result_id] += rrf_score
        
        # 결과 생성 및 정렬
        ranked_results = []
        for result_id, final_score in result_scores.items():
            result, features, strategy = result_objects[result_id]
            
            ranked_result = RankedResult(
                content=result,
                final_score=final_score,
                features=features,
                explanation=f"RRF fusion across {len(search_results)} strategies"
            )
            ranked_results.append(ranked_result)
        
        ranked_results.sort(key=lambda x: x.final_score, reverse=True)
        return ranked_results
    
    def _bayesian_fusion(
        self, 
        search_results: Dict[str, List[Dict[str, Any]]], 
        all_results: List[Tuple[Dict[str, Any], RankingFeatures, str]]
    ) -> List[RankedResult]:
        """베이지안 융합"""
        
        # 각 전략의 신뢰도 계산 (과거 성능 기반)
        strategy_reliability = {
            'semantic': 0.85,
            'keyword': 0.75,
            'metadata': 0.70,
            'cross_modal': 0.90,
            'adaptive': 0.80
        }
        
        result_scores = defaultdict(list)
        result_objects = {}
        
        # 각 결과에 대해 베이지안 점수 계산
        for result, features, strategy in all_results:
            result_id = id(result)
            result_objects[result_id] = (result, features, strategy)
            
            # 전략 신뢰도 고려한 가중 점수
            reliability = strategy_reliability.get(strategy, 0.5)
            weighted_score = features.confidence * reliability
            
            result_scores[result_id].append(weighted_score)
        
        # 베이지안 평균 계산
        ranked_results = []
        for result_id, scores in result_scores.items():
            result, features, strategy = result_objects[result_id]
            
            # 베이지안 평균: (α + Σx) / (α + n)
            alpha = self.bayesian_alpha
            bayesian_score = (alpha + sum(scores)) / (alpha + len(scores))
            
            ranked_result = RankedResult(
                content=result,
                final_score=bayesian_score,
                features=features,
                explanation=f"Bayesian fusion with α={alpha}"
            )
            ranked_results.append(ranked_result)
        
        ranked_results.sort(key=lambda x: x.final_score, reverse=True)
        return ranked_results
    
    def _weighted_fusion(
        self, 
        search_results: Dict[str, List[Dict[str, Any]]], 
        all_results: List[Tuple[Dict[str, Any], RankingFeatures, str]]
    ) -> List[RankedResult]:
        """가중치 기반 융합"""
        
        result_scores = defaultdict(float)
        result_objects = {}
        
        for result, features, strategy in all_results:
            result_id = id(result)
            result_objects[result_id] = (result, features, strategy)
            
            # 전략별 가중치 적용
            weight = self.strategy_weights.get(strategy, 0.1)
            weighted_score = features.confidence * weight
            
            result_scores[result_id] += weighted_score
        
        # 결과 생성 및 정렬
        ranked_results = []
        for result_id, final_score in result_scores.items():
            result, features, strategy = result_objects[result_id]
            
            ranked_result = RankedResult(
                content=result,
                final_score=final_score,
                features=features,
                explanation=f"Weighted fusion with strategy weights"
            )
            ranked_results.append(ranked_result)
        
        ranked_results.sort(key=lambda x: x.final_score, reverse=True)
        return ranked_results
    
    def _diversity_aware_ranking(self, all_results: List[Tuple[Dict[str, Any], RankingFeatures, str]]) -> List[RankedResult]:
        """다양성 고려 랭킹 (MMR - Maximal Marginal Relevance)"""
        
        if not all_results:
            return []
        
        # 초기 점수순으로 정렬
        sorted_results = sorted(all_results, key=lambda x: x[1].confidence, reverse=True)
        
        selected_results = []
        remaining_results = sorted_results.copy()
        
        # MMR 파라미터 (λ=0.7은 관련성 70%, 다양성 30%)
        lambda_param = 0.7
        
        # 첫 번째 결과는 가장 높은 점수의 것을 선택
        if remaining_results:
            best_result = remaining_results.pop(0)
            selected_results.append(best_result)
        
        # 나머지 결과들을 다양성을 고려하여 선택
        while remaining_results and len(selected_results) < 20:  # 최대 20개
            best_mmr_score = -1
            best_result = None
            best_idx = -1
            
            for idx, (result, features, strategy) in enumerate(remaining_results):
                # 관련성 점수
                relevance_score = features.confidence
                
                # 다양성 점수 (이미 선택된 결과들과의 차이)
                diversity_score = self._calculate_diversity_score(features, selected_results)
                
                # MMR 점수 계산
                mmr_score = lambda_param * relevance_score + (1 - lambda_param) * diversity_score
                
                if mmr_score > best_mmr_score:
                    best_mmr_score = mmr_score
                    best_result = (result, features, strategy)
                    best_idx = idx
            
            if best_result:
                selected_results.append(best_result)
                remaining_results.pop(best_idx)
            else:
                break
        
        # RankedResult 객체로 변환
        ranked_results = []
        for result, features, strategy in selected_results:
            ranked_result = RankedResult(
                content=result,
                final_score=features.confidence,  # 원래 관련성 점수 유지
                features=features,
                explanation=f"Diversity-aware ranking (MMR λ={lambda_param})"
            )
            ranked_results.append(ranked_result)
        
        return ranked_results
    
    def _calculate_diversity_score(self, features: RankingFeatures, selected_results: List) -> float:
        """선택된 결과들과의 다양성 점수 계산"""
        if not selected_results:
            return 1.0
        
        diversity_scores = []
        
        for selected_result, selected_features, _ in selected_results:
            # 카테고리 다양성
            category_diversity = 0.5 if features.category != selected_features.category else 0.0
            
            # 모달리티 다양성  
            modality_diversity = 0.3 if features.modality != selected_features.modality else 0.0
            
            # 전략 다양성
            strategy_diversity = 0.2 if features.source_strategy != selected_features.source_strategy else 0.0
            
            total_diversity = category_diversity + modality_diversity + strategy_diversity
            diversity_scores.append(total_diversity)
        
        # 평균 다양성 반환
        return sum(diversity_scores) / len(diversity_scores) if diversity_scores else 0.0
    
    def _hybrid_ensemble_ranking(
        self, 
        search_results: Dict[str, List[Dict[str, Any]]], 
        all_results: List[Tuple[Dict[str, Any], RankingFeatures, str]]
    ) -> List[RankedResult]:
        """하이브리드 앙상블 랭킹 (여러 방법 조합)"""
        
        # 여러 랭킹 방법으로 점수 계산
        methods_weights = {
            RankingMethod.SCORE_BASED: 0.2,
            RankingMethod.RECIPROCAL_RANK: 0.3,
            RankingMethod.BAYESIAN_FUSION: 0.3,
            RankingMethod.WEIGHTED_FUSION: 0.2
        }
        
        # 각 방법별 결과 수집
        method_results = {}
        for method, weight in methods_weights.items():
            try:
                if method == RankingMethod.SCORE_BASED:
                    method_results[method] = self._score_based_ranking(all_results)
                elif method == RankingMethod.RECIPROCAL_RANK:
                    method_results[method] = self._reciprocal_rank_fusion(search_results, all_results)
                elif method == RankingMethod.BAYESIAN_FUSION:
                    method_results[method] = self._bayesian_fusion(search_results, all_results)
                elif method == RankingMethod.WEIGHTED_FUSION:
                    method_results[method] = self._weighted_fusion(search_results, all_results)
            except Exception as e:
                logger.warning(f"Failed to compute {method}: {e}")
                continue
        
        # 앙상블 점수 계산
        result_ensemble_scores = defaultdict(float)
        result_objects = {}
        
        for method, results in method_results.items():
            weight = methods_weights[method]
            
            for rank, ranked_result in enumerate(results):
                result_id = id(ranked_result.content)
                result_objects[result_id] = ranked_result
                
                # 순위 기반 점수 (높은 순위일수록 높은 점수)
                rank_score = 1.0 / (rank + 1)
                ensemble_score = weight * (ranked_result.final_score * 0.7 + rank_score * 0.3)
                
                result_ensemble_scores[result_id] += ensemble_score
        
        # 최종 결과 생성
        final_results = []
        for result_id, ensemble_score in result_ensemble_scores.items():
            original_result = result_objects[result_id]
            
            final_result = RankedResult(
                content=original_result.content,
                final_score=ensemble_score,
                features=original_result.features,
                explanation=f"Hybrid ensemble of {len(method_results)} ranking methods"
            )
            final_results.append(final_result)
        
        # 앙상블 점수순으로 정렬
        final_results.sort(key=lambda x: x.final_score, reverse=True)
        
        # 다양성 필터링 적용 (상위 결과들 중에서)
        if len(final_results) > 5:
            diversity_filtered = self._apply_diversity_filter(final_results[:20])  # 상위 20개에서 다양성 고려
            return diversity_filtered
        
        return final_results
    
    def _apply_diversity_filter(self, results: List[RankedResult]) -> List[RankedResult]:
        """결과 리스트에 다양성 필터 적용"""
        if len(results) <= 5:
            return results
        
        filtered_results = [results[0]]  # 첫 번째는 무조건 포함
        seen_categories = {results[0].features.category}
        seen_modalities = {results[0].features.modality}
        
        for result in results[1:]:
            # 카테고리나 모달리티가 다르면 우선 포함
            if (result.features.category not in seen_categories or 
                result.features.modality not in seen_modalities):
                filtered_results.append(result)
                seen_categories.add(result.features.category)
                seen_modalities.add(result.features.modality)
                
                if len(filtered_results) >= 10:  # 최대 10개까지
                    break
        
        # 아직 부족하면 점수순으로 추가
        for result in results:
            if result not in filtered_results and len(filtered_results) < 10:
                filtered_results.append(result)
        
        return filtered_results

def create_result_ranker(method: str = "hybrid_ensemble") -> ResultRanker:
    """편의 함수: ResultRanker 인스턴스 생성"""
    try:
        ranking_method = RankingMethod(method)
    except ValueError:
        logger.warning(f"Unknown ranking method: {method}, using hybrid_ensemble")
        ranking_method = RankingMethod.HYBRID_ENSEMBLE
    
    return ResultRanker(default_method=ranking_method)

# 사용 예시
if __name__ == "__main__":
    # 테스트용 더미 데이터
    sample_results = {
        'semantic': [
            {
                'text': 'Pneumonia is a lung infection that inflames air sacs.',
                'score': 0.95,
                'metadata': {'category': 'pneumonia', 'relevance_score': 0.9}
            },
            {
                'text': 'Chest X-ray shows infiltrates in lower lobe.',
                'score': 0.87,
                'metadata': {'category': 'pneumonia', 'relevance_score': 0.8}
            }
        ],
        'keyword': [
            {
                'text': 'Patient presents with pneumonia symptoms including fever and cough.',
                'score': 0.82,
                'metadata': {'category': 'pneumonia', 'relevance_score': 0.85}
            },
            {
                'text': 'Pleural effusion detected on imaging studies.',
                'score': 0.78,
                'metadata': {'category': 'effusion', 'relevance_score': 0.75}
            }
        ],
        'cross_modal': [
            {
                'text': 'Radiological findings consistent with pneumonia.',
                'image_path': '/path/to/xray.jpg',
                'score': 0.91,
                'metadata': {'category': 'pneumonia', 'image_confidence': 0.88, 'relevance_score': 0.9}
            }
        ]
    }
    
    # ResultRanker 테스트
    ranker = ResultRanker(RankingMethod.HYBRID_ENSEMBLE)
    
    print("=== 하이브리드 앙상블 랭킹 테스트 ===")
    ranked_results = ranker.rank_results(sample_results, top_k=5)
    
    for i, result in enumerate(ranked_results, 1):
        print(f"\n{i}. 순위: {result.rank}")
        print(f"   점수: {result.final_score:.4f}")
        print(f"   카테고리: {result.features.category}")
        print(f"   모달리티: {result.features.modality}")
        print(f"   설명: {result.explanation}")
        print(f"   내용: {result.content.get('text', '')[:100]}...")
    
    print("\n=== 다양한 랭킹 방법 비교 ===")
    methods = [
        RankingMethod.SCORE_BASED,
        RankingMethod.RECIPROCAL_RANK,
        RankingMethod.BAYESIAN_FUSION,
        RankingMethod.DIVERSITY_AWARE
    ]
    
    for method in methods:
        print(f"\n--- {method.value} ---")
        try:
            results = ranker.rank_results(sample_results, method=method, top_k=3)
            for i, result in enumerate(results, 1):
                print(f"  {i}. {result.final_score:.3f} - {result.features.category} - {result.content.get('text', '')[:50]}...")
        except Exception as e:
            print(f"  오류: {e}")
    
    print("\n=== 특성 추출 테스트 ===")
    extractor = FeatureExtractor()
    
    test_result = {
        'text': 'Patient diagnosed with pneumonia. Chest X-ray shows bilateral infiltrates.',
        'score': 0.92,
        'metadata': {'category': 'pneumonia', 'relevance_score': 0.88}
    }
    
    features = extractor.extract_features(test_result, 'semantic')
    print(f"추출된 특성:")
    print(f"  - 시맨틱 점수: {features.semantic_score}")
    print(f"  - 질병 매칭 점수: {features.disease_match_score}")
    print(f"  - 키워드 점수: {features.keyword_score}")
    print(f"  - 카테고리: {features.category}")
    print(f"  - 모달리티: {features.modality}")
    print(f"  - 전체 신뢰도: {features.confidence}")
    
    print("\n✅ result_ranker.py 테스트 완료!")