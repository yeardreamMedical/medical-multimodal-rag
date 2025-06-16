# search/__init__.py
# 검색 모듈 초기화 및 외부 인터페이스

"""
Medical Multimodal Search Module

이 모듈은 의료 텍스트와 이미지를 통합하여 검색하는 멀티모달 검색 시스템을 제공합니다.

주요 기능:
- 텍스트 기반 의학 지식 검색 (쿼리 확장 포함)
- 이미지 기반 흉부 X-ray 검색 (BioViL-T 모델)
- 질병명 자동 추출 및 매칭
- 검색 결과 통합 및 컨텍스트 생성

Example:
    >>> from search import SearchEngine
    >>> engine = SearchEngine()
    >>> result = engine.search_text("폐렴 진단")
    >>> print(result['korean_diagnosis'])
"""

from .search_engine import (
    SearchEngine,
    SearchConfig,
    QueryProcessor,
    DiseaseExtractor,
    ImageSearcher,
    ContextBuilder,
    SearchTester,
    create_search_engine,
    quick_test,
    run_full_evaluation
)

# 버전 정보
__version__ = "1.0.0"
__author__ = "sol kim"
__email__ = "kimsol1134@naver.com"

# 모듈 레벨에서 사용할 주요 클래스들
__all__ = [
    # 메인 클래스
    "SearchEngine",
    "SearchConfig", 
    "SearchTester",
    
    # 컴포넌트 클래스
    "QueryProcessor",
    "DiseaseExtractor", 
    "ImageSearcher",
    "ContextBuilder",
    
    # 편의 함수
    "create_search_engine",
    "quick_test",
    "run_full_evaluation"
]

# 모듈 레벨 편의 함수
def get_version():
    """모듈 버전 반환"""
    return __version__

def get_supported_diseases():
    """지원되는 질병 목록 반환"""
    return list(SearchConfig.DISEASE_INFO.keys())

def get_disease_info():
    """질병 정보 딕셔너리 반환"""
    return SearchConfig.DISEASE_INFO.copy()

# 모듈 로딩 시 기본 설정 확인
def _check_environment():
    """환경 설정 확인"""
    import os
    from dotenv import load_dotenv
    
    load_dotenv()
    
    required_vars = ["PINECONE_API_KEY", "OPENAI_API_KEY"]
    missing_vars = []
    
    for var in required_vars:
        if not os.getenv(var):
            missing_vars.append(var)
    
    if missing_vars:
        import warnings
        warnings.warn(
            f"환경변수가 설정되지 않았습니다: {', '.join(missing_vars)}. "
            f".env 파일을 확인하거나 환경변수를 직접 설정해주세요.",
            UserWarning
        )
        return False
    
    return True

# 환경 설정 자동 확인
_env_check_passed = _check_environment()

# 디버그 정보
def get_module_info():
    """모듈 정보 반환"""
    return {
        "version": __version__,
        "author": __author__,
        "supported_diseases": len(get_supported_diseases()),
        "environment_check": _env_check_passed,
        "main_classes": len(__all__)
    }

# 모듈 임포트 시 간단한 정보 출력 (선택사항)
if __name__ != "__main__":
    # 모듈이 임포트될 때만 실행
    pass  # 필요시 로딩 메시지 추가

# 예제 사용법을 위한 독스트링
"""
사용 예제:

1. 기본 검색:
    from search import SearchEngine
    
    engine = SearchEngine()
    result = engine.search_text("폐렴 진단")
    print(f"진단: {result['korean_diagnosis']}")

2. 빠른 테스트:
    from search import quick_test
    
    quick_test("pleural effusion")

3. 전체 평가:
    from search import run_full_evaluation
    
    results = run_full_evaluation()
    print(f"정확도: {results['accuracy']:.1f}%")

4. 모듈 정보:
    from search import get_module_info, get_supported_diseases
    
    print(get_module_info())
    print(f"지원 질병: {get_supported_diseases()}")
"""