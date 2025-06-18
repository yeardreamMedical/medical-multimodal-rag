import sys
# search_engine.py가 있는 경로를 시스템 경로에 추가
sys.path.append('./search') 

from search.search_engine import SearchEngine
from generation.context_builder import LLMContextBuilder

def main():
    print("🚀 Phase 5: Step 1.5 - ContextBuilder 연동 및 이미지 경로 확인 테스트")
    
    # 1. Phase 4의 검색 엔진 실행
    print("\n[1/4] SearchEngine으로 '폐렴' 정보를 검색합니다...")
    engine = SearchEngine()
    search_result = engine.search_text("폐렴")

    if "error" in search_result:
        print(f"❌ 검색 실패: {search_result['error']}")
        return

    print("✅ 검색 성공!")

    # 2. ContextBuilder로 컨텍스트 생성
    print("\n[2/4] LLMContextBuilder로 검색 결과를 변환합니다...")
    try:
        builder = LLMContextBuilder(search_result)
        llm_context = builder.build_context_for_llm()
        print("✅ LLM용 컨텍스트 생성 성공!")
    except ValueError as e:
        print(f"❌ 컨텍스트 생성 실패: {e}")
        return

    # 3. 생성된 컨텍스트 확인
    print("\n[3/4] 생성된 최종 LLM용 컨텍스트:")
    print("-" * 60)
    print(llm_context)
    print("-" * 60)

    # 4. 이미지 경로 추출 기능 확인
    print("\n[4/4] 문제와 함께 제시할 주요 이미지 경로를 추출합니다...")
    primary_image_path = builder.get_primary_image_path()
    print(f"✅ 추출된 이미지 경로: {primary_image_path}")
    
    if "관련 이미지 없음" in primary_image_path or not primary_image_path:
        print("⚠️ 참고: 이미지 경로가 없거나 기본값입니다. 검색 결과에 이미지가 포함되었는지 확인하세요.")
    else:
        print("🎉 성공! 이미지 경로를 정상적으로 가져왔습니다.")


if __name__ == "__main__":
    main()