# context/context_builder.py
from typing import Dict, Any

class LLMContextBuilder:
    """
    SearchEngine의 검색 결과를 LLM이 이해하기 쉬운
    구조화된 컨텍스트로 변환합니다.
    """
    def __init__(self, search_result: Dict[str, Any]):
        """
        LLMContextBuilder를 초기화합니다.

        Args:
            search_result (Dict[str, Any]): SearchEngine.search_text()의 반환값.
        """
        if not isinstance(search_result, dict) or "error" in search_result:
            raise ValueError("유효하지 않거나 오류를 포함한 검색 결과입니다.")
        self.result = search_result

    def build_context_for_llm(self) -> str:
        """
        문제 생성 또는 문제 풀이를 위한 통합 컨텍스트를 생성합니다.
        Markdown 형식을 사용하여 LLM의 이해도를 높입니다.
        """
        # 필수 정보 추출
        diagnosis = self.result.get('korean_diagnosis', '진단명 없음')
        confidence = self.result.get('confidence', '낮음')
        text_content = self.result.get('text_content', '관련 텍스트 정보 없음.')
        image_info = self.result.get('image_info', '관련 이미지 정보 없음.')
        all_diseases = self.result.get('all_diseases', [])

        # 컨텍스트 문자열 생성
        context_parts = [
            "# [의료 정보 컨텍스트]",
            "이 정보는 RAG 시스템을 통해 검증된 소스로부터 추출되었습니다.",
            "",
            "## 1. 핵심 진단 정보",
            f"- **주요 추정 진단명**: {diagnosis}",
            f"- **검색 신뢰도**: {confidence.upper()}",
            "",
            "## 2. 관련 의학 지식 (텍스트 기반)",
            f"요약: {text_content}",
            "",
            "## 3. 관련 이미지 정보 (이미지 기반)",
            f"요약: {image_info}",
            "",
            "## 4. 고려된 기타 질병",
            f"- {', '.join(all_diseases) if all_diseases else '없음'}"
        ]

        return "\n".join(context_parts)

    def get_primary_image_path(self) -> str:
        """
        문제 생성 시 함께 제시할 가장 관련성 높은 이미지의 경로를 반환합니다.
        """
        image_results = self.result.get('image_info', {}) # search_result의 image_info가 아닌 image_results를 봐야함
        
        # search_engine.py의 search_text -> image_searcher.search_by_diseases -> 반환값인 image_results를 봐야함
        # 하지만 현재 context_builder는 search_text의 최종 결과인 context만 받으므로, image_results를 직접 접근할 수 없음.
        # search_engine.py의 ContextBuilder.create_context 에서 'images' 키로 이미지 리스트를 전달하도록 수정 필요.
        # 우선은 임시 방편으로 image_info 딕셔너리 내에 path가 있다고 가정.
        # TODO: SearchEngine의 ContextBuilder가 'images' 리스트를 context에 포함하도록 수정할 것.

        images = self.result.get('images', [])
        if images and isinstance(images, list) and 'image_path' in images[0]:
            return images[0]['image_path']
        
        return "관련 이미지 없음"