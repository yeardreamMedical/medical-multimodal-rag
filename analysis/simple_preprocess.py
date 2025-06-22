import re
from bs4 import BeautifulSoup

def preprocess_question_text(text: str) -> str:
    """질문 텍스트를 분석용으로 간단 전처리한다 (GUI 의존성 없음)."""
    if not isinstance(text, str):
        return ""

    # 1. HTML 태그 제거
    text = BeautifulSoup(text, "html.parser").get_text()

    # 2. 보기 번호(1), 2.  같은 패턴 제거
    text = re.sub(r"\s+\d[).]\s+", " ", text)

    # 3. 안내성 한국어 괄호 제거 – 그림, 사진, 도표 등
    text = re.sub(r"\((?:그림|사진|도표|표|보기|참조|단,)\s*[^)]*\)", "", text)

    # 4. 제어 문자 → 공백
    text = re.sub(r"[\n\r\t]", " ", text)

    # 5. 중복 공백 정리
    text = re.sub(r"\s+", " ", text).strip()

    return text 