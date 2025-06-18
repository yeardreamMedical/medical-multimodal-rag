# context/prompt_engineer.py

class PromptEngineer:
    """
    의료 문제 생성 및 풀이를 위한 LLM 프롬프트를 생성합니다.
    """
    def __init__(self, context: str, image_path: str = None):
        """
        PromptEngineer를 초기화합니다.

        Args:
            context (str): LLMContextBuilder가 생성한 구조화된 컨텍스트.
            image_path (str, optional): 문제와 함께 제시될 이미지 경로. Defaults to None.
        """
        if not context or not isinstance(context, str):
            raise ValueError("유효한 컨텍스트 문자열을 제공해야 합니다.")
        self.context = context
        self.image_path = image_path

    def create_question_generation_prompt(self) -> str:
        """
        의사 국가고시 스타일의 5지선다형 문제 생성 프롬프트를 반환합니다.
        """
        prompt = f"""
# [지시문]
당신은 한국 의사 국가고시 출제위원입니다. 제공된 [의료 정보 컨텍스트]를 바탕으로, 임상 추론 능력을 평가할 수 있는 고품질의 5지선다형 객관식 문제 1개를 생성하세요.

# [규칙]
1.  **문제 형식**: 실제 의사 국가고시처럼 환자의 증례(나이, 성별, 주소, 현병력 등)를 먼저 제시하고, 그에 대한 질문을 하는 형식을 사용하세요.
2.  **내용 기반**: 반드시 주어진 [의료 정보 컨텍스트] 내의 사실만을 활용하여 문제를 만들어야 합니다. 컨텍스트에 없는 내용을 추측하거나 꾸며내지 마세요.
3.  **이미지 연계**: 이 문제에는 `{self.image_path}` 이미지가 함께 제시될 예정입니다. 질문은 이 이미지를 해석해야만 풀 수 있도록 구성하는 것이 좋습니다.
4.  **보기 구성**: 5개의 보기(①, ②, ③, ④, ⑤)를 만드세요. 정답은 1개여야 하며, 오답은 정답과 관련이 있어 보이지만 명백히 틀린 내용이어야 합니다.
5.  **출력 형식**: 아래 JSON 형식을 엄격히 준수하여 다른 설명 없이 JSON 객체만 출력하세요.
    - `question`: 문제 본문 (환자 증례 포함)
    - `options`: 보기 5개를 담은 문자열 리스트 (예: ["보기 1", "보기 2", ...])
    - `answer`: 정답에 해당하는 보기의 인덱스 (0부터 시작하는 숫자)
    - `explanation`: 왜 이것이 정답인지, 다른 보기들은 왜 오답인지 상세히 설명.

---
[의료 정보 컨텍스트]
{self.context}
---

# [출력 (JSON 형식만 허용)]
"""
        return prompt.strip()

    def create_question_solving_prompt(self, question: str, options: list) -> str:
        """
        주어진 문제를 풀기 위한 프롬프트를 반환합니다.
        """
        # 보기 목록을 번호와 함께 포맷팅
        formatted_options = "\n".join([f"  {i+1}. {opt}" for i, opt in enumerate(options)])
        
        prompt = f"""
# [지시문]
당신은 주어진 정보를 바탕으로 추론하는 뛰어난 영상의학과 전문의입니다. 제공된 [의료 정보 컨텍스트]와 [문제]를 참고하여 가장 적절한 답을 선택하세요.

# [규칙]
1.  **핵심 근거**: 당신의 판단은 반드시 [의료 정보 컨텍스트]를 최우선 근거로 삼아야 합니다. 당신의 사전 지식보다 컨텍스트의 내용을 신뢰하세요.
2.  **정답 형식**: 가장 가능성이 높은 보기의 '번호' 하나만을 다른 설명 없이 숫자로만 답하세요. (예: 3)

---
[의료 정보 컨텍스트]
{self.context}
---

# [문제]
{question}

# [보기]
{formatted_options}

# [정답 (숫자만 출력)]
"""
        return prompt.strip()