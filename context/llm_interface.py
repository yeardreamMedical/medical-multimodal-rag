# context/llm_interface.py
import os
import google.generativeai as genai
from dotenv import load_dotenv
import json

# .env 파일에서 환경 변수 로드
load_dotenv()

class GeminiInterface:
    """
    Google Gemini 1.5 Pro 모델과의 상호작용을 담당합니다.
    """
    def __init__(self):
        """
        API 키를 설정하고 Gemini 모델을 초기화합니다.
        """
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("환경 변수 파일(.env)에 GEMINI_API_KEY를 설정해주세요.")
        
        genai.configure(api_key=api_key)
        
        # 최신 모델 사용
        self.model = genai.GenerativeModel('gemini-1.5-pro-latest')

    def generate_question_json(self, prompt: str) -> dict:
        """
        주어진 프롬프트를 Gemini 모델로 보내고, 결과를 JSON(dict)으로 파싱하여 반환합니다.
        
        Args:
            prompt (str): PromptEngineer가 생성한 최종 프롬프트.

        Returns:
            dict: 생성된 질문, 보기, 정답, 해설을 담은 딕셔너리.
                  파싱 실패 시 에러 정보를 담은 딕셔너리를 반환합니다.
        """
        try:
            print("  - Gemini API에 프롬프트 전송 중...")
            response = self.model.generate_content(prompt)
            raw_text = response.text
            print("  - Gemini API로부터 응답 수신 완료.")

            # LLM 응답에서 JSON 부분만 정확히 추출 (```json ... ``` 처리)
            if '```json' in raw_text:
                json_text = raw_text.split('```json')[1].split('```')[0].strip()
            else:
                json_text = raw_text.strip()
            
            # JSON 텍스트를 파이썬 딕셔너리로 파싱
            parsed_json = json.loads(json_text)
            return parsed_json

        except json.JSONDecodeError:
            print(f"❌ JSON 파싱 실패! LLM 응답:\n{raw_text}")
            return {"error": "JSON 파싱 실패", "raw_response": raw_text}
        except Exception as e:
            print(f"❌ Gemini API 호출 중 오류 발생: {e}")
            return {"error": f"API 호출 오류: {e}"}