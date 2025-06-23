#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
## KorMedMCQA 의사 국가고시 데이터셋 평가 스크립트 (Gemini 1.5 Pro)

이 스크립트는 'KorMedMCQA' 데이터셋의 'doctor-test.csv' 파일을 사용하여, 
Google의 Gemini 1.5 Pro 모델의 의료 지식 및 추론 능력을 평가합니다.

**주요 기능:**
1.  **Few-shot 프롬프팅**: 5개의 예시 문제(Few-shot)를 프롬프트에 포함하여 모델이 문제 형식을 학습하고 더 정확한 답변을 생성하도록 유도합니다.
2.  **견고한 답안 추출**: 정규표현식을 사용하여 모델의 다양한 출력 형태로부터 정답 선택지(A-E)를 안정적으로 추출합니다.
3.  **데이터 검증**: 평가 시작 전, 데이터셋의 정답 인덱스와 선택지 데이터의 유효성을 검사하여 평가 오류를 최소화합니다.
4.  **상세 로깅**: 평가 과정에서 모델이 틀린 문제들을 `evaluation_results/` 디렉터리에 'doctor_wrong_answers_fixed.jsonl' 파일로 저장하여, 오답 분석을 용이하게 합니다.
5.  **디버깅 모드**: `--debug` 플래그를 통해 단일 문제에 대한 프롬프트와 모델 응답을 직접 확인할 수 있어, 프롬프트 엔지니어링 및 테스트에 유용합니다.

**실행 방법:**
- 전체 평가: `python medical-multimodal-rag/evaluation/doctor_choice_eval.py`
- 상위 10개 문제만 평가: `python medical-multimodal-rag/evaluation/doctor_choice_eval.py --top_n 10`
- 디버그 모드 실행: `python medical-multimodal-rag/evaluation/doctor_choice_eval.py --debug`
"""
# --- 기본 라이브러리 임포트 ---
import os, re, argparse, json, time, pathlib
from typing import List, Dict

# --- 외부 라이브러리 임포트 ---
from tqdm import tqdm  # 진행 상황을 시각적으로 보여주는 프로그레스 바 라이브러리
import pandas as pd  # 데이터 조작 및 분석을 위한 라이브러리
import google.generativeai as genai  # Google Gemini API 라이브러리
from dotenv import load_dotenv  # .env 파일에서 환경 변수를 로드하는 라이브러리

# --- 환경 설정 및 상수 정의 ---

# .env 파일에 저장된 환경 변수를 로드합니다. (주로 API 키 관리용)
load_dotenv()
# Gemini API 키를 환경 변수에서 가져와 설정합니다.
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
# 사용할 Gemini 모델의 이름을 지정합니다.
MODEL_NAME = "gemini-1.5-pro"

# Few-shot 프롬프팅에 사용될 5개의 예제입니다. (KorMedMCQA 논문에서 사용된 예제와 동일)
# 모델에게 문제 형식(질문, 보기, 정답)을 명확히 알려주어 성능을 높이는 역할을 합니다.
FEW_SHOT_EXAMPLES: List[Dict] = [
    # 예제 1
    {
        "question": "13개월남아가, 형이 하루 전 홍역으로 진단을 받아서 병원에 왔다. 평소건강하며 현재특별한증상은없다. 맥박100회/분, 호흡 28회/분, 체온36.7°C이다. 예방접종기록은다음과같다. 조치는? BCG 1회B형간염백신3회DTaP-폴리오-Hib혼합백신3회폐렴구균단백결합백신4회일본뇌염백신1회",
        "choices": {"A": "경과관찰", "B": "비타민A", "C": "단클론항체", "D": "면역글로불린", "E": "홍역-볼거리-풍진백신"},
        "answer": "E"
    },
    # 예제 2
    {
        "question": "55세여자가 2일전부터 열이난다며 병원에 왔다. 며칠 전부터 평소보다소변을자주본다고한다. 1개월전에도 비슷한증상이있었다고한다. 혈압 130/80 mmHg, 맥박105회/분, 호흡 18회/분, 체온38.5°C이다. 왼쪽 갈비척추각에 압통이있다. 소변검사 결과는다음과같다. 가능성이높은원인미생물은? 백혈구10～20/고배율시야, 적혈구0～5/고배율시야",
        "choices": {"A": "대장균(Escherichia coli)", "B": "창자알균(Enterococcus faecalis)", "C": "프로테우스미라빌리스(Proteus mirabilis)", "D": "우레아플라스마우레알리티쿰(Ureaplasma urealyticum)", "E": "스타필로코쿠스사프로피티쿠스(Staphylococcus saprophyticus)"},
        "answer": "A"
    },
    # 예제 3
    {
        "question": " 8세 남아가 쉽게 짜증을 내고 친구를 자주 때려 병원에 왔다. 학교에서는 선생님의 지시를 따르지못하고집에서 자주울고떼를 썼으며 밤늦게까지안 자서엄마가감당하기힘들었다. 5세때한차례경련을했고이후로괜찮아 치료를 받지않았다. 지능지수는49이다. 치료는?",
        "choices": {"A": "졸피뎀", "B": "도네페질", "C": "리스페리돈", "D": "아고멜라틴", "E": "알프라졸람"},
        "answer": "C"
    },
    # 예제 4
    {
        "question": "50세 남자가 10일전부터 서서히 배가불러온다고병원에 왔다. 누나와형이간질환으로약을복용중이라고한다. 혈압 110/70 mmHg, 맥박80회/분, 호흡 18회/분, 체온36.7°C이다. 결막은창백하고공막에 황달이있다. 앞가슴에 거미혈관종이보인다. 복부는팽만하고이동둔탁음이있다. 복부에 압통이나 반동압통은없다. 양쪽 정강뼈앞 오목부종이 있다. 검사 결과는다음과같다. 복부컴퓨터단층촬영사진이다. 치료는? 혈액: 백혈구4,700/mm^3, 혈색소10.4 g/dL,혈소판59,000/mm^3 프로트롬빈시간(INR) 2.1, 알부민 1.8 g/dL,총빌리루빈 4.2 mg/dL, 아스파르테이트아미노전달효소69U/L 알라닌아미노전달효소18 U/L HBsAg (+), anti-HBs (-), anti-HCV (-) 알파태아단백질3.9 ng/mL (참고치, <8.5) 복수:총단백질1.2 g/dL, 알부민 0.4 g/dL,백혈구100/mm^3,아데노신탈아미노효소(ADA) 9 U/L",
        "choices": {"A": "리팜핀", "B": "세포탁심", "C": "락툴로오스", "D": "스피로놀락톤", "E": "메트로니다졸"},
        "answer": "D"
    },
    # 예제 5
    {
        "question": "35세 여자가 2주전부터 배가 불러온다며 병원에 왔다. 2개월동안 체중이4 kg 줄었다고한다. 혈압 110/70 mmHg,맥박97회/분, 호흡 21회/분, 체온38.1°C이다. 의식은명료하다. 배는팽만하고이동둔탁음이있다. 배에 가벼운압통은 있으나 반동압통은없다. 검사 결과는다음과같다. 원인은?혈액: 혈색소11.2g/dL, 백혈구4,200/mm^3, 혈소판169,000/mm^3 총단백질6.0 g/dL, 알부민 3.3 g/dL, 총빌리루빈 1.2 mg/dL, 아스파르테이트아미노전달효소34 U/L 알라닌아미노전달효소21 U/L 아밀라아제61 U/L, 리파제30 U/L 복수: 백혈구1,500/mm^3 (중성구5%, 림프구90%) 단백질3.1 g/dL, 알부민 2.5 g/dL,아데노신탈아미노효소(ADA) 85 U/L 소변: 단백질(-), 적혈구0～2/고배율시야",
        "choices": {"A": "간경화증", "B": "복막결핵", "C": "콩팥증후군", "D": "급성이자염", "E": "복막암종증"},
        "answer": "B"
    },
]

def extract_choice(output: str) -> str:
    """
    LLM의 다양한 출력 텍스트에서 정답 선택지(A-E)를 추출합니다.
    Gemini 모델의 응답 형식(`**A.` 등)에 더 잘 대응하도록 수정되었습니다.

    KorMedMCQA 논문의 Figure 3에서 제시된 순서와 유사하게,
    가장 명확한 패턴부터 덜 명확한 패턴 순으로 정규표현식을 적용하여
    추출의 정확성과 안정성을 높입니다.

    Args:
        output (str): LLM이 생성한 원본 텍스트.

    Returns:
        str: 추출된 정답 선택지 (A, B, C, D, E). 추출에 실패하면 빈 문자열을 반환.
    """
    # 패턴 1: Gemini 모델이 자주 사용하는 마크다운 강조 형식 (`**A.`, `**A**`)을 최우선으로 처리
    # 이런 형식의 답변은 보통 설명의 시작 부분에 위치하므로, 가장 먼저 찾은 결과를 정답으로 간주합니다.
    matches = re.findall(r'\*{2}\s*([ABCDE])', output, re.IGNORECASE)
    if matches:
        return matches[0].upper()

    # 패턴 2: "정답: A", "정답 A" 와 같이 명시적인 키워드가 있는 경우
    # '정답' 키워드가 명확히 있으므로, 가장 마지막에 나온 것을 최종 답변으로 신뢰할 수 있습니다.
    matches = re.findall(r'정답[:\s]*([ABCDE])', output, re.IGNORECASE)
    if matches:
        return matches[-1].upper()
    
    # 패턴 3: "정답은 A입니다" 와 같은 문장 형식
    matches = re.findall(r'정답은\s*([ABCDE])\s*입니다', output, re.IGNORECASE)
    if matches:
        return matches[-1].upper()
    
    # 패턴 4: "A." 로 시작하는 문장 형식.
    # 설명문에 포함된 'A.', 'B.' 등을 오탐지할 수 있으므로, 가장 먼저 나온 결과만 사용합니다.
    matches = re.findall(r'\b([ABCDE])\.', output, re.IGNORECASE)
    if matches:
        return matches[0].upper()
    
    # 패턴 5: A, B, C, D, E 알파벳 단독. 가장 위험한 패턴이므로 최후에 사용하며,
    # 역시 오탐지를 피하기 위해 가장 먼저 나온 결과만 사용합니다.
    matches = re.findall(r'\b([ABCDE])\b', output, re.IGNORECASE)
    if matches:
        return matches[0].upper()
    
    # 모든 패턴에서 정답을 찾지 못한 경우, 빈 문자열을 반환합니다.
    return ''

def build_prompt(question: str, choices: Dict[str, str]) -> str:
    """
    하나의 문제와 선택지 딕셔너리를 LLM에게 전달할 프롬프트 형식으로 구성합니다.

    Args:
        question (str): 문제 텍스트.
        choices (Dict[str, str]): 선택지 (예: {"A": "선택지1", "B": "선택지2"}).

    Returns:
        str: 형식화된 프롬프트 문자열.
    """
    # 선택지 딕셔너리를 "A. 내용", "B. 내용" ... 형태의 문자열로 변환합니다.
    choices_str = "\n".join([f"{k}. {v}" for k, v in choices.items()])
    return f"질문: {question}\n\n보기:\n{choices_str}"

def make_5shot_prompt(target_q: str, target_choices: Dict[str, str]) -> str:
    """
    5개의 Few-shot 예제와 실제 문제를 결합하여 최종 프롬프트를 생성합니다.
    모델이 오해하지 않도록 예제 부분과 실제 문제 부분을 명확히 분리하고,
    하나의 정답만 출력하도록 지시문을 강화합니다.

    Args:
        target_q (str): 모델이 풀어야 할 실제 문제의 텍스트.
        target_choices (Dict[str, str]): 실제 문제의 선택지.

    Returns:
        str: Gemini API에 전달될 완전한 프롬프트 문자열.
    """
    chunks = ["다음은 문제와 정답의 예시입니다.", "---"]
    
    # FEW_SHOT_EXAMPLES에 정의된 예제들을 순회하며 프롬프트에 추가합니다.
    for i, ex in enumerate(FEW_SHOT_EXAMPLES, 1):
        chunks.append(f"<예제 {i}>")
        chunks.append(build_prompt(ex["question"], ex["choices"]))
        chunks.append(f"정답: {ex['answer']}")
        chunks.append("")
    
    # 예제가 끝났음을 명확히 알리고, 실제 과제를 제시합니다.
    chunks.append("---")
    chunks.append("이제, 아래 문제에 대한 정답을 보기에서 골라 A, B, C, D, E 중 하나의 알파벳으로만 답해주십시오.")
    chunks.append("다른 어떤 설명도 추가하지 마십시오.")
    chunks.append("\n<풀어야 할 문제>")
    # 모든 예제가 추가된 후, 모델이 풀어야 할 실제 문제를 추가합니다.
    chunks.append(build_prompt(target_q, target_choices))
    chunks.append("정답:")
    
    # 모든 조각(chunk)들을 개행 문자로 합쳐 하나의 완성된 프롬프트로 만듭니다.
    return "\n".join(chunks)

def gemini_completion(prompt: str) -> str:
    """
    주어진 프롬프트를 Gemini API로 보내고, 모델의 응답을 반환합니다.

    Args:
        prompt (str): `make_5shot_prompt`에서 생성된 최종 프롬프트.

    Returns:
        str: 모델이 생성한 텍스트. API 호출 중 오류가 발생하면 빈 문자열을 반환.
    """
    try:
        # 지정된 모델을 사용하여 API 요청 객체를 생성합니다.
        model = genai.GenerativeModel(MODEL_NAME)
        # 프롬프트와 생성 옵션을 전달하여 콘텐츠 생성을 요청합니다.
        response = model.generate_content(
            prompt,
            generation_config={
                "temperature": 0.0,  # 온도를 0으로 설정하여, 가장 확률이 높은 단어를 선택하게 함 (일관성 있는 답변 유도)
                "top_p": 1.0,        # Nucleus sampling, 1.0은 비활성화.
                "max_output_tokens": 1024, # 최대 생성 토큰 수
            }
        )
        # 응답 텍스트의 앞뒤 공백을 제거하고 반환합니다.
        return response.text.strip()
    except Exception as e:
        # API 호출 중 예외 발생 시, 오류 메시지를 출력하고 빈 문자열을 반환합니다.
        print(f"API 호출 오류: {e}")
        return ""

def debug_single_question():
    """
    단일 질문을 사용하여 프롬프트 생성 및 API 호출 과정을 디버깅합니다.

    전체 데이터셋을 실행하지 않고도 프롬프트 구조, 모델 응답, 답안 추출 로직을
    빠르게 테스트할 수 있습니다.
    """
    # 테스트용 질문과 선택지를 정의합니다.
    test_q = "2개월 남아가 BCG예방접종 1개월 뒤 주사 부위에 이상반응이 생겨서 예방접종을 실시한 소아청소년과의원을 찾아왔다."
    test_choices = {
        "A": "대한의사협회장",
        "B": "보건복지부장관", 
        "C": "남아 소재지 관할 보건소장",
        "D": "남아 소재지 관할 시장 ∙ 군수 ∙ 구청장",
        "E": "남아 소재지 관할 시 ∙ 도지사"
    }
    
    # 5-shot 프롬프트를 생성합니다.
    prompt = make_5shot_prompt(test_q, test_choices)
    print("=== PROMPT (last 500 chars) ===")
    print(prompt[-500:])  # 프롬프트가 너무 길기 때문에 마지막 500자만 출력하여 확인합니다.
    print("\n=== RESPONSE ===")
    
    # Gemini API를 호출하여 응답을 받습니다.
    response = gemini_completion(prompt)
    print(response)
    
    # 응답에서 정답을 추출합니다.
    extracted = extract_choice(response)
    print(f"\n=== EXTRACTED: '{extracted}' ===")
    print(f"GT (Ground Truth) should be: D")

def validate_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    평가에 사용될 DataFrame의 데이터 유효성을 검사하고 정제합니다.

    잘못된 정답 인덱스나 비어있는 선택지 등 오류를 유발할 수 있는 데이터를
    미리 제거하여 평가의 신뢰도를 높입니다.

    Args:
        df (pd.DataFrame): 원본 데이터프레임.

    Returns:
        pd.DataFrame: 유효성 검사가 완료된 데이터프레임.
    """
    print("=== 데이터 검증 시작 ===")
    
    original_len = len(df)
    print(f"원본 데이터 행 수: {original_len}개")
    
    # 정답 인덱스('answer' 컬럼)가 1~5 사이의 유효한 값을 갖는지 확인합니다.
    # CSV 파일에서 정답은 1,2,3,4,5로 표기되어 있습니다.
    valid_mask = (df['answer'] >= 1) & (df['answer'] <= 5)
    invalid_count = len(df) - valid_mask.sum()
    
    if invalid_count > 0:
        print(f"오류: 유효하지 않은 'answer' 값 발견: {invalid_count}개")
        print("해당 데이터:")
        # 유효하지 않은 데이터를 순회하며 인덱스와 값을 출력합니다.
        for idx, row in df[~valid_mask].iterrows():
            print(f"  - 행 {idx}: answer = {row['answer']}")
        
        # 유효한 데이터만 남기고, 잘못된 데이터는 제거합니다.
        df = df[valid_mask].copy() # .copy()를 사용하여 SettingWithCopyWarning 방지
    
    # A, B, C, D, E 선택지 컬럼에 null(빈 값)이 있는지 확인합니다.
    choice_cols = ['A', 'B', 'C', 'D', 'E']
    for col in choice_cols:
        null_mask = df[col].isnull()
        if null_mask.any(): # null 값이 하나라도 있으면
            print(f"오류: 선택지 '{col}'에 null 값 발견: {null_mask.sum()}개")
            df = df[~null_mask].copy() # null 값이 포함된 행 제거
    
    print(f"정제 후 데이터 행 수: {len(df)}개")
    print(f"제거된 데이터 행 수: {original_len - len(df)}개")
    print("=== 데이터 검증 완료 ===")
    
    return df

def evaluate(top_n: int = None, debug: bool = False):
    """
    메인 평가 로직을 실행합니다.

    Args:
        top_n (int, optional): 평가할 문제의 최대 개수. None이면 전체 데이터셋을 평가합니다.
        debug (bool, optional): True이면 디버그 모드(`debug_single_question`)를 실행합니다.
    """
    
    # 디버그 모드가 활성화된 경우, 단일 질문 테스트 함수만 실행하고 종료합니다.
    if debug:
        debug_single_question()
        return
    
    try:
        # 현재 스크립트 파일의 위치를 기준으로 'doctor-test.csv' 파일 경로를 설정합니다.
        csv_path = pathlib.Path(__file__).parent / "doctor-test.csv"
        df = pd.read_csv(csv_path)
    except FileNotFoundError:
        print(f"오류: 평가 파일 '{csv_path}'을(를) 찾을 수 없습니다.")
        return
    
    # 데이터 유효성 검사 및 정제를 수행합니다.
    df = validate_data(df)
    
    if len(df) == 0:
        print("오류: 평가를 진행할 유효한 데이터가 없습니다.")
        return
    
    # 데이터프레임을 딕셔너리 리스트 형태로 변환하여 사용하기 쉽게 만듭니다.
    test_data = df.to_dict('records')
    
    # top_n 인자가 주어졌으면, 데이터셋을 해당 개수만큼만 잘라냅니다.
    if top_n:
        test_data = test_data[:top_n]
    
    print(f"\n=== 평가 시작 (총 {len(test_data)}개 문제) ===")
    
    total, correct = 0, 0
    wrong_log = []  # 틀린 문제들을 기록할 리스트

    # tqdm을 사용하여 프로그레스 바와 함께 각 문제를 순회합니다.
    for item in tqdm(test_data, desc="평가 진행 중"):
        q_text = item["question"]
        
        # 선택지 딕셔너리를 구성합니다. (예: {'A': '내용', 'B': '내용', ...})
        choices_dict = {letter: str(item[letter]).strip() for letter in ['A', 'B', 'C', 'D', 'E']}
        
        # CSV의 정답 인덱스(1~5)를 알파벳(A~E)으로 변환합니다.
        # chr(65)는 'A'입니다. (1 -> A, 2 -> B, ...)
        answer_idx = item["answer"]
        gt = chr(65 + answer_idx - 1)

        # 5-shot 프롬프트를 생성하고 API를 호출하여 모델의 답변을 받습니다.
        full_prompt = make_5shot_prompt(q_text, choices_dict)
        gen_out = gemini_completion(full_prompt)
        # 모델의 답변에서 최종 선택지를 추출합니다.
        pred = extract_choice(gen_out)

        # 예측과 정답을 비교합니다.
        is_ok = (pred == gt)
        correct += int(is_ok)  # 맞았으면 1, 틀렸으면 0을 더함
        total += 1

        # 각 문제의 채점 결과를 즉시 출력합니다.
        status = "✓ (정답)" if is_ok else "✗ (오답)"
        print(f"{status} - 문제 {total}: 정답={gt}, 예측={pred}")
        
        if not is_ok:
            # 틀린 경우, 상세 정보를 로그에 기록합니다.
            wrong_log.append({
                "id": f"item_{total}",
                "question": q_text[:100] + "...", # 질문이 너무 길 수 있으므로 100자만 저장
                "choices": choices_dict,
                "gt": gt,
                "pred": pred,
                "model_output": gen_out[:300] + "..." # 모델 출력도 300자만 저장
            })

        # API rate limit을 준수하기 위해 1초 대기합니다.
        time.sleep(1.0)

    # 최종 결과 출력
    if total > 0:
        acc = correct / total * 100
        print(f"\n🎯 최종 정확도: {acc:.2f}% (총 {total} 문제 중 {correct} 문제 정답)")
    
    # 틀린 문제 로그를 파일로 저장
    if wrong_log:
        out_dir = pathlib.Path("evaluation_results")
        out_dir.mkdir(exist_ok=True) # evaluation_results 디렉토리가 없으면 생성
        file_path = out_dir / "doctor_wrong_answers_fixed.jsonl"
        with open(file_path, "w", encoding="utf-8") as f:
            for row in wrong_log:
                # json.dumps를 사용하여 딕셔너리를 JSON 문자열로 변환하여 파일에 씁니다.
                # ensure_ascii=False는 한글이 깨지지 않게 하기 위함입니다.
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(f"오답 노트 저장 완료: {file_path} ({len(wrong_log)}개)")

if __name__ == "__main__":
    # 커맨드라인 인자를 파싱하기 위한 설정
    parser = argparse.ArgumentParser(description="Gemini 1.5 Pro 모델로 KorMedMCQA-doctor 데이터셋을 평가합니다.")
    parser.add_argument("--top_n", type=int, help="평가할 문제의 수를 제한합니다 (예: --top_n 10).")
    parser.add_argument("--debug", action="store_true", help="단일 문제로 디버깅 모드를 실행합니다.")
    args = parser.parse_args()
    
    # 파싱된 인자들을 evaluate 함수에 전달하여 평가를 시작합니다.
    evaluate(top_n=args.top_n, debug=args.debug)