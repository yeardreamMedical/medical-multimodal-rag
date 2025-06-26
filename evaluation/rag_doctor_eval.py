#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
## KorMedMCQA 의사 국가고시 데이터셋 RAG 기반 평가 스크립트 (수정 완료)

이 스크립트는 'doctor-test.csv' 문제 데이터셋에 대해 
'search_engine.py'의 벡터 검색 기능을 활용한 RAG(Retrieval-Augmented Generation) 
성능을 평가합니다.

**주요 기능:**
1.  **RAG 파이프라인**: 각 문제에 대해 먼저 벡터 DB에서 관련 정보를 검색하고, 
    검색된 내용을 바탕으로 LLM이 문제를 풀도록 합니다.
2.  **동적 프롬프트 생성**: 검색된 컨텍스트의 유무에 따라 LLM에게 다른 지침을 제공하는
    프롬프트를 동적으로 생성합니다.
3.  **성능 평가 및 로깅**: RAG 파이프라인의 전체 정확도를 측정하고, 틀린 문제에 대한
    상세 정보(사용한 컨텍스트, 모델 답변 등)를 JSONL 파일로 저장하여 분석을 돕습니다.

**실행 방법:**
- 프로젝트 루트(medical-multimodal-rag) 폴더에서 실행:
  `python -m evaluation.rag_doctor_eval`
- 상위 5개 문제만 평가:
  `python -m evaluation.rag_doctor_eval --top_n 5`
"""
# --- 기본 라이브러리 임포트 ---
import os, re, argparse, json, time, pathlib, sys
from typing import Dict, Optional 

# --- 외부 라이브러리 임포트 ---
from tqdm import tqdm
import pandas as pd
import google.generativeai as genai
from dotenv import load_dotenv

### 경로 설정 시작 ###
# 현재 파일의 위치를 기준으로 프로젝트 루트 디렉토리 경로를 계산합니다.
project_root = pathlib.Path(__file__).resolve().parents[1]

# sys.path에 프로젝트 루트를 추가하여 다른 폴더의 모듈을 임포트할 수 있도록 합니다.
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))
### 경로 설정 끝 ###


### 사용자 정의 모듈 임포트 수정 시작 ###
# 모든 임포트 구문을 스크립트 상단으로 이동하고, 전체 경로를 명시합니다.
from search.search_engine import SearchEngine
from evaluation.doctor_choice_eval import extract_choice, gemini_completion, validate_data
### 사용자 정의 모듈 임포트 수정 끝 ###


# --- 환경 설정 ---
load_dotenv(dotenv_path=project_root / '.env')
api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    raise ValueError("GEMINI_API_KEY가 .env 파일에 설정되지 않았습니다.")
genai.configure(api_key=api_key)    

def build_rag_prompt(question: str, choices: Dict[str, str], context: Optional[str]) -> str:
    """
    RAG 파이프라인을 위한 동적 프롬프트를 생성합니다.
    """
    choices_str = "\n".join([f"{k}. {v}" for k, v in choices.items()])

    if context and context != "관련 텍스트 정보 없음":
        instruction = """당신은 의학 전문가입니다. 주어진 [참고 정보]를 검토하고, 만약 이 정보가 [문제] 해결에 직접적이고 정확하게 관련이 있다면 이를 최우선으로 활용하여 답을 찾으세요.
하지만, [참고 정보]가 문제와 관련이 없거나, 사실과 다르거나, 오해의 소지가 있다고 판단되면, 이 정보를 무시하고 당신의 의학 전문 지식에만 의존하여 가장 적절한 답을 고르세요."""
        context_section = f"---\n[참고 정보]\n{context}\n---"
    else:
        instruction = "당신은 의학 전문가입니다. 당신의 의학 지식을 총동원하여 아래 [문제]에 가장 적절한 답을 [보기]에서 하나만 고르세요."
        context_section = ""

    prompt = f"""[지침]
{instruction}
답변은 반드시 A, B, C, D, E 중 하나의 알파벳으로만 출력해야 하며, 다른 어떤 부가 설명도 추가하지 마세요.

{context_section}

[문제]
{question}

[보기]
{choices_str}

[정답]:"""
    
    return prompt.strip()

def generate_search_query(question: str, model_name: str = "gemini-1.5-pro") -> str:
    """
    LLM을 사용하여 문제에서 검색에 가장 적합한 핵심 키워드나 질문을 생성합니다.
    """
    prompt = f"""다음 의학 문제에서 벡터 검색을 위한 가장 핵심적인 키워드나 짧은 질문을 1~2개 생성해줘. 법률이나 규정과 관련된 문제라면 관련 법률명을 포함해줘. 다른 설명 없이 키워드나 질문만 출력해줘.

[문제]
{question}

[검색용 쿼리]:"""
    
    try:
        # gemini_completion 함수는 이미 doctor_choice_eval.py에 존재하므로 재사용합니다.
        search_query = gemini_completion(prompt, model_name)
        # LLM이 불필요한 문구를 추가할 경우를 대비해 간단히 처리
        search_query = search_query.replace("*", "").replace("`", "").strip()
        print(f"    - 생성된 검색 쿼리: {search_query}")
        return search_query
    except Exception as e:
        print(f"    - 쿼리 생성 중 오류 발생: {e}")
        return question # 실패 시 원본 질문 사용
    


# --- 쿼리 확장이 적용된 evaluate_with_rag 함수 (전체 수정 버전) ---
def evaluate_with_rag(top_n: int = None, model_name: str = "gemini-1.5-pro"):
    print("--- RAG 평가 시스템 시작 (쿼리 확장 적용됨) ---")
    
    print("\n[1/5] 검색 엔진 초기화 중...")
    try:
        search_engine = SearchEngine()
        print("✅ 검색 엔진 초기화 완료")
    except Exception as e:
        print(f"❌ 검색 엔진 초기화 실패: {e}")
        return

    print("\n[2/5] 평가 데이터 로드 중...")
    try:
        data_path = pathlib.Path(__file__).parent
        csv_path = data_path / "doctor-test.csv"
        df = pd.read_csv(csv_path)
        df = validate_data(df)
        if len(df) == 0:
            print("❌ 평가를 진행할 유효한 데이터가 없습니다.")
            return
        test_data = df.to_dict('records')
        print(f"✅ 데이터 로드 및 검증 완료: {len(test_data)}개 문제")
    except FileNotFoundError:
        print(f"❌ 오류: 평가 파일 '{csv_path}'을(를) 찾을 수 없습니다.")
        return
    except Exception as e:
        print(f"❌ 데이터 로드 중 오류 발생: {e}")
        return

    if top_n:
        test_data = test_data[:top_n]
        print(f"ℹ️  --top_n={top_n} 설정에 따라 상위 {len(test_data)}개 문제만 평가합니다.")

    print(f"\n[3/5] RAG 기반 평가 루프 시작 (모델: {model_name})")
    
    total, correct = 0, 0
    wrong_log = []

    for item in tqdm(test_data, desc="RAG 평가 진행 중"):
        total += 1
        q_text = item["question"]
        choices_dict = {letter: str(item[letter]).strip() for letter in ['A', 'B', 'C', 'D', 'E']}
        gt = chr(65 + item["answer"] - 1)

        # --- KeyError 수정된 부분 ---
        # item['id'] 대신 카운터 변수 total을 사용합니다.
        print(f"\n[문제 {total}]")
        print("  - 쿼리 확장 중...")
        search_query = generate_search_query(q_text, model_name)
        print(f"  - 생성된 검색 쿼리: '{search_query}'")
        
        search_result = search_engine.search_text(query=search_query, top_k=3)
        retrieved_context = search_result.get("text_content")
        print(f"  - 검색된 정보: {'있음' if retrieved_context else '없음'}")
        
        rag_prompt = build_rag_prompt(q_text, choices_dict, retrieved_context)
        gen_out = gemini_completion(rag_prompt, model_name)
        pred = extract_choice(gen_out)

        is_ok = (pred == gt)
        if is_ok:
            correct += 1
            print(f"  - 결과: 정답 (예측: {pred}, 정답: {gt})")
        else:
            print(f"  - 결과: 오답 (예측: {pred}, 정답: {gt})")
            
            # --- KeyError 수정된 부분 ---
            # 오답 로그 저장 시에도 동적으로 생성된 ID를 사용합니다.
            wrong_log.append({
                "id": f"item_{total}", 
                "question": q_text,
                "choices": choices_dict,
                "generated_query": search_query,
                "retrieved_context": retrieved_context,
                "gt": gt,
                "pred": pred,
                "model_output": gen_out
            })

        time.sleep(1.0)
    
    print("\n[4/5] 평가 완료. 결과 집계 중...")

    if total > 0:
        acc = correct / total * 100
        print(f"\n🎯 최종 정확도: {acc:.2f}% (총 {total} 문제 중 {correct} 문제 정답)")
    
    if wrong_log:
        out_dir = pathlib.Path(__file__).parent / "evaluation_results"
        out_dir.mkdir(exist_ok=True)
        model_short_name = model_name.split('/')[-1]
        file_path = out_dir / f"rag_wrong_answers_query_expansion_{model_short_name}.jsonl"
        with open(file_path, "w", encoding="utf-8") as f:
            for row in wrong_log:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(f"💾 오답 노트 저장 완료: {file_path} ({len(wrong_log)}개)")

    print("\n[5/5] RAG 평가 시스템 종료")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RAG 기반으로 KorMedMCQA-doctor 데이터셋을 평가합니다.")
    parser.add_argument("--model", type=str, default="gemini-1.5-pro", help="평가에 사용할 모델 이름 (예: gemini-1.5-pro, gemini-1.5-flash)")
    parser.add_argument("--top_n", type=int, help="평가할 문제의 수를 제한합니다 (예: --top_n 10).")
    args = parser.parse_args()
    
    current_dir = pathlib.Path(__file__).parent
    required_files = ["doctor-test.csv", "doctor_choice_eval.py"]
    missing_files = [f for f in required_files if not (current_dir / f).exists()]
    
    if missing_files:
        print("❌ 오류: 평가에 필요한 파일이 'evaluation' 폴더에 없습니다.")
        for f in missing_files:
            print(f"  - {f} 파일을 찾을 수 없습니다.")
    else:
        evaluate_with_rag(top_n=args.top_n, model_name=args.model)