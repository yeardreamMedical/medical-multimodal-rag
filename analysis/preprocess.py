# -*- coding: utf-8 -*-
"""
데이터과학 전문가용 데이터 전처리 및 임베딩 GUI 애플리케이션
Description:
    이 애플리케이션은 tkinter를 사용하여 사용자 친화적인 GUI를 제공합니다.
    OpenAI API를 효율적으로 활용하기 위해 배치 처리(Batch Processing)를 구현하고,
    유의미한 정보를 보존하는 정교한 전처리 로직을 적용했습니다.
Author:
    데이터과학 및 파이썬 전문가 (100k USD/day)
Date:
    2024-06-15 (Rev. 2024-06-17, 전처리 로직 정교화)
"""
import os
import re
import json
import logging
import threading
import queue
from pathlib import Path
from datetime import datetime
import tkinter as tk
from tkinter import ttk, filedialog, scrolledtext
from typing import List, Dict, Optional

# 서드파티 라이브러리 임포트
from dotenv import load_dotenv
from openai import OpenAI, APIError
from datasets import load_dataset
from bs4 import BeautifulSoup

# --- 1. 상수 및 코어 로직 (Constants & Core Logic) ---

EMBEDDING_MODEL = 'text-embedding-3-small'
LOCAL_OUTPUT_DIR_NAME = 'train'
HF_OUTPUT_DIR_NAME = 'hf_train'
OUTPUT_FILENAME_LOCAL = 'train_embedded.jsonl'
OUTPUT_FILENAME_HF = 'hf_train_embedded.jsonl'
BATCH_SIZE = 1024

def preprocess_question_text(text: str) -> str:
    """
    질문 텍스트에 대해 정교화된 전처리 규칙을 순서대로 적용합니다.
    """
    if not isinstance(text, str): return ""

    # 1. HTML 태그 제거
    text = BeautifulSoup(text, "html.parser").get_text()
    
    # 2. 보기 번호 제거 (문자열 전체에서 " 1) ", " 2. " 등 패턴을 공백으로 치환)
    text = re.sub(r'\s+\d\)\s+', ' ', text)
    text = re.sub(r'\s+\d\.\s+', ' ', text)
    
    # 3. 안내성 괄호만 제거 (예: (그림 참조), (보기)) / 영문, 숫자 등이 포함된 괄호는 보존
    text = re.sub(r'\((?:그림|사진|도표|표|보기|참조|단,)\s*[^)]*\)', '', text)
    
    # 4. 줄바꿈, 탭 등 제어 문자 제거 -> 공백으로 대체
    text = re.sub(r'[\n\r\t]', ' ', text)
    
    # 5. 중복 공백 제거 및 양 끝 공백 제거
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

def get_embeddings_batch(client: OpenAI, texts_batch: List[str], model: str) -> Optional[List[List[float]]]:
    """
    주어진 텍스트 배치에 대해 OpenAI 임베딩을 생성하고 소수점 4자리로 반올림합니다.
    """
    try:
        response = client.embeddings.create(input=texts_batch, model=model)
        batched_embeddings = [d.embedding for d in response.data]
        rounded_embeddings = [[round(val, 4) for val in emb] for emb in batched_embeddings]
        return rounded_embeddings
    except APIError as e:
        logging.error(f"OpenAI API 호출 중 에러 발생 (배치 처리): {e}")
        return None
    except Exception as e:
        logging.error(f"임베딩 배치 생성 중 예기치 않은 에러 발생: {e}")
        return None

# --- 2. GUI 애플리케이션 클래스 (이하 코드는 이전과 동일) ---
class Application(ttk.Frame):
    def __init__(self, master):
        super().__init__(master, padding="10")
        self.master = master
        self.master.title("데이터 전처리 및 임베딩 도구 (전처리 정교화 v3)")
        self.master.geometry("800x650")
        self.grid(sticky=(tk.W, tk.E, tk.N, tk.S))
        self.master.columnconfigure(0, weight=1)
        self.master.rowconfigure(0, weight=1)

        # 변수 초기화
        self.input_dir_path = tk.StringVar()
        self.api_key_status = tk.StringVar(value="API 키 확인 중...")
        self.status_text = tk.StringVar(value="대기 중")
        self.client = None
        self.log_queue = queue.Queue()

        self.create_widgets()
        self.setup_logging()
        self.master.after(100, self.process_queue)
        self.load_api_key()

    def create_widgets(self):
        """GUI의 모든 위젯을 생성하고 배치합니다."""
        # 컨트롤 프레임
        control_frame = ttk.LabelFrame(self, text="작업 설정", padding="10")
        control_frame.grid(row=0, column=0, sticky=(tk.W, tk.E), padx=5, pady=5)
        control_frame.columnconfigure(1, weight=1)

        ttk.Label(control_frame, text="OpenAI API 키:").grid(row=0, column=0, sticky=tk.W, padx=5, pady=2)
        ttk.Label(control_frame, textvariable=self.api_key_status, foreground="blue").grid(row=0, column=1, sticky=tk.W, padx=5, pady=2)

        ttk.Label(control_frame, text="로컬 JSON 폴더:").grid(row=1, column=0, sticky=tk.W, padx=5, pady=2)
        self.folder_entry = ttk.Entry(control_frame, textvariable=self.input_dir_path, state='readonly', width=70)
        self.folder_entry.grid(row=1, column=1, sticky=(tk.W, tk.E), padx=5, pady=2)
        self.select_button = ttk.Button(control_frame, text="폴더 선택", command=self.select_directory)
        self.select_button.grid(row=1, column=2, sticky=tk.E, padx=5, pady=2)
        
        ttk.Label(control_frame, text="배치 크기:").grid(row=2, column=0, sticky=tk.W, padx=5, pady=2)
        ttk.Label(control_frame, text=f"{BATCH_SIZE}개 항목 / API 호출").grid(row=2, column=1, sticky=tk.W, padx=5, pady=2)


        # 실행 프레임
        action_frame = ttk.Frame(self, padding="10")
        action_frame.grid(row=1, column=0, sticky=(tk.W, tk.E), padx=5, pady=5)
        action_frame.columnconfigure(0, weight=1)

        self.start_button = ttk.Button(action_frame, text="처리 시작", command=self.start_processing, state='disabled')
        self.start_button.grid(row=0, column=0, ipady=5, ipadx=10)

        # 진행 상태 프레임
        progress_frame = ttk.LabelFrame(self, text="진행 상태", padding="10")
        progress_frame.grid(row=2, column=0, sticky=(tk.W, tk.E), padx=5, pady=5)
        progress_frame.columnconfigure(0, weight=1)

        ttk.Label(progress_frame, textvariable=self.status_text).grid(row=0, column=0, sticky=(tk.W, tk.E))
        self.progress_bar = ttk.Progressbar(progress_frame, orient='horizontal', mode='determinate')
        self.progress_bar.grid(row=1, column=0, sticky=(tk.W, tk.E), pady=5)

        # 로그 프레임
        log_frame = ttk.LabelFrame(self, text="실시간 로그", padding="10")
        log_frame.grid(row=3, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), padx=5, pady=5)
        self.rowconfigure(3, weight=1)
        log_frame.columnconfigure(0, weight=1); log_frame.rowconfigure(0, weight=1)

        self.log_text = scrolledtext.ScrolledText(log_frame, state='disabled', wrap=tk.WORD, height=15, bg="#f0f0f0", fg="black")
        self.log_text.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

    def setup_logging(self):
        log_handler = QueueHandler(self.log_queue)
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - [%(levelname)s] - %(message)s', datefmt='%H:%M:%S', handlers=[log_handler])

    def process_queue(self):
        try:
            while True: self.display_log(self.log_queue.get_nowait())
        except queue.Empty: pass
        finally: self.master.after(100, self.process_queue)

    def display_log(self, record):
        msg = logging.getLogger().handlers[0].format(record)
        self.log_text.configure(state='normal')
        self.log_text.insert(tk.END, msg + '\n')
        self.log_text.configure(state='disabled')
        self.log_text.yview(tk.END)

    def load_api_key(self):
        logging.info("'.env' 파일에서 환경 변수를 로드합니다.")
        load_dotenv()
        api_key = os.getenv("OPENAI_API_KEY")
        if api_key:
            self.client = OpenAI(api_key=api_key)
            self.api_key_status.set(f"로드 성공 (키 일부: ...{api_key[-4:]})")
            logging.info("OpenAI API 키 로드 성공.")
        else:
            self.api_key_status.set("로드 실패! .env 파일을 확인해주세요.")
            logging.error("환경 변수 'OPENAI_API_KEY'를 찾을 수 없습니다.")
            
    def select_directory(self):
        path = filedialog.askdirectory(title="JSON 파일이 있는 폴더를 선택하세요")
        if path:
            self.input_dir_path.set(path)
            logging.info(f"사용자 선택 폴더: {path}")
            if self.client: self.start_button.config(state='normal')

    def start_processing(self):
        if not self.input_dir_path.get():
            logging.warning("작업을 시작하기 전에 폴더를 선택해주세요.")
            return
        self.start_button.config(state='disabled')
        self.select_button.config(state='disabled')
        self.progress_bar['value'] = 0
        self.log_text.configure(state='normal'); self.log_text.delete(1.0, tk.END); self.log_text.configure(state='disabled')

        self.processing_thread = threading.Thread(target=self.run_processing_logic, args=(Path(self.input_dir_path.get()),))
        self.processing_thread.daemon = True
        self.processing_thread.start()

    def _process_and_write_batch(self, batch_items: List[Dict], batch_texts: List[str], f_out) -> int:
        if not batch_items:
            return 0
        embeddings = get_embeddings_batch(self.client, batch_texts, EMBEDDING_MODEL)
        if embeddings and len(embeddings) == len(batch_items):
            for item, embedding in zip(batch_items, embeddings):
                item["embedding"] = embedding
                f_out.write(json.dumps(item, ensure_ascii=False) + '\n')
            return len(batch_items)
        else:
            logging.error(f"{len(batch_items)}개 아이템의 배치 처리에 실패했습니다. 이 배치를 건너뜁니다.")
            return 0

    def run_processing_logic(self, input_dir: Path):
        try:
            now_str = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
            base_output_dir = Path(now_str)
            local_output_dir = base_output_dir / LOCAL_OUTPUT_DIR_NAME
            hf_output_dir = base_output_dir / HF_OUTPUT_DIR_NAME
            local_output_dir.mkdir(exist_ok=True, parents=True)
            hf_output_dir.mkdir(exist_ok=True, parents=True)
            logging.info(f"결과물 저장 폴더 생성 완료: '{base_output_dir}'")

            self.process_local_files_threaded(input_dir, local_output_dir)
            self.process_hf_dataset_threaded(hf_output_dir)

            logging.info("🎉 모든 작업이 성공적으로 완료되었습니다.")
            self.status_text.set("모든 작업 완료!")

        except Exception as e:
            logging.critical(f"처리 중 치명적인 오류 발생: {e}", exc_info=True)
            self.status_text.set(f"오류 발생: {e}")
        finally:
            self.start_button.config(state='normal')
            self.select_button.config(state='normal')

    def process_local_files_threaded(self, input_dir, output_dir):
        self.status_text.set("로컬 JSON 파일 목록을 수집하는 중...")
        logging.info(f"'{input_dir}' 폴더에서 로컬 JSON 파일 처리를 시작합니다.")
        json_files = list(input_dir.rglob('*.json'))
        if not json_files:
            logging.warning(f"'{input_dir}'에서 JSON 파일을 찾지 못했습니다.")
            return

        total_items_to_process = 0
        all_valid_items = []
        logging.info("처리할 전체 아이템 수를 계산합니다...")
        for file_path in json_files:
            try:
                with open(file_path, 'r', encoding='utf-8-sig') as f:
                    data = json.load(f)
                items = data if isinstance(data, list) else [data]
                for item in items:
                    if item.get("q_type") == 1 and "question" in item:
                        all_valid_items.append(item)
                        total_items_to_process += 1
            except Exception as e:
                logging.error(f"파일 '{file_path}' 사전 읽기 중 오류: {e}")
        logging.info(f"총 {total_items_to_process}개의 유효한 로컬 아이템을 처리합니다.")
        
        output_path = output_dir / OUTPUT_FILENAME_LOCAL
        processed_count = 0
        batch_items, batch_texts = [], []

        with open(output_path, 'w', encoding='utf-8') as f_out:
            for item in all_valid_items:
                preprocessed_text = preprocess_question_text(item["question"])
                item["preprocessed_question"] = preprocessed_text
                batch_items.append(item)
                batch_texts.append(preprocessed_text)
                
                if len(batch_items) >= BATCH_SIZE:
                    self.status_text.set(f"로컬 파일: {len(batch_items)}개 아이템 배치 처리 중...")
                    num_processed = self._process_and_write_batch(batch_items, batch_texts, f_out)
                    processed_count += num_processed
                    self.progress_bar['value'] = processed_count / total_items_to_process * 100
                    batch_items.clear(); batch_texts.clear()
            
            if batch_items:
                self.status_text.set(f"로컬 파일: 마지막 {len(batch_items)}개 아이템 배치 처리 중...")
                num_processed = self._process_and_write_batch(batch_items, batch_texts, f_out)
                processed_count += num_processed

        self.progress_bar['value'] = 100
        logging.info(f"로컬 JSON 파일 처리 완료. 총 {processed_count}개 항목을 '{output_path}'에 저장했습니다.")

    def process_hf_dataset_threaded(self, output_dir):
        self.status_text.set("Hugging Face 데이터셋 다운로드 중...")
        logging.info("Hugging Face 데이터셋 'sean0042/KorMedMCQA' (doctor) 로드를 시작합니다.")
        dataset = load_dataset("sean0042/KorMedMCQA", "doctor", split='train')
        total_items = len(dataset)
        logging.info(f"데이터셋 로드 완료. 총 {total_items}개의 데이터를 처리합니다.")
        
        output_path = output_dir / OUTPUT_FILENAME_HF
        processed_count = 0
        batch_items, batch_texts = [], []

        with open(output_path, 'w', encoding='utf-8') as f_out:
            for i, item_dict in enumerate(dataset):
                if "question" in item_dict:
                    preprocessed_text = preprocess_question_text(item_dict["question"])
                    item_dict["preprocessed_question"] = preprocessed_text
                    batch_items.append(item_dict)
                    batch_texts.append(preprocessed_text)

                if len(batch_items) >= BATCH_SIZE or (i + 1) == total_items:
                    if not batch_items: continue
                    self.status_text.set(f"HF 데이터: {processed_count+1}-{processed_count+len(batch_items)}/{total_items} 처리 중...")
                    num_processed = self._process_and_write_batch(batch_items, batch_texts, f_out)
                    processed_count += num_processed
                    self.progress_bar['value'] = processed_count / total_items * 100
                    batch_items.clear(); batch_texts.clear()
        
        logging.info(f"Hugging Face 데이터셋 처리 완료. 총 {processed_count}개 항목을 '{output_path}'에 저장했습니다.")

class QueueHandler(logging.Handler):
    def __init__(self, log_queue):
        super().__init__()
        self.log_queue = log_queue

    def emit(self, record):
        self.log_queue.put(record)

# --- 3. 애플리케이션 실행 (Application Execution) ---
if __name__ == "__main__":
    root = tk.Tk()
    app = Application(master=root)
    app.mainloop()
