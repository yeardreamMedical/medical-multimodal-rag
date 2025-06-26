# -*- coding: utf-8 -*-
"""
데이터 탐색 및 시각화 전문가용 GUI 애플리케이션
Description:
    이 애플리케이션은 두 개의 jsonl 데이터셋을 입력받아, 텍스트 통계,
    어휘 다양성, 임베딩 기반 의미 공간 분석을 수행하고 그 결과를 시각화합니다.
    모든 분석 과정은 체계적으로 자동화되어 있으며, 결과는 타임스탬프 폴더에 저장됩니다.
Author:
    데이터과학 및 파이썬 전문가 (100k USD/day)
Date:
    2024-06-17
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

# --- 데이터 분석 및 시각화 라이브러리 ---
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg') # GUI 백엔드 충돌 방지
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.manifold import TSNE
from sklearn.metrics.pairwise import cosine_similarity
from konlpy.tag import Okt
from wordcloud import WordCloud

# 한국어 폰트 설정 (맑은 고딕)
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False


# --- 1. 분석 로직 함수 (Core Analysis Functions) ---

def analyze_text_length(df: pd.DataFrame, dataset_name: str) -> pd.DataFrame:
    """데이터프레임의 'question' 필드에 대한 텍스트 길이 통계를 계산합니다."""
    logging.info(f"[{dataset_name}] 텍스트 길이 분석 시작...")
    # 글자 수
    df['char_count'] = df['question'].str.len()
    # 단어 수 (공백 기준)
    df['word_count'] = df['question'].str.split().str.len()
    # 문장 수 (마침표, 물음표, 느낌표 기준)
    df['sentence_count'] = df['question'].str.count(r'[.?!]') + 1
    
    stats = df[['char_count', 'word_count', 'sentence_count']].describe().round(2)
    logging.info(f"[{dataset_name}] 텍스트 길이 기술 통계:\n{stats}")
    return df

def visualize_length_distributions(df1: pd.DataFrame, df2: pd.DataFrame, output_dir: Path):
    """두 데이터셋의 텍스트 길이 분포를 시각화합니다."""
    logging.info("텍스트 길이 분포 시각화 생성 중...")
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    metrics = ['char_count', 'word_count', 'sentence_count']
    titles = ['글자 수 분포', '단어 수 분포', '문장 수 분포']

    for ax, metric, title in zip(axes, metrics, titles):
        sns.histplot(df1[metric], ax=ax, color='skyblue', label='Dataset 1', kde=True, stat="density")
        sns.histplot(df2[metric], ax=ax, color='salmon', label='Dataset 2', kde=True, stat="density")
        ax.set_title(title)
        ax.legend()
    
    plt.tight_layout()
    save_path = output_dir / "1_text_length_distribution.png"
    plt.savefig(save_path)
    plt.close()
    logging.info(f"시각화 저장 완료: {save_path}")

def analyze_lexical_diversity(df: pd.DataFrame, okt: Okt, dataset_name: str) -> dict:
    """데이터셋의 어휘 다양성(TTR) 및 상위 키워드(TF-IDF)를 분석합니다."""
    logging.info(f"[{dataset_name}] 어휘 다양성 및 키워드 분석 시작...")
    
    # 토큰화 (시간이 걸릴 수 있음)
    tokens = [okt.morphs(text) for text in df['question']]
    
    # TTR 및 고유 어휘 수 계산
    all_tokens = [token for sublist in tokens for token in sublist]
    total_tokens = len(all_tokens)
    unique_tokens = len(set(all_tokens))
    ttr = unique_tokens / total_tokens if total_tokens > 0 else 0
    
    results = {
        'total_tokens': total_tokens,
        'unique_tokens': unique_tokens,
        'ttr': ttr
    }
    logging.info(f"[{dataset_name}] 어휘 분석 결과: {results}")
    
    # TF-IDF 계산
    corpus = [' '.join(token_list) for token_list in tokens]
    vectorizer = TfidfVectorizer(max_features=2000, min_df=5, max_df=0.5)
    tfidf_matrix = vectorizer.fit_transform(corpus)
    
    feature_names = vectorizer.get_feature_names_out()
    scores = tfidf_matrix.sum(axis=0).A1
    tfidf_scores = dict(zip(feature_names, scores))
    
    # 상위 30개 키워드
    top_keywords = sorted(tfidf_scores.items(), key=lambda x: x[1], reverse=True)[:30]
    results['top_keywords'] = top_keywords
    results['tfidf_scores_for_wc'] = tfidf_scores

    logging.info(f"[{dataset_name}] 상위 키워드 (TF-IDF):\n{top_keywords[:10]}")
    return results

def visualize_lexical_analysis(res1: dict, res2: dict, output_dir: Path):
    """어휘 분석 결과를 막대그래프와 워드클라우드로 시각화합니다."""
    logging.info("어휘 분석 결과 시각화 생성 중...")
    
    # 1. 정량 지표 비교 (막대그래프)
    metrics_df = pd.DataFrame({
        'Dataset 1': [res1['unique_tokens'], res1['ttr']],
        'Dataset 2': [res2['unique_tokens'], res2['ttr']]
    }, index=['고유 어휘 수', '어휘 다양성 (TTR)'])
    
    fig, ax = plt.subplots(1, 2, figsize=(12, 5))
    metrics_df.loc[['고유 어휘 수']].plot(kind='bar', ax=ax[0], title='고유 어휘 수 비교', rot=0)
    metrics_df.loc[['어휘 다양성 (TTR)']].plot(kind='bar', ax=ax[1], title='어휘 다양성(TTR) 비교', rot=0)
    plt.tight_layout()
    save_path = output_dir / "2a_lexical_metrics_comparison.png"
    plt.savefig(save_path)
    plt.close()
    
    # 2. 워드 클라우드
    font_path = "c:/Windows/Fonts/malgun.ttf" # 윈도우 맑은 고딕 기준
    wc = WordCloud(width=800, height=400, background_color='white', font_path=font_path)
    
    fig, axes = plt.subplots(2, 1, figsize=(12, 10))
    wc.generate_from_frequencies(res1['tfidf_scores_for_wc'])
    axes[0].imshow(wc, interpolation='bilinear')
    axes[0].set_title('Dataset 1 핵심 키워드 (Word Cloud)')
    axes[0].axis('off')
    
    wc.generate_from_frequencies(res2['tfidf_scores_for_wc'])
    axes[1].imshow(wc, interpolation='bilinear')
    axes[1].set_title('Dataset 2 핵심 키워드 (Word Cloud)')
    axes[1].axis('off')
    
    plt.tight_layout()
    save_path = output_dir / "2b_keyword_wordclouds.png"
    plt.savefig(save_path)
    plt.close()
    logging.info(f"시각화 저장 완료: {save_path}")

def analyze_cosine_similarity(df1: pd.DataFrame, df2: pd.DataFrame, output_dir: Path):
    """데이터셋 내/간 코사인 유사도 분포를 분석하고 시각화합니다."""
    logging.info("코사인 유사도 분석 시작...")
    
    emb1 = np.array(df1['embedding'].tolist())
    emb2 = np.array(df2['embedding'].tolist())
    
    # 데이터셋 1 내부 유사도
    sim_intra1 = cosine_similarity(emb1)
    intra1_scores = sim_intra1[np.triu_indices(len(emb1), k=1)]
    
    # 데이터셋 2 내부 유사도
    sim_intra2 = cosine_similarity(emb2)
    intra2_scores = sim_intra2[np.triu_indices(len(emb2), k=1)]
    
    # 데이터셋 간 유사도
    sim_inter = cosine_similarity(emb1, emb2)
    inter_scores = sim_inter.flatten()
    
    # 시각화
    sim_df = pd.concat([
        pd.DataFrame({'score': intra1_scores, 'type': 'Intra-Dataset 1'}),
        pd.DataFrame({'score': intra2_scores, 'type': 'Intra-Dataset 2'}),
        pd.DataFrame({'score': inter_scores, 'type': 'Inter-Dataset'})
    ])
    
    plt.figure(figsize=(10, 6))
    sns.violinplot(data=sim_df, x='type', y='score')
    plt.title('질문 임베딩 코사인 유사도 분포')
    plt.ylabel('Cosine Similarity')
    plt.xlabel('비교 유형')
    
    save_path = output_dir / "3_cosine_similarity_distribution.png"
    plt.savefig(save_path)
    plt.close()
    logging.info(f"시각화 저장 완료: {save_path}")
    
def visualize_t_sne(df1: pd.DataFrame, df2: pd.DataFrame, output_dir: Path):
    """t-SNE를 사용하여 임베딩 공간을 2D로 시각화합니다."""
    logging.info("t-SNE 분석 시작... (시간이 다소 소요될 수 있습니다)")
    
    emb1 = np.array(df1['embedding'].tolist())
    emb2 = np.array(df2['embedding'].tolist())
    
    # 데이터셋 크기를 줄여서 t-SNE 실행 (샘플링)
    sample_size = min(2000, len(emb1), len(emb2))
    idx1 = np.random.choice(len(emb1), sample_size, replace=False)
    idx2 = np.random.choice(len(emb2), sample_size, replace=False)
    
    embeddings = np.vstack([emb1[idx1], emb2[idx2]])
    labels = np.array([0] * sample_size + [1] * sample_size)
    
    tsne = TSNE(n_components=2, perplexity=30, n_iter=1000, random_state=42, verbose=1)
    tsne_results = tsne.fit_transform(embeddings)
    
    tsne_df = pd.DataFrame({
        'x': tsne_results[:, 0],
        'y': tsne_results[:, 1],
        'label': ['Dataset 1' if l == 0 else 'Dataset 2' for l in labels]
    })
    
    plt.figure(figsize=(10, 8))
    sns.scatterplot(data=tsne_df, x='x', y='y', hue='label', alpha=0.7, s=20)
    plt.title('t-SNE 기반 질문 임베딩 2D 시각화')
    plt.xlabel('t-SNE Dimension 1')
    plt.ylabel('t-SNE Dimension 2')
    plt.legend(title='Dataset')
    
    save_path = output_dir / "4_tSNE_embedding_visualization.png"
    plt.savefig(save_path)
    plt.close()
    logging.info(f"시각화 저장 완료: {save_path}")


# --- 2. GUI 애플리케이션 클래스 ---
class DataExplorerApp(ttk.Frame):
    def __init__(self, master):
        super().__init__(master, padding="10")
        self.master = master
        self.master.title("데이터 탐색 및 분석 도구 v1.0")
        self.master.geometry("800x600")
        self.grid(sticky=(tk.W, tk.E, tk.N, tk.S))
        self.master.columnconfigure(0, weight=1); self.master.rowconfigure(0, weight=1)

        self.file1_path = tk.StringVar()
        self.file2_path = tk.StringVar()
        self.status_text = tk.StringVar(value="대기 중. 분석할 두 개의 .jsonl 파일을 선택해주세요.")
        self.log_queue = queue.Queue()

        self.okt = Okt() # Okt 초기화

        self.create_widgets()
        self.setup_logging()
        self.master.after(100, self.process_queue)

    def create_widgets(self):
        # 파일 선택 프레임
        file_frame = ttk.LabelFrame(self, text="파일 선택", padding="10")
        file_frame.grid(row=0, column=0, sticky=(tk.W, tk.E), padx=5, pady=5)
        file_frame.columnconfigure(1, weight=1)

        ttk.Button(file_frame, text="Dataset 1 (.jsonl) 선택", command=lambda: self.select_file(self.file1_path)).grid(row=0, column=0, padx=5, pady=5)
        ttk.Entry(file_frame, textvariable=self.file1_path, state='readonly').grid(row=0, column=1, sticky=(tk.W, tk.E), padx=5)

        ttk.Button(file_frame, text="Dataset 2 (.jsonl) 선택", command=lambda: self.select_file(self.file2_path)).grid(row=1, column=0, padx=5, pady=5)
        ttk.Entry(file_frame, textvariable=self.file2_path, state='readonly').grid(row=1, column=1, sticky=(tk.W, tk.E), padx=5)

        # 실행 프레임
        action_frame = ttk.Frame(self, padding="10")
        action_frame.grid(row=1, column=0, sticky=tk.E, padx=5, pady=5)
        self.analyze_button = ttk.Button(action_frame, text="분석 시작", command=self.start_analysis, state='disabled')
        self.analyze_button.pack()

        # 상태 및 로그 프레임
        status_frame = ttk.LabelFrame(self, text="진행 상태 및 로그", padding="10")
        status_frame.grid(row=2, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), padx=5, pady=5)
        self.rowconfigure(2, weight=1)
        status_frame.columnconfigure(0, weight=1); status_frame.rowconfigure(1, weight=1)

        ttk.Label(status_frame, textvariable=self.status_text).grid(row=0, column=0, sticky=(tk.W, tk.E))
        self.log_text = scrolledtext.ScrolledText(status_frame, state='disabled', wrap=tk.WORD, height=15)
        self.log_text.grid(row=1, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), pady=5)

    def select_file(self, path_var):
        filepath = filedialog.askopenfilename(title="jsonl 파일을 선택하세요", filetypes=[("JSON Lines", "*.jsonl"), ("All files", "*.*")])
        if filepath:
            path_var.set(filepath)
            if self.file1_path.get() and self.file2_path.get():
                self.analyze_button.config(state='normal')

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

    def start_analysis(self):
        self.analyze_button.config(state='disabled')
        self.log_text.config(state='normal'); self.log_text.delete(1.0, tk.END); self.log_text.config(state='disabled')
        
        threading.Thread(target=self.run_full_analysis, daemon=True).start()

    def run_full_analysis(self):
        try:
            self.status_text.set("분석을 시작합니다...")
            output_dir = Path(f"analysis_결과_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}")
            output_dir.mkdir(exist_ok=True)
            logging.info(f"결과 저장 폴더: {output_dir}")

            # 데이터 로딩
            self.status_text.set("Dataset 1 로딩 중...")
            df1 = pd.read_json(self.file1_path.get(), lines=True)
            logging.info(f"Dataset 1 로드 완료: {len(df1)}개 항목")

            self.status_text.set("Dataset 2 로딩 중...")
            df2 = pd.read_json(self.file2_path.get(), lines=True)
            logging.info(f"Dataset 2 로드 완료: {len(df2)}개 항목")

            # Phase 1: 텍스트 분석
            self.status_text.set("Phase 1: 텍스트 길이 분석 중...")
            df1_len = analyze_text_length(df1.copy(), "Dataset 1")
            df2_len = analyze_text_length(df2.copy(), "Dataset 2")
            visualize_length_distributions(df1_len, df2_len, output_dir)

            self.status_text.set("Phase 1: 어휘 다양성 분석 중...")
            lex_res1 = analyze_lexical_diversity(df1, self.okt, "Dataset 1")
            lex_res2 = analyze_lexical_diversity(df2, self.okt, "Dataset 2")
            visualize_lexical_analysis(lex_res1, lex_res2, output_dir)

            # Phase 2: 임베딩 분석
            self.status_text.set("Phase 2: 코사인 유사도 분석 중...")
            analyze_cosine_similarity(df1, df2, output_dir)
            
            self.status_text.set("Phase 2: t-SNE 2D 시각화 중... (시간 소요)")
            visualize_t_sne(df1, df2, output_dir)

            self.status_text.set(f"🎉 모든 분석 완료! 결과가 '{output_dir}' 폴더에 저장되었습니다.")
            logging.info("모든 분석이 성공적으로 완료되었습니다.")
        
        except Exception as e:
            logging.critical(f"분석 중 치명적인 오류 발생: {e}", exc_info=True)
            self.status_text.set(f"오류 발생: {e}")
        finally:
            self.analyze_button.config(state='normal')

class QueueHandler(logging.Handler):
    def __init__(self, log_queue): super().__init__(); self.log_queue = log_queue
    def emit(self, record): self.log_queue.put(record)

# --- 3. 애플리케이션 실행 ---
if __name__ == "__main__":
    root = tk.Tk()
    app = DataExplorerApp(master=root)
    app.mainloop()
