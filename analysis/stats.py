import tkinter as tk
from tkinter import filedialog, messagebox
import json
import os
import re
import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize # <<<<<<<<<<<<<<<< [수정된 부분] 함수를 직접 사용하기 위해 추가합니다.
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.manifold import TSNE
from wordcloud import WordCloud

# ==============================================================================
# 0. 초기 설정 및 헬퍼 함수
# ==============================================================================

# NLTK 토크나이저 다운로드 (최초 1회 실행)
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    print("NLTK 'punkt' 토크나이저를 찾을 수 없습니다. 다운로드를 시작합니다...")
    nltk.download('punkt')
    nltk.download('punkt_tab')
    print("다운로드 완료.")

# 한글 폰트 설정 (Windows: Malgun Gothic, Mac: AppleGothic)
try:
    plt.rc('font', family='Malgun Gothic')
    plt.rcParams['axes.unicode_minus'] = False 
    FONT_PATH = 'malgun.ttf'
except:
    try:
        plt.rc('font', family='AppleGothic')
        plt.rcParams['axes.unicode_minus'] = False
        FONT_PATH = '/System/Library/Fonts/Supplemental/AppleGothic.ttf'
    except:
        print("경고: 'Malgun Gothic' 또는 'AppleGothic' 폰트를 찾을 수 없습니다. 워드클라우드 및 차트의 한글이 깨질 수 있습니다.")
        FONT_PATH = None

def clean_text(text: str, remove_choice_prefix: bool = False) -> str:
    """
    텍스트에서 불필요한 공백과 줄바꿈을 제거하고,
    필요 시 보기 번호(1)·2) …) 접두어를 삭제합니다.

    Parameters
    ----------
    text : str
        원본 문자열
    remove_choice_prefix : bool, optional
        True 이면 '1) ' ~ '5) ' 과 같은 보기 번호 접두어를 제거합니다.
    """
    if not isinstance(text, str):
        return ""

    # 보기 번호 접두어 제거 (선택)
    if remove_choice_prefix:
        text = re.sub(r'^\s*[1-5][)\.]?\s*', '', text)

    # HTML 태그 제거 (예: <br>, <div> 등)
    text = re.sub(r'<[^>]+>', '', text)
    # 괄호 안 내용 제거 (예: "(그림 참조)", "(보기)")
    text = re.sub(r'\([^)]*\)', '', text)
    # 줄바꿈, 탭 등 제어 문자 제거
    text = re.sub(r'[\r\n\t]', ' ', text)
    # 특수 문자 정리
    text = re.sub(r'[•●▪︎]', ' ', text)
    # 중복 공백 제거 및 양 끝 공백 제거
    text = re.sub(r'\s+', ' ', text).strip()

    return text

# ----------------------------------------------------------------------
# 객관식문제(5지선다) 전용 파서
# ----------------------------------------------------------------------
def parse_json_questions(root_folder: str, status_updater=lambda msg: None):
    """
    여러 폴더에 흩어진 JSON 형식 5지선다 문제를 표준 dict 리스트로 변환합니다.
    유효성 검사:
      1) 보기 5개 필수
      2) answer_idx 가 0~4
    """
    records = []
    if not os.path.isdir(root_folder):
        status_updater(f"오류: 폴더를 찾을 수 없습니다: {root_folder}")
        return records

    json_paths = [
        os.path.join(r, f)
        for r, _, fs in os.walk(root_folder)
        for f in fs if f.endswith('.json')
    ]
    choice_pat = re.compile(r'^\s*([1-5])\)\s*(.*)$')

    for p in json_paths:
        fname = os.path.basename(p)
        try:
            with open(p, 'r', encoding='utf-8-sig') as f:
                obj = json.load(f)
        except Exception as e:
            status_updater(f"오류: {fname} JSON 로드 실패 – {e}")
            continue

        q_raw = obj.get('question', '')
        a_raw = obj.get('answer', '')
        qa_id = obj.get('qa_id')
        domain = obj.get('domain')
        q_type = obj.get('q_type')

        if not q_raw or not a_raw:
            status_updater(f"경고: {fname} – question/answer 필드 누락. 건너뜀.")
            continue

        # question 줄 분리
        stem_lines, choice_lines = [], []
        for ln in q_raw.splitlines():
            if choice_pat.match(ln):
                choice_lines.append(ln)
            else:
                stem_lines.append(ln)

        stem = clean_text(' '.join(stem_lines))

        # 보기 추출
        choices = [''] * 5
        for ln in choice_lines:
            m = choice_pat.match(ln)
            if m:
                idx = int(m.group(1)) - 1
                choices[idx] = clean_text(m.group(2), remove_choice_prefix=False)

        if '' in choices:
            status_updater(f"오류: {fname} – 5개 보기를 모두 찾지 못함. 건너뜀.")
            continue

        # 답안 파싱
        m_ans = choice_pat.match(a_raw)
        if not m_ans:
            status_updater(f"오류: {fname} – answer 필드 형식 불일치. 건너뜀.")
            continue
        answer_idx = int(m_ans.group(1)) - 1
        answer_text = clean_text(m_ans.group(2))

        # 유효성 검사
        if len(choices) != 5:
            status_updater(f"오류: {fname} – 보기 수 5개 아님 ({len(choices)}개).")
            continue
        if not (0 <= answer_idx < 5):
            status_updater(f"오류: {fname} – answer_idx 범위 오류 ({answer_idx}).")
            continue

        records.append({
            'qa_id': qa_id,
            'domain': domain,
            'q_type': q_type,
            'stem': stem,
            'choices': choices,
            'answer_idx': answer_idx,
            'answer_text': answer_text,
            'source_file': fname
        })
    return records


def parse_csv_questions(csv_path: str, status_updater=lambda msg: None):
    """
    CSV 형식 5지선다 문제를 표준 dict 리스트로 변환합니다.
    필수 열: question, A, B, C, D, E, answer
    """
    records = []
    if not os.path.isfile(csv_path):
        status_updater(f"오류: CSV 파일을 찾을 수 없습니다: {csv_path}")
        return records

    try:
        df = pd.read_csv(csv_path, encoding='utf-8-sig')
    except Exception as e:
        status_updater(f"오류: CSV 로드 실패 – {e}")
        return records

    required_cols = ['question', 'A', 'B', 'C', 'D', 'E', 'answer']
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        status_updater(f"오류: CSV 필수 열 누락 – {missing}")
        return records

    for _, row in df.iterrows():
        stem = clean_text(str(row['question']))
        choices = [clean_text(str(row[c]), remove_choice_prefix=True) for c in ['A', 'B', 'C', 'D', 'E']]
        answer_idx = int(row['answer']) - 1

        # 유효성 검사
        if len(choices) != 5:
            status_updater("오류: CSV – 보기 수 5개 아님.")
            continue
        if not (0 <= answer_idx < 5):
            status_updater("오류: CSV – answer_idx 범위 오류.")
            continue

        records.append({
            'qa_id': row.get('qa_id', None),
            'domain': row.get('domain', None),
            'q_type': row.get('q_type', 1),
            'stem': stem,
            'choices': choices,
            'answer_idx': answer_idx,
            'answer_text': choices[answer_idx],
            'source_file': os.path.basename(csv_path)
        })
    return records


def records_to_dataframe(records: list) -> pd.DataFrame:
    """표준 dict 리스트를 DataFrame 으로 변환합니다."""
    return pd.DataFrame(records)


def load_and_parse_json_data(folder_path, status_updater):
    """
    지정된 폴더에서 모든 JSON 파일을 재귀적으로 탐색하고 파싱합니다.
    """
    data_list = []
    if not os.path.isdir(folder_path):
        status_updater(f"오류: 폴더를 찾을 수 없습니다: {folder_path}")
        return []

    file_paths = [os.path.join(root, file) for root, _, files in os.walk(folder_path) for file in files if file.endswith('.json')]
    
    for i, file_path in enumerate(file_paths):
        file_name = os.path.basename(file_path)
        if (i + 1) % 50 == 0:
             status_updater(f"파일 로드 중... ({i+1}/{len(file_paths)})")
        try:
            with open(file_path, 'r', encoding='utf-8-sig') as f: # json encoding 변경
                data = json.load(f)
                if data.get("q_type") != 1: # 객관식 문제로만 비교
                    continue
                text_content = None
                embedding_vector = None

                if "original_text" in data and "embedding" in data:
                    text_content = data.get("original_text")
                    embedding_vector = data.get("embedding")
                elif "text" in data and "embeddings" in data:
                    text_content = data.get("text")
                    embedding_vector = data.get("embeddings")

                if text_content and embedding_vector is not None and len(embedding_vector) > 0:
                    data_list.append({
                        'filename': file_name,
                        'text': clean_text(text_content),
                        'embedding': np.array(embedding_vector, dtype=float).flatten() 
                    })
                else:
                    status_updater(f"경고: {file_name}에서 유효한 텍스트 또는 임베딩을 찾지 못했습니다. 건너뜁니다.")
        except json.JSONDecodeError:
            status_updater(f"오류: {file_name} 파일이 유효한 JSON 형식이 아닙니다.")
        except Exception as e:
            status_updater(f"오류: {file_name} 파일 처리 중 예외 발생: {e}")
    return data_list

# ==============================================================================
# Phase 1: 데이터 기초 특성 분석
# ==============================================================================

def analyze_text_lengths(data_group_1, data_group_2, output_dir, status_updater):
    """1) 텍스트 길이 분포 분석"""
    status_updater("\n[1단계] 텍스트 길이 분포 분석을 시작합니다...")
    
    def get_stats(data, group_name):
        return pd.DataFrame([
            {
                '글자 수': len(d['text']),
                '단어 수': len(word_tokenize(d['text'])),
                '문장 수': len(sent_tokenize(d['text'])),
                '그룹': group_name
            } for d in data if d['text']
        ])

    df_len_g1 = get_stats(data_group_1, '그룹 1')
    df_len_g2 = get_stats(data_group_2, '그룹 2')
    df_lengths = pd.concat([df_len_g1, df_len_g2])

    status_updater("\n[그룹 1] 텍스트 길이 기술 통계량:")
    status_updater(df_len_g1.describe().to_string())
    status_updater("\n[그룹 2] 텍스트 길이 기술 통계량:")
    status_updater(df_len_g2.describe().to_string())

    length_metrics = ['글자 수', '단어 수', '문장 수']
    for metric in length_metrics:
        plt.figure(figsize=(12, 7))
        sns.histplot(data=df_lengths, x=metric, hue='그룹', kde=True, palette='viridis', alpha=0.6, multiple='stack')
        plt.title(f'📊 그룹별 {metric} 분포 비교 (히스토그램)', fontsize=16)
        plt.savefig(os.path.join(output_dir, f'1_length_{metric}_histogram.png'))
        plt.close()

        plt.figure(figsize=(10, 7))
        sns.violinplot(data=df_lengths, x='그룹', y=metric, palette='viridis', inner='quartile')
        plt.title(f'🎻 그룹별 {metric} 분포 비교 (바이올린 플롯)', fontsize=16)
        plt.savefig(os.path.join(output_dir, f'1_length_{metric}_violinplot.png'))
        plt.close()

    status_updater("[1단계] 분석 완료. 결과가 저장되었습니다.")

def analyze_vocabulary_and_keywords(data_group_1, data_group_2, output_dir, status_updater):
    """2) 어휘 다양성 및 핵심 키워드 분석"""
    status_updater("\n[2단계] 어휘 다양성 및 핵심 키워드 분석을 시작합니다...")
    results = {}
    
    for i, (group_data, group_name) in enumerate([(data_group_1, "그룹 1"), (data_group_2, "그룹 2")]):
        all_texts = [d['text'] for d in group_data if d['text']]
        if not all_texts:
            status_updater(f"경고: {group_name}에 분석할 텍스트가 없습니다.")
            continue
            
        all_tokens = [word for text in all_texts for word in word_tokenize(text)]
        unique_words = set(all_tokens)
        ttr = len(unique_words) / len(all_tokens) if all_tokens else 0
        
        results[group_name] = {'고유 어휘 수': len(unique_words), 'TTR': ttr}
        status_updater(f"\n[{group_name}] 어휘 다양성: 고유 어휘 수={len(unique_words)}, TTR={ttr:.4f}")

        try:
            vectorizer = TfidfVectorizer(max_features=2000, tokenizer=word_tokenize, stop_words=None)
            tfidf_matrix = vectorizer.fit_transform(all_texts)
            feature_names = vectorizer.get_feature_names_out()
            avg_tfidf_scores = tfidf_matrix.mean(axis=0).A1
            
            top_indices = avg_tfidf_scores.argsort()[::-1][:20]
            top_keywords = {feature_names[i]: avg_tfidf_scores[i] for i in top_indices}
            
            keywords_df = pd.DataFrame(list(top_keywords.items()), columns=['키워드', '평균 TF-IDF'])
            status_updater(f"[{group_name}] 상위 TF-IDF 키워드:")
            status_updater(keywords_df.to_string(index=False))
            keywords_df.to_csv(os.path.join(output_dir, f'2_{group_name}_keywords.csv'), index=False, encoding='utf-8-sig')

            if top_keywords and FONT_PATH:
                wc = WordCloud(font_path=FONT_PATH, background_color='white', width=800, height=400, collocations=False).generate_from_frequencies(top_keywords)
                plt.figure(figsize=(10, 5))
                plt.imshow(wc, interpolation='bilinear')
                plt.title(f'☁️ [{group_name}] 핵심 키워드 워드클라우드', fontsize=16)
                plt.axis('off')
                plt.savefig(os.path.join(output_dir, f'2_{group_name}_wordcloud.png'))
                plt.close()
        except Exception as e:
            status_updater(f"경고: [{group_name}] TF-IDF 또는 워드클라우드 생성 중 오류: {e}")

    if results:
        df_metrics = pd.DataFrame.from_dict(results, orient='index').reset_index().rename(columns={'index': '그룹'})
        df_melted = df_metrics.melt(id_vars='그룹', var_name='지표', value_name='값')
        plt.figure(figsize=(12, 7))
        sns.barplot(data=df_melted, x='지표', y='값', hue='그룹', palette='coolwarm')
        plt.title('📊 그룹별 어휘 다양성 지표 비교', fontsize=16)
        plt.yscale('log')
        plt.ylabel('값 (로그 스케일)')
        plt.savefig(os.path.join(output_dir, '2_vocab_diversity_barchart.png'))
        plt.close()

    status_updater("[2단계] 분석 완료. 결과가 저장되었습니다.")


# ==============================================================================
# Phase 2: 임베딩 기반 의미 공간 분석
# ==============================================================================

def analyze_cosine_similarity(data_group_1, data_group_2, output_dir, status_updater, sample_size=500):
    """3) 코사인 유사도 분포 분석"""
    status_updater("\n[3단계] 코사인 유사도 분석을 시작합니다...")
    
    embeddings_g1 = np.array([d['embedding'] for d in data_group_1 if 'embedding' in d and len(d['embedding']) > 0])
    embeddings_g2 = np.array([d['embedding'] for d in data_group_2 if 'embedding' in d and len(d['embedding']) > 0])

    if len(embeddings_g1) < 2 or len(embeddings_g2) < 2:
        status_updater("오류: 유사도 분석을 위해 각 그룹에 최소 2개 이상의 유효한 임베딩이 필요합니다.")
        return

    # 데이터가 너무 클 경우 샘플링
    if len(embeddings_g1) > sample_size:
        status_updater(f"그룹 1이 너무 커서({len(embeddings_g1)}개) {sample_size}개로 샘플링합니다.")
        indices = np.random.choice(len(embeddings_g1), sample_size, replace=False)
        embeddings_g1 = embeddings_g1[indices]
    if len(embeddings_g2) > sample_size:
        status_updater(f"그룹 2가 너무 커서({len(embeddings_g2)}개) {sample_size}개로 샘플링합니다.")
        indices = np.random.choice(len(embeddings_g2), sample_size, replace=False)
        embeddings_g2 = embeddings_g2[indices]
        
    sim_data = []

    # 그룹 1 내부 유사도
    sim_matrix_g1 = cosine_similarity(embeddings_g1)
    intra_sim_g1 = sim_matrix_g1[np.triu_indices_from(sim_matrix_g1, k=1)]
    sim_data.extend([(s, '그룹 1 내부') for s in intra_sim_g1])
    
    # 그룹 2 내부 유사도
    sim_matrix_g2 = cosine_similarity(embeddings_g2)
    intra_sim_g2 = sim_matrix_g2[np.triu_indices_from(sim_matrix_g2, k=1)]
    sim_data.extend([(s, '그룹 2 내부') for s in intra_sim_g2])

    # 그룹 간 유사도
    inter_sim = cosine_similarity(embeddings_g1, embeddings_g2).flatten()
    sim_data.extend([(s, '그룹 간') for s in inter_sim])

    df_sim = pd.DataFrame(sim_data, columns=['코사인 유사도', '비교 유형'])

    status_updater(f"\n그룹 1 내부 평균 유사도: {np.mean(intra_sim_g1):.4f}")
    status_updater(f"그룹 2 내부 평균 유사도: {np.mean(intra_sim_g2):.4f}")
    status_updater(f"그룹 간 평균 유사도: {np.mean(inter_sim):.4f}")
    
    plt.figure(figsize=(12, 8))
    sns.violinplot(data=df_sim, x='비교 유형', y='코사인 유사도', palette='plasma')
    plt.title('🎻 그룹 내/간 코사인 유사도 분포', fontsize=16)
    plt.savefig(os.path.join(output_dir, '3_cosine_similarity_violinplot.png'))
    plt.close()

    status_updater("[3단계] 분석 완료. 결과가 저장되었습니다.")

def analyze_embedding_clusters_tsne(data_group_1, data_group_2, output_dir, status_updater):
    """4) 2D 임베딩 공간 시각화 (t-SNE)"""
    status_updater("\n[4단계] t-SNE 임베딩 공간 시각화를 시작합니다...")

    embeddings_g1 = np.array([d['embedding'] for d in data_group_1 if 'embedding' in d and len(d['embedding']) > 0])
    embeddings_g2 = np.array([d['embedding'] for d in data_group_2 if 'embedding' in d and len(d['embedding']) > 0])

    if len(embeddings_g1) == 0 or len(embeddings_g2) == 0:
        status_updater("오류: t-SNE 분석을 위해 각 그룹에 최소 1개 이상의 유효한 임베딩이 필요합니다.")
        return

    all_embeddings = np.vstack([embeddings_g1, embeddings_g2])
    labels = ['그룹 1'] * len(embeddings_g1) + ['그룹 2'] * len(embeddings_g2)

    if all_embeddings.shape[1] <= 2:
        status_updater("경고: 임베딩 차원이 이미 2차원 이하입니다. t-SNE를 건너뛰고 바로 시각화합니다.")
        tsne_results = all_embeddings
    else:
        perplexity = min(30, len(all_embeddings) - 2)
        if perplexity < 5:
            status_updater(f"경고: 데이터 포인트({len(all_embeddings)}개)가 너무 적어 t-SNE 분석이 불안정할 수 있습니다.")
            perplexity = max(1, len(all_embeddings) - 2)

        tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42, n_iter=1000, learning_rate='auto', init='pca')
        tsne_results = tsne.fit_transform(all_embeddings)

    df_tsne = pd.DataFrame({'x': tsne_results[:, 0], 'y': tsne_results[:, 1], '그룹': labels})

    plt.figure(figsize=(12, 10))
    sns.scatterplot(
        data=df_tsne,
        x='x', y='y',
        hue='그룹',
        palette='bright',
        alpha=0.7,
        s=50
    )
    plt.title('🗺️ t-SNE 기반 2D 임베딩 공간 시각화', fontsize=16)
    plt.legend(title='데이터 그룹')
    plt.xlabel('t-SNE Component 1')
    plt.ylabel('t-SNE Component 2')
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, '4_tsne_embedding_space.png'))
    plt.close()
    
    status_updater("[4단계] 분석 완료. 결과가 저장되었습니다.")


# ==============================================================================
# GUI 및 메인 실행 로직
# ==============================================================================

class NLPAnalysisApp:
    def __init__(self, master):
        self.master = master
        master.title("텍스트 비교 분석 도구 (v1.2 - 오류 수정)")
        master.geometry("700x550")

        self.folder_path1 = tk.StringVar()
        self.folder_path2 = tk.StringVar()
        
        self.create_widgets()

    def create_widgets(self):
        main_frame = tk.Frame(self.master, padx=10, pady=10)
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        folder_frame = tk.LabelFrame(main_frame, text="입력 데이터 설정", padx=10, pady=10)
        folder_frame.pack(fill=tk.X, pady=5)
        
        tk.Label(folder_frame, text="그룹 1 데이터 폴더:").grid(row=0, column=0, sticky='w', pady=2)
        tk.Entry(folder_frame, textvariable=self.folder_path1, width=60).grid(row=1, column=0, sticky='we')
        tk.Button(folder_frame, text="폴더 선택", command=lambda: self.select_folder(self.folder_path1)).grid(row=1, column=1, padx=5)

        tk.Label(folder_frame, text="그룹 2 데이터 폴더:").grid(row=2, column=0, sticky='w', pady=2)
        tk.Entry(folder_frame, textvariable=self.folder_path2, width=60).grid(row=3, column=0, sticky='we')
        tk.Button(folder_frame, text="폴더 선택", command=lambda: self.select_folder(self.folder_path2)).grid(row=3, column=1, padx=5)

        folder_frame.grid_columnconfigure(0, weight=1)

        self.run_button = tk.Button(main_frame, text="분석 실행", command=self.run_analysis, font=("Helvetica", 12, "bold"), bg="#4CAF50", fg="white")
        self.run_button.pack(pady=10, fill=tk.X)

        status_frame = tk.LabelFrame(main_frame, text="분석 로그", padx=10, pady=10)
        status_frame.pack(fill=tk.BOTH, expand=True)

        self.status_text = tk.Text(status_frame, height=15, state='disabled', wrap='word', bg="#f0f0f0")
        scrollbar = tk.Scrollbar(status_frame, command=self.status_text.yview)
        self.status_text.config(yscrollcommand=scrollbar.set)
        
        self.status_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        self.update_status("분석 준비 완료. 두 개의 데이터 폴더를 선택하고 '분석 실행' 버튼을 누르세요.")

    def select_folder(self, entry_var):
        folder_selected = filedialog.askdirectory()
        if folder_selected:
            entry_var.set(folder_selected)
            self.update_status(f"폴더 선택됨: {folder_selected}")

    def update_status(self, message):
        self.status_text.config(state='normal')
        self.status_text.insert(tk.END, message + "\n")
        self.status_text.see(tk.END)
        self.status_text.config(state='disabled')
        self.master.update_idletasks()

    def run_analysis(self):
        folder1 = self.folder_path1.get()
        folder2 = self.folder_path2.get()

        if not folder1 or not folder2:
            messagebox.showwarning("입력 오류", "두 개의 데이터 폴더를 모두 선택해주세요.")
            return

        self.run_button.config(state='disabled', text="분석 중...")
        try:
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            output_dir = f"analysis_results_{timestamp}"
            os.makedirs(output_dir, exist_ok=True)

            self.update_status("="*50)
            self.update_status(f"분석을 시작합니다. 결과는 '{output_dir}' 폴더에 저장됩니다.")
            self.update_status("="*50)
            
            self.update_status("\n[데이터 로딩] 그룹 1 데이터 로드를 시작합니다...")
            data_group_1 = load_and_parse_json_data(folder1, self.update_status)
            self.update_status(f"그룹 1에서 {len(data_group_1)}개 파일 로드 완료.")
            
            self.update_status("\n[데이터 로딩] 그룹 2 데이터 로드를 시작합니다...")
            data_group_2 = load_and_parse_json_data(folder2, self.update_status)
            self.update_status(f"그룹 2에서 {len(data_group_2)}개 파일 로드 완료.")

            if not data_group_1 or not data_group_2:
                messagebox.showerror("데이터 오류", "하나 또는 두 그룹 모두에서 유효한 데이터를 로드하지 못했습니다. 폴더 경로와 JSON 파일 형식을 확인하세요.")
                raise ValueError("데이터 로드 실패")

            analyze_text_lengths(data_group_1, data_group_2, output_dir, self.update_status)
            analyze_vocabulary_and_keywords(data_group_1, data_group_2, output_dir, self.update_status)
            analyze_cosine_similarity(data_group_1, data_group_2, output_dir, self.update_status)
            analyze_embedding_clusters_tsne(data_group_1, data_group_2, output_dir, self.update_status)

            self.update_status("\n" + "="*50)
            self.update_status("모든 분석이 성공적으로 완료되었습니다!")
            self.update_status(f"결과물은 '{os.path.abspath(output_dir)}' 폴더에서 확인하세요.")
            self.update_status("="*50)
            messagebox.showinfo("분석 완료", f"모든 분석이 완료되었습니다. 결과를 '{output_dir}' 폴더에서 확인하세요.")

        except Exception as e:
            self.update_status(f"\n치명적인 오류 발생: {e}")
            messagebox.showerror("분석 오류", f"분석 중 오류가 발생했습니다: {e}")
        finally:
            self.run_button.config(state='normal', text="분석 실행")

if __name__ == "__main__":
    root = tk.Tk()
    app = NLPAnalysisApp(root)
    root.mainloop()
