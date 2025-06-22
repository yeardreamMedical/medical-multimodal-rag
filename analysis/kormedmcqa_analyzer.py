#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
KorMedMCQA CSV 데이터에 대한 텍스트 통계 분석 스크립트
- 질문(question) 열에 대해 텍스트 통계 분석
- 문자 수, 단어 수, 문장 수, 어휘 다양성 등 분석
- 시각화 결과를 PNG 파일로 저장

사용법:
  python kormedmcqa_analyzer.py --data_path "../data/KorMedMCQA/doctor/doctor-train.csv" --out_dir "kormedmcqa_results"

Tk 백엔드가 없는 환경에서도 실행 가능하도록 Agg 백엔드를 사용
"""
from __future__ import annotations
import argparse
import pandas as pd
import numpy as np
from pathlib import Path
import os
import re

# Matplotlib 설정 (Tkinter 의존성 제거)
import matplotlib
matplotlib.use("Agg")  # Tk 없이 동작하는 백엔드
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import font_manager

# 한글 폰트 설정 (개선된 버전)
import platform

# 디버깅용 함수: 사용 가능한 한글 폰트 목록 출력
def print_available_fonts():
    print("\n=== 사용 가능한 한글 폰트 목록 ===")
    system_fonts = sorted([f.name for f in font_manager.fontManager.ttflist if any(keyword in f.name.lower() for keyword in ['gothic', 'gulim', 'batang', 'dotum', 'malgun', 'apple', '나눔', '산돌', '고딕'])])
    for i, font in enumerate(system_fonts):
        print(f"{i+1}. {font}")
    print("=" * 35)

# Mac용 한글 폰트 설정 (방법 1-3 시도)
font_installed = False

# 방법 1: AppleGothic 폰트 사용 (많은 Mac에 기본 설치)
try:
    if platform.system() == 'Darwin':  # macOS
        font_path = '/System/Library/Fonts/Supplemental/AppleGothic.ttf'
        font_prop = font_manager.FontProperties(fname=font_path)
        plt.rcParams['font.family'] = font_prop.get_name()
        plt.rcParams['axes.unicode_minus'] = False  # 마이너스 폰트 깨짐 방지
        print("✅ 한글 폰트 설정 성공 (AppleGothic)")
        font_installed = True
except Exception as e:
    print(f"방법 1 실패: {e}")

# 방법 2: AppleSDGothicNeo 사용
if not font_installed:
    try:
        if platform.system() == 'Darwin':  # macOS
            font_path = '/System/Library/Fonts/AppleSDGothicNeo.ttc'
            font_prop = font_manager.FontProperties(fname=font_path)
            plt.rcParams['font.family'] = font_prop.get_name()
            plt.rcParams['axes.unicode_minus'] = False
            print("✅ 한글 폰트 설정 성공 (AppleSDGothicNeo)")
            font_installed = True
    except Exception as e:
        print(f"방법 2 실패: {e}")

# 방법 3: 시스템별 기본 설정 방식
if not font_installed:
    try:
        if platform.system() == 'Darwin':  # macOS
            plt.rc('font', family='AppleGothic')
        elif platform.system() == 'Windows':
            plt.rc('font', family='Malgun Gothic')
        else:  # Linux
            plt.rc('font', family='NanumGothic')
        plt.rcParams['axes.unicode_minus'] = False
        print("✅ 한글 폰트 설정 성공 (일반적인 방식)")
        font_installed = True
    except Exception as e:
        print(f"방법 3 실패: {e}")

# 방법 4: 폰트 목록 출력 및 안내
if not font_installed:
    print("⚠️ 한글 폰트 설정 실패")
    print_available_fonts()
    print("코드의 font_path 변수를 위 목록의 폰트 경로로 수정해주세요.")

# 한국어 형태소 분석기 
try:
    from konlpy.tag import Okt
    okt = Okt()
    print("✅ KoNLPy Okt 모듈 로드 성공")
except Exception as e:
    print(f"⚠️ KoNLPy Okt 로드 실패: {e}")
    print("명사 추출 및 어휘 다양성 분석이 비활성화됩니다.")
    okt = None

# 텍스트 전처리 함수
def preprocess_text(text: str) -> str:
    """문제 텍스트를 분석용으로 간단 전처리"""
    if not isinstance(text, str):
        return ""
    
    # HTML 태그 제거 (BeautifulSoup 의존성 제거)
    text = re.sub(r'<[^>]+>', '', text)
    
    # 보기 번호(1), 2.  같은 패턴 제거
    text = re.sub(r"\s+\d[).]\s+", " ", text)
    
    # 안내성 한국어 괄호 제거 – 그림, 사진, 도표 등
    text = re.sub(r"\((?:그림|사진|도표|표|보기|참조|단,)\s*[^)]*\)", "", text)
    
    # 제어 문자 → 공백
    text = re.sub(r"[\n\r\t]", " ", text)
    
    # 중복 공백 정리
    text = re.sub(r"\s+", " ", text).strip()
    
    return text

# CSV 파일에서 데이터 로드
def load_csv_data(file_path: str) -> pd.DataFrame:
    """CSV 파일을 로드하여 question 열만 추출"""
    try:
        # UTF-8-BOM 또는 UTF-8 인코딩 시도
        df = pd.read_csv(file_path, encoding='utf-8-sig')
    except:
        # 다른 인코딩 시도
        try:
            df = pd.read_csv(file_path, encoding='utf-8')
        except:
            df = pd.read_csv(file_path, encoding='cp949')
    
    if 'question' not in df.columns:
        raise ValueError(f"CSV 파일에 'question' 열이 없습니다: {file_path}")
    
    print(f"✅ 데이터 로드 완료: {len(df)} 행")
    return df

# 기본 통계 계산
def calculate_text_stats(df: pd.DataFrame) -> pd.DataFrame:
    """문자 수, 단어 수, 문장 수 계산"""
    df['char_count'] = df['question'].str.len()
    df['word_count'] = df['question'].str.split().str.len()
    # 문장 종결 문자(.) 기준으로 문장 수 계산, 없으면 1로 간주
    df['sentence_count'] = df['question'].str.count(r'[.?!]') + 1
    return df

# 어휘 다양성 분석 (TTR)
def analyze_vocabulary(texts: list[str]) -> dict:
    """어휘 다양성 분석 (TTR, 명사 비율 등)"""
    if okt is None:
        return {"total_nouns": 0, "unique_nouns": 0, "TTR": 0}
    
    full_text = " ".join(texts)
    nouns = okt.nouns(full_text)
    
    if not nouns:
        return {"total_nouns": 0, "unique_nouns": 0, "TTR": 0}
    
    total = len(nouns)
    unique = len(set(nouns))
    ttr = unique / total if total > 0 else 0
    
    return {
        "total_nouns": total,
        "unique_nouns": unique,
        "TTR": ttr
    }

# 히스토그램 및 그래프 생성
def create_visualizations(df: pd.DataFrame, vocab_stats: dict, out_dir: Path, dataset_name: str):
    """통계 시각화 함수"""
    # 1. 문자 수 분포
    plt.figure(figsize=(10, 6))
    sns.histplot(data=df, x='char_count', kde=True, color='steelblue')
    plt.title(f'{dataset_name} 문제 문자 수 분포')
    plt.xlabel('문자 수')
    plt.ylabel('빈도')
    plt.tight_layout()
    plt.savefig(out_dir / 'char_count_dist.png')
    plt.close()
    
    # 2. 단어 수 분포
    plt.figure(figsize=(10, 6))
    sns.histplot(data=df, x='word_count', kde=True, color='darkgreen')
    plt.title(f'{dataset_name} 문제 단어 수 분포')
    plt.xlabel('단어 수')
    plt.ylabel('빈도')
    plt.tight_layout()
    plt.savefig(out_dir / 'word_count_dist.png')
    plt.close()
    
    # 3. 문장 수 분포
    plt.figure(figsize=(10, 6))
    sns.histplot(data=df, x='sentence_count', kde=True, color='darkorange')
    plt.title(f'{dataset_name} 문제 문장 수 분포')
    plt.xlabel('문장 수')
    plt.ylabel('빈도')
    plt.tight_layout()
    plt.savefig(out_dir / 'sentence_count_dist.png')
    plt.close()
    
    # 4. 어휘 다양성 (TTR) 도표
    if okt is not None:
        plt.figure(figsize=(8, 6))
        data = {
            '지표': ['총 명사 수', '고유 명사 수', 'TTR'],
            '값': [vocab_stats['total_nouns'], vocab_stats['unique_nouns'], vocab_stats['TTR']]
        }
        bar_df = pd.DataFrame(data)
        ax = sns.barplot(data=bar_df, x='지표', y='값', palette='viridis')
        
        # 값 라벨 추가
        for i, v in enumerate(bar_df['값']):
            value_text = f"{v:.4f}" if i == 2 else f"{int(v)}"
            ax.text(i, v + 0.1, value_text, ha='center')
            
        plt.title(f'{dataset_name} 어휘 다양성')
        plt.tight_layout()
        plt.savefig(out_dir / 'vocabulary_stats.png')
        plt.close()

# 요약 통계 텍스트 파일 생성
def save_summary_stats(df: pd.DataFrame, vocab_stats: dict, out_dir: Path, dataset_name: str):
    """요약 통계를 텍스트 파일로 저장"""
    summary_path = out_dir / 'summary_stats.txt'
    
    with open(summary_path, 'w', encoding='utf-8') as f:
        f.write(f"# {dataset_name} 텍스트 통계 요약\n\n")
        f.write(f"## 기본 통계\n")
        f.write(f"- 총 문항 수: {len(df)}\n")
        
        # 문자 수 통계
        f.write(f"\n## 문자 수 통계\n")
        f.write(f"- 평균: {df['char_count'].mean():.1f}\n")
        f.write(f"- 중앙값: {df['char_count'].median():.1f}\n")
        f.write(f"- 표준편차: {df['char_count'].std():.1f}\n")
        f.write(f"- 최소값: {df['char_count'].min()}\n")
        f.write(f"- 최대값: {df['char_count'].max()}\n")
        
        # 단어 수 통계
        f.write(f"\n## 단어 수 통계\n")
        f.write(f"- 평균: {df['word_count'].mean():.1f}\n")
        f.write(f"- 중앙값: {df['word_count'].median():.1f}\n")
        f.write(f"- 표준편차: {df['word_count'].std():.1f}\n")
        f.write(f"- 최소값: {df['word_count'].min()}\n")
        f.write(f"- 최대값: {df['word_count'].max()}\n")
        
        # 문장 수 통계
        f.write(f"\n## 문장 수 통계\n")
        f.write(f"- 평균: {df['sentence_count'].mean():.1f}\n")
        f.write(f"- 중앙값: {df['sentence_count'].median():.1f}\n")
        f.write(f"- 표준편차: {df['sentence_count'].std():.1f}\n")
        f.write(f"- 최소값: {df['sentence_count'].min()}\n")
        f.write(f"- 최대값: {df['sentence_count'].max()}\n")
        
        # 어휘 다양성
        if okt is not None:
            f.write(f"\n## 어휘 다양성\n")
            f.write(f"- 총 명사 수: {vocab_stats['total_nouns']}\n")
            f.write(f"- 고유 명사 수: {vocab_stats['unique_nouns']}\n")
            f.write(f"- Type-Token Ratio (TTR): {vocab_stats['TTR']:.4f}\n")
        
    print(f"✅ 요약 통계를 {summary_path}에 저장했습니다.")

def main():
    # 인자 파싱
    parser = argparse.ArgumentParser(description="KorMedMCQA CSV 파일의 질문 텍스트 분석")
    parser.add_argument("--data_path", required=True, help="KorMedMCQA CSV 파일 경로")
    parser.add_argument("--out_dir", default="kormedmcqa_results", help="결과 저장 폴더")
    args = parser.parse_args()
    
    # 데이터셋 이름 추출 (파일 이름에서)
    dataset_name = Path(args.data_path).stem
    
    # 출력 디렉토리 생성
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"데이터 경로: {args.data_path}")
    print(f"출력 폴더: {out_dir}")
    
    # 1. CSV 데이터 로드
    df = load_csv_data(args.data_path)
    
    # 2. 텍스트 전처리
    df['clean_question'] = df['question'].apply(preprocess_text)
    
    # 3. 기본 통계 계산
    df = calculate_text_stats(df)
    
    # 4. 어휘 다양성 분석 
    vocab_stats = analyze_vocabulary(df['clean_question'].tolist())
    
    # 5. 시각화 생성
    create_visualizations(df, vocab_stats, out_dir, dataset_name)
    
    # 6. 요약 통계 저장
    save_summary_stats(df, vocab_stats, out_dir, dataset_name)
    
    print(f"✅ 분석 완료! 결과가 {out_dir} 폴더에 저장되었습니다.")

if __name__ == "__main__":
    main() 