"""
PDF Renamer - OCR、Google Cloud Vision、およびOpenAIを使用してPDFファイルを
内容に基づいて自動的にリネームするための強化ツール

このアプリケーションはOCRを使用してPDFからテキストを抽出し、AIで内容を分析し、
カスタマイズ可能なルールに従って意味のあるファイル名を生成します。

作者: Claude（ユーザーのオリジナルに基づく）
日付: 2025年5月9日
"""

import os
import sys
import time
import logging
import tkinter as tk
from tkinter import ttk, messagebox, scrolledtext
from tkinter.filedialog import askdirectory
import threading
import queue
import subprocess
import platform
from datetime import datetime
from pathlib import Path
import re
import yaml
import tempfile
from dotenv import load_dotenv
import openai
from pdf2image import convert_from_path
from pdf2image.exceptions import PDFInfoNotInstalledError, PDFPageCountError
from google.cloud import vision
from concurrent.futures import ThreadPoolExecutor, as_completed
import google.api_core.exceptions
from openai import OpenAI
import pytesseract
from PIL import Image
import pdf2image
import json
import traceback
import shutil

# ============================================================================
# 設定とセットアップ
# ============================================================================

# スクリプトディレクトリの設定
SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_ENV_PATH = SCRIPT_DIR / '.env'
DEFAULT_YAML_PATH = SCRIPT_DIR / 'rename_rules.yaml'
DEFAULT_IMAGE_PATH = SCRIPT_DIR / 'temp_images'
APP_VERSION = "2025年5月10日バージョン"

# ログの設定
log_file_path = os.path.join(SCRIPT_DIR, "pdf-renamer.log")

# 既存のハンドラーをクリア
for handler in logging.root.handlers[:]:
    logging.root.removeHandler(handler)

# ログの基本設定
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file_path, encoding="utf-8", mode='w'),  # mode='w'で毎回新規作成
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

# 初期化ログの出力
logger.debug("=" * 50)
logger.debug("プログラム開始")
logger.debug(f"ログファイルの出力先: {log_file_path}")
logger.debug(f"Pythonバージョン: {sys.version}")
logger.debug(f"実行ディレクトリ: {os.getcwd()}")
logger.debug("=" * 50)

# 定数とグローバル変数
SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_ENV_PATH = SCRIPT_DIR / '.env'
DEFAULT_YAML_PATH = SCRIPT_DIR / 'rename_rules.yaml'
DEFAULT_IMAGE_PATH = SCRIPT_DIR / 'temp_images'
APP_VERSION = "2025年5月10日バージョン"


# ============================================================================
# ヘルパークラス
# ============================================================================

class DefaultDict(dict):
    """存在しないキーに対してデフォルト値を返すカスタム辞書。"""
    def __missing__(self, key):
        return "不明"


class ConfigManager:
    """アプリケーション設定と環境変数を管理するクラス。"""
    
    def __init__(self, env_path=None):
        """オプションの環境ファイルパスで設定マネージャーを初期化します。"""
        self.env_path = env_path or DEFAULT_ENV_PATH
        self.config = {}
        self.load_config()
        
    def load_config(self):
        """.envファイルから設定を読み込みます。"""
        try:
            load_dotenv(self.env_path)
            
            # 必要な設定
            self.config['PDF_FOLDER_PATH'] = os.getenv('PDF_FOLDER_PATH', '')
            self.config['IMAGE_FILE_PATH'] = os.getenv('IMAGE_FILE_PATH', str(DEFAULT_IMAGE_PATH))
            self.config['YAML_FILE'] = os.getenv('YAML_FILE', str(DEFAULT_YAML_PATH))
            self.config['GOOGLE_APPLICATION_CREDENTIALS'] = os.getenv('GOOGLE_APPLICATION_CREDENTIALS', '')
            self.config['OPENAI_API_KEY'] = os.getenv('OPENAI_API_KEY', '')
            self.config['POPPLER_PATH'] = os.getenv('POPPLER_PATH', self._find_poppler_path())
            
            # AIモデル設定
            self.config['OPENAI_MODEL'] = os.getenv('OPENAI_MODEL', 'gpt-4.1')
            self.config['OPENAI_TEMPERATURE'] = float(os.getenv('OPENAI_TEMPERATURE', '0.2'))
            
            # オプション設定（デフォルト値あり）
            self.config['PERSONS'] = os.getenv('PERSONS', '').split(',')
            self.config['DEFAULT_PERSON'] = os.getenv('DEFAULT_PERSON', '担当者自動設定')
            self.config['ORGANIZATION_NAME'] = os.getenv('ORGANIZATION_NAME', 'DefaultOrganization')
            self.config['TITLE'] = os.getenv('TITLE', 'Untitled')
            
            # 画像ディレクトリがない場合は作成する
            os.makedirs(self.config['IMAGE_FILE_PATH'], exist_ok=True)
            
            logger.info(f"設定を{self.env_path}から読み込みました")
            
        except Exception as e:
            logger.error(f"設定の読み込みエラー: {e}")
            raise
            
    def _find_poppler_path(self):
        """一般的なインストール場所からPopplerのパスを見つけます。"""
        possible_paths = [
            "C:\\Program Files\\Poppler\\bin",
            "C:\\Program Files (x86)\\Poppler\\bin",
            "C:\\poppler-24.08.0\\Library\\bin",
        ]
        
        for path in possible_paths:
            if os.path.exists(path) and os.path.isfile(os.path.join(path, "pdftoppm.exe")):
                logger.debug(f"Popplerが見つかりました: {path}")
                return path
        
        # PopplerがPATHにある場合は空文字を返す（システムPATHを使用）
        try:
            result = subprocess.run(["pdftoppm", "-v"], capture_output=True, text=True, check=False)
            if result.returncode == 0:
                logger.debug("PopplerがシステムPATHで見つかりました")
                return ""
        except Exception:
            pass
            
        logger.debug("Popplerのパスが見つかりませんでした")
        return ""
    
    def validate_config(self):
        """必要な設定がすべて存在し、有効であることを確認します。"""
        required_configs = {
            'PDF_FOLDER_PATH': "PDFフォルダパス",
            'IMAGE_FILE_PATH': "一時画像フォルダパス",
            'YAML_FILE': "YAMLルールファイルパス",
            'GOOGLE_APPLICATION_CREDENTIALS': "Google Cloud認証情報パス",
            'OPENAI_API_KEY': "OpenAI APIキー",
        }
        
        missing = []
        for key, desc in required_configs.items():
            if not self.config.get(key):
                missing.append(desc)
                
        # Popplerインストールの検証
        if not self.config.get('POPPLER_PATH'):
            try:
                # pdftoppmがPATHにあるかチェック
                result = subprocess.run(["pdftoppm", "-v"], capture_output=True, text=True, check=False)
                if result.returncode != 0:
                    missing.append("Popplerインストール（PDF変換に必要）")
            except Exception:
                missing.append("Popplerインストール（PDF変換に必要）")
        else:
            poppler_path = self.config['POPPLER_PATH']
            pdftoppm_path = os.path.join(poppler_path, "pdftoppm.exe")
            if not os.path.exists(pdftoppm_path):
                missing.append(f"{pdftoppm_path}のPopplerの実行ファイル")
                
        # YAMLファイルが存在するか検証
        yaml_path = Path(self.config['YAML_FILE'])
        if not yaml_path.exists():
            missing.append(f"{yaml_path}のYAMLルールファイル")
                
        return missing
    
    def get(self, key, default=None):
        """キーによって設定値を取得します。"""
        return self.config.get(key, default)
    
    def set(self, key, value):
        """設定値を設定します。"""
        self.config[key] = value
        
    def save_config(self):
        """設定を.envファイルに保存します。"""
        try:
            with open(self.env_path, 'w', encoding='utf-8') as f:
                for key, value in self.config.items():
                    if key == 'PERSONS':
                        f.write(f"{key}={','.join(value)}\n")
                    else:
                        f.write(f"{key}={value}\n")
                        
            logger.info(f"設定を{self.env_path}に保存しました")
            
        except Exception as e:
            logger.error(f"設定の保存エラー: {e}")
            raise


class PDFProcessor:
    """OCRとAIを使用してPDFファイルを処理し、意味のあるファイル名を生成するクラス。"""
    
    def __init__(self, config_manager, selected_person, status_queue, business_card_mode=False):
        """PDFProcessorの初期化"""
        self.config_manager = config_manager
        self.selected_person = selected_person
        self.status_queue = status_queue
        self.business_card_mode = business_card_mode
        self.yaml_rules = None
        self.load_yaml_rules()
        
        # ログの設定
        logger.debug("PDFProcessorを初期化しました")
        logger.debug(f"選択された担当者: {selected_person}")
        logger.debug(f"名刺読み取りモード: {business_card_mode}")
        logger.debug(f"設定マネージャー: {config_manager}")
        
    def load_yaml_rules(self):
        """YAMLファイルからリネームルールを読み込みます。"""
        try:
            yaml_path = self.config_manager.get('YAML_FILE')
            logger.debug(f"YAMLファイルのパス: {yaml_path}")
            
            if not yaml_path:
                raise ValueError("YAMLファイルのパスが設定されていません")
                
            if not os.path.exists(yaml_path):
                raise FileNotFoundError(f"YAMLファイルが見つかりません: {yaml_path}")
                
            logger.debug(f"YAMLファイルを読み込み中: {yaml_path}")
            
            try:
                with open(yaml_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                    logger.debug(f"YAMLファイルの内容:\n{content[:500]}...")  # 最初の500文字を表示
            except UnicodeDecodeError:
                # UTF-8で読み込めない場合は、CP932（Windows日本語）で試行
                with open(yaml_path, 'r', encoding='cp932') as f:
                    content = f.read()
                    logger.debug(f"YAMLファイルの内容 (CP932):\n{content[:500]}...")
            
            try:
                self.yaml_rules = yaml.safe_load(content)
            except yaml.YAMLError as e:
                logger.error(f"YAMLの解析エラー: {e}")
                raise ValueError(f"YAMLファイルの形式が不正です: {e}")
                
            if not self.yaml_rules:
                raise ValueError("YAMLファイルが空です")
                
            if not isinstance(self.yaml_rules, dict):
                raise ValueError("YAMLファイルのルート要素が辞書形式ではありません")
                
            if "ファイル命名のルール" not in self.yaml_rules:
                raise ValueError("YAMLファイルに「ファイル命名のルール」セクションがありません")
                
            rules = self.yaml_rules["ファイル命名のルール"]
            if not isinstance(rules, list):
                raise ValueError("「ファイル命名のルール」がリスト形式ではありません")
                
            logger.debug(f"YAMLルールを読み込みました: {len(rules)}個のルール")
            
            # デフォルトルールの設定
            default_rule = {
                "ルール": {
                    "説明": "デフォルトルール",
                    "正規表現": ".*",
                    "書類の種類": "その他",
                    "命名ルール": "{日付} {担当者} {タイトル}"
                }
            }
            
            # ルールの検証
            valid_rules = []
            for i, rule in enumerate(rules, 1):
                if not rule:
                    logger.warning(f"ルール {i} がNoneです")
                    continue
                    
                if not isinstance(rule, dict):
                    logger.warning(f"ルール {i} が辞書形式ではありません")
                    continue
                    
                if "ルール" in rule:
                    current_rule = rule["ルール"]
                    if not isinstance(current_rule, dict):
                        logger.warning(f"ルール {i} の「ルール」セクションが辞書形式ではありません")
                        continue
                else:
                    current_rule = rule
                    
                if "正規表現" not in current_rule:
                    logger.warning(f"ルール {i} に「正規表現」がありません")
                    continue
                    
                if "命名ルール" not in current_rule:
                    logger.warning(f"ルール {i} に「命名ルール」がありません")
                    continue
                    
                valid_rules.append(rule)
                logger.debug(f"ルール {i}: {current_rule.get('説明', 'Unnamed rule')}")
            
            if not valid_rules:
                raise ValueError("有効なルールが1つも見つかりません")
                
            # デフォルトルールを追加
            valid_rules.append(default_rule)
            self.yaml_rules["ファイル命名のルール"] = valid_rules
            logger.info(f"有効なルール {len(valid_rules)} 個を読み込みました（デフォルトルール含む）")
                
        except Exception as e:
            logger.exception(f"YAMLルールの読み込みエラー: {e}")
            self.status_queue.put(f"エラー: YAML設定の読み込みに失敗しました: {e}\n")
            self.yaml_rules = None
    
    def process_pdf(self, pdf_file_path):
        """単一のPDFファイルを処理します：画像に変換、OCR、リネーム。"""
        temp_jpeg_path = None
        try:
            logger.debug(f"PDFを処理中: {pdf_file_path}")
            
            # 1. PDFをJPEGに変換
            temp_jpeg_path = self._pdf_to_jpeg(pdf_file_path)
            if not temp_jpeg_path:
                self.status_queue.put(f"処理失敗: {pdf_file_path} (PDF変換エラー)\n")
                return False
                
            # 2. OCRでテキストを抽出
            ocr_result = self._ocr_jpeg(temp_jpeg_path)
            if not ocr_result:
                self.status_queue.put(f"処理失敗: {pdf_file_path} (OCRエラー)\n")
                return False
                
            # 3. OCR結果に基づいて新しいファイル名を生成
            new_pdf_name = self._propose_file_name(ocr_result)
            if not new_pdf_name:
                self.status_queue.put(f"処理失敗: {pdf_file_path} (ファイル名提案エラー)\n")
                return False
                
            # 4. PDFファイルのリネーム
            new_path = self._rename_pdf(pdf_file_path, new_pdf_name)
            if new_path:
                self.status_queue.put(f"リネーム完了: {os.path.basename(new_path)}\n")
                return True
            else:
                self.status_queue.put(f"処理失敗: {pdf_file_path} (リネームエラー)\n")
                return False
                
        except Exception as e:
            logger.exception(f"PDFの処理エラー: {pdf_file_path}")
            self.status_queue.put(f"処理エラー: {pdf_file_path} - {str(e)}\n")
            return False
            
        finally:
            # 一時ファイルのクリーンアップ
            if temp_jpeg_path and os.path.exists(temp_jpeg_path):
                try:
                    os.remove(temp_jpeg_path)
                    logger.debug(f"一時ファイルを削除しました: {temp_jpeg_path}")
                except Exception as e:
                    logger.exception(f"一時ファイルの削除エラー: {temp_jpeg_path}")
    
    def _pdf_to_jpeg(self, pdf_file_path):
        """PDFの最初のページをJPEG画像に変換します。"""
        try:
            logger.debug(f"PDFをJPEGに変換中: {pdf_file_path}")
            
            poppler_path = self.config_manager.get('POPPLER_PATH')
            
            # 指定されたフォルダ内に一意の一時ファイル名を作成
            temp_jpeg_path = os.path.join(
                self.config_manager.get('IMAGE_FILE_PATH'),
                f"temp_{datetime.now().strftime('%Y%m%d%H%M%S%f')}_{os.path.basename(pdf_file_path)}.jpg"
            )
            
            # PDFの最初のページをJPEG画像に変換
            kwargs = {
                'first_page': 1,
                'last_page': 1,
                'dpi': 400,  # OCR結果向上のため解像度を上げる
                'grayscale': False,  # カラー画像を使用
                'thread_count': 1,  # スレッド数を制限して安定性を向上
                'use_pdftocairo': True,  # pdftocairoを使用して品質を向上
            }
            
            if poppler_path:
                kwargs['poppler_path'] = poppler_path
                
            try:
                images = convert_from_path(pdf_file_path, **kwargs)
                if not images:
                    raise ValueError("PDFから画像が抽出されませんでした")
                    
                images[0].save(temp_jpeg_path, 'JPEG', quality=95)  # OCR向上のため高品質に設定
                logger.debug(f"PDFをJPEGに変換しました: {temp_jpeg_path}")
                return temp_jpeg_path
                
            except PDFPageCountError:
                logger.warning(f"PDFの変換に失敗しました - 破損している可能性があります: {pdf_file_path}")
                return None
                
        except PDFInfoNotInstalledError:
            logger.error("Popplerが正しくインストールまたは設定されていません")
            self.status_queue.put("エラー: Popplerが正しく設定されていません。\n")
            return None
            
        except Exception as e:
            logger.exception(f"PDFをJPEGに変換するエラー: {pdf_file_path}")
            return None
    
    def _ocr_jpeg(self, image_path):
        """Google Cloud Vision APIを使用してJPEG画像からテキストを抽出します。"""
        try:
            logger.debug(f"画像に対してOCRを実行中: {image_path}")
            
            # Vision APIクライアントを初期化
            client = vision.ImageAnnotatorClient()
            
            # 画像ファイルを読み込む
            with open(image_path, 'rb') as image_file:
                content = image_file.read()
                
            # 画像オブジェクトを作成しテキスト検出を実行
            image = vision.Image(content=content)
            
            # 日本語のヒントを設定
            image_context = vision.ImageContext(
                language_hints=["ja"]
            )
            
            # document_text_detectionを使用して日本語テキストの認識精度を向上
            response = client.document_text_detection(
                image=image,
                image_context=image_context
            )
            
            # エラーをチェック
            if response.error.message:
                raise Exception(f"Google Cloud Vision APIエラー: {response.error.message}")
                
            # OCRテキストを抽出して返す
            if response.text_annotations:
                result = response.text_annotations[0].description
                logger.debug(f"OCRに成功し、{len(result)}文字を抽出しました")
                # OCR結果をログファイルに出力
                logger.info(f"OCR結果:\n{'-' * 50}\n{result}\n{'-' * 50}")
                return result
            else:
                logger.warning(f"画像内にテキストが検出されませんでした: {image_path}")
                return ""
                
        except google.api_core.exceptions.PermissionDenied as e:
            logger.error(f"Google Cloud Vision APIのアクセス権が拒否されました: {e}")
            self.status_queue.put(
                "エラー: Google Cloud Vision APIの課金が有効になっていません。\n"
                "https://console.developers.google.com/billing/enable で課金を有効にしてください。\n"
            )
            return None
            
        except Exception as e:
            logger.exception(f"OCRエラー: {image_path}")
            return None
    
    def _convert_wareki_to_western(self, text):
        """和暦を西暦に変換します。"""
        # 年号なしの和暦の変換（令和を想定）
        text = re.sub(r'(\d+)年', lambda m: str(2018 + int(m.group(1))) + '年', text)
        
        # 令和の変換 (令和: 2019-)
        text = re.sub(r'令和(\d+)年', lambda m: str(2018 + int(m.group(1))) + '年', text)
        # 平成の変換 (平成: 1989-2019)
        text = re.sub(r'平成(\d+)年', lambda m: str(1988 + int(m.group(1))) + '年', text)
        # 昭和の変換 (昭和: 1926-1989)
        text = re.sub(r'昭和(\d+)年', lambda m: str(1925 + int(m.group(1))) + '年', text)
        # 大正の変換 (大正: 1912-1926)
        text = re.sub(r'大正(\d+)年', lambda m: str(1911 + int(m.group(1))) + '年', text)
        return text
        
    def _extract_death_date(self, text):
        """テキストから死亡日を抽出してYYYY年MM月DD日形式に変換します。"""
        # 死亡年月日時のパターン
        patterns = [
            r'死亡年月日時\s*令和(\d+)年(\d+)月(\d+)日',
            r'死亡年月日\s*令和(\d+)年(\d+)月(\d+)日',
            r'死亡年月日時\s*平成(\d+)年(\d+)月(\d+)日',
            r'死亡年月日\s*平成(\d+)年(\d+)月(\d+)日',
            r'死亡年月日時\s*昭和(\d+)年(\d+)月(\d+)日',
            r'死亡年月日\s*昭和(\d+)年(\d+)月(\d+)日',
            r'死亡年月日時\s*令和(\d+)年(\d+)月中旬頃',
            r'死亡年月日\s*令和(\d+)年(\d+)月中旬頃',
            # 年号なしのパターンを追加
            r'死亡年月日時\s*(\d+)年(\d+)月(\d+)日',
            r'死亡年月日\s*(\d+)年(\d+)月(\d+)日',
            r'死亡年月日時\s*(\d+)年(\d+)月中旬頃',
            r'死亡年月日\s*(\d+)年(\d+)月中旬頃'
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text)
            if match:
                if len(match.groups()) == 3:
                    year = int(match.group(1))
                    month = match.group(2).zfill(2)
                    
                    # 日付が「中旬頃」の場合は15日とする
                    if '中旬頃' in match.group(0):
                        day = '15'
                    else:
                        day = match.group(3).zfill(2)
                    
                    # 年号なしの場合は令和として扱う
                    if not any(era in pattern for era in ['令和', '平成', '昭和', '大正']):
                        year = 2018 + year
                    
                    return f"{year}年{month}月{day}日"
                
        return None

    def _extract_name(self, text, alt_keys=None):
        """テキストから故人の名前を抽出します。alt_keysで追加のキーワードも探索できます。"""
        # 死亡者の氏名のパターン
        patterns = [
            r'死亡者の氏名\s*([^\n]+)',
            r'死亡者氏名\s*([^\n]+)',
            r'死亡者\s*氏名\s*([^\n]+)',
            r'死亡者\s*([^\n]+)',
            r'本人署名\s*([^\n]+)'  # 門徒誓約書用のパターンを追加
        ]
        if alt_keys:
            for key in alt_keys:
                patterns.append(rf'{key}\s*([^\n]+)')
        for pattern in patterns:
            match = re.search(pattern, text)
            if match:
                name = match.group(1).strip()
                # 余分な空白を削除
                name = re.sub(r'\s+', ' ', name)
                # 性別やその他の情報が含まれている場合は除去
                name = re.sub(r'\s*[男女]\s*$', '', name)
                return name
        return None

    def _extract_cremation_date(self, text):
        """テキストから火葬日を抽出してYYYY年MM月DD日形式に変換します。"""
        # 火葬年月日のパターン
        patterns = [
            r'火葬年月日\s*令和(\d+)年(\d+)月(\d+)日',
            r'火葬\s*年月日\s*令和(\d+)年(\d+)月(\d+)日',
            r'火葬年月日\s*平成(\d+)年(\d+)月(\d+)日',
            r'火葬\s*年月日\s*平成(\d+)年(\d+)月(\d+)日',
            r'火葬年月日\s*昭和(\d+)年(\d+)月(\d+)日',
            r'火葬\s*年月日\s*昭和(\d+)年(\d+)月(\d+)日',
            # 年号なしのパターンを追加
            r'火葬年月日\s*(\d+)年(\d+)月(\d+)日',
            r'火葬\s*年月日\s*(\d+)年(\d+)月(\d+)日'
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text)
            if match:
                year = int(match.group(1))
                month = match.group(2).zfill(2)
                day = match.group(3).zfill(2)
                
                # 年号なしの場合は令和として扱う
                if not any(era in pattern for era in ['令和', '平成', '昭和', '大正']):
                    year = 2018 + year
                
                return f"{year}年{month}月{day}日"
                
        return None

    def _extract_organization(self, ocr_result):
        """OCRテキストから組織名を抽出します。"""
        try:
            logger.debug("OCRテキストから組織名を抽出中")
            
            # 文書内の一般的な組織識別子
            patterns = [
                # 日本語パターン
                r'発行元[：:]\s*([^\n]+)',
                r'発行者[：:]\s*([^\n]+)',
                r'請求元[：:]\s*([^\n]+)',
                r'取引先[：:]\s*([^\n]+)',
                r'会社名[：:]\s*([^\n]+)',
                r'事業者名[：:]\s*([^\n]+)',
                r'企業名[：:]\s*([^\n]+)',
                r'組織名[：:]\s*([^\n]+)',
                r'屋号[：:]\s*([^\n]+)',
                r'法人名[：:]\s*([^\n]+)',
                r'支払先[：:]\s*([^\n]+)',
                r'振込先[：:]\s*([^\n]+)',
                r'販売元[：:]\s*([^\n]+)',
                
                # より広いパターン（より具体性が低いが、より多くのケースを捕捉できる可能性がある）
                r'株式会社([^\n]{1,20})',  # 「株式会社」に続く会社名
                r'有限会社([^\n]{1,20})',  # 「有限会社」に続く会社名
                r'合同会社([^\n]{1,20})',  # 「合同会社」に続く会社名
                
                # 標準的なレターヘッドの位置を探す（文書の上部）
                r'^([^\n]{1,50}株式会社[^\n]{0,30})\n',
                r'^([^\n]{1,50}有限会社[^\n]{0,30})\n',
                r'^([^\n]{1,50}合同会社[^\n]{0,30})\n',
                
                # その他のケース用のより一般的なパターン
                r'(\S+商店)',
                r'(\S+店)',
                
                # 葬儀や宗教関連文書用の特別なパターン
                r'葬儀社[：:]\s*([^\n]+)',
                r'((?:\S+葬儀社|葬儀\S+|セレモニー\S+))',
                r'((?:\S+寺院|寺院\S+|寺院\s+\S+))',
                r'((?:\S+石材|石材\S+))'
            ]
            
            # 各パターンを試す
            for pattern in patterns:
                match = re.search(pattern, ocr_result)
                if match:
                    org_name = match.group(1).strip()
                    
                    # 抽出した名前をクリーンアップ
                    org_name = re.sub(r'\s+', ' ', org_name)  # 空白を正規化
                    org_name = re.sub(r'[、。．・]$', '', org_name)  # 末尾の句読点を削除
                    
                    # 長すぎる場合は切り詰める
                    if len(org_name) > 30:
                        org_name = org_name[:30] + "..."
                    
                    logger.debug(f"組織名を見つけました: {org_name}")
                    return org_name
            
            # より詳細なアプローチとしてOpenAIを使用
            if self.config_manager.get('OPENAI_API_KEY'):
                try:
                    openai.api_key = self.config_manager.get('OPENAI_API_KEY')
                    
                    # OCR結果が長すぎる場合は切り詰める
                    max_length = 2000  # OpenAIのトークン制限
                    truncated_ocr = ocr_result[:max_length] if len(ocr_result) > max_length else ocr_result
                    
                    prompt = f"""
                    以下のOCRテキストから、組織名または発行元の名前を抽出してください。
                    組織名が見つからない場合は「不明」と返してください。
                    回答は組織名のみを返し、余計な説明は不要です。

                    OCRテキスト:
                    {truncated_ocr}
                    """
                    
                    response = openai.chat.completions.create(
                        model=self.config_manager.get('OPENAI_MODEL', 'gpt-4.1'),
                        messages=[
                            {"role": "system", "content": "OCRテキストから組織名を抽出します。組織名のみを返すか、見つからない場合は「不明」と返してください。"},
                            {"role": "user", "content": prompt}
                        ],
                        temperature=0.1,
                        max_tokens=50
                    )
                    
                    org_name = response.choices[0].message.content.strip()
                    
                    # OpenAIがより長い説明を返した場合、組織名だけを抽出
                    if len(org_name.split()) > 4:
                        # これは長い説明かもしれないので、最初の行やフレーズを取得
                        org_name = org_name.split("\n")[0].strip()
                        if len(org_name) > 30:
                            org_name = org_name[:30] + "..."
                    
                    logger.debug(f"OpenAIで抽出された組織名: {org_name}")
                    
                    if org_name == "不明" or "見つかりません" in org_name:
                        org_name = "不明"
                    
                    return org_name
                    
                except Exception as e:
                    logger.error(f"組織名抽出のためのOpenAI使用中のエラー: {e}")
            
            # 組織名が見つからない場合はデフォルトを返す
            logger.debug("OCRテキスト内に組織名が見つかりませんでした")
            return "不明"
            
        except Exception as e:
            logger.exception("組織名抽出エラー")
            return "不明"

    def _propose_file_name(self, ocr_result):
        """OCR結果とYAMLルールに基づいてファイル名を提案します。"""
        try:
            logger.debug("OCR結果に基づいてファイル名を提案中")
            
            if not ocr_result or ocr_result.strip() == "":
                logger.error("OCR結果が空です")
                return f"{datetime.now().strftime('%Y年%m月%d日')}_不明.pdf"
            
            if not self.yaml_rules:
                logger.error("YAMLルールが読み込まれていません")
                return f"{datetime.now().strftime('%Y年%m月%d日')}_エラー.pdf"
            
            rules = self.yaml_rules.get("ファイル命名のルール", [])
            if not rules:
                logger.error("YAMLにルールが見つかりません")
                return f"{datetime.now().strftime('%Y年%m月%d日')}_エラー.pdf"
                
            logger.debug(f"YAMLに{len(rules)}個のルールが見つかりました")
            
            # OCR結果の正規化：一般的なOCRエラーを修正
            ocr_result_normalized = self._normalize_ocr_result(ocr_result)
            
            # 日付を抽出
            western_date = self._extract_date(ocr_result_normalized)
            format_params = {
                "日付": western_date,
                "担当者": self.selected_person,
                "組織名": "不明",
                "タイトル": "不明"
            }
            
            # 名刺読み取りモードの場合は名刺用のルールを無条件で適用
            if self.business_card_mode:
                logger.info("名刺読み取りモード: 名刺用のルールを適用します")
                for rule in rules:
                    if rule.get("書類の種類") == "名刺":
                        return self._process_rule(rule, western_date, format_params, ocr_result_normalized)
            
            # YAMLルールに基づいて書類の種類を判定
            doc_type = None
            matching_rule = None
            
            for i, rule in enumerate(rules, 1):
                if not rule:
                    logger.warning(f"ルール {i} がNoneです")
                    continue
                    
                current_rule = rule.get("ルール", rule) if "ルール" in rule else rule
                if not current_rule:
                    logger.warning(f"ルール {i} のcurrent_ruleがNoneです")
                    continue
                    
                logger.debug(f"ルール {i} をチェック中: {current_rule.get('説明', 'Unnamed rule')}")
                
                if "正規表現" in current_rule:
                    pattern = current_rule["正規表現"]
                    logger.debug(f"パターンを適用中: {pattern}")
                    logger.debug(f"テキスト: {ocr_result_normalized[:200]}...")
                    match = re.search(pattern, ocr_result_normalized, re.IGNORECASE)
                    if match:
                        logger.info(f"ルールが一致しました: {current_rule.get('説明', 'Unnamed rule')}")
                        logger.info(f"適用された正規表現: {pattern}")
                        logger.info(f"マッチした文字列: {match.group(0)}")
                        return self._process_rule(current_rule, western_date, format_params, ocr_result_normalized)
                    else:
                        logger.debug(f"パターンに一致しませんでした: {pattern}")
            
            # 特定のルールに一致しない場合はデフォルトルールを適用
            logger.info("特定のルールに一致しませんでした。デフォルトルールを適用します。")
            default_rule_dict = next((r for r in rules if "デフォルト" in r), {})
            default_rule = default_rule_dict.get("デフォルト", {})
            
            if not default_rule:
                logger.warning("デフォルトルールが見つかりませんでした")
                return f"{western_date}_不明.pdf"
            
            return self._process_rule(default_rule, western_date, format_params, ocr_result_normalized)
            
        except Exception as e:
            logger.exception("ファイル名提案エラー")
            return f"{datetime.now().strftime('%Y年%m月%d日')}_エラー.pdf"
    
    def _normalize_ocr_result(self, ocr_result):
        """OCR結果を正規化します。"""
        # 改行を空白に置換
        normalized = re.sub(r'\n+', ' ', ocr_result)
        # 連続する空白を1つに置換
        normalized = re.sub(r'\s+', ' ', normalized)
        # 前後の空白を削除
        normalized = normalized.strip()
        
        logger.debug(f"正規化前のOCR結果: {ocr_result}")
        logger.debug(f"正規化後のOCR結果: {normalized}")
        
        return normalized

    def _auto_detect_person(self, ocr_text):
        """OCRテキストの内容に基づいて担当者を自動検出します。"""
        person_keywords = {
            "前田至法": ["翠江会", "前田至法"],
            "前田百音": ["安田女子中学高等学校", "前田百音"],
            "前田蓮生": ["広島学院", "前田蓮生"],
            "前田ツヤ": ["前田ツヤ"],
            "前田至正": ["前田至正"],
            "前田純代": ["前田純代"]
        }
        
        for person, keywords in person_keywords.items():
            for keyword in keywords:
                if keyword in ocr_text:
                    logger.debug(f"自動検出された担当者: {person}、キーワード: {keyword}に基づく")
                    return person
        
        logger.debug("担当者が自動検出されませんでした")
        return ""  # 一致がない場合は空文字
    
    def _extract_date(self, text):
        """テキストから日付を抽出してYYYY年MM月DD日形式に変換します。"""
        # 様々な日付形式を試す
        
        # 令和〇年〇月〇日 形式（優先度を上げる）
        date_match = re.search(r'令和\s*(\d+)\s*年\s*(\d+)\s*月\s*(\d+)\s*日', text)
        if date_match:
            year = 2018 + int(date_match.group(1))
            month = date_match.group(2).zfill(2)
            day = date_match.group(3).zfill(2)
            logger.debug(f"令和の日付を検出: {year}年{month}月{day}日")
            return f"{year}年{month}月{day}日"
            
        # 令〇年〇月〇日 形式（OCR誤認識対応）
        date_match = re.search(r'令\s*(\d+)\s*年\s*(\d+)\s*月\s*(\d+)\s*日', text)
        if date_match:
            year = 2018 + int(date_match.group(1))
            month = date_match.group(2).zfill(2)
            day = date_match.group(3).zfill(2)
            logger.debug(f"令の日付を検出: {year}年{month}月{day}日")
            return f"{year}年{month}月{day}日"
            
        # 平成〇年〇月〇日 形式
        date_match = re.search(r'平成\s*(\d+)\s*年\s*(\d+)\s*月\s*(\d+)\s*日', text)
        if date_match:
            year = 1988 + int(date_match.group(1))
            month = date_match.group(2).zfill(2)
            day = date_match.group(3).zfill(2)
            logger.debug(f"平成の日付を検出: {year}年{month}月{day}日")
            return f"{year}年{month}月{day}日"
            
        # 昭和〇年〇月〇日 形式
        date_match = re.search(r'昭和\s*(\d+)\s*年\s*(\d+)\s*月\s*(\d+)\s*日', text)
        if date_match:
            year = 1925 + int(date_match.group(1))
            month = date_match.group(2).zfill(2)
            day = date_match.group(3).zfill(2)
            logger.debug(f"昭和の日付を検出: {year}年{month}月{day}日")
            return f"{year}年{month}月{day}日"
            
        # YYYY年MM月DD日 形式
        date_match = re.search(r'(\d{4})年(\d{1,2})月(\d{1,2})日', text)
        if date_match:
            year = date_match.group(1)
            month = date_match.group(2).zfill(2)
            day = date_match.group(3).zfill(2)
            logger.debug(f"西暦の日付を検出: {year}年{month}月{day}日")
            return f"{year}年{month}月{day}日"
            
        # YYYY/MM/DD または YYYY-MM-DD 形式
        date_match = re.search(r'(\d{4})[/-](\d{1,2})[/-](\d{1,2})', text)
        if date_match:
            year = date_match.group(1)
            month = date_match.group(2).zfill(2)
            day = date_match.group(3).zfill(2)
            logger.debug(f"スラッシュ区切りの日付を検出: {year}年{month}月{day}日")
            return f"{year}年{month}月{day}日"
            
        # 日付が見つからない場合は現在の日付を使用
        current_date = datetime.now().strftime('%Y年%m月%d日')
        logger.debug(f"日付が見つからないため現在の日付を使用: {current_date}")
        return current_date
    
    def _process_rule(self, rule, western_date, format_params, ocr_result):
        """ルールを処理し、ファイル名を生成します。"""
        try:
            logger.debug(f"ルール処理を開始: {rule.get('説明', 'Unnamed rule')}")
            
            # 名刺の場合は特別な処理
            if rule.get('書類の種類') == '名刺':
                logger.debug(f"名刺の処理を開始: {ocr_result}")
                
                # AIに名刺情報の抽出を依頼
                prompt = f"""
                以下の名刺のOCR結果から、以下の情報を抽出してください：

                1. 氏名漢字：名刺に記載されている漢字の氏名
                2. 氏名ふりがな：名刺に記載されているふりがな（括弧内の文字列）
                3. 勤務先：会社名や組織名
                4. 役職：役職名や学位
                   - 役職は完全な形で抽出してください
                   - 例：「特任准教授 博士（情報科学）」の場合、「特任准教授 博士（情報科学）」と完全に抽出してください
                   - 「特任准教授」や「教授」「准教授」などの役職名は必ず含めてください
                   - 学位（博士、修士など）がある場合は、役職と合わせて記載してください
                   - 役職と学位の間は半角スペースで区切ってください
                   - 注意：役職名は省略せず、完全な形で抽出してください
                   - 注意：名刺に記載されている役職名は必ず含めてください
                   - 注意：役職名が見つからない場合は、空文字列を返してください
                   - 重要：「特任准教授」という役職名がある場合は、必ず「特任准教授」を含めてください
                   - 重要：役職名は「特任准教授」「教授」「准教授」「講師」などの完全な形で抽出してください
                   - 重要：役職名を省略したり、一部だけを抽出したりしないでください

                抽出した情報は以下のJSON形式で返してください：
                {{
                    "氏名漢字": "抽出した氏名漢字",
                    "氏名ふりがな": "抽出したふりがな（括弧内の文字列）",
                    "勤務先": "抽出した勤務先",
                    "役職": "抽出した役職（学位を含む）"
                }}

                OCR結果：
                {ocr_result}
                """

                try:
                    # OpenAI APIを呼び出して情報を抽出
                    response = self._call_openai_api(prompt)
                    if response:
                        # JSON文字列をパース
                        import json
                        extracted_info = json.loads(response)
                        
                        # 抽出した情報をformat_paramsに設定
                        format_params.update(extracted_info)
                        logger.debug(f"AIによる名刺情報の抽出結果: {extracted_info}")
                    else:
                        logger.error("AIによる名刺情報の抽出に失敗しました")
                        return None
                except Exception as e:
                    logger.error(f"AIによる名刺情報の抽出中にエラーが発生: {str(e)}")
                    return None
            
            # 命名ルールの取得
            naming_rule = rule.get('命名ルール')
            if not naming_rule:
                logger.error("命名ルールが指定されていません")
                return None

            # ファイル名の生成
            try:
                new_name = naming_rule.format(**format_params)
                logger.debug(f"生成されたファイル名: {new_name}")
                return new_name
            except KeyError as e:
                logger.error(f"フォーマットパラメータの不足: {e}")
                return None
            except Exception as e:
                logger.error(f"ファイル名生成中にエラーが発生: {e}")
                return None

        except Exception as e:
            logger.error(f"ルール処理中にエラーが発生しました: {str(e)}")
            logger.exception("詳細なエラー情報:")
            return None
    
    def _extract_value(self, text, key):
        """OCRテキストから特定の値を抽出します。"""
        patterns = {
            "内容": [
                r"内容[：:]\s*([^\n]+)",
                r"件名[：:]\s*([^\n]+)",
                r"タイトル[：:]\s*([^\n]+)"
            ],
            "金額": [
                r"金額[：:]\s*(\d+(?:,\d+)*)",
                r"合計[：:]\s*(\d+(?:,\d+)*)",
                r"￥\s*(\d+(?:,\d+)*)"
            ],
            "取引先": [
                r"取引先[：:]\s*([^\n]+)",
                r"発行元[：:]\s*([^\n]+)",
                r"発行者[：:]\s*([^\n]+)"
            ],
            "申請者": [
                r"申請者[：:]\s*([^\n]+)",
                r"申請人[：:]\s*([^\n]+)",
                r"氏名[：:]\s*([^\n]+)"
            ],
            "年度": [
                r"年度[：:]\s*([^\n]+)",
                r"(\d{4})年度",
                r"令和(\d+)年度"
            ]
        }
        
        if key in patterns:
            for pattern in patterns[key]:
                match = re.search(pattern, text)
                if match:
                    value = match.group(1).strip()
                    if key == "金額":
                        value = value.replace(",", "")
                    elif key == "年度" and "令和" in pattern:
                        value = f"令和{value}年度"
                    return value
        
        return None
    
    def _rename_pdf(self, pdf_file_path, new_pdf_name):
        """PDFファイルを新しい名前でリネームします。"""
        try:
            logger.debug(f"リネーム中: {pdf_file_path} -> {new_pdf_name}")
            
            if not os.path.exists(pdf_file_path):
                logger.error(f"ファイルが存在しません: {pdf_file_path}")
                return False
            
            # ファイル名のサニタイズ（無効な文字を削除）
            new_pdf_name = re.sub(r'[\\/:*?"<>|&]', '_', new_pdf_name)
            
            # .pdf拡張子を確保
            if not new_pdf_name.lower().endswith('.pdf'):
                new_pdf_name += '.pdf'
            
            # ターゲットパスの取得
            directory = os.path.dirname(pdf_file_path)
            new_file_path = os.path.join(directory, new_pdf_name)
            
            # 名前が同じ場合はスキップ
            if os.path.normpath(pdf_file_path) == os.path.normpath(new_file_path):
                logger.info(f"ファイルはすでに正しい名前です: {pdf_file_path}")
                return new_file_path
            
            # 名前の衝突を処理するためにカウンターを追加
            if os.path.exists(new_file_path):
                base_name, ext = os.path.splitext(new_pdf_name)
                counter = 1
                while os.path.exists(os.path.join(directory, f"{base_name}_{counter}{ext}")):
                    counter += 1
                new_file_path = os.path.join(directory, f"{base_name}_{counter}{ext}")
            
            # リネームを実行
            os.rename(pdf_file_path, new_file_path)
            logger.info(f"ファイルは正常にリネームされました: {new_file_path}")
            
            # ファイルシステムの更新を許可するための小さな遅延
            time.sleep(0.5)
            
            return new_file_path
            
        except Exception as e:
            logger.exception(f"ファイルのリネームエラー: {pdf_file_path}")
            return False
    
    def _extract_submit_date(self, text):
        """テキストから提出日を抽出してYYYY年MM月DD日形式に変換します。"""
        # 提出日のパターン
        patterns = [
            r'提出日\s*西暦\s*(\d{4})年(\d{1,2})月(\d{1,2})日',
            r'提出日\s*令和(\d+)年(\d+)月(\d+)日',
            r'提出日\s*平成(\d+)年(\d+)月(\d+)日',
            r'提出日\s*昭和(\d+)年(\d+)月(\d+)日',
            r'提出日\s*(\d{4})年(\d{1,2})月(\d{1,2})日',
            r'西暦\s*(\d{4})年(\d{1,2})月(\d{1,2})日\s*提出',
            r'令和(\d+)年(\d+)月(\d+)日\s*提出',
            r'平成(\d+)年(\d+)月(\d+)日\s*提出',
            r'昭和(\d+)年(\d+)月(\d+)日\s*提出',
            r'(\d{4})年(\d{1,2})月(\d{1,2})日\s*提出',
            r'西暦\s*(\d{4})年(\d{1,2})月(\d{1,2})日'  # 西暦で始まるパターンを追加
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text)
            if match:
                if "西暦" in pattern:
                    year = match.group(1)
                elif "令和" in pattern:
                    year = str(2018 + int(match.group(1)))
                elif "平成" in pattern:
                    year = str(1988 + int(match.group(1)))
                elif "昭和" in pattern:
                    year = str(1925 + int(match.group(1)))
                else:
                    year = match.group(1)
                    
                month = match.group(2).zfill(2)
                day = match.group(3).zfill(2)
                
                return f"{year}年{month}月{day}日"
                
        return None

    def _call_openai_api(self, prompt):
        """OpenAI APIを呼び出してファイル名を生成します。"""
        try:
            logger.debug("OpenAI API呼び出しを開始します")
            logger.debug(f"プロンプト: {prompt}")

            # 設定からAPIキーを取得
            api_key = self.config_manager.get('OPENAI_API_KEY')
            if not api_key:
                logger.error("OpenAI APIキーが設定されていません")
                return None

            logger.debug("OpenAIクライアントを初期化します")
            # OpenAIクライアントの初期化
            client = OpenAI(api_key=api_key)

            # モデル名を取得（デフォルトはgpt-4）
            model = self.config_manager.get('OPENAI_MODEL', 'gpt-4')
            logger.debug(f"使用するモデル: {model}")

            # APIリクエストの送信
            logger.debug("APIリクエストを送信します")
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": "あなたはPDFファイルの名前を生成するアシスタントです。与えられたOCR結果から適切な情報を抽出し、指定された形式でファイル名を生成してください。"},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                max_tokens=150
            )

            # レスポンスからテキストを抽出
            if response.choices and len(response.choices) > 0:
                result = response.choices[0].message.content.strip()
                logger.debug(f"APIレスポンス: {result}")
                return result
            else:
                logger.error("OpenAI APIからの応答が空です")
                return None

        except Exception as e:
            logger.error(f"OpenAI API呼び出し中にエラーが発生しました: {str(e)}")
            logger.exception("詳細なエラー情報:")
            return None


class PDFRenamerApp:
    """PDFリネーマーのメインアプリケーションクラス。"""
    
    def __init__(self, root):
        """アプリケーションUIとコンポーネントを初期化します。"""
        self.root = root
        self.config_manager = ConfigManager()
        self.status_queue = queue.Queue()
        self.selected_person = tk.StringVar(value=self.config_manager.get('DEFAULT_PERSON'))
        self.selected_model = tk.StringVar(value=self.config_manager.get('OPENAI_MODEL'))
        self.processing_flag = tk.BooleanVar(value=False)
        self.business_card_mode = tk.BooleanVar(value=False)  # 名刺読み取りモード用の変数を追加
        
        # UIのセットアップ
        self._setup_ui()
        
        # 起動時に設定を検証
        self._validate_config_on_startup()
    
    def _setup_ui(self):
        """ユーザーインターフェースをセットアップします。"""
        self.root.title(f"PDF Renamer {APP_VERSION}")
        self.root.geometry("700x600")
        self.root.minsize(600, 400)
        
        # ウィンドウを中央に配置
        screen_width = self.root.winfo_screenwidth()
        screen_height = self.root.winfo_screenheight()
        x = (screen_width - 700) // 2
        y = (screen_height - 600) // 2 - 100  # 中央よりやや上に調整
        self.root.geometry(f"700x600+{x}+{y}")
        
        # スタイルの作成と設定
        style = ttk.Style()
        style.configure("TButton", padding=6, relief="flat", font=("Helvetica", 10))
        style.configure("TLabel", font=("Helvetica", 10))
        style.configure("Header.TLabel", font=("Helvetica", 12, "bold"))
        
        # パディング付きのメインフレーム
        main_frame = ttk.Frame(self.root, padding=10)
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # メニューバー
        self._create_menu()
        
        # 担当者選択フレーム
        person_frame = ttk.Frame(main_frame)
        person_frame.pack(fill=tk.X, pady=(0, 10))
        
        person_label = ttk.Label(
            person_frame,
            text="担当者: ",
            style="TLabel"
        )
        person_label.pack(side=tk.LEFT)
        
        persons = self.config_manager.get('PERSONS', [])
        if not persons or (len(persons) == 1 and not persons[0]):
            persons = ["担当者自動設定"]
            
        person_menu = ttk.Combobox(
            person_frame,
            textvariable=self.selected_person,
            values=["担当者自動設定"] + persons,
            state="readonly",
            width=20
        )
        person_menu.pack(side=tk.LEFT, padx=(5, 0))
        person_menu.bind("<<ComboboxSelected>>", self._on_person_changed)
        
        # フォルダ選択フレーム
        folder_frame = ttk.Frame(main_frame)
        folder_frame.pack(fill=tk.X, pady=(0, 10))
        
        folder_label = ttk.Label(
            folder_frame,
            text="PDFフォルダ: ",
            style="TLabel"
        )
        folder_label.pack(side=tk.LEFT)
        
        self.folder_var = tk.StringVar(value=self.config_manager.get('PDF_FOLDER_PATH', ''))
        folder_entry = ttk.Entry(
            folder_frame,
            textvariable=self.folder_var,
            width=50
        )
        folder_entry.pack(side=tk.LEFT, padx=(5, 5), fill=tk.X, expand=True)
        
        browse_button = ttk.Button(
            folder_frame,
            text="参照...",
            command=self._browse_folder
        )
        browse_button.pack(side=tk.LEFT)
        
        # 進捗セクション
        progress_frame = ttk.Frame(main_frame)
        progress_frame.pack(fill=tk.X, pady=(0, 10))
        
        self.progress_label = ttk.Label(
            progress_frame,
            text="進捗: 0%",
            style="TLabel"
        )
        self.progress_label.pack(anchor=tk.W)
        
        self.progress_bar = ttk.Progressbar(
            progress_frame,
            orient=tk.HORIZONTAL,
            length=100,
            mode='determinate'
        )
        self.progress_bar.pack(fill=tk.X)
        
        # ステータステキストエリア
        status_frame = ttk.LabelFrame(main_frame, text="処理状況")
        status_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 10))
        
        self.status_text = scrolledtext.ScrolledText(
            status_frame,
            wrap=tk.WORD,
            width=80,
            height=15
        )
        self.status_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # ボタンセクション
        button_frame = ttk.Frame(main_frame)
        button_frame.pack(fill=tk.X)
        
        self.rename_button = ttk.Button(
            button_frame,
            text="リネーム開始",
            command=self._start_renaming
        )
        self.rename_button.pack(side=tk.LEFT, padx=(0, 10))
        
        self.open_folder_button = ttk.Button(
            button_frame,
            text="フォルダを開く",
            command=lambda: self._open_file(self.folder_var.get())
        )
        self.open_folder_button.pack(side=tk.LEFT)
        
        # ステータスキューのポーリングを開始
        self._poll_status_queue()
    
    def _create_menu(self):
        """アプリケーションメニューバーを作成します。"""
        menubar = tk.Menu(self.root)
        self.root.config(menu=menubar)
        
        # ファイルメニュー
        file_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="ファイル", menu=file_menu)
        file_menu.add_command(label="PDFフォルダ選択...", command=self._browse_folder)
        file_menu.add_separator()
        file_menu.add_command(label=".envを編集", command=lambda: self._open_file(self.config_manager.env_path))
        file_menu.add_command(label="YAMLルールを編集", command=lambda: self._open_file(self.config_manager.get('YAML_FILE')))
        file_menu.add_separator()
        file_menu.add_command(label="終了", command=self.root.quit)
        
        # 設定メニュー
        settings_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="設定", menu=settings_menu)
        
        # 名刺読み取りモードの設定を追加
        settings_menu.add_checkbutton(
            label="名刺の読み取り",
            variable=self.business_card_mode,
            command=self._toggle_business_card_mode
        )
        settings_menu.add_separator()
        
        # .envバックアップメニュー
        settings_menu.add_command(label=".envをバックアップ", command=self._backup_env_file)
        settings_menu.add_command(label="rename_rules.yamlをバックアップ", command=self._backup_yaml_file)
        settings_menu.add_separator()
        
        # 担当者サブメニュー
        person_menu = tk.Menu(settings_menu, tearoff=0)
        settings_menu.add_cascade(label="担当者の設定", menu=person_menu)
        
        # 自動設定オプションを追加
        person_menu.add_radiobutton(
            label="担当者自動設定",
            value="担当者自動設定",
            variable=self.selected_person,
            command=lambda: self._set_person("担当者自動設定")
        )
        
        person_menu.add_separator()
        
        # 設定からすべての担当者を追加
        persons = self.config_manager.get('PERSONS', [])
        for person in persons:
            if person:  # 空の文字列をスキップ
                person_menu.add_radiobutton(
                    label=person,
                    value=person,
                    variable=self.selected_person,
                    command=lambda p=person: self._set_person(p)
                )
        
        # AIモデルサブメニュー
        ai_model_menu = tk.Menu(settings_menu, tearoff=0)
        settings_menu.add_cascade(label="AIモデルの設定", menu=ai_model_menu)
        
        # 利用可能なAIモデル
        available_models = [
            "gpt-4.1",
            "gpt-4",
            "gpt-3.5-turbo",
            "gpt-3.5-turbo-16k"
        ]
        
        # モデル選択オプションを追加
        for model in available_models:
            ai_model_menu.add_radiobutton(
                label=model,
                value=model,
                variable=self.selected_model,
                command=lambda m=model: self._set_ai_model(m)
            )
        
        # ヘルプメニュー
        help_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="ヘルプ", menu=help_menu)
        help_menu.add_command(label="バージョン情報", command=self._show_about)
        help_menu.add_command(label="使い方", command=self._show_help)
    
    def _browse_folder(self):
        """PDFフォルダを選択するためのフォルダブラウザダイアログを開きます。"""
        current_folder = self.folder_var.get()
        folder = askdirectory(
            initialdir=current_folder if current_folder else "/",
            title="PDFファイルのあるフォルダを選択"
        )
        
        if folder:
            self.folder_var.set(folder)
            self.config_manager.set('PDF_FOLDER_PATH', folder)
            # 選択したフォルダを表示するためにステータステキストを更新
            self._add_to_status(f"PDFフォルダを設定しました: {folder}\n")
    
    def _open_file(self, file_path):
        """ファイルやフォルダをデフォルトのアプリケーションで開きます。"""
        if not file_path:
            messagebox.showerror("エラー", "ファイルまたはフォルダが指定されていません。")
            return
            
        try:
            logger.debug(f"ファイルを開いています: {file_path}")
            
            if platform.system() == "Windows":
                os.startfile(file_path)
            elif platform.system() == "Linux":
                subprocess.run(["xdg-open", file_path], check=True)
            elif platform.system() == "Darwin":  # macOS
                subprocess.run(["open", file_path], check=True)
                
        except Exception as e:
            logger.exception(f"ファイルを開くエラー: {file_path}")
            messagebox.showerror("エラー", f"ファイルを開けませんでした: {e}")
    
    def _set_person(self, person):
        """選択された担当者を設定しUIを更新します。"""
        self.selected_person.set(person)
        logger.info(f"担当者を設定: {person}")
        self._add_to_status(f"担当者を「{person}」に設定しました\n")
    
    def _on_person_changed(self, event):
        """担当者の選択変更イベントを処理します。"""
        self._set_person(self.selected_person.get())
    
    def _add_to_status(self, message):
        """ステータステキストエリアにメッセージを追加します。"""
        self.status_text.insert(tk.END, message)
        self.status_text.see(tk.END)
    
    def _poll_status_queue(self):
        """ステータスキューをポーリングしUIを更新します。"""
        try:
            while not self.status_queue.empty():
                message = self.status_queue.get(0)
                self._add_to_status(message)
                
        except queue.Empty:
            pass
            
        finally:
            # 次のポーリングをスケジュール
            self.root.after(100, self._poll_status_queue)
    
    def _validate_config_on_startup(self):
        """アプリケーション起動時に設定を検証します。"""
        missing = self.config_manager.validate_config()
        
        if missing:
            message = "以下の設定が不足しています:\n\n" + "\n".join([f"- {item}" for item in missing])
            message += "\n\n設定ファイル (.env) を編集して必要な情報を追加してください。"
            
            messagebox.showwarning("設定の問題", message)
    
    def _start_renaming(self):
        """PDFリネーム処理を開始します。"""
        if self.processing_flag.get():
            messagebox.showinfo("処理中", "現在処理中です。完了するまでお待ちください。")
            return
            
        # UIからPDFフォルダを更新
        pdf_folder = self.folder_var.get()
        self.config_manager.set('PDF_FOLDER_PATH', pdf_folder)
        
        # フォルダが存在するか検証
        if not pdf_folder or not os.path.isdir(pdf_folder):
            messagebox.showerror("エラー", f"PDFフォルダが存在しません: {pdf_folder}")
            return
            
        # フォルダ内のPDFファイルを検索
        pdf_files = [f for f in os.listdir(pdf_folder) if f.lower().endswith('.pdf')]
        
        if not pdf_files:
            messagebox.showinfo("情報", f"フォルダ内にPDFファイルがありません: {pdf_folder}")
            return
            
        # UIをリセット
        self.progress_bar['value'] = 0
        self.progress_label.config(text="進捗: 0%")
        self.status_text.delete(1.0, tk.END)
        self._add_to_status(f"PDFフォルダ: {pdf_folder}\n")
        self._add_to_status(f"PDF数: {len(pdf_files)}個\n")
        self._add_to_status(f"担当者: {self.selected_person.get()}\n")
        self._add_to_status("リネーム処理を開始します...\n")
        self._add_to_status("-" * 50 + "\n")
        
        # 処理フラグを設定
        self.processing_flag.set(True)
        
        # 処理中はスタートボタンを無効化
        self.rename_button.config(state="disabled")
        
        # 別のスレッドで処理を開始
        threading.Thread(
            target=self._process_pdf_files,
            args=(pdf_folder, pdf_files),
            daemon=True
        ).start()
    
    def _process_pdf_files(self, pdf_folder, pdf_files):
        """別のスレッドですべてのPDFファイルを処理します。"""
        try:
            processor = PDFProcessor(
                self.config_manager,
                self.selected_person.get(),
                self.status_queue,
                self.business_card_mode.get()  # 名刺読み取りモードの状態を渡す
            )
            
            total_files = len(pdf_files)
            completed = 0
            successful = 0
            
            # 最適なワーカー数を計算
            max_workers = min(os.cpu_count() or 4, 4)  # APIレート制限を避けるために4ワーカーに制限
            
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = {
                    executor.submit(
                        processor.process_pdf, 
                        os.path.join(pdf_folder, pdf_file)
                    ): pdf_file 
                    for pdf_file in pdf_files
                }
                
                for future in as_completed(futures):
                    try:
                        result = future.result()
                        if result:
                            successful += 1
                    except Exception as e:
                        pdf_file = futures[future]
                        error_msg = f"処理エラー: {pdf_file} - {str(e)}\n"
                        logger.error(error_msg)
                        self.status_queue.put(error_msg)
                        
                    completed += 1
                    progress_value = (completed / total_files) * 100
                    
                    # スレッドセーフな方法でUIを更新
                    self.root.after(0, lambda val=progress_value: 
                        self._update_progress(val, completed, successful, total_files))
            
            # 最終ステータス更新
            summary = (
                f"\n{'-' * 50}\n"
                f"処理完了: {completed}/{total_files} ファイル\n"
                f"成功: {successful} ファイル\n"
                f"失敗: {completed - successful} ファイル\n"
            )
            self.status_queue.put(summary)
            
        except Exception as e:
            logger.exception("PDF処理スレッドのエラー")
            self.status_queue.put(f"処理全体のエラー: {str(e)}\n")
            
        finally:
            # スレッドセーフな方法でUIをリセット
            self.root.after(0, self._reset_after_processing)
    
    def _update_progress(self, value, completed, successful, total):
        """プログレスバーとラベルを更新します。"""
        self.progress_bar['value'] = value
        self.progress_label.config(
            text=f"進捗: {int(value)}% ({completed}/{total}件, 成功: {successful}件)"
        )
    
    def _reset_after_processing(self):
        """処理完了後にUI状態をリセットします。"""
        self.processing_flag.set(False)
        self.rename_button.config(state="normal")
    
    def _show_about(self):
        """バージョン情報ダイアログを表示します。"""
        about_text = (
            f"PDF Renamer {APP_VERSION}\n\n"
            "このアプリケーションはPDFファイルをOCRにより解析し、\n"
            "内容に応じて自動的にリネームします。\n\n"
            "• Google Cloud Vision API: OCR処理\n"
            "• OpenAI API: ファイル名生成\n"
            "• YAMLルール: カスタマイズ可能なリネームルール\n\n"
            "© 2025"
        )
        messagebox.showinfo("バージョン情報", about_text)
    
    def _show_help(self):
        """ヘルプダイアログを表示します。"""
        help_text = (
            "使い方：\n\n"
            "1. PDFフォルダを選択します。\n"
            "2. 担当者を選択します（自動設定も可能）。\n"
            "3. 「リネーム開始」ボタンをクリックします。\n\n"
            "フォルダ内のすべてのPDFファイルが処理され、内容に基づいて\n"
            "自動的にリネームされます。処理状況は画面下部に表示されます。\n\n"
            "設定：\n"
            "• .envファイル: 基本設定\n"
            "• rename_rules.yaml: リネームルール\n\n"
            "これらのファイルは「ファイル」メニューから編集できます。"
        )
        messagebox.showinfo("使い方", help_text)
    
    def _set_ai_model(self, model):
        """AIモデルを設定し、設定を更新します。"""
        try:
            self.config_manager.set('OPENAI_MODEL', model)
            self.config_manager.save_config()
            logger.info(f"AIモデルを設定: {model}")
            self._add_to_status(f"AIモデルを「{model}」に設定しました\n")
        except Exception as e:
            logger.error(f"AIモデル設定エラー: {e}")
            messagebox.showerror("エラー", f"AIモデルの設定に失敗しました: {e}")

    def _backup_env_file(self):
        """現在の.envファイルをバックアップします。"""
        try:
            env_path = self.config_manager.env_path
            if not os.path.exists(env_path):
                messagebox.showerror("エラー", ".envファイルが見つかりません。")
                return

            # バックアップファイル名を生成（日本語の日時形式）
            timestamp = datetime.now().strftime('%Y年%m月%d日%H時%M分%S秒')
            
            # 保存されているバックアップ先を取得（なければデフォルトのbackupsディレクトリ）
            saved_backup_dir = self.config_manager.get('BACKUP_DIR')
            default_backup_dir = saved_backup_dir if saved_backup_dir else os.path.join(SCRIPT_DIR, 'backups')
            
            # バックアップ先のディレクトリを選択
            backup_dir = askdirectory(
                initialdir=default_backup_dir,
                title=".envファイルのバックアップ先を選択"
            )
            
            if not backup_dir:  # キャンセルされた場合
                return
                
            # バックアップ先を.envに保存
            self.config_manager.set('BACKUP_DIR', backup_dir)
            self.config_manager.save_config()
            
            os.makedirs(backup_dir, exist_ok=True)
            backup_path = os.path.join(backup_dir, f'.env.backup({timestamp})')

            # ファイルをコピー
            shutil.copy2(env_path, backup_path)
            
            logger.info(f".envファイルをバックアップしました: {backup_path}")
            self._add_to_status(f".envファイルをバックアップしました: {backup_path}\n")
            messagebox.showinfo("成功", f".envファイルをバックアップしました:\n{backup_path}")

        except Exception as e:
            logger.exception(".envファイルのバックアップ中にエラーが発生しました")
            messagebox.showerror("エラー", f".envファイルのバックアップに失敗しました:\n{str(e)}")

    def _backup_yaml_file(self):
        """rename_rules.yamlファイルをバックアップします。"""
        try:
            yaml_path = self.config_manager.get('YAML_FILE')
            if not os.path.exists(yaml_path):
                messagebox.showerror("エラー", "YAMLルールファイルが見つかりません。")
                return

            # バックアップファイル名を生成（日本語の日時形式）
            timestamp = datetime.now().strftime('%Y年%m月%d日%H時%M分%S秒')
            
            # 保存されているバックアップ先を取得（なければデフォルトのbackupsディレクトリ）
            saved_backup_dir = self.config_manager.get('BACKUP_DIR')
            default_backup_dir = saved_backup_dir if saved_backup_dir else os.path.join(SCRIPT_DIR, 'backups')
            
            # バックアップ先のディレクトリを選択
            backup_dir = askdirectory(
                initialdir=default_backup_dir,
                title="YAMLルールファイルのバックアップ先を選択"
            )
            
            if not backup_dir:  # キャンセルされた場合
                return
                
            # バックアップ先をYAMLに保存
            self.config_manager.set('BACKUP_DIR', backup_dir)
            self.config_manager.save_config()
            
            os.makedirs(backup_dir, exist_ok=True)
            backup_path = os.path.join(backup_dir, f'rename_rules.yaml.backup({timestamp})')

            # ファイルをコピー
            shutil.copy2(yaml_path, backup_path)
            
            logger.info(f"YAMLルールファイルをバックアップしました: {backup_path}")
            self._add_to_status(f"YAMLルールファイルをバックアップしました: {backup_path}\n")
            messagebox.showinfo("成功", f"YAMLルールファイルをバックアップしました:\n{backup_path}")

        except Exception as e:
            logger.exception("YAMLルールファイルのバックアップ中にエラーが発生しました")
            messagebox.showerror("エラー", f"YAMLルールファイルのバックアップに失敗しました:\n{str(e)}")

    def _toggle_business_card_mode(self):
        """名刺読み取りモードの切り替えを処理します。"""
        mode = "有効" if self.business_card_mode.get() else "無効"
        logger.info(f"名刺読み取りモードを{mode}に設定")
        self._add_to_status(f"名刺読み取りモードを{mode}に設定しました\n")


def main():
    """アプリケーションのエントリーポイント。"""
    try:
        # ルートウィンドウの設定
        root = tk.Tk()
        app = PDFRenamerApp(root)
        
        # メインループを開始
        root.mainloop()
        
    except Exception as e:
        logger.exception("アプリケーションエラー")
        messagebox.showerror("アプリケーションエラー", str(e))


if __name__ == "__main__":
    main()
