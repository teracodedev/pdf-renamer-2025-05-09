"""
PDF Renamer - OCR、Google Cloud Vision、およびOpenAIを使用してPDFファイルを
内容に基づいて自動的にリネームするための強化ツール（パフォーマンス最適化版）        

このアプリケーションはOCRを使用してPDFからテキストを抽出し、AIで内容を分析し、
カスタマイズ可能なルールに従って意味のあるファイル名を生成します。
。

作者: Claude（ユーザーのオリジナルに基づく）
日付: 2025年6月4日（パフォーマンス最適化版）
"""

import os
import sys
import time
import logging
import tkinter as tk
from tkinter import ttk, messagebox, scrolledtext, filedialog
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
import google.api_core.exceptions
from openai import OpenAI
import pytesseract
from PIL import Image
import pdf2image
import json
import traceback
import shutil
import gc
from concurrent.futures import ThreadPoolExecutor, as_completed

# ============================================================================
# 設定とセットアップ
# ============================================================================

# スクリプトディレクトリの設定
SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_ENV_PATH = SCRIPT_DIR / '.env'
DEFAULT_YAML_PATH = SCRIPT_DIR / 'rename_rules.yaml'
DEFAULT_IMAGE_PATH = SCRIPT_DIR / 'temp_images'
APP_VERSION = "2025年6月4日バージョン（パフォーマンス最適化版・GPT-5対応・高速処理）"

# ログの設定
log_file_path = os.path.join(SCRIPT_DIR, "pdf-renamer.log")

# 既存のハンドラーをクリア
for handler in logging.root.handlers[:]:
    logging.root.removeHandler(handler)

# ログの基本設定
logging.basicConfig(
    level=logging.DEBUG,  # DEBUGログを有効化
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file_path, encoding="utf-8", mode='w'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)  # DEBUGログを有効化

# 初期化ログの出力
logger.debug("=" * 50)
logger.debug("プログラム開始")
logger.debug(f"ログファイルの出力先: {log_file_path}")
logger.debug(f"Pythonバージョン: {sys.version}")
logger.debug(f"実行ディレクトリ: {os.getcwd()}")
logger.debug("=" * 50)

class ConfigManager:
    """アプリケーション設定と環境変数を管理するクラス。"""
    
    def __init__(self, env_path=None):
        self.env_path = env_path or DEFAULT_ENV_PATH
        self.config = {}
        self.load_config()
        
    def load_config(self):
        try:
            logger.debug(f".envファイルの絶対パス: {os.path.abspath(self.env_path)}")
            if os.path.exists(self.env_path):
                logger.debug(f".envファイルが存在します: {self.env_path}")
            else:
                logger.warning(f".envファイルが存在しません: {self.env_path}")
            logger.debug(f".envファイルの読み込みを開始します: {self.env_path}")
            load_dotenv(self.env_path)
            logger.debug(f".envファイルの読み込みが完了しました: {self.env_path}")
            
            # 必要な設定
            self.config['PDF_FOLDER_PATH'] = os.getenv('PDF_FOLDER_PATH', '')
            self.config['IMAGE_FILE_PATH'] = os.getenv('IMAGE_FILE_PATH', str(DEFAULT_IMAGE_PATH))
            self.config['YAML_FILE'] = os.getenv('YAML_FILE', str(DEFAULT_YAML_PATH))
            self.config['GOOGLE_APPLICATION_CREDENTIALS'] = os.getenv('GOOGLE_APPLICATION_CREDENTIALS', '')
            self.config['OPENAI_API_KEY'] = os.getenv('OPENAI_API_KEY', '')
            self.config['POPPLER_PATH'] = self._find_poppler_path()
            
            # AIモデル設定
            self.config['OPENAI_MODEL'] = os.getenv('OPENAI_MODEL', 'gpt-5-mini')
            self.config['OPENAI_TEMPERATURE'] = float(os.getenv('OPENAI_TEMPERATURE', '0.0'))
            
            # オプション設定
            self.config['PERSONS'] = os.getenv('PERSONS', '').split(',')
            self.config['DEFAULT_PERSON'] = os.getenv('DEFAULT_PERSON', '担当者自動設定')
            self.config['ORGANIZATION_NAME'] = os.getenv('ORGANIZATION_NAME', 'DefaultOrganization')
            self.config['TITLE'] = os.getenv('TITLE', 'Untitled')
            
            # 画像ディレクトリの作成
            os.makedirs(self.config['IMAGE_FILE_PATH'], exist_ok=True)
            
            logger.info(f"設定を{self.env_path}から読み込みました")
            logger.debug("現在の設定値:")
            for k, v in self.config.items():
                logger.debug(f"  {k} = {v}")
            
        except Exception as e:
            logger.error(f"設定の読み込みエラー: {e}")
            raise
            
    def _find_poppler_path(self):
        system = platform.system()
        if system == "Windows":
            possible_paths = [
                "C:\\Program Files\\Poppler\\bin",
                "C:\\Program Files (x86)\\Poppler\\bin",
                "C:\\poppler-24.08.0\\Library\\bin",
            ]
            exe_name = "pdftoppm.exe"
        else:
            # Mac/Linux
            possible_paths = [
                "/opt/homebrew/bin",
                "/usr/local/bin",
                "/usr/bin",
            ]
            exe_name = "pdftoppm"
        for path in possible_paths:
            if os.path.exists(path) and os.path.isfile(os.path.join(path, exe_name)):
                logger.debug(f"Popplerが見つかりました: {path}")
                return path
        # システムPATHにあるか確認
        try:
            result = subprocess.run([exe_name, "-v"], capture_output=True, text=True, check=False)
            if result.returncode == 0:
                logger.debug("PopplerがシステムPATHで見つかりました")
                return ""
        except Exception:
            pass
        logger.debug("Popplerのパスが見つかりませんでした")
        return ""
    
    def validate_config(self):
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
        # Popplerの存在確認
        system = platform.system()
        poppler_path = self.config.get('POPPLER_PATH')
        if system == "Windows":
            exe_name = "pdftoppm.exe"
        else:
            exe_name = "pdftoppm"
        if not poppler_path:
            # システムPATHにあるか確認
            try:
                result = subprocess.run([exe_name, "-v"], capture_output=True, text=True, check=False)
                if result.returncode != 0:
                    missing.append("Popplerインストール（PDF変換に必要）")
            except Exception:
                missing.append("Popplerインストール（PDF変換に必要）")
        else:
            pdftoppm_path = os.path.join(poppler_path, exe_name)
            if not os.path.exists(pdftoppm_path):
                missing.append(f"{pdftoppm_path}のPopplerの実行ファイル")
        yaml_path = Path(self.config['YAML_FILE'])
        if not yaml_path.exists():
            missing.append(f"{yaml_path}のYAMLルールファイル")
        return missing
    
    def get(self, key, default=None):
        return self.config.get(key, default)
    
    def set(self, key, value):
        self.config[key] = value
        
    def save_config(self):
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
        self.config_manager = config_manager
        self.selected_person = selected_person
        self.status_queue = status_queue
        self.business_card_mode = business_card_mode
        self.processed_files = set()
        self.rules = self.load_yaml_rules()
        self.openai_client = OpenAI(api_key=self.config_manager.get('OPENAI_API_KEY'))
        self.vision_client = vision.ImageAnnotatorClient()
        self._rules_count_displayed = False  # ルール総数表示フラグを追加
        
        # パフォーマンス最適化のための設定
        self.image_cache = {}
        self.ocr_cache = {}
        self.max_cache_size = 200  # キャッシュサイズを増加
        self.dpi = 120  # DPIをさらに下げて処理を高速化
        self.thread_pool_size = 2  # 並列処理のスレッド数
        self.batch_size = 3  # バッチ処理サイズ
        
        logger.debug("PDFProcessorを初期化しました")
        logger.debug(f"選択された担当者: {selected_person}")
        logger.debug(f"名刺読み取りモード: {business_card_mode}")
        logger.info(f"PDFProcessor初期化 - 名刺読み取りモード: {business_card_mode}")
        
        if business_card_mode:
            logger.info("名刺読み取りモードが有効です - 名刺専用のファイル名生成を行います")
        else:
            logger.info("通常モードで動作します")
        
    def load_yaml_rules(self):
        """YAMLファイルからリネームルールを読み込みます。"""
        try:
            with open(self.config_manager.get('YAML_FILE'), 'r', encoding='utf-8') as f:
                rules = yaml.safe_load(f)
            logger.debug(f"YAMLルールを読み込みました: {len(rules['ファイル命名のルール'])}個のルール")
            return rules['ファイル命名のルール']
        except Exception as e:
            logger.error(f"YAMLルールの読み込みエラー: {e}")
            raise

    def _check_applicable_rules(self, ocr_text):
        """OCRテキストに対して適用可能なルールを確認します。"""
        applicable_rules = []
        total_rules = len(self.rules)
        
        # ルールの総数は初回のみ表示
        if not self._rules_count_displayed:
            self.status_queue.put(f"ルールの総数: {total_rules}個")
            logger.info(f"ルールの総数: {total_rules}個")
            self._rules_count_displayed = True
        
        # 名刺読み取りモードの場合の特別処理
        if self.business_card_mode:
            logger.info("名刺読み取りモード: 名刺のリネームルールを優先的に適用します")
            # 名刺のリネームルールを探す
            business_card_rule = None
            for rule in self.rules:
                if rule.get('説明') == "名刺のリネームルール":
                    business_card_rule = rule
                    break
            
            if business_card_rule:
                logger.info("名刺のリネームルールが見つかりました")
                self.status_queue.put("名刺読み取りモード: 名刺のリネームルールを適用します")
                return [business_card_rule]  # 名刺ルールのみを返す
            else:
                logger.warning("名刺のリネームルールが見つかりませんでした")
        
        # 通常のルール適用処理
        for rule in self.rules:
            try:
                pattern = rule.get('正規表現', '')
                if pattern:
                    if re.search(pattern, ocr_text, re.IGNORECASE):
                        applicable_rules.append(rule)
            except Exception as e:
                error_msg = f"ルール適用エラー: {rule.get('説明', '説明なし')} - {str(e)}"
                self.status_queue.put(error_msg)
                logger.error(error_msg)
        
        # 適用可能なルールを優先順位でソート（優先順位が低いほど先に適用）
        if applicable_rules:
            applicable_rules.sort(key=lambda x: x.get('優先順位', 999))
            logger.info(f"適用可能なルールを優先順位でソートしました: {[rule.get('説明', '説明なし') for rule in applicable_rules]}")
        
        if not applicable_rules:
            self.status_queue.put("適用可能なルールが見つかりませんでした")
            logger.warning("適用可能なルールが見つかりませんでした")
        
        return applicable_rules
            
    def process_pdf(self, pdf_file_path):
        """PDFファイルを処理し、新しいファイル名を生成します。"""
        image_paths = []
        start_time = time.time()
        logger.info("\n==============================")
        logger.info(f"[START] PDF処理: {os.path.basename(pdf_file_path)} (開始時刻: {time.strftime('%H:%M:%S')})")
        temp_dir = None
        temp_pdf_path = None
        try:
            # 一時ディレクトリにPDFをコピーして作業
            temp_dir = tempfile.mkdtemp()
            temp_pdf_path = os.path.join(temp_dir, os.path.basename(pdf_file_path))
            shutil.copy2(pdf_file_path, temp_pdf_path)
            work_pdf_path = temp_pdf_path
            logger.info(f"PDFを一時ディレクトリにコピーして作業します: {work_pdf_path}")

            # PDFを画像に変換
            pdf_start = time.time()
            image_paths = self._pdf_to_jpeg(work_pdf_path)
            pdf_time = time.time() - pdf_start
            logger.info(f"PDF→画像変換完了: {pdf_time:.2f}秒")
            if not image_paths:
                raise Exception("PDFの画像変換に失敗しました")
            # OCR処理を並列実行（最適化版）
            ocr_start = time.time()
            ocr_results = []
            
            # 並列処理でOCRを実行
            with ThreadPoolExecutor(max_workers=self.thread_pool_size) as executor:
                # 各画像のOCR処理を並列で実行
                future_to_path = {
                    executor.submit(self._ocr_jpeg, path): path 
                    for path in image_paths
                }
                
                # 結果を順次取得
                for i, future in enumerate(as_completed(future_to_path)):
                    path = future_to_path[future]
                    try:
                        result = future.result()
                        if result:
                            ocr_results.append(result)
                            logger.info(f"OCR処理成功 ({i+1}/{len(image_paths)}): {os.path.basename(path)}")
                        else:
                            logger.warning(f"OCR処理失敗 ({i+1}/{len(image_paths)}): {os.path.basename(path)}")
                    except Exception as e:
                        logger.error(f"OCR処理エラー ({i+1}/{len(image_paths)}): {os.path.basename(path)} - {e}")
            
            ocr_time = time.time() - ocr_start
            logger.info(f"OCR処理完了: {ocr_time:.2f}秒")
            if not ocr_results:
                raise Exception("OCR処理に失敗しました")
            # OCR結果を結合
            combined_ocr = "\n".join(filter(None, ocr_results))
            logger.info(f"OCR結果文字数: {len(combined_ocr)}文字")
            logger.info(f"OCR結果内容:\n{combined_ocr}")
            # 適用可能なルールを確認
            rules_start = time.time()
            applicable_rules = self._check_applicable_rules(combined_ocr)
            rules_time = time.time() - rules_start
            logger.info(f"ルール適用完了: {rules_time:.2f}秒 (適用ルール数: {len(applicable_rules)})")
            # ChatGPT-4でファイル名を生成
            ai_start = time.time()
            new_name = self._generate_filename_with_gpt4(combined_ocr, work_pdf_path, applicable_rules)
            ai_time = time.time() - ai_start
            logger.info(f"AI処理完了: {ai_time:.2f}秒")
            # 担当者が「該当者なし」の場合は（該当者なし）を除去
            if new_name and self.selected_person == "該当者なし":
                new_name = re.sub(r'[（(]該当者なし[）)]$', '', new_name).rstrip()
            # ファイル名の正規化
            new_name = self._normalize_filename(new_name)
            # ファイルのリネーム
            if new_name:
                rename_start = time.time()
                # 元のディレクトリに移動（Windowsエクスプローラーと同じルールで連番を付ける）
                ext = os.path.splitext(pdf_file_path)[1]
                base_filename = f"{new_name}{ext}"
                
                # Windowsエクスプローラーと同じルールでユニークなファイル名を生成
                unique_filename = self._get_unique_filename(pdf_file_path, base_filename)
                new_final_path = os.path.join(os.path.dirname(pdf_file_path), unique_filename)
                logger.info(f"リネーム後のファイルを元の場所に移動: {new_final_path}")
                
                # リネーム前とリネーム後のファイル名が同じ場合はスキップ
                if os.path.basename(pdf_file_path) == unique_filename:
                    logger.info(f"ファイル名が同じなので移動処理をスキップ: {os.path.basename(pdf_file_path)}")
                    # ファイル名が同じでも、一時ディレクトリのファイルを元の場所に移動して元ファイルを置き換える
                    temp_file_path = os.path.join(temp_dir, f"{new_name}{ext}")
                    if os.path.exists(temp_file_path):
                        # 既存ファイルがあれば削除
                        if os.path.exists(new_final_path):
                            try:
                                os.remove(new_final_path)
                                logger.info(f"既存ファイルを削除しました: {new_final_path}")
                            except OSError as e:
                                logger.warning(f"既存ファイル削除に失敗しましたが続行します: {e}")
                        # 一時ファイルを元の場所に移動
                        shutil.move(temp_file_path, new_final_path)
                        logger.info(f"一時ファイルを元の場所に移動しました: {new_final_path}")
                        # 元のファイルの削除は不要（shutil.moveで既に移動済み）
                    rename_time = time.time() - rename_start
                    logger.info(f"ファイルリネーム完了: {rename_time:.2f}秒 (成功: True)")
                    return True
                
                # 一時ファイルをリネーム
                temp_file_path = os.path.join(temp_dir, f"{new_name}{ext}")
                if os.path.exists(work_pdf_path):
                    shutil.move(work_pdf_path, temp_file_path)
                    logger.info(f"一時ファイルをリネームしました: {os.path.basename(work_pdf_path)} -> {new_name}{ext}")
                
                # 元のディレクトリに移動（上書きではなく、ユニークな名前で保存）
                if os.path.exists(temp_file_path):
                    shutil.move(temp_file_path, new_final_path)
                    logger.info(f"リネーム後のファイルを移動しました: {new_final_path}")
                    
                    # 元のファイルを削除
                    if os.path.exists(pdf_file_path):
                        try:
                            os.remove(pdf_file_path)
                            logger.info(f"元のファイルを削除しました: {pdf_file_path}")
                        except OSError as e:
                            logger.error(f"元のファイル削除に失敗しました: {e}")
                            self.status_queue.put(f"警告: 元のファイルの削除に失敗しました - {os.path.basename(pdf_file_path)}")
                    else:
                        logger.warning(f"元のファイルが見つかりません: {pdf_file_path}")
                    
                    rename_time = time.time() - rename_start
                    logger.info(f"ファイルリネーム完了: {rename_time:.2f}秒 (成功: True)")
                    return True
                else:
                    logger.warning(f"一時ファイルが見つかりません: {temp_file_path}")
                    rename_time = time.time() - rename_start
                    logger.info(f"ファイルリネーム完了: {rename_time:.2f}秒 (成功: False)")
                    return False
            return False
        except Exception as e:
            total_time = time.time() - start_time
            logger.error(f"PDF処理エラー ({pdf_file_path}): {e} (総処理時間: {total_time:.2f}秒)")
            self.status_queue.put(f"エラー: {os.path.basename(pdf_file_path)} - {str(e)}")
            return False
        finally:
            cleanup_start = time.time()
            self._cleanup_temp_files(image_paths)
            if temp_dir and os.path.exists(temp_dir):
                try:
                    # 一時ディレクトリ内のファイルを確認
                    temp_files = os.listdir(temp_dir)
                    logger.info(f"一時ディレクトリ内のファイル: {temp_files}")
                    
                    shutil.rmtree(temp_dir)
                    logger.info(f"一時ディレクトリを削除しました: {temp_dir}")
                except Exception as e:
                    logger.error(f"一時ディレクトリ削除エラー: {e}")
                    # 個別にファイルを削除してみる
                    try:
                        for file in os.listdir(temp_dir):
                            file_path = os.path.join(temp_dir, file)
                            if os.path.isfile(file_path):
                                os.remove(file_path)
                                logger.info(f"個別ファイル削除: {file_path}")
                        os.rmdir(temp_dir)
                        logger.info(f"空の一時ディレクトリを削除しました: {temp_dir}")
                    except Exception as cleanup_error:
                        logger.error(f"個別ファイル削除エラー: {cleanup_error}")
            cleanup_time = time.time() - cleanup_start
            logger.info(f"一時ファイル削除完了: {cleanup_time:.2f}秒")
            
            # メモリクリーンアップ
            gc.collect()
            
            total_time = time.time() - start_time
            logger.info(f"[END] PDF処理: {os.path.basename(pdf_file_path)} (終了時刻: {time.strftime('%H:%M:%S')}) (総処理時間: {total_time:.2f}秒)")
            logger.info("==============================\n")
            
    def _pdf_to_jpeg(self, pdf_file_path):
        """PDFファイルをJPEG画像に変換します（最初の1ページのみ）。"""
        try:
            # キャッシュをチェック
            if pdf_file_path in self.image_cache:
                return self.image_cache[pdf_file_path]

            # 最初の1ページのみ変換（最適化設定）
            images = convert_from_path(
                pdf_file_path,
                poppler_path=self.config_manager.get('POPPLER_PATH'),
                dpi=self.dpi,  # DPIを下げて処理を高速化
                thread_count=2,  # 並列処理を有効化（パフォーマンス向上）
                first_page=1,
                last_page=1,
                fmt='jpeg',  # フォーマットを明示的に指定
                jpegopt={'quality': 60, 'optimize': True}  # JPEG品質を最適化
            )
            
            image_paths = []
            for i, image in enumerate(images):
                image_path = os.path.join(
                    self.config_manager.get('IMAGE_FILE_PATH'),
                    f"temp_{int(time.time() * 1000)}_{os.getpid()}_{i}.jpg"
                )
                # 画像の圧縮率を上げてファイルサイズを削減（さらに最適化）
                image.save(image_path, "JPEG", quality=60, optimize=True, progressive=True)
                image_paths.append(image_path)
            
            # キャッシュに保存（LRU方式で最適化）
            if len(self.image_cache) >= self.max_cache_size:
                # 最も古いエントリを削除
                oldest_key = next(iter(self.image_cache))
                del self.image_cache[oldest_key]
            self.image_cache[pdf_file_path] = image_paths
                
            return image_paths
            
        except Exception as e:
            logger.error(f"PDF変換エラー: {e}")
            return []
            
    def _ocr_jpeg(self, image_path):
        """JPEG画像からテキストを抽出します。"""
        start_time = time.time()
        try:
            # キャッシュをチェック
            if image_path in self.ocr_cache:
                logger.info(f"OCRキャッシュヒット: {os.path.basename(image_path)}")
                return self.ocr_cache[image_path]

            logger.info(f"Google Vision API呼び出し開始: {os.path.basename(image_path)}")
            with open(image_path, 'rb') as image_file:
                content = image_file.read()
                
            image = vision.Image(content=content)
            # Visionクライアントを再利用（毎回新規生成を避ける）
            if not hasattr(self, '_vision_client'):
                self._vision_client = vision.ImageAnnotatorClient()
            response = self._vision_client.text_detection(
                image=image,
                image_context={"language_hints": ["ja"]}  # 日本語を優先
            )
            
            if response.error.message:
                raise Exception(f"Google Vision APIエラー: {response.error.message}")
                
            result = None
            if response.text_annotations:
                result = response.text_annotations[0].description
            
            # キャッシュに保存（LRU方式で最適化）
            if len(self.ocr_cache) >= self.max_cache_size:
                # 最も古いエントリを削除
                oldest_key = next(iter(self.ocr_cache))
                del self.ocr_cache[oldest_key]
            self.ocr_cache[image_path] = result
            
            ocr_time = time.time() - start_time
            logger.info(f"OCR処理完了: {os.path.basename(image_path)} ({ocr_time:.2f}秒)")
            
            # OCR結果をログに出力
            if result:
                logger.info(f"OCR結果文字数: {len(result)}文字")
                logger.info(f"OCR結果内容:\n{result}")
            else:
                logger.warning("OCR結果が空です")
                
            return result
            
        except Exception as e:
            ocr_time = time.time() - start_time
            logger.error(f"OCRエラー: {e} ({ocr_time:.2f}秒)")
            return None
            
    def _generate_filename_with_gpt4(self, ocr_text, pdf_file_path, applicable_rules):
        """ChatGPT-4を使用してファイル名を生成します。"""
        start_time = time.time()
        try:
            current_model = self.config_manager.get('OPENAI_MODEL')
            logger.info(f"OpenAI API呼び出し開始: {os.path.basename(pdf_file_path)}")
            logger.info(f"使用モデル: {current_model}")
            logger.info(f"名刺読み取りモード: {self.business_card_mode}")
            
            # 現在の日付を取得
            current_date = datetime.now().strftime("%Y年%m月%d日")
            
            # 名刺読み取りモードの場合の特別なプロンプト
            if self.business_card_mode:
                # 名刺のリネームルールのプロンプトを使用
                business_card_rule = None
                for rule in applicable_rules:
                    if rule.get('説明') == "名刺のリネームルール":
                        business_card_rule = rule
                        break
                
                if business_card_rule:
                    # YAMLファイルの名刺ルールのプロンプトを使用
                    rule_prompt = business_card_rule.get('プロンプト', '')
                    # プレースホルダーを置換
                    prompt = rule_prompt.replace('{ocr_result}', ocr_text)
                    prompt = prompt.replace('{書類の種類}', business_card_rule.get('書類の種類', '名刺'))
                    prompt = prompt.replace('{命名ルール}', business_card_rule.get('命名ルール', ''))
                    prompt = prompt.replace('{担当者}', self.selected_person)
                    
                    logger.info("名刺読み取りモード: YAMLファイルの名刺ルールプロンプトを使用")
                    self.status_queue.put("名刺読み取りモード: 名刺専用のプロンプトを使用してファイル名を生成します")
                else:
                    # フォールバック用の名刺プロンプト
                    prompt = f"""
                    以下のOCR結果は名刺から読み取られたものです。名刺の内容に基づいて適切なファイル名を生成してください。
                    
                    OCR結果:
                    {ocr_text}
                    
                    現在の日付: {current_date}
                    担当者: {self.selected_person}
                    
                    名刺の情報（会社名、役職、氏名など）を適切に組み合わせて、
                    以下の形式でファイル名を生成してください：
                    - 会社名_氏名_役職
                    - 氏名_会社名
                    - 会社名_氏名
                    など、名刺の内容に最も適した形式を選択してください。
                    
                    以下の形式でJSONを返してください：
                    {{"filename": "生成されたファイル名", "used_rule": "名刺読み取りモード"}}
                    """
            else:
                # 通常のプロンプト
                prompt = f"""
                以下のOCR結果から、適切なファイル名を生成してください。
                
                OCR結果:
                {ocr_text}
                
                現在の日付: {current_date}
                担当者: {self.selected_person}
                
                以下の適用可能なルールに従ってファイル名を生成してください：
                {json.dumps(applicable_rules, ensure_ascii=False, indent=2)}
                
                以下の形式でJSONを返してください：
                {{"filename": "生成されたファイル名", "used_rule": "使用したルールの説明"}}
                """
            
            # OpenAI APIにリクエスト
            try:
                # GPT-5モデルの場合はtemperatureを調整
                temperature = self.config_manager.get('OPENAI_TEMPERATURE')
                if current_model.startswith('gpt-5'):
                    # GPT-5モデルはtemperature=1.0のみサポート
                    temperature = 1.0
                    logger.info(f"GPT-5モデルのため、temperatureを1.0に調整: {current_model}")
                
                response = self.openai_client.chat.completions.create(
                    model=current_model,
                    messages=[
                        {"role": "system", "content": "あなたは文書の内容を分析し、適切なファイル名を生成する専門家です。"},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=temperature
                )
            except Exception as api_error:
                logger.error(f"OpenAI API呼び出しエラー: {api_error}")
                logger.error(f"エラータイプ: {type(api_error).__name__}")
                
                # GPT-5-nanoが利用できない場合はGPT-5-miniにフォールバック
                if current_model == "gpt-5-nano":
                    logger.warning("GPT-5-nanoが利用できません。GPT-5-miniにフォールバックします。")
                    self.status_queue.put("警告: GPT-5-nanoが利用できません。GPT-5-miniに切り替えます。")
                    
                    # GPT-5-miniで再試行
                    try:
                        response = self.openai_client.chat.completions.create(
                            model="gpt-5-mini",
                            messages=[
                                {"role": "system", "content": "あなたは文書の内容を分析し、適切なファイル名を生成する専門家です。"},
                                {"role": "user", "content": prompt}
                            ],
                            temperature=1.0  # GPT-5-miniはtemperature=1.0のみサポート
                        )
                        logger.info("GPT-5-miniでの再試行が成功しました")
                        self.status_queue.put("GPT-5-miniでの処理が成功しました")
                    except Exception as fallback_error:
                        logger.error(f"GPT-5-miniでの再試行も失敗: {fallback_error}")
                        raise fallback_error
                else:
                    # その他のモデルの場合はエラーをそのまま投げる
                    raise api_error
            
            # レスポンスの解析
            result = response.choices[0].message.content
            try:
                result_dict = json.loads(result)
                # 使用されたルールを表示
                if "used_rule" in result_dict:
                    rule_summary = "適用されたルール:"
                    rule_summary += f"\n- {result_dict['used_rule']}"
                    self.status_queue.put(rule_summary)
                    logger.info(rule_summary)
                return result_dict.get('filename')
            except json.JSONDecodeError:
                # JSONとして解析できない場合は、そのままのテキストを返す
                return result.strip()
                
        except Exception as e:
            ai_time = time.time() - start_time
            logger.error(f"OpenAI APIエラー: {e} ({ai_time:.2f}秒)")
            logger.error(f"エラーの詳細: {traceback.format_exc()}")
            
            # エラーメッセージをステータスキューに送信
            error_msg = f"AI処理エラー: {str(e)}"
            self.status_queue.put(error_msg)
            
            # モデルが利用できない場合の追加情報
            if "model" in str(e).lower() or "not found" in str(e).lower():
                self.status_queue.put("ヒント: 選択したAIモデルが利用できない可能性があります")
                self.status_queue.put("設定メニューから別のモデルを試してください")
            
            return None
        finally:
            ai_time = time.time() - start_time
            logger.info(f"OpenAI API処理完了: {os.path.basename(pdf_file_path)} ({ai_time:.2f}秒)")
            
    def _normalize_filename(self, filename):
        """ファイル名を正規化します。"""
        if not filename:
            return None
            
        # ファイル名に使用できない文字を置換
        invalid_chars = r'[<>:"/\\|?*]'
        filename = re.sub(invalid_chars, '_', filename)
        
        # 先頭と末尾の空白を削除
        filename = filename.strip()
        
        # 連続する空白を1つに
        filename = re.sub(r'\s+', ' ', filename)
        
        return filename
        
    def _get_unique_filename(self, base_path, filename):
        """Windowsエクスプローラーと同じルールでユニークなファイル名を生成します。"""
        directory = os.path.dirname(base_path)
        name, ext = os.path.splitext(filename)
        
        # 最初に元の名前を試す
        test_path = os.path.join(directory, filename)
        if not os.path.exists(test_path):
            return filename
        
        # 連番を付けて試す (1), (2), (3)...
        counter = 1
        while True:
            new_filename = f"{name} ({counter}){ext}"
            test_path = os.path.join(directory, new_filename)
            if not os.path.exists(test_path):
                return new_filename
            counter += 1
    
    def _rename_pdf(self, pdf_file_path, new_name):
        """PDFファイルをリネームします（バックグラウンドスレッドで実行）。"""
        result = {'success': False}
        def do_rename():
            try:
                # ファイルの存在確認
                if not os.path.exists(pdf_file_path):
                    error_msg = f"元ファイルが存在しません: {pdf_file_path}"
                    logger.error(error_msg)
                    self.status_queue.put(error_msg)
                    result['success'] = False
                    return
                
                # 拡張子を保持
                ext = os.path.splitext(pdf_file_path)[1]
                base_filename = f"{new_name}{ext}"
                
                # Windowsエクスプローラーと同じルールでユニークなファイル名を生成
                unique_filename = self._get_unique_filename(pdf_file_path, base_filename)
                new_path = os.path.join(os.path.dirname(pdf_file_path), unique_filename)
                
                logger.info(f"リネーム処理開始: {os.path.basename(pdf_file_path)} -> {unique_filename}")
                logger.info(f"元ファイルパス: {pdf_file_path}")
                logger.info(f"新ファイルパス: {new_path}")
                
                # リネーム前とリネーム後のファイル名が同じ場合はスキップ
                if pdf_file_path == new_path:
                    logger.info(f"ファイル名が同じなのでリネーム処理をスキップ: {os.path.basename(pdf_file_path)}")
                    self.status_queue.put(f"リネームスキップ: {os.path.basename(pdf_file_path)} (ファイル名変更なし)")
                    result['success'] = True
                    return
                
                # ファイルの移動（上書きではなく、ユニークな名前で保存）
                shutil.move(pdf_file_path, new_path)
                logger.info(f"ファイルをリネームしました: {os.path.basename(pdf_file_path)} -> {unique_filename}")
                self.status_queue.put(f"リネーム成功: {unique_filename}")
                result['success'] = True
            except Exception as e:
                logger.error(f"ファイルリネームエラー: {e}")
                logger.error(f"元ファイルパス: {pdf_file_path}")
                logger.error(f"新ファイルパス: {new_path if 'new_path' in locals() else '未定義'}")
                self.status_queue.put(f"リネームエラー: {os.path.basename(pdf_file_path)} - {str(e)}")
                result['success'] = False
        t = threading.Thread(target=do_rename)
        t.start()
        t.join()  # 呼び出し元で同期的に待つ（process_pdfの流れを変えないため）
        return result['success']
            
    def _cleanup_temp_files(self, file_paths):
        """一時ファイルを削除します（最適化版）。"""
        # バッチ削除でパフォーマンス向上
        temp_dir = self.config_manager.get('IMAGE_FILE_PATH')
        files_to_remove = []
        
        # 指定されたファイルパスを削除対象に追加
        for path in file_paths:
            if path and os.path.exists(path):
                files_to_remove.append(path)
        
        # temp_imagesディレクトリ内の古い一時ファイルも削除対象に追加
        try:
            if os.path.exists(temp_dir):
                for file in os.listdir(temp_dir):
                    if file.startswith('temp_') and file.endswith('.jpg'):
                        file_path = os.path.join(temp_dir, file)
                        # ファイルの作成時刻をチェック（1時間以上古いファイルのみ削除）
                        if os.path.exists(file_path):
                            file_age = time.time() - os.path.getctime(file_path)
                            if file_age > 3600:  # 1時間
                                files_to_remove.append(file_path)
        except Exception as e:
            logger.error(f"一時ファイル一覧取得エラー: {e}")
        
        # バッチ削除実行
        for file_path in files_to_remove:
            try:
                if os.path.exists(file_path):
                    os.remove(file_path)
                    logger.debug(f"一時ファイル削除: {os.path.basename(file_path)}")
            except Exception as e:
                logger.error(f"一時ファイル削除エラー: {e}")

class PDFRenamerApp:
    """PDFリネーマーのGUIアプリケーション。"""
    
    def __init__(self, root):
        self.root = root
        self.root.title("PDF Renamer")
        self.config_manager = ConfigManager()
        self.status_queue = queue.Queue(maxsize=1000)  # キューサイズを制限
        self.selected_person = self.config_manager.get('DEFAULT_PERSON')
        self.business_card_mode = False
        
        # 処理状態の管理
        self.is_processing = False
        self.processing_complete = False
        
        # パフォーマンス最適化のための設定
        self.status_update_interval = 30  # ステータス更新間隔（ミリ秒）をさらに短縮
        self.processing_queue = queue.Queue()  # 処理キュー
        self.status_batch_size = 10  # ステータスメッセージのバッチサイズ
        self.last_status_update = 0  # 最後のステータス更新時刻
        
        self._setup_ui()
        self._validate_config_on_startup()
        self._poll_status_queue()
        
        # ウィンドウ終了時のイベントをバインド
        self.root.protocol("WM_DELETE_WINDOW", self._on_closing)
        
    def _setup_ui(self):
        """UIの初期設定を行います。"""
        # メインフレーム
        main_frame = ttk.Frame(self.root, padding="5")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # グリッドの重み設定
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=1)
        
        # フォルダ選択
        ttk.Label(main_frame, text="PDFフォルダ:").grid(row=0, column=0, sticky=tk.W)
        self.folder_path = tk.StringVar(value=self.config_manager.get('PDF_FOLDER_PATH'))
        ttk.Entry(main_frame, textvariable=self.folder_path, width=35).grid(row=0, column=1, padx=5)
        ttk.Button(main_frame, text="参照", command=self._browse_folder).grid(row=0, column=2)
        
        # 担当者選択
        ttk.Label(main_frame, text="担当者:").grid(row=1, column=0, sticky=tk.W)
        self.person_var = tk.StringVar(value=self.selected_person)
        person_combo = ttk.Combobox(main_frame, textvariable=self.person_var, width=32)
        person_combo['values'] = self.config_manager.get('PERSONS')
        person_combo.grid(row=1, column=1, padx=5, pady=5)
        person_combo.bind('<<ComboboxSelected>>', self._on_person_changed)
        
        # 名刺モード
        self.business_card_var = tk.BooleanVar(value=self.business_card_mode)
        self.business_card_checkbox = ttk.Checkbutton(
            main_frame,
            text="名刺読み取りモード",
            variable=self.business_card_var,
            command=self._toggle_business_card_mode,
            style="TCheckbutton"
        )
        self.business_card_checkbox.grid(row=2, column=0, columnspan=3, sticky=tk.N, pady=(5, 0))
        
        # チェックボックスの状態を確認するためのデバッグ情報
        logger.info(f"名刺チェックボックス作成: {self.business_card_checkbox}")
        logger.info(f"チェックボックスの状態: {self.business_card_var.get()}")
        
        # 名刺モードの説明ラベル
        self.business_card_label = ttk.Label(
            main_frame,
            text="名刺読み取りモードが有効の場合、「名刺のリネームルール」が優先的に適用されます",
            font=("", 8),
            foreground="gray"
        )
        self.business_card_label.grid(row=3, column=0, columnspan=3, sticky=tk.N, pady=(0, 5))
        
        # AIモデル情報ラベル
        self.ai_model_label = ttk.Label(
            main_frame,
            text=f"現在のAIモデル: {self.config_manager.get('OPENAI_MODEL', 'gpt-5-mini')}",
            font=("", 8),
            foreground="blue"
        )
        self.ai_model_label.grid(row=4, column=0, columnspan=3, sticky=tk.N, pady=(0, 5))
        
        # チェックボックスの状態変更を監視
        self.business_card_var.trace_add('write', self._on_business_card_mode_changed)
        
        # 名刺モードの初期状態をログに出力
        logger.info(f"名刺読み取りモード初期状態: {self.business_card_mode}")
        
        # 開始/停止ボタン
        self.start_button = ttk.Button(
            main_frame,
            text="リネーム開始",
            command=self._start_renaming
        )
        self.start_button.grid(row=5, column=0, columnspan=3, pady=10)
        
        # 進捗バー
        self.progress_var = tk.DoubleVar()
        self.progress_bar = ttk.Progressbar(
            main_frame,
            variable=self.progress_var,
            maximum=100
        )
        self.progress_bar.grid(row=6, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=5)
        
        # ステータス表示
        self.status_text = scrolledtext.ScrolledText(main_frame, height=12, width=50)
        self.status_text.grid(row=7, column=0, columnspan=3, pady=5, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # ステータス表示エリアの拡張設定
        main_frame.rowconfigure(7, weight=1)
        
        # 終了ボタン
        self.exit_button = ttk.Button(
            main_frame,
            text="終了",
            command=self._exit_app
        )
        self.exit_button.grid(row=8, column=0, columnspan=3, pady=5)
        
        # メニューの作成
        self._create_menu()
        
    def _create_menu(self):
        """メニューバーを作成します。"""
        menubar = tk.Menu(self.root)
        self.root.config(menu=menubar)
        
        # ファイルメニュー
        file_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="ファイル", menu=file_menu)
        file_menu.add_command(label="フォルダを開く", command=self._browse_folder)
        file_menu.add_separator()
        file_menu.add_command(label="終了", command=self._on_closing)
        
        # 設定メニュー
        settings_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="設定", menu=settings_menu)
        
        # AIモデル選択
        self.model_menu = tk.Menu(settings_menu, tearoff=0)
        settings_menu.add_cascade(label="AIモデル", menu=self.model_menu)
        
        # モデル選択用の変数
        self.model_var = tk.StringVar(value=self.config_manager.get('OPENAI_MODEL', 'gpt-5-mini'))
        
        # ラジオボタンスタイルのメニューアイテム
        self.model_menu.add_radiobutton(
            label="GPT-5-mini (推奨)", 
            variable=self.model_var, 
            value="gpt-5-mini",
            command=lambda: self._set_ai_model("gpt-5-mini")
        )
        self.model_menu.add_radiobutton(
            label="GPT-5-nano (軽量)", 
            variable=self.model_var, 
            value="gpt-5-nano",
            command=lambda: self._set_ai_model("gpt-5-nano")
        )
        self.model_menu.add_separator()
        self.model_menu.add_command(label="モデル確認", command=self._check_current_model)
        self.model_menu.add_command(label="利用可能モデル一覧", command=self._list_available_models)

        # ルールメニュー
        rules_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="ルール", menu=rules_menu)
        rules_menu.add_command(label="ルールの保存", command=self._save_rules)
        rules_menu.add_command(label="ルールの読み込み", command=self._load_rules)
        rules_menu.add_command(label="環境変数の保存", command=self._save_env_vars)
        
        # ヘルプメニュー
        help_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="ヘルプ", menu=help_menu)
        help_menu.add_command(label="使い方", command=self._show_help)
        help_menu.add_command(label="バージョン情報", command=self._show_about)
        
    def _browse_folder(self):
        """PDFフォルダを選択します。"""
        folder_path = askdirectory()
        if folder_path:
            self.folder_path.set(folder_path)
            self.config_manager.set('PDF_FOLDER_PATH', folder_path)
            self.config_manager.save_config()
            self._load_pdf_files(folder_path)
            
    def _load_pdf_files(self, folder_path):
        """PDFファイルの一覧を読み込みます。"""
        try:
            pdf_files = [f for f in os.listdir(folder_path) if f.lower().endswith('.pdf')]
            self._add_to_status(f"{len(pdf_files)}個のPDFファイルが見つかりました")
        except Exception as e:
            self._add_to_status(f"エラー: {str(e)}")
            
    def _on_person_changed(self, event):
        """担当者が変更された時の処理。"""
        self.selected_person = self.person_var.get()
        
    def _add_to_status(self, message):
        """ステータスメッセージを追加します。"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.status_text.insert(tk.END, f"[{timestamp}] {message}\n")
        self.status_text.see(tk.END)
        self.root.update_idletasks()
        
    def _poll_status_queue(self):
        """ステータスキューを監視します（最適化版）。"""
        try:
            messages = []
            current_time = time.time()
            
            # バッチサイズまたは時間間隔でメッセージを取得
            while len(messages) < self.status_batch_size:
                try:
                    message = self.status_queue.get_nowait()
                    messages.append(message)
                except queue.Empty:
                    break
            
            if messages:
                # メッセージをまとめて追加（バッチ処理）
                timestamp = datetime.now().strftime("%H:%M:%S")
                batch_text = ""
                for message in messages:
                    batch_text += f"[{timestamp}] {message}\n"
                
                self.status_text.insert(tk.END, batch_text)
                self.status_text.see(tk.END)
                
                # UI更新を最適化（必要時のみ）
                if current_time - self.last_status_update > 0.1:  # 100ms間隔
                    self.root.update_idletasks()
                    self.last_status_update = current_time
            
            # 処理完了の検知とUI更新（重複を避けるため簡素化）
            if not self.is_processing and self.processing_complete:
                logger.info("_poll_status_queueで処理完了を検知しました")
                # 処理完了フラグをリセット（一度だけ実行）
                self.processing_complete = False
                
                # 完了メッセージを表示（重複を避けるため条件付き）
                if not hasattr(self, '_completion_message_shown'):
                    self._add_to_status("処理が完了しました")
                    self._completion_message_shown = True
                    
                    # ボタンの状態を再確認・修正
                    if self.start_button['text'] != "リネーム開始":
                        logger.warning("ボタンが「リネーム開始」になっていないため、強制修正します")
                        self._set_start_button_to_ready()
                    
                    # 完了ダイアログを表示
                    # try:
                    #     messagebox.showinfo("処理完了", "処理が完了しました")
                    # except Exception as dialog_error:
                    #     logger.error(f"ダイアログ表示エラー: {dialog_error}")
                
        except Exception as e:
            logger.error(f"ステータスキュー処理エラー: {e}")
        finally:
            # 処理中の場合は短い間隔で、そうでなければ長い間隔で監視
            interval = self.status_update_interval if self.is_processing else 200
            self.root.after(interval, self._poll_status_queue)
            
    def _validate_config_on_startup(self):
        """起動時の設定検証を行います。"""
        missing = self.config_manager.validate_config()
        if missing:
            messagebox.showwarning(
                "設定エラー",
                "以下の設定が不足しています：\n" + "\n".join(missing)
            )
            
    def _start_renaming(self):
        """リネーム処理を開始します。"""
        if self.is_processing:
            self._stop_processing()
            return
            
        folder_path = self.folder_path.get()
        if not folder_path:
            messagebox.showwarning("エラー", "PDFフォルダを選択してください")
            return
            
        try:
            # ログファイルを再作成
            if os.path.exists(log_file_path):
                try:
                    os.remove(log_file_path)
                    print(f"既存のログファイルを削除しました: {log_file_path}")
                except Exception as e:
                    print(f"ログファイルの削除に失敗しました: {e}")

            # 既存のハンドラーをクリア
            for handler in logging.root.handlers[:]:
                logging.root.removeHandler(handler)

            # ログの基本設定を再設定
            logging.basicConfig(
                level=logging.DEBUG,
                format='%(asctime)s - %(levelname)s - %(message)s',
                handlers=[
                    logging.FileHandler(log_file_path, encoding="utf-8", mode='w'),
                    logging.StreamHandler()
                ]
            )

            logger.debug("=" * 50)
            logger.debug("リネーム処理開始")
            logger.debug(f"ログファイルの出力先: {log_file_path}")
            logger.debug("=" * 50)

            pdf_files = [f for f in os.listdir(folder_path) if f.lower().endswith('.pdf')]
            logger.info(f"フォルダ内の全ファイル: {os.listdir(folder_path)}")
            logger.info(f"検出されたPDFファイル: {pdf_files}")
            
            if not pdf_files:
                messagebox.showinfo("情報", "PDFファイルが見つかりません")
                return
                
            self._start_processing(folder_path, pdf_files)
            
        except Exception as e:
            messagebox.showerror("エラー", str(e))
            
    def _start_processing(self, pdf_folder, pdf_files):
        logger.info("ボタンを「処理を停止」に設定します（_start_processing）")
        self.is_processing = True
        self.processing_complete = False
        self._set_start_button_to_stop()
        
        # 名刺読み取りモードの状態をログに出力
        logger.info(f"名刺読み取りモード: {self.business_card_mode}")
        self._add_to_status(f"処理開始: {len(pdf_files)}個のファイルを処理します")
        if self.business_card_mode:
            self._add_to_status("名刺読み取りモードが有効です")
        
        # バックグラウンドスレッドで処理を実行
        thread = threading.Thread(
            target=self._process_pdf_files,
            args=(pdf_folder, pdf_files),
            daemon=True
        )
        thread.start()
        logger.info("バックグラウンドスレッドを開始しました")
        
    def _process_pdf_files(self, pdf_folder, pdf_files):
        """PDFファイルの処理を実行します。"""
        total_files = len(pdf_files)
        completed = 0
        successful = 0
        
        logger.info(f"PDFファイル処理開始: 合計{total_files}個のファイル")
        logger.info(f"処理対象ファイル: {pdf_files}")
        
        # 名刺読み取りモードの状態をログに出力
        logger.info(f"PDFProcessor作成時の名刺読み取りモード: {self.business_card_mode}")
        
        # 単一のPDFProcessorインスタンスを作成（全ファイルで共有）
        processor = PDFProcessor(
            self.config_manager,
            self.selected_person,
            self.status_queue,
            self.business_card_mode
        )
        
        try:
            # ファイルを順次処理（並列処理によるAPI制限を回避）
            for i, pdf_file in enumerate(pdf_files):
                if not self.is_processing:  # 停止がリクエストされた場合
                    logger.info("処理停止がリクエストされました")
                    break
                    
                try:
                    logger.info(f"ファイル処理開始 ({i+1}/{total_files}): {pdf_file}")
                    pdf_path = os.path.join(pdf_folder, pdf_file)
                    
                    # ファイルの存在確認と詳細ログ
                    logger.info(f"処理対象ファイルパス: {pdf_path}")
                    if os.path.exists(pdf_path):
                        file_size = os.path.getsize(pdf_path)
                        logger.info(f"ファイル存在確認: サイズ={file_size}バイト")
                    else:
                        logger.error(f"ファイルが存在しません: {pdf_path}")
                        self.status_queue.put(f"エラー: {pdf_file} - ファイルが存在しません")
                        continue
                    
                    # PDF処理を実行（タイムアウト付き）
                    start_time = time.time()
                    if processor.process_pdf(pdf_path):
                        successful += 1
                        processing_time = time.time() - start_time
                        logger.info(f"ファイル処理成功: {pdf_file} (処理時間: {processing_time:.2f}秒)")
                    else:
                        processing_time = time.time() - start_time
                        logger.warning(f"ファイル処理失敗: {pdf_file} (処理時間: {processing_time:.2f}秒)")
                        
                except Exception as e:
                    processing_time = time.time() - start_time if 'start_time' in locals() else 0
                    self.status_queue.put(f"エラー: {pdf_file} - {str(e)}")
                    logger.error(f"ファイル処理エラー ({pdf_file}): {e} (処理時間: {processing_time:.2f}秒)")
                finally:
                    completed += 1
                    # 進捗更新
                    progress_percent = (completed / total_files) * 100
                    self.root.after(0, lambda: self._update_progress(
                        progress_percent, completed, successful, total_files
                    ))
                    
                    # 定期的なメモリクリーンアップ（5ファイルごと）
                    if completed % 5 == 0:
                        gc.collect()
                        logger.debug(f"メモリクリーンアップ実行: {completed}/{total_files} ファイル処理完了")
                    
        except Exception as e:
            logger.error(f"処理全体のエラー: {e}")
            self.status_queue.put(f"処理エラー: {str(e)}")
        finally:
            # メモリクリーンアップ
            gc.collect()
            
            logger.info(f"全処理完了: {successful}/{total_files} ファイルが成功")
            
            # 処理状態をリセット
            self.is_processing = False
            self.processing_complete = True
            logger.info("処理状態をリセットしました")
            
            # UI更新をメインスレッドで実行
            try:
                self.root.after(0, self._processing_completed, successful, total_files)
                logger.info("処理完了処理をスケジュールしました")
                
                # 追加の安全策：少し遅れてボタン状態を再確認
                self.root.after(100, self._ensure_button_state)
                
            except Exception as e:
                logger.error(f"処理完了処理のスケジュールエラー: {e}")
                # エラーが発生した場合は直接実行
                try:
                    self._processing_completed(successful, total_files)
                except Exception as direct_error:
                    logger.error(f"処理完了処理の直接実行エラー: {direct_error}")
                    # 最終手段としてボタンのみ更新
                    self._set_start_button_to_ready()
        
    def _stop_processing(self):
        """処理を停止します。"""
        try:
            self.is_processing = False
            self._add_to_status("処理停止がリクエストされました...")
            self._set_start_button_to_ready()
            self._add_to_status("処理が停止されました")
        except Exception as e:
            logger.error(f"処理停止エラー: {e}")
            self.is_processing = False
            self._set_start_button_to_ready()
        
    def _update_progress(self, value, completed, successful, total):
        """進捗状況を更新します。"""
        try:
            self.progress_var.set(min(value, 100))
            self.status_queue.put(f"進捗: {completed}/{total} (成功: {successful})")
            
            # UIの更新を強制
            self.root.update_idletasks()
            
        except Exception as e:
            logger.error(f"進捗更新エラー: {e}")
        
    def _processing_completed(self, successful, total):
        """処理完了時の処理を一元化します。"""
        logger.info(f"_processing_completedメソッドが呼び出されました: 成功={successful}, 総数={total}")
        try:
            # 処理状態をリセット
            self.is_processing = False
            self.processing_complete = True
            
            # 完了メッセージフラグをリセット
            if hasattr(self, '_completion_message_shown'):
                delattr(self, '_completion_message_shown')
            
            # 進捗バーを100%に設定
            self.progress_var.set(100)
            
            # 完了メッセージをステータスキューに送信
            self.status_queue.put(f"処理完了: {successful}/{total} ファイルが正常に処理されました")
            
            # ボタンを「リネーム開始」に戻す（即座に実行）
            self._set_start_button_to_ready()
            
            # UIの強制更新
            self.root.update()
            
            logger.info("処理完了処理が正常に完了しました")
                
        except Exception as e:
            logger.error(f"処理完了処理エラー: {e}")
            # エラーが発生してもボタンを戻す
            self._set_start_button_to_ready()
        
    def _reset_after_processing(self):
        """処理完了後のリセットを行います。"""
        try:
            # 進捗バーをリセット
            self.progress_var.set(0)
            
            # メモリの解放を促す
            gc.collect()
            
            # UI更新
            self.root.update_idletasks()
            
        except Exception as e:
            logger.error(f"リセット処理エラー: {e}")
        
    def _show_about(self):
        """バージョン情報を表示します。"""
        messagebox.showinfo(
            "バージョン情報",
            f"PDF Renamer {APP_VERSION}\n\n"
            "OCR、Google Cloud Vision、およびOpenAIを使用して\n"
            "PDFファイルを内容に基づいて自動的にリネームするツール\n\n"
            "パフォーマンス最適化版\n"
            "対応モデル: GPT-5-mini, GPT-5-nano\n"
            "モデル選択: 有効"
        )
        
    def _show_help(self):
        """使い方を表示します。"""
        help_text = """
使い方:

1. 「参照」ボタンをクリックしてPDFフォルダを選択
2. 担当者を選択（必要な場合）
3. 名刺読み取りモードを選択（必要な場合）
4. 「リネーム開始」ボタンをクリック

AIモデル設定:
- 設定メニューから「AIモデル」を選択
- GPT-5-mini (推奨): 高速で効率的な処理
- GPT-5-nano (軽量): 超軽量で低コストな処理
- モデル確認: 現在のモデルの利用可能性をテスト
- 利用可能モデル一覧: 実際に利用可能なモデルを確認

注意事項:
- PDFファイルは自動的に内容に基づいてリネームされます
- 処理中は進捗状況が表示されます
- エラーが発生した場合はログに記録されます
- 処理中に「処理を停止」ボタンで中断できます

修正点（パフォーマンス最適化版）:
- DPIを120に下げてPDF→画像変換を高速化
- 画像保存のqualityを60に下げてファイルサイズ削減
- 並列処理によるOCR処理の高速化（ThreadPoolExecutor使用）
- LRUキャッシュ方式によるメモリ効率の改善
- 定期的なガベージコレクションによるメモリ管理
- バッチ処理によるUI応答性の向上
- Visionクライアントの再利用によるAPI呼び出し最適化
- 一時ファイルの効率的な管理
- ステータス更新の最適化（バッチ処理）
- GPT-5-miniとGPT-5-nanoの選択可能
- モデル選択機能を有効化
        """
        messagebox.showinfo("使い方", help_text)
        
    def _set_ai_model(self, model):
        """AIモデルを設定します。"""
        # モデルの存在確認
        if not self._check_model_availability(model):
            self._add_to_status(f"警告: {model}が利用できない可能性があります")
            self._add_to_status("GPT-5-miniにフォールバックすることをお勧めします")
        
        self.config_manager.set('OPENAI_MODEL', model)
        self.config_manager.save_config()
        self._add_to_status(f"AIモデルを{model}に変更しました")
        
        # AIモデル情報ラベルの更新
        self.ai_model_label.config(text=f"現在のAIモデル: {model}")
        
        # メニューのラジオボタン状態を更新
        self.model_var.set(model)
        
        # モデル変更後の説明を追加
        if model == "gpt-5-mini":
            self._add_to_status("GPT-5-mini: 高速で効率的な処理が可能です")
        elif model == "gpt-5-nano":
            self._add_to_status("GPT-5-nano: 超軽量で低コストな処理が可能です")
    
    def _check_model_availability(self, model):
        """モデルの利用可能性を確認します。"""
        try:
            # 簡単なテストリクエストでモデルの存在を確認
            test_client = OpenAI(api_key=self.config_manager.get('OPENAI_API_KEY'))
            
            # GPT-5モデルの場合はtemperatureを調整
            temperature = 0.0
            if model.startswith('gpt-5'):
                temperature = 1.0
                logger.info(f"GPT-5モデルのため、temperatureを1.0に調整: {model}")
            
            # GPT-5モデルの場合はmax_completion_tokensを使用
            if model.startswith('gpt-5'):
                test_response = test_client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "user", "content": "test"}
                    ],
                    max_completion_tokens=10,
                    temperature=temperature
                )
            else:
                test_response = test_client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "user", "content": "test"}
                    ],
                    max_tokens=1,
                    temperature=temperature
                )
            logger.info(f"モデル {model} の利用可能性確認: 成功")
            return True
        except Exception as e:
            logger.warning(f"モデル {model} の利用可能性確認: 失敗 - {e}")
            logger.warning(f"エラーの詳細: {traceback.format_exc()}")
            return False
    
    def _check_current_model(self):
        """現在のモデルの利用可能性を確認します。"""
        current_model = self.config_manager.get('OPENAI_MODEL')
        self._add_to_status(f"現在のモデル {current_model} の利用可能性を確認中...")
        
        if self._check_model_availability(current_model):
            self._add_to_status(f"✓ {current_model} は利用可能です")
            messagebox.showinfo("モデル確認", f"{current_model} は正常に利用できます。")
        else:
            self._add_to_status(f"✗ {current_model} は利用できません")
            
            # エラーの詳細をログから取得して表示
            try:
                # ログファイルから最新のエラー情報を取得
                log_file = os.path.join(SCRIPT_DIR, "pdf_renamer.log")
                if os.path.exists(log_file):
                    with open(log_file, 'r', encoding='utf-8') as f:
                        lines = f.readlines()
                        # 最新のエラーログを探す
                        for line in reversed(lines):
                            if f"モデル {current_model} の利用可能性確認: 失敗" in line:
                                # 次の行にエラーの詳細がある可能性
                                error_details = []
                                for i, log_line in enumerate(lines):
                                    if f"モデル {current_model} の利用可能性確認: 失敗" in log_line:
                                        # エラー行の前後数行を取得
                                        start = max(0, i-2)
                                        end = min(len(lines), i+3)
                                        error_details = lines[start:end]
                                        break
                                
                                if error_details:
                                    error_text = "".join(error_details)
                                    self._add_to_status("エラーの詳細:")
                                    self._add_to_status(error_text.strip())
                                break
            except Exception as log_error:
                logger.error(f"ログファイル読み込みエラー: {log_error}")
            
            # 一般的な原因と対処法を表示
            self._add_to_status("")
            self._add_to_status("AIモデルが利用できない一般的な原因:")
            self._add_to_status("1. モデル名が間違っている (正しい名前: gpt-5-mini, gpt-5-nano)")
            self._add_to_status("2. アカウントの利用制限")
            self._add_to_status("3. APIキーの権限不足")
            self._add_to_status("4. インターネット接続の問題")
            self._add_to_status("5. GPT-5モデルはtemperature=1.0のみサポート")
            self._add_to_status("")
            self._add_to_status("対処法:")
            self._add_to_status("- GPT-5-miniを使用する (推奨)")
            self._add_to_status("- GPT-5-nanoを使用する (軽量)")
            self._add_to_status("- OpenAIの公式サイトで利用可能なモデルを確認")
            self._add_to_status("- APIキーの権限を確認")
            self._add_to_status("- 利用可能モデル一覧で確認")
            
            messagebox.showwarning(
                "モデル確認", 
                f"{current_model} は利用できません。\n\n"
                "詳細なエラー情報はステータス画面とログファイルを確認してください。\n\n"
                "GPT-5-miniに切り替えることをお勧めします。\n"
                "設定メニューから「AIモデル」→「GPT-5-mini (推奨)」を選択してください。"
            )
    
    def _list_available_models(self):
        """利用可能なモデルの一覧を表示します。"""
        try:
            self._add_to_status("利用可能なモデルを確認中...")
            
            # OpenAI APIから利用可能なモデル一覧を取得
            client = OpenAI(api_key=self.config_manager.get('OPENAI_API_KEY'))
            models = client.models.list()
            
            # 利用可能なモデルをフィルタリング（GPT系のみ）
            gpt_models = []
            for model in models.data:
                if model.id.startswith('gpt-'):
                    gpt_models.append(model.id)
            
            if gpt_models:
                self._add_to_status("利用可能なGPTモデル:")
                for model in sorted(gpt_models):
                    self._add_to_status(f"  - {model}")
                
                # 推奨モデルを表示
                self._add_to_status("")
                self._add_to_status("推奨モデル:")
                if "gpt-5-mini" in gpt_models:
                    self._add_to_status("  - gpt-5-mini (高速・効率的)")
                if "gpt-5-nano" in gpt_models:
                    self._add_to_status("  - gpt-5-nano (超軽量・低コスト)")
                if "gpt-4o" in gpt_models:
                    self._add_to_status("  - gpt-4o (高品質)")
                    
            else:
                self._add_to_status("利用可能なGPTモデルが見つかりませんでした")
                self._add_to_status("APIキーまたはアカウントの設定を確認してください")
                
        except Exception as e:
            error_msg = f"モデル一覧取得エラー: {str(e)}"
            self._add_to_status(error_msg)
            logger.error(error_msg)
            
            # 一般的なモデル名を表示
            self._add_to_status("")
            self._add_to_status("一般的なGPT-5モデル名:")
            self._add_to_status("  - gpt-5-mini (軽量版・推奨)")
            self._add_to_status("  - gpt-5-nano (超軽量版)")
            self._add_to_status("  - gpt-4o (高品質版)")
            self._add_to_status("  - gpt-4o-mini (軽量版)")
            self._add_to_status("")
            self._add_to_status("注意: GPT-5モデルはtemperature=1.0のみサポート")
            self._add_to_status("正しいモデル名を使用してください")
    

    def _save_rules(self):
        """ルールを保存します。"""
        try:
            # デフォルトのファイル名を生成
            timestamp = datetime.now().strftime("%Y年%m月%d日%H時%M分%S秒")
            default_filename = f"rename_rules.backup({timestamp}).yaml"
            
            # ファイル保存ダイアログを表示
            file_path = filedialog.asksaveasfilename(
                initialfile=default_filename,
                filetypes=[("YAML files", "*.yaml"), ("YAML files", "*.yml"), ("All files", "*.*")],
                title="ルールファイルの保存"
            )
            
            if file_path:
                # 拡張子がない場合や.ymlの場合は.yamlに変更
                if not file_path.endswith('.yaml') and not file_path.endswith('.yml'):
                    file_path += '.yaml'
                elif file_path.endswith('.yml'):
                    file_path = file_path[:-4] + '.yaml'
                
                # 現在のルールを読み込む
                with open(self.config_manager.get('YAML_FILE'), 'r', encoding='utf-8') as source:
                    rules_content = source.read()
                
                # 新しいファイルに保存
                with open(file_path, 'w', encoding='utf-8') as target:
                    target.write(rules_content)
                
                self._add_to_status(f"ルールを保存しました: {os.path.basename(file_path)}")
                messagebox.showinfo("保存完了", f"ルールを保存しました:\n{file_path}")
            else:
                self._add_to_status("ルールの保存をキャンセルしました")
                
        except Exception as e:
            error_msg = f"ルールの保存中にエラーが発生しました: {str(e)}"
            logger.error(error_msg)
            self._add_to_status(error_msg)
            messagebox.showerror("エラー", error_msg)
    
    def _load_rules(self):
        """ルールを読み込みます。"""
        try:
            # ファイル選択ダイアログを表示
            file_path = filedialog.askopenfilename(
                filetypes=[("YAML files", "*.yaml"), ("YAML files", "*.yml"), ("All files", "*.*")],
                title="ルールファイルの読み込み"
            )
            
            if file_path:
                # バックアップを作成
                backup_path = self.config_manager.get('YAML_FILE') + '.backup'
                with open(self.config_manager.get('YAML_FILE'), 'r', encoding='utf-8') as current:
                    current_content = current.read()
                
                with open(backup_path, 'w', encoding='utf-8') as backup:
                    backup.write(current_content)
                
                self._add_to_status(f"現在のルールをバックアップしました: {os.path.basename(backup_path)}")
                
                # 選択したファイルを読み込んでコピー
                with open(file_path, 'r', encoding='utf-8') as source:
                    rules_content = source.read()
                
                # 現在のルールファイルに保存
                with open(self.config_manager.get('YAML_FILE'), 'w', encoding='utf-8') as target:
                    target.write(rules_content)
                
                # YAMLの形式を検証
                try:
                    yaml.safe_load(rules_content)
                    self._add_to_status(f"ルールを読み込みました: {os.path.basename(file_path)}")
                    messagebox.showinfo("読み込み完了", f"ルールを読み込みました:\n{file_path}\n\n現在のルールは {os.path.basename(backup_path)} にバックアップされています。")
                except yaml.YAMLError as yaml_error:
                    # YAML形式が無効な場合、バックアップから復元
                    with open(backup_path, 'r', encoding='utf-8') as backup:
                        backup_content = backup.read()
                    with open(self.config_manager.get('YAML_FILE'), 'w', encoding='utf-8') as target:
                        target.write(backup_content)
                    
                    error_msg = f"YAMLファイルの形式が正しくありません: {str(yaml_error)}"
                    logger.error(error_msg)
                    self._add_to_status(error_msg)
                    messagebox.showerror("エラー", error_msg + "\nルールを復元しました。")
            else:
                self._add_to_status("ルールの読み込みをキャンセルしました")
                
        except Exception as e:
            error_msg = f"ルールの読み込み中にエラーが発生しました: {str(e)}"
            logger.error(error_msg)
            self._add_to_status(error_msg)
            messagebox.showerror("エラー", error_msg)
        
    def _save_env_vars(self):
        """環境変数を保存します。"""
        try:
            # デフォルトのファイル名を生成
            timestamp = datetime.now().strftime("%Y年%m月%d日%H時%M分%S秒")
            default_filename = f"backup({timestamp}).env"
            
            # ファイル保存ダイアログを表示
            file_path = filedialog.asksaveasfilename(
                initialfile=default_filename,
                filetypes=[("Environment files", "*.env"), ("All files", "*.*")],
                title="環境変数ファイルの保存"
            )
            
            if file_path:
                # 現在の環境変数を読み込む
                with open(self.config_manager.env_path, 'r', encoding='utf-8') as source:
                    env_content = source.read()
                
                # 新しいファイルに保存
                with open(file_path, 'w', encoding='utf-8') as target:
                    target.write(env_content)
                
                self._add_to_status(f"環境変数を保存しました: {os.path.basename(file_path)}")
                messagebox.showinfo("保存完了", f"環境変数を保存しました:\n{file_path}")
            else:
                self._add_to_status("環境変数の保存をキャンセルしました")
                
        except Exception as e:
            error_msg = f"環境変数の保存中にエラーが発生しました: {str(e)}"
            logger.error(error_msg)
            self._add_to_status(error_msg)
            messagebox.showerror("エラー", error_msg)
        
    def _toggle_business_card_mode(self):
        """名刺モードを切り替えます。"""
        self.business_card_mode = self.business_card_var.get()
        logger.info(f"名刺読み取りモードを切り替え: {self.business_card_mode}")
        self._add_to_status(
            "名刺読み取りモードを" +
            ("有効" if self.business_card_mode else "無効") +
            "にしました"
        )
        
    def _on_business_card_mode_changed(self, *args):
        """名刺読み取りモードの状態変更を監視します。"""
        current_mode = self.business_card_var.get()
        if current_mode != self.business_card_mode:
            self.business_card_mode = current_mode
            logger.info(f"名刺読み取りモード状態変更を検出: {self.business_card_mode}")
            self._add_to_status(
                "名刺読み取りモードが" +
                ("有効" if self.business_card_mode else "無効") +
                "になりました"
            )
        
    def _on_closing(self):
        """アプリケーション終了時の処理。"""
        try:
            # 処理中の場合は停止
            if self.is_processing:
                self._stop_processing()
                # 少し待つ
                time.sleep(1)
            
            # 一時ファイルのクリーンアップ
            temp_dir = self.config_manager.get('IMAGE_FILE_PATH')
            if os.path.exists(temp_dir):
                try:
                    for file in os.listdir(temp_dir):
                        if file.startswith('temp_') and file.endswith('.jpg'):
                            os.remove(os.path.join(temp_dir, file))
                except Exception as e:
                    logger.error(f"一時ファイルクリーンアップエラー: {e}")
            
            # ログの最終出力
            logger.info("アプリケーションを終了します")
            
            # ウィンドウを破棄
            self.root.quit()
            self.root.destroy()
            
        except Exception as e:
            logger.error(f"終了処理エラー: {e}")
            # 強制終了
            self.root.quit()

    def _set_start_button_to_ready(self):
        logger.info("_set_start_button_to_ready called")
        def _set():
            try:
                logger.info(f"UIスレッドでボタンを「リネーム開始」にリセット（現在: {self.start_button['text']}）")
                self.start_button.config(text="リネーム開始", state="normal")
                logger.info(f"UIスレッドでボタンを「リネーム開始」に変更完了（現在: {self.start_button['text']}）")
                
                # UIの強制更新（複数の方法で試行）
                try:
                    self.root.update_idletasks()
                    logger.info("ボタン変更後のupdate_idletasksを実行しました")
                except Exception as update_error:
                    logger.error(f"update_idletasksエラー: {update_error}")
                
                try:
                    self.root.update()
                    logger.info("ボタン変更後のupdateを実行しました")
                except Exception as update_error:
                    logger.error(f"updateエラー: {update_error}")
                
                # ボタンの状態を確認
                logger.info(f"ボタン状態確認: text='{self.start_button['text']}', state='{self.start_button['state']}'")
                
            except Exception as e:
                logger.error(f"_set_start_button_to_ready _set() error: {e}")
        
        # 即座に実行を試行
        try:
            _set()
        except Exception as immediate_error:
            logger.warning(f"即座実行に失敗、スケジュール実行に切り替え: {immediate_error}")
            self.root.after(0, _set)

    def _set_start_button_to_stop(self):
        def _set():
            logger.info(f"UIスレッドでボタンを「処理を停止」にリセット（現在: {self.start_button['text']}）")
            self.start_button.config(text="処理を停止", state="normal")
            logger.info(f"UIスレッドでボタンを「処理を停止」に変更完了（現在: {self.start_button['text']}）")
        self.root.after(0, _set)

    def _ensure_button_state(self):
        """ボタンの状態を確認し、必要に応じて修正します。"""
        try:
            if not self.is_processing and self.start_button['text'] != "リネーム開始":
                logger.warning("ボタン状態の不整合を検出、修正します")
                self._set_start_button_to_ready()
            else:
                logger.info("ボタン状態は正常です")
        except Exception as e:
            logger.error(f"ボタン状態確認エラー: {e}")
    
    def _exit_app(self):
        """アプリケーションを終了します（確認なし）。"""
        try:
            # 処理中の場合は停止
            if self.is_processing:
                self._stop_processing()
                # 少し待つ
                time.sleep(0.5)
            
            # 一時ファイルのクリーンアップ
            temp_dir = self.config_manager.get('IMAGE_FILE_PATH')
            if os.path.exists(temp_dir):
                try:
                    for file in os.listdir(temp_dir):
                        if file.startswith('temp_') and file.endswith('.jpg'):
                            os.remove(os.path.join(temp_dir, file))
                except Exception as e:
                    logger.error(f"一時ファイルクリーンアップエラー: {e}")
            
            # ログの最終出力
            logger.info("アプリケーションを終了します（終了ボタンから）")
            
            # ウィンドウを破棄
            self.root.quit()
            self.root.destroy()
            
        except Exception as e:
            logger.error(f"終了処理エラー: {e}")
            # 強制終了
            self.root.quit()

def main():
    """メイン関数"""
    try:
        # Tkinterのルートウィンドウを作成
        root = tk.Tk()
        
        # ウィンドウサイズを設定
        window_width = 647  # 588から1割増加
        window_height = 420
        root.geometry(f"{window_width}x{window_height}")
        root.minsize(554, 350)  # 最小サイズも1割増加
        
        # 画面の中央に配置
        screen_width = root.winfo_screenwidth()
        screen_height = root.winfo_screenheight()
        x = (screen_width - window_width) // 2
        y = (screen_height - window_height) // 2
        root.geometry(f"+{x}+{y}")
        
        # アプリケーションを初期化
        app = PDFRenamerApp(root)
        
        # メインループを開始
        root.mainloop()
        
    except Exception as e:
        logger.error(f"メイン関数エラー: {e}")
        print(f"アプリケーション開始エラー: {e}")
    finally:
        # 最終的なクリーンアップ
        logger.info("プログラム終了")

if __name__ == "__main__":
    main()