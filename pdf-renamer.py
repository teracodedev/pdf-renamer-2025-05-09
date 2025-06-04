"""
PDF Renamer - OCR、Google Cloud Vision、およびOpenAIを使用してPDFファイルを
内容に基づいて自動的にリネームするための強化ツール（フリーズ問題修正版）

このアプリケーションはOCRを使用してPDFからテキストを抽出し、AIで内容を分析し、
カスタマイズ可能なルールに従って意味のあるファイル名を生成します。

作者: Claude（ユーザーのオリジナルに基づく）
日付: 2025年6月4日（フリーズ問題修正版）
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
import gc

# ============================================================================
# 設定とセットアップ
# ============================================================================

# スクリプトディレクトリの設定
SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_ENV_PATH = SCRIPT_DIR / '.env'
DEFAULT_YAML_PATH = SCRIPT_DIR / 'rename_rules.yaml'
DEFAULT_IMAGE_PATH = SCRIPT_DIR / 'temp_images'
APP_VERSION = "2025年6月4日バージョン（フリーズ問題修正版）"

# ログの設定
log_file_path = os.path.join(SCRIPT_DIR, "pdf-renamer.log")

# 既存のログファイルを削除
if os.path.exists(log_file_path):
    try:
        os.remove(log_file_path)
        print(f"既存のログファイルを削除しました: {log_file_path}")
    except Exception as e:
        print(f"ログファイルの削除に失敗しました: {e}")

# 既存のハンドラーをクリア
for handler in logging.root.handlers[:]:
    logging.root.removeHandler(handler)

# ログの基本設定
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file_path, encoding="utf-8", mode='w'),
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

class ConfigManager:
    """アプリケーション設定と環境変数を管理するクラス。"""
    
    def __init__(self, env_path=None):
        self.env_path = env_path or DEFAULT_ENV_PATH
        self.config = {}
        self.load_config()
        
    def load_config(self):
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
            
            # オプション設定
            self.config['PERSONS'] = os.getenv('PERSONS', '').split(',')
            self.config['DEFAULT_PERSON'] = os.getenv('DEFAULT_PERSON', '担当者自動設定')
            self.config['ORGANIZATION_NAME'] = os.getenv('ORGANIZATION_NAME', 'DefaultOrganization')
            self.config['TITLE'] = os.getenv('TITLE', 'Untitled')
            
            # 画像ディレクトリの作成
            os.makedirs(self.config['IMAGE_FILE_PATH'], exist_ok=True)
            
            logger.info(f"設定を{self.env_path}から読み込みました")
            
        except Exception as e:
            logger.error(f"設定の読み込みエラー: {e}")
            raise
            
    def _find_poppler_path(self):
        possible_paths = [
            "C:\\Program Files\\Poppler\\bin",
            "C:\\Program Files (x86)\\Poppler\\bin",
            "C:\\poppler-24.08.0\\Library\\bin",
        ]
        
        for path in possible_paths:
            if os.path.exists(path) and os.path.isfile(os.path.join(path, "pdftoppm.exe")):
                logger.debug(f"Popplerが見つかりました: {path}")
                return path
        
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
                
        if not self.config.get('POPPLER_PATH'):
            try:
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
        
        logger.debug("PDFProcessorを初期化しました")
        logger.debug(f"選択された担当者: {selected_person}")
        logger.debug(f"名刺読み取りモード: {business_card_mode}")
        
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
            
    def process_pdf(self, pdf_file_path):
        """PDFファイルを処理し、新しいファイル名を生成します。"""
        image_paths = []
        try:
            # PDFを画像に変換
            image_paths = self._pdf_to_jpeg(pdf_file_path)
            if not image_paths:
                raise Exception("PDFの画像変換に失敗しました")
                
            # OCR処理
            ocr_results = []
            for image_path in image_paths:
                ocr_result = self._ocr_jpeg(image_path)
                if ocr_result:
                    ocr_results.append(ocr_result)
                    
            if not ocr_results:
                raise Exception("OCR処理に失敗しました")
                
            # OCR結果を結合
            combined_ocr = "\n".join(ocr_results)
            
            # ChatGPT-4でファイル名を生成
            new_name = self._generate_filename_with_gpt4(combined_ocr, pdf_file_path)
            
            # ファイル名の正規化
            new_name = self._normalize_filename(new_name)
            
            # ファイルのリネーム
            if new_name:
                self._rename_pdf(pdf_file_path, new_name)
                return True
                
            return False
            
        except Exception as e:
            logger.error(f"PDF処理エラー ({pdf_file_path}): {e}")
            self.status_queue.put(f"エラー: {os.path.basename(pdf_file_path)} - {str(e)}")
            return False
        finally:
            # 一時ファイルの削除
            self._cleanup_temp_files(image_paths)
            
    def _pdf_to_jpeg(self, pdf_file_path):
        """PDFファイルをJPEG画像に変換します。"""
        try:
            images = convert_from_path(
                pdf_file_path,
                poppler_path=self.config_manager.get('POPPLER_PATH'),
                dpi=300
            )
            
            image_paths = []
            for i, image in enumerate(images):
                image_path = os.path.join(
                    self.config_manager.get('IMAGE_FILE_PATH'),
                    f"temp_{int(time.time() * 1000)}_{os.getpid()}_{i}.jpg"
                )
                image.save(image_path, "JPEG")
                image_paths.append(image_path)
                
            return image_paths
            
        except Exception as e:
            logger.error(f"PDF変換エラー: {e}")
            return []
            
    def _ocr_jpeg(self, image_path):
        """JPEG画像からテキストを抽出します。"""
        try:
            with open(image_path, 'rb') as image_file:
                content = image_file.read()
                
            image = vision.Image(content=content)
            response = self.vision_client.text_detection(image=image)
            
            if response.error.message:
                raise Exception(f"Google Vision APIエラー: {response.error.message}")
                
            if response.text_annotations:
                return response.text_annotations[0].description
                
            return None
            
        except Exception as e:
            logger.error(f"OCRエラー: {e}")
            return None
            
    def _generate_filename_with_gpt4(self, ocr_text, pdf_file_path):
        """ChatGPT-4を使用してファイル名を生成します。"""
        try:
            # 現在の日付を取得
            current_date = datetime.now().strftime("%Y年%m月%d日")
            
            # プロンプトの作成
            prompt = f"""
            以下のOCR結果から、適切なファイル名を生成してください。
            
            OCR結果:
            {ocr_text}
            
            現在の日付: {current_date}
            担当者: {self.selected_person}
            
            以下のルールに従ってファイル名を生成してください：
            {json.dumps(self.rules, ensure_ascii=False, indent=2)}
            
            ファイル名は以下の形式で出力してください：
            {{"filename": "生成されたファイル名"}}
            """
            
            # ChatGPT-4にリクエスト
            response = self.openai_client.chat.completions.create(
                model=self.config_manager.get('OPENAI_MODEL'),
                messages=[
                    {"role": "system", "content": "あなたは文書の内容を分析し、適切なファイル名を生成する専門家です。"},
                    {"role": "user", "content": prompt}
                ],
                temperature=self.config_manager.get('OPENAI_TEMPERATURE')
            )
            
            # レスポンスの解析
            result = response.choices[0].message.content
            try:
                result_dict = json.loads(result)
                return result_dict.get('filename')
            except json.JSONDecodeError:
                # JSONとして解析できない場合は、そのままのテキストを返す
                return result.strip()
                
        except Exception as e:
            logger.error(f"ChatGPT-4 APIエラー: {e}")
            return None
            
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
        
    def _rename_pdf(self, pdf_file_path, new_name):
        """PDFファイルをリネームします。"""
        try:
            # 拡張子を保持
            ext = os.path.splitext(pdf_file_path)[1]
            new_path = os.path.join(os.path.dirname(pdf_file_path), f"{new_name}{ext}")
            
            # 同名ファイルが存在する場合は番号を付加
            counter = 1
            while os.path.exists(new_path):
                new_path = os.path.join(
                    os.path.dirname(pdf_file_path),
                    f"{new_name}_{counter}{ext}"
                )
                counter += 1
                
            # ファイルの移動
            shutil.move(pdf_file_path, new_path)
            logger.info(f"ファイルをリネームしました: {os.path.basename(pdf_file_path)} -> {os.path.basename(new_path)}")
            self.status_queue.put(f"リネーム成功: {os.path.basename(new_path)}")
            
        except Exception as e:
            logger.error(f"ファイルリネームエラー: {e}")
            self.status_queue.put(f"リネームエラー: {os.path.basename(pdf_file_path)} - {str(e)}")
            
    def _cleanup_temp_files(self, file_paths):
        """一時ファイルを削除します。"""
        for path in file_paths:
            try:
                if path and os.path.exists(path):
                    os.remove(path)
                    logger.debug(f"一時ファイルを削除しました: {path}")
            except Exception as e:
                logger.error(f"一時ファイル削除エラー: {e}")

class PDFRenamerApp:
    """PDFリネーマーのGUIアプリケーション。"""
    
    def __init__(self, root):
        self.root = root
        self.root.title("PDF Renamer")
        self.config_manager = ConfigManager()
        self.status_queue = queue.Queue()
        self.selected_person = self.config_manager.get('DEFAULT_PERSON')
        self.business_card_mode = False
        
        # 処理状態の管理
        self.is_processing = False
        self.processing_complete = False
        self.executor = None
        
        self._setup_ui()
        self._validate_config_on_startup()
        self._poll_status_queue()
        
        # ウィンドウ終了時のイベントをバインド
        self.root.protocol("WM_DELETE_WINDOW", self._on_closing)
        
    def _setup_ui(self):
        """UIの初期設定を行います。"""
        # メインフレーム
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # フォルダ選択
        ttk.Label(main_frame, text="PDFフォルダ:").grid(row=0, column=0, sticky=tk.W)
        self.folder_path = tk.StringVar(value=self.config_manager.get('PDF_FOLDER_PATH'))
        ttk.Entry(main_frame, textvariable=self.folder_path, width=50).grid(row=0, column=1, padx=5)
        ttk.Button(main_frame, text="参照", command=self._browse_folder).grid(row=0, column=2)
        
        # 担当者選択
        ttk.Label(main_frame, text="担当者:").grid(row=1, column=0, sticky=tk.W)
        self.person_var = tk.StringVar(value=self.selected_person)
        person_combo = ttk.Combobox(main_frame, textvariable=self.person_var, width=47)
        person_combo['values'] = self.config_manager.get('PERSONS')
        person_combo.grid(row=1, column=1, padx=5, pady=5)
        person_combo.bind('<<ComboboxSelected>>', self._on_person_changed)
        
        # 名刺モード
        self.business_card_var = tk.BooleanVar(value=self.business_card_mode)
        ttk.Checkbutton(
            main_frame,
            text="名刺読み取りモード",
            variable=self.business_card_var,
            command=self._toggle_business_card_mode
        ).grid(row=2, column=1, sticky=tk.W, pady=5)
        
        # 開始/停止ボタン
        self.start_button = ttk.Button(
            main_frame,
            text="リネーム開始",
            command=self._start_renaming
        )
        self.start_button.grid(row=3, column=1, pady=10)
        
        # 進捗バー
        self.progress_var = tk.DoubleVar()
        self.progress_bar = ttk.Progressbar(
            main_frame,
            variable=self.progress_var,
            maximum=100
        )
        self.progress_bar.grid(row=4, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=5)
        
        # ステータス表示
        self.status_text = scrolledtext.ScrolledText(main_frame, height=10, width=60)
        self.status_text.grid(row=5, column=0, columnspan=3, pady=5)
        
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
        model_menu = tk.Menu(settings_menu, tearoff=0)
        settings_menu.add_cascade(label="AIモデル", menu=model_menu)
        model_menu.add_command(label="GPT-4.1", command=lambda: self._set_ai_model("gpt-4.1"))
        model_menu.add_command(label="GPT-4", command=lambda: self._set_ai_model("gpt-4"))
        
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
        """ステータスキューを監視します。"""
        try:
            messages = []
            while True:
                try:
                    message = self.status_queue.get_nowait()
                    messages.append(message)
                except queue.Empty:
                    break
            
            if messages:
                # メッセージをまとめて追加
                timestamp = datetime.now().strftime("%H:%M:%S")
                for message in messages:
                    self.status_text.insert(tk.END, f"[{timestamp}] {message}\n")
                self.status_text.see(tk.END)
                self.root.update_idletasks()
                
        except Exception as e:
            logger.error(f"ステータスキュー処理エラー: {e}")
        finally:
            # 処理中の場合は短い間隔で、そうでなければ長い間隔で監視
            interval = 100 if self.is_processing else 500
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
            pdf_files = [f for f in os.listdir(folder_path) if f.lower().endswith('.pdf')]
            if not pdf_files:
                messagebox.showinfo("情報", "PDFファイルが見つかりません")
                return
                
            self._start_processing(folder_path, pdf_files)
            
        except Exception as e:
            messagebox.showerror("エラー", str(e))
            
    def _start_processing(self, pdf_folder, pdf_files):
        """PDFファイルの処理を開始します。"""
        self.is_processing = True
        self.processing_complete = False
        self.start_button.config(text="処理を停止", state="normal")
        
        self._add_to_status(f"処理開始: {len(pdf_files)}個のファイルを処理します")
        
        # バックグラウンドスレッドで処理を実行
        thread = threading.Thread(
            target=self._process_pdf_files,
            args=(pdf_folder, pdf_files),
            daemon=True
        )
        thread.start()
        
    def _process_pdf_files(self, pdf_folder, pdf_files):
        """PDFファイルの処理を実行します。"""
        total_files = len(pdf_files)
        completed = 0
        successful = 0
        
        def process_single_file(pdf_file):
            nonlocal completed, successful
            try:
                if not self.is_processing:  # 停止がリクエストされた場合
                    return False
                    
                pdf_path = os.path.join(pdf_folder, pdf_file)
                processor = PDFProcessor(
                    self.config_manager,
                    self.selected_person,
                    self.status_queue,
                    self.business_card_mode
                )
                
                if processor.process_pdf(pdf_path):
                    successful += 1
                    return True
                    
            except Exception as e:
                self.status_queue.put(f"エラー: {pdf_file} - {str(e)}")
                logger.error(f"ファイル処理エラー ({pdf_file}): {e}")
            finally:
                completed += 1
                # 進捗更新
                progress_percent = (completed / total_files) * 100
                self.root.after(0, lambda: self._update_progress(
                    progress_percent, completed, successful, total_files
                ))
            
            return False
        
        try:
            # ThreadPoolExecutorを使用してファイルを並列処理
            max_workers = min(3, len(pdf_files))  # 最大3つのワーカーに制限
            self.executor = ThreadPoolExecutor(max_workers=max_workers)
            
            futures = []
            for pdf_file in pdf_files:
                if not self.is_processing:  # 停止がリクエストされた場合
                    break
                future = self.executor.submit(process_single_file, pdf_file)
                futures.append(future)
            
            # すべてのタスクの完了を待つ
            for future in as_completed(futures):
                if not self.is_processing:  # 停止がリクエストされた場合
                    break
                try:
                    future.result(timeout=300)  # 5分のタイムアウト
                except Exception as e:
                    logger.error(f"Future処理エラー: {e}")
                    
        except Exception as e:
            logger.error(f"処理全体のエラー: {e}")
            self.status_queue.put(f"処理エラー: {str(e)}")
        finally:
            # リソースのクリーンアップ
            if self.executor:
                self.executor.shutdown(wait=False)  # 即座にシャットダウン
                self.executor = None
                
            # メモリクリーンアップ
            gc.collect()
            
            # UI更新をメインスレッドで実行
            self.root.after(0, lambda: self._processing_completed(successful, total_files))
        
    def _stop_processing(self):
        """処理を停止します。"""
        self.is_processing = False
        self._add_to_status("処理停止がリクエストされました...")
        
        # ExecutorPoolを強制終了
        if self.executor:
            self.executor.shutdown(wait=False)
            self.executor = None
            
        self.start_button.config(text="リネーム開始", state="normal")
        self._add_to_status("処理が停止されました")
        
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
        """処理完了時の処理を行います。"""
        try:
            self.is_processing = False
            self.processing_complete = True
            
            # UIの更新
            self.start_button.config(text="リネーム開始", state="normal")
            self.progress_var.set(100)
            
            # 完了メッセージ
            self._add_to_status(f"処理完了: {successful}/{total} ファイルが正常に処理されました")
            
            # 完了ダイアログを表示
            messagebox.showinfo(
                "処理完了",
                f"処理が完了しました\n成功: {successful}/{total}"
            )
            
            # リセット処理
            self._reset_after_processing()
            
        except Exception as e:
            logger.error(f"処理完了処理エラー: {e}")
            
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
            "フリーズ問題修正版"
        )
        
    def _show_help(self):
        """使い方を表示します。"""
        help_text = """
使い方:

1. 「参照」ボタンをクリックしてPDFフォルダを選択
2. 担当者を選択（必要な場合）
3. 名刺読み取りモードを選択（必要な場合）
4. 「リネーム開始」ボタンをクリック

注意事項:
- PDFファイルは自動的に内容に基づいてリネームされます
- 処理中は進捗状況が表示されます
- エラーが発生した場合はログに記録されます
- 処理中に「処理を停止」ボタンで中断できます

修正点（フリーズ問題対策）:
- 適切なリソース管理とクリーンアップ
- ThreadPoolExecutorの適切な終了処理
- メモリリークの防止
- UI応答性の改善
        """
        messagebox.showinfo("使い方", help_text)
        
    def _set_ai_model(self, model):
        """AIモデルを設定します。"""
        self.config_manager.set('OPENAI_MODEL', model)
        self.config_manager.save_config()
        self._add_to_status(f"AIモデルを{model}に変更しました")
        
    def _toggle_business_card_mode(self):
        """名刺モードを切り替えます。"""
        self.business_card_mode = self.business_card_var.get()
        self._add_to_status(
            "名刺読み取りモードを" +
            ("有効" if self.business_card_mode else "無効") +
            "にしました"
        )
        
    def _on_closing(self):
        """アプリケーション終了時の処理。"""
        try:
            # 処理中の場合は停止
            if self.is_processing:
                self._stop_processing()
                # 少し待つ
                time.sleep(1)
            
            # ExecutorPoolを確実に終了
            if self.executor:
                self.executor.shutdown(wait=False)
                self.executor = None
            
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

def main():
    """メイン関数"""
    try:
        # Tkinterのルートウィンドウを作成
        root = tk.Tk()
        
        # ウィンドウサイズを設定
        root.geometry("700x600")
        root.minsize(600, 500)
        
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