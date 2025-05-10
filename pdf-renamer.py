"""
PDF Renamer - An enhanced tool for automatically renaming PDF files
based on their content using OCR, Google Cloud Vision, and OpenAI.

This application extracts text from PDFs using OCR, analyzes the content
with AI, and generates meaningful filenames according to customizable rules.

Author: Claude (based on original by user)
Date: May 9, 2025
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

# ============================================================================
# Configuration and Setup
# ============================================================================

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("pdf_renamer.log", encoding="utf-8"),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

# Constants and global variables
SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_ENV_PATH = SCRIPT_DIR / '.env'
DEFAULT_YAML_PATH = SCRIPT_DIR / 'rename_rules.yaml'
DEFAULT_IMAGE_PATH = SCRIPT_DIR / 'temp_images'
APP_VERSION = "2025年5月10日バージョン"


# ============================================================================
# Helper Classes
# ============================================================================

class DefaultDict(dict):
    """Custom dictionary that returns a default value for missing keys."""
    def __missing__(self, key):
        return "不明"


class ConfigManager:
    """Manages application configuration and environment variables."""
    
    def __init__(self, env_path=None):
        """Initialize configuration manager with optional environment file path."""
        self.env_path = env_path or DEFAULT_ENV_PATH
        self.config = {}
        self.load_config()
        
    def load_config(self):
        """Load configuration from .env file."""
        try:
            load_dotenv(self.env_path)
            
            # Required settings
            self.config['PDF_FOLDER_PATH'] = os.getenv('PDF_FOLDER_PATH', '')
            self.config['IMAGE_FILE_PATH'] = os.getenv('IMAGE_FILE_PATH', str(DEFAULT_IMAGE_PATH))
            self.config['YAML_FILE'] = os.getenv('YAML_FILE', str(DEFAULT_YAML_PATH))
            self.config['GOOGLE_APPLICATION_CREDENTIALS'] = os.getenv('GOOGLE_APPLICATION_CREDENTIALS', '')
            self.config['OPENAI_API_KEY'] = os.getenv('OPENAI_API_KEY', '')
            self.config['POPPLER_PATH'] = os.getenv('POPPLER_PATH', self._find_poppler_path())
            
            # AI Model settings
            self.config['OPENAI_MODEL'] = os.getenv('OPENAI_MODEL', 'gpt-4.1')
            self.config['OPENAI_TEMPERATURE'] = float(os.getenv('OPENAI_TEMPERATURE', '0.2'))
            
            # Optional settings with defaults
            self.config['PERSONS'] = os.getenv('PERSONS', '').split(',')
            self.config['DEFAULT_PERSON'] = os.getenv('DEFAULT_PERSON', '担当者自動設定')
            self.config['ORGANIZATION_NAME'] = os.getenv('ORGANIZATION_NAME', 'DefaultOrganization')
            self.config['TITLE'] = os.getenv('TITLE', 'Untitled')
            
            # Create image directory if it doesn't exist
            os.makedirs(self.config['IMAGE_FILE_PATH'], exist_ok=True)
            
            logger.info(f"Configuration loaded from {self.env_path}")
            
        except Exception as e:
            logger.error(f"Error loading configuration: {e}")
            raise
            
    def _find_poppler_path(self):
        """Find Poppler path from common installation locations."""
        possible_paths = [
            "C:\\Program Files\\Poppler\\bin",
            "C:\\Program Files (x86)\\Poppler\\bin",
            "C:\\poppler-24.08.0\\Library\\bin",
        ]
        
        for path in possible_paths:
            if os.path.exists(path) and os.path.isfile(os.path.join(path, "pdftoppm.exe")):
                logger.debug(f"Poppler found at: {path}")
                return path
        
        # If Poppler is in PATH, return empty string (use system PATH)
        try:
            result = subprocess.run(["pdftoppm", "-v"], capture_output=True, text=True, check=False)
            if result.returncode == 0:
                logger.debug("Poppler found in system PATH")
                return ""
        except Exception:
            pass
            
        logger.debug("Poppler path not found")
        return ""
    
    def validate_config(self):
        """Validate that all required configuration is present and valid."""
        required_configs = {
            'PDF_FOLDER_PATH': "PDF folder path",
            'IMAGE_FILE_PATH': "Temporary image folder path",
            'YAML_FILE': "YAML rules file path",
            'GOOGLE_APPLICATION_CREDENTIALS': "Google Cloud credentials path",
            'OPENAI_API_KEY': "OpenAI API key",
        }
        
        missing = []
        for key, desc in required_configs.items():
            if not self.config.get(key):
                missing.append(desc)
                
        # Validate Poppler installation
        if not self.config.get('POPPLER_PATH'):
            try:
                # Check if pdftoppm is in PATH
                result = subprocess.run(["pdftoppm", "-v"], capture_output=True, text=True, check=False)
                if result.returncode != 0:
                    missing.append("Poppler installation (needed for PDF conversion)")
            except Exception:
                missing.append("Poppler installation (needed for PDF conversion)")
        else:
            poppler_path = self.config['POPPLER_PATH']
            pdftoppm_path = os.path.join(poppler_path, "pdftoppm.exe")
            if not os.path.exists(pdftoppm_path):
                missing.append(f"Poppler executable at {pdftoppm_path}")
                
        # Validate YAML file exists
        yaml_path = Path(self.config['YAML_FILE'])
        if not yaml_path.exists():
            missing.append(f"YAML rules file at {yaml_path}")
                
        return missing
    
    def get(self, key, default=None):
        """Get configuration value by key."""
        return self.config.get(key, default)
    
    def set(self, key, value):
        """Set configuration value."""
        self.config[key] = value
        
    def save_config(self):
        """Save configuration to .env file."""
        try:
            with open(self.env_path, 'w', encoding='utf-8') as f:
                for key, value in self.config.items():
                    if key == 'PERSONS':
                        f.write(f"{key}={','.join(value)}\n")
                    else:
                        f.write(f"{key}={value}\n")
                        
            logger.info(f"Configuration saved to {self.env_path}")
            
        except Exception as e:
            logger.error(f"Error saving configuration: {e}")
            raise


class PDFProcessor:
    """Processes PDF files using OCR and AI to generate meaningful filenames."""
    
    def __init__(self, config_manager, selected_person, status_queue):
        """Initialize with configuration and processing queues."""
        self.config = config_manager
        self.selected_person = selected_person
        self.status_queue = status_queue
        self.yaml_rules = None
        self.load_yaml_rules()
        
    def load_yaml_rules(self):
        """Load rename rules from YAML file."""
        try:
            yaml_path = self.config.get('YAML_FILE')
            if not yaml_path or not os.path.exists(yaml_path):
                raise FileNotFoundError(f"YAML file not found: {yaml_path}")
                
            with open(yaml_path, 'r', encoding='utf-8') as f:
                self.yaml_rules = yaml.safe_load(f)
                
            logger.debug(f"YAML rules loaded from {yaml_path}")
            
        except Exception as e:
            logger.error(f"Error loading YAML rules: {e}")
            self.status_queue.put(f"エラー: YAML設定の読み込みに失敗しました: {e}\n")
            self.yaml_rules = None
    
    def process_pdf(self, pdf_file_path):
        """Process a single PDF file: convert to image, OCR, rename."""
        temp_jpeg_path = None
        try:
            logger.debug(f"Processing PDF: {pdf_file_path}")
            
            # 1. Convert PDF to JPEG
            temp_jpeg_path = self._pdf_to_jpeg(pdf_file_path)
            if not temp_jpeg_path:
                self.status_queue.put(f"処理失敗: {pdf_file_path} (PDF変換エラー)\n")
                return False
                
            # 2. Extract text with OCR
            ocr_result = self._ocr_jpeg(temp_jpeg_path)
            if not ocr_result:
                self.status_queue.put(f"処理失敗: {pdf_file_path} (OCRエラー)\n")
                return False
                
            # 3. Generate new filename based on OCR results
            new_pdf_name = self._propose_file_name(ocr_result)
            if not new_pdf_name:
                self.status_queue.put(f"処理失敗: {pdf_file_path} (ファイル名提案エラー)\n")
                return False
                
            # 4. Rename the PDF file
            new_path = self._rename_pdf(pdf_file_path, new_pdf_name)
            if new_path:
                self.status_queue.put(f"リネーム完了: {os.path.basename(new_path)}\n")
                return True
            else:
                self.status_queue.put(f"処理失敗: {pdf_file_path} (リネームエラー)\n")
                return False
                
        except Exception as e:
            logger.exception(f"Error processing PDF: {pdf_file_path}")
            self.status_queue.put(f"処理エラー: {pdf_file_path} - {str(e)}\n")
            return False
            
        finally:
            # Clean up temporary files
            if temp_jpeg_path and os.path.exists(temp_jpeg_path):
                try:
                    os.remove(temp_jpeg_path)
                    logger.debug(f"Temporary file deleted: {temp_jpeg_path}")
                except Exception as e:
                    logger.exception(f"Error deleting temporary file: {temp_jpeg_path}")
    
    def _pdf_to_jpeg(self, pdf_file_path):
        """Convert the first page of a PDF to JPEG image."""
        try:
            logger.debug(f"Converting PDF to JPEG: {pdf_file_path}")
            
            poppler_path = self.config.get('POPPLER_PATH')
            
            # Create a unique temporary filename in the designated folder
            temp_jpeg_path = os.path.join(
                self.config.get('IMAGE_FILE_PATH'),
                f"temp_{datetime.now().strftime('%Y%m%d%H%M%S%f')}_{os.path.basename(pdf_file_path)}.jpg"
            )
            
            # Convert the first page of the PDF to a JPEG image
            kwargs = {
                'first_page': 1,
                'last_page': 1,
                'dpi': 300,  # Increased DPI for better OCR results
            }
            
            if poppler_path:
                kwargs['poppler_path'] = poppler_path
                
            try:
                images = convert_from_path(pdf_file_path, **kwargs)
                if not images:
                    raise ValueError("No images extracted from PDF")
                    
                images[0].save(temp_jpeg_path, 'JPEG', quality=95)  # Higher quality for better OCR
                logger.debug(f"PDF converted to JPEG: {temp_jpeg_path}")
                return temp_jpeg_path
                
            except PDFPageCountError:
                logger.warning(f"Failed to convert PDF - may be corrupted: {pdf_file_path}")
                return None
                
        except PDFInfoNotInstalledError:
            logger.error("Poppler not properly installed or configured")
            self.status_queue.put("エラー: Popplerが正しく設定されていません。\n")
            return None
            
        except Exception as e:
            logger.exception(f"Error converting PDF to JPEG: {pdf_file_path}")
            return None
    
    def _ocr_jpeg(self, image_path):
        """Extract text from a JPEG image using Google Cloud Vision API."""
        try:
            logger.debug(f"Performing OCR on image: {image_path}")
            
            # Initialize the Vision API client
            client = vision.ImageAnnotatorClient()
            
            # Read the image file
            with open(image_path, 'rb') as image_file:
                content = image_file.read()
                
            # Create an image object and perform text detection
            image = vision.Image(content=content)
            response = client.text_detection(image=image)
            
            # Check for errors
            if response.error.message:
                raise Exception(f"Google Cloud Vision API error: {response.error.message}")
                
            # Extract and return the OCR text
            if response.text_annotations:
                result = response.text_annotations[0].description
                logger.debug(f"OCR successful with {len(result)} characters extracted")
                return result
            else:
                logger.warning(f"No text detected in image: {image_path}")
                return ""
                
        except google.api_core.exceptions.PermissionDenied as e:
            logger.error(f"Google Cloud Vision API permission denied: {e}")
            self.status_queue.put(
                "エラー: Google Cloud Vision APIの課金が有効になっていません。\n"
                "https://console.developers.google.com/billing/enable で課金を有効にしてください。\n"
            )
            return None
            
        except Exception as e:
            logger.exception(f"OCR error: {image_path}")
            return None
    
    def _convert_wareki_to_western(self, text):
        """Convert Japanese era dates to Western (Gregorian) dates."""
        # 令和の変換 (Reiwa: 2019-)
        text = re.sub(r'令和(\d+)年', lambda m: str(2018 + int(m.group(1))) + '年', text)
        # 平成の変換 (Heisei: 1989-2019)
        text = re.sub(r'平成(\d+)年', lambda m: str(1988 + int(m.group(1))) + '年', text)
        # 昭和の変換 (Showa: 1926-1989)
        text = re.sub(r'昭和(\d+)年', lambda m: str(1925 + int(m.group(1))) + '年', text)
        return text
        
    def _propose_file_name(self, ocr_result):
        """Generate a filename proposal based on OCR results and YAML rules."""
        try:
            logger.debug("Proposing filename based on OCR results")
            
            if not ocr_result or ocr_result.strip() == "":
                logger.error("OCR result is empty")
                return f"{datetime.now().strftime('%Y年%m月%d日')}_不明.pdf"
                
            if not self.yaml_rules:
                logger.error("YAML rules not loaded")
                return f"{datetime.now().strftime('%Y年%m月%d日')}_エラー.pdf"
            
            rules = self.yaml_rules.get("ファイル命名のルール", [])
            logger.debug(f"Found {len(rules)} rules in YAML")
            
            # Normalize OCR result: fix common OCR errors
            ocr_result_normalized = self._normalize_ocr_result(ocr_result)
            
            # Set up format parameters with defaults
            current_date = datetime.now().strftime('%Y年%m月%d日')
            format_params = DefaultDict({
                "担当者": self.selected_person.get(),
                "日付": current_date,
                "今日の日付": current_date,
                "組織名": self.config.get('ORGANIZATION_NAME'),
                "タイトル": self.config.get('TITLE'),
                "内容": "不明",
                "金額": "不明",
                "取引先": "不明",
                "申請者": "不明",
                "年度": "不明"
            })
            
            # Auto-assign person if needed
            if self.selected_person.get() == "担当者自動設定":
                format_params["担当者"] = self._auto_detect_person(ocr_result_normalized)
            
            # Extract a date from the OCR result
            western_date = self._extract_date(ocr_result_normalized)
            logger.debug(f"Extracted date: {western_date}")
            
            # Try to find a matching rule
            sorted_rules = sorted(
                rules, 
                key=lambda x: x.get("ルール", {}).get("優先順位", float("inf")) 
                if isinstance(x, dict) and "ルール" in x 
                else x.get("優先順位", float("inf"))
            )
            
            for rule in sorted_rules:
                current_rule = rule.get("ルール", rule) if "ルール" in rule else rule
                logger.debug(f"Checking rule: {current_rule.get('説明', 'Unnamed rule')}")
                
                if "正規表現" in current_rule and re.search(
                    current_rule["正規表現"], ocr_result_normalized, re.IGNORECASE):
                    
                    logger.debug(f"Rule matched: {current_rule.get('説明', 'Unnamed rule')}")
                    return self._process_rule(current_rule, western_date, format_params, ocr_result_normalized)
            
            # Apply default rule if no specific rule matched
            logger.debug("No specific rule matched, applying default rule")
            default_rule_dict = next((r for r in rules if "デフォルト" in r), {})
            default_rule = default_rule_dict.get("デフォルト", {})
            
            if not default_rule:
                logger.warning("No default rule found in YAML")
                return f"{western_date}_不明.pdf"
                
            return self._process_rule(default_rule, western_date, format_params, ocr_result_normalized)
            
        except Exception as e:
            logger.exception("Error proposing filename")
            return f"{datetime.now().strftime('%Y年%m月%d日')}_エラー.pdf"
    
    def _normalize_ocr_result(self, ocr_result):
        """Normalize OCR result to fix common OCR errors."""
        # Replace misrecognized era names
        normalized = re.sub(r'(今和|合和)', '令和', ocr_result)
        
        # Fix common OCR errors with spaces
        normalized = re.sub(r'(\d)\s+(\d)', r'\1\2', normalized)  # Remove spaces between digits
        normalized = re.sub(r'年\s+(\d)', r'年\1', normalized)    # Fix "年 X" to "年X"
        normalized = re.sub(r'月\s+(\d)', r'月\1', normalized)    # Fix "月 X" to "月X"
        
        return normalized
    
    def _auto_detect_person(self, ocr_text):
        """Automatically detect the person responsible based on OCR text content."""
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
                    logger.debug(f"Auto-detected person: {person} based on keyword: {keyword}")
                    return person
        
        logger.debug("No person automatically detected")
        return ""  # Empty string if no match
    
    def _extract_date(self, text):
        """Extract a date from text and convert to YYYY年MM月DD日 format."""
        # Try various date formats
        
        # 令和〇年〇月〇日 format
        date_match = re.search(r'令和\s*(\d+)\s*年\s*(\d+)\s*月\s*(\d+)\s*日', text)
        if date_match:
            year = 2018 + int(date_match.group(1))
            month = date_match.group(2).zfill(2)
            day = date_match.group(3).zfill(2)
            return f"{year}年{month}月{day}日"
            
        # 平成〇年〇月〇日 format
        date_match = re.search(r'平成\s*(\d+)\s*年\s*(\d+)\s*月\s*(\d+)\s*日', text)
        if date_match:
            year = 1988 + int(date_match.group(1))
            month = date_match.group(2).zfill(2)
            day = date_match.group(3).zfill(2)
            return f"{year}年{month}月{day}日"
            
        # 昭和〇年〇月〇日 format
        date_match = re.search(r'昭和\s*(\d+)\s*年\s*(\d+)\s*月\s*(\d+)\s*日', text)
        if date_match:
            year = 1925 + int(date_match.group(1))
            month = date_match.group(2).zfill(2)
            day = date_match.group(3).zfill(2)
            return f"{year}年{month}月{day}日"
            
        # YYYY年MM月DD日 format
        date_match = re.search(r'(\d{4})年(\d{1,2})月(\d{1,2})日', text)
        if date_match:
            year = date_match.group(1)
            month = date_match.group(2).zfill(2)
            day = date_match.group(3).zfill(2)
            return f"{year}年{month}月{day}日"
            
        # YYYY/MM/DD or YYYY-MM-DD format
        date_match = re.search(r'(\d{4})[/-](\d{1,2})[/-](\d{1,2})', text)
        if date_match:
            year = date_match.group(1)
            month = date_match.group(2).zfill(2)
            day = date_match.group(3).zfill(2)
            return f"{year}年{month}月{day}日"
            
        # If no date found, use current date
        return datetime.now().strftime('%Y年%m月%d日')
    
    def _process_rule(self, rule, date, format_params, ocr_result):
        """Process a rule to generate a filename."""
        doc_type = rule.get("書類の種類", "不明")
        naming_rule = rule.get("命名ルール", "{日付}_{内容}")
        
        try:
            # Format the prompt with all available parameters
            prompt = rule.get("プロンプト", "").format(
                書類の種類=doc_type,
                命名ルール=naming_rule,
                ocr_result=ocr_result,
                日付=date,
                今日の日付=format_params["今日の日付"],
                担当者=format_params["担当者"],
                取引先=format_params["取引先"],
                タイトル=format_params["タイトル"],
                金額=format_params["金額"],
                申請者=format_params["申請者"],
                年度=format_params["年度"],
                内容=format_params["内容"],
                組織名=format_params.get("組織名", "")
            )
            
            logger.debug(f"Sending prompt to OpenAI: {prompt[:100]}...")
            
            # Call OpenAI API
            openai.api_key = self.config.get('OPENAI_API_KEY')
            response = openai.chat.completions.create(
                model=self.config.get('OPENAI_MODEL'),
                messages=[{"role": "user", "content": prompt}],
                temperature=self.config.get('OPENAI_TEMPERATURE')
            )
            
            # Parse response
            file_name_raw = response.choices[0].message.content.strip()
            logger.debug(f"OpenAI response: {file_name_raw}")
            
            # Extract the filename (expected to be between ** markers)
            match = re.search(r'\*\*(.+?)\*\*', file_name_raw)
            file_name = match.group(1) if match else file_name_raw
            
            if not file_name or file_name == "ファイル名":
                file_name = date
            
            # Post-process the filename
            file_name = self._convert_wareki_to_western(file_name)
            file_name = file_name.replace("（不明）", "").replace("(不明)", "")
            
            # Ensure file has .pdf extension
            if not file_name.lower().endswith('.pdf'):
                file_name += '.pdf'
                
            logger.info(f"Proposed filename: {file_name}")
            return file_name
            
        except Exception as e:
            logger.exception(f"Error processing rule: {doc_type}")
            return f"{date}_エラー.pdf"
    
    def _rename_pdf(self, pdf_file_path, new_pdf_name):
        """Rename a PDF file with the new name."""
        try:
            logger.debug(f"Renaming: {pdf_file_path} -> {new_pdf_name}")
            
            if not os.path.exists(pdf_file_path):
                logger.error(f"File does not exist: {pdf_file_path}")
                return False
            
            # Sanitize filename (remove invalid characters)
            new_pdf_name = re.sub(r'[\\/:*?"<>|&]', '_', new_pdf_name)
            
            # Ensure .pdf extension
            if not new_pdf_name.lower().endswith('.pdf'):
                new_pdf_name += '.pdf'
            
            # Get the target path
            directory = os.path.dirname(pdf_file_path)
            new_file_path = os.path.join(directory, new_pdf_name)
            
            # Skip if names are the same
            if os.path.normpath(pdf_file_path) == os.path.normpath(new_file_path):
                logger.info(f"File already has the correct name: {pdf_file_path}")
                return new_file_path
            
            # Handle name collisions by adding a counter
            if os.path.exists(new_file_path):
                base_name, ext = os.path.splitext(new_pdf_name)
                counter = 1
                while os.path.exists(os.path.join(directory, f"{base_name}_{counter}{ext}")):
                    counter += 1
                new_file_path = os.path.join(directory, f"{base_name}_{counter}{ext}")
            
            # Perform the rename
            os.rename(pdf_file_path, new_file_path)
            logger.info(f"File renamed successfully: {new_file_path}")
            
            # Small delay to allow file system to update
            time.sleep(0.5)
            
            return new_file_path
            
        except Exception as e:
            logger.exception(f"Error renaming file: {pdf_file_path}")
            return False


class PDFRenamerApp:
    """Main application class for the PDF Renamer."""
    
    def __init__(self, root):
        """Initialize the application UI and components."""
        self.root = root
        self.config_manager = ConfigManager()
        self.status_queue = queue.Queue()
        self.selected_person = tk.StringVar(value=self.config_manager.get('DEFAULT_PERSON'))
        self.selected_model = tk.StringVar(value=self.config_manager.get('OPENAI_MODEL'))
        self.processing_flag = tk.BooleanVar(value=False)
        
        # Set up the UI
        self._setup_ui()
        
        # Validate configuration on startup
        self._validate_config_on_startup()
    
    def _setup_ui(self):
        """Set up the user interface."""
        self.root.title(f"PDF Renamer {APP_VERSION}")
        self.root.geometry("700x600")
        self.root.minsize(600, 400)
        
        # Center the window
        screen_width = self.root.winfo_screenwidth()
        screen_height = self.root.winfo_screenheight()
        x = (screen_width - 700) // 2
        y = (screen_height - 600) // 2 - 100  # Adjust to be slightly above center
        self.root.geometry(f"700x600+{x}+{y}")
        
        # Create and configure a style
        style = ttk.Style()
        style.configure("TButton", padding=6, relief="flat", font=("Helvetica", 10))
        style.configure("TLabel", font=("Helvetica", 10))
        style.configure("Header.TLabel", font=("Helvetica", 12, "bold"))
        
        # Main frame with padding
        main_frame = ttk.Frame(self.root, padding=10)
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Menu bar
        self._create_menu()
        
        # Person selector frame
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
        
        # Folder selection frame
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
        
        # Progress section
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
        
        # Status text area
        status_frame = ttk.LabelFrame(main_frame, text="処理状況")
        status_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 10))
        
        self.status_text = scrolledtext.ScrolledText(
            status_frame,
            wrap=tk.WORD,
            width=80,
            height=15
        )
        self.status_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Button section
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
        
        # Start status queue polling
        self._poll_status_queue()
    
    def _create_menu(self):
        """Create the application menu bar."""
        menubar = tk.Menu(self.root)
        self.root.config(menu=menubar)
        
        # File menu
        file_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="ファイル", menu=file_menu)
        file_menu.add_command(label="PDFフォルダ選択...", command=self._browse_folder)
        file_menu.add_separator()
        file_menu.add_command(label=".envを編集", command=lambda: self._open_file(self.config_manager.env_path))
        file_menu.add_command(label="YAMLルールを編集", command=lambda: self._open_file(self.config_manager.get('YAML_FILE')))
        file_menu.add_separator()
        file_menu.add_command(label="終了", command=self.root.quit)
        
        # Settings menu
        settings_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="設定", menu=settings_menu)
        
        # Person submenu
        person_menu = tk.Menu(settings_menu, tearoff=0)
        settings_menu.add_cascade(label="担当者の設定", menu=person_menu)
        
        # Add auto-setting option
        person_menu.add_radiobutton(
            label="担当者自動設定",
            value="担当者自動設定",
            variable=self.selected_person,
            command=lambda: self._set_person("担当者自動設定")
        )
        
        person_menu.add_separator()
        
        # Add all persons from config
        persons = self.config_manager.get('PERSONS', [])
        for person in persons:
            if person:  # Skip empty strings
                person_menu.add_radiobutton(
                    label=person,
                    value=person,
                    variable=self.selected_person,
                    command=lambda p=person: self._set_person(p)
                )
        
        # AI Model submenu
        ai_model_menu = tk.Menu(settings_menu, tearoff=0)
        settings_menu.add_cascade(label="AIモデルの設定", menu=ai_model_menu)
        
        # Available AI models
        available_models = [
            "gpt-4.1",
            "gpt-4",
            "gpt-3.5-turbo",
            "gpt-3.5-turbo-16k"
        ]
        
        # Add model selection options
        for model in available_models:
            ai_model_menu.add_radiobutton(
                label=model,
                value=model,
                variable=self.selected_model,
                command=lambda m=model: self._set_ai_model(m)
            )
        
        # Help menu
        help_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="ヘルプ", menu=help_menu)
        help_menu.add_command(label="バージョン情報", command=self._show_about)
        help_menu.add_command(label="使い方", command=self._show_help)
    
    def _browse_folder(self):
        """Open a folder browser dialog to select the PDF folder."""
        current_folder = self.folder_var.get()
        folder = askdirectory(
            initialdir=current_folder if current_folder else "/",
            title="PDFファイルのあるフォルダを選択"
        )
        
        if folder:
            self.folder_var.set(folder)
            self.config_manager.set('PDF_FOLDER_PATH', folder)
            # Update the status text to show the selected folder
            self._add_to_status(f"PDFフォルダを設定しました: {folder}\n")
    
    def _open_file(self, file_path):
        """Open a file or folder in the default application."""
        if not file_path:
            messagebox.showerror("エラー", "ファイルまたはフォルダが指定されていません。")
            return
            
        try:
            logger.debug(f"Opening file: {file_path}")
            
            if platform.system() == "Windows":
                os.startfile(file_path)
            elif platform.system() == "Linux":
                subprocess.run(["xdg-open", file_path], check=True)
            elif platform.system() == "Darwin":  # macOS
                subprocess.run(["open", file_path], check=True)
                
        except Exception as e:
            logger.exception(f"Error opening file: {file_path}")
            messagebox.showerror("エラー", f"ファイルを開けませんでした: {e}")
    
    def _set_person(self, person):
        """Set the selected person and update the UI."""
        self.selected_person.set(person)
        logger.info(f"Person set to: {person}")
        self._add_to_status(f"担当者を「{person}」に設定しました\n")
    
    def _on_person_changed(self, event):
        """Handle person selection change event."""
        self._set_person(self.selected_person.get())
    
    def _add_to_status(self, message):
        """Add a message to the status text area."""
        self.status_text.insert(tk.END, message)
        self.status_text.see(tk.END)
    
    def _poll_status_queue(self):
        """Poll the status queue and update the UI."""
        try:
            while not self.status_queue.empty():
                message = self.status_queue.get(0)
                self._add_to_status(message)
                
        except queue.Empty:
            pass
            
        finally:
            # Schedule next poll
            self.root.after(100, self._poll_status_queue)
    
    def _validate_config_on_startup(self):
        """Validate the configuration when the application starts."""
        missing = self.config_manager.validate_config()
        
        if missing:
            message = "以下の設定が不足しています:\n\n" + "\n".join([f"- {item}" for item in missing])
            message += "\n\n設定ファイル (.env) を編集して必要な情報を追加してください。"
            
            messagebox.showwarning("設定の問題", message)
    
    def _start_renaming(self):
        """Start the PDF renaming process."""
        if self.processing_flag.get():
            messagebox.showinfo("処理中", "現在処理中です。完了するまでお待ちください。")
            return
            
        # Update PDF folder from UI
        pdf_folder = self.folder_var.get()
        self.config_manager.set('PDF_FOLDER_PATH', pdf_folder)
        
        # Validate folder exists
        if not pdf_folder or not os.path.isdir(pdf_folder):
            messagebox.showerror("エラー", f"PDFフォルダが存在しません: {pdf_folder}")
            return
            
        # Find PDF files in the folder
        pdf_files = [f for f in os.listdir(pdf_folder) if f.lower().endswith('.pdf')]
        
        if not pdf_files:
            messagebox.showinfo("情報", f"フォルダ内にPDFファイルがありません: {pdf_folder}")
            return
            
        # Reset UI
        self.progress_bar['value'] = 0
        self.progress_label.config(text="進捗: 0%")
        self.status_text.delete(1.0, tk.END)
        self._add_to_status(f"PDFフォルダ: {pdf_folder}\n")
        self._add_to_status(f"PDF数: {len(pdf_files)}個\n")
        self._add_to_status(f"担当者: {self.selected_person.get()}\n")
        self._add_to_status("リネーム処理を開始します...\n")
        self._add_to_status("-" * 50 + "\n")
        
        # Set processing flag
        self.processing_flag.set(True)
        
        # Disable the start button during processing
        self.rename_button.config(state="disabled")
        
        # Start processing in a separate thread
        threading.Thread(
            target=self._process_pdf_files,
            args=(pdf_folder, pdf_files),
            daemon=True
        ).start()
    
    def _process_pdf_files(self, pdf_folder, pdf_files):
        """Process all PDF files in a separate thread."""
        try:
            processor = PDFProcessor(self.config_manager, self.selected_person, self.status_queue)
            
            total_files = len(pdf_files)
            completed = 0
            successful = 0
            
            # Calculate optimal number of workers
            max_workers = min(os.cpu_count() or 4, 4)  # Limit to 4 workers to avoid API rate limits
            
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
                    
                    # Update UI in a thread-safe way
                    self.root.after(0, lambda val=progress_value: 
                        self._update_progress(val, completed, successful, total_files))
            
            # Final status update
            summary = (
                f"\n{'-' * 50}\n"
                f"処理完了: {completed}/{total_files} ファイル\n"
                f"成功: {successful} ファイル\n"
                f"失敗: {completed - successful} ファイル\n"
            )
            self.status_queue.put(summary)
            
        except Exception as e:
            logger.exception("Error in PDF processing thread")
            self.status_queue.put(f"処理全体のエラー: {str(e)}\n")
            
        finally:
            # Reset UI in a thread-safe way
            self.root.after(0, self._reset_after_processing)
    
    def _update_progress(self, value, completed, successful, total):
        """Update progress bar and label."""
        self.progress_bar['value'] = value
        self.progress_label.config(
            text=f"進捗: {int(value)}% ({completed}/{total}件, 成功: {successful}件)"
        )
    
    def _reset_after_processing(self):
        """Reset UI state after processing is complete."""
        self.processing_flag.set(False)
        self.rename_button.config(state="normal")
    
    def _show_about(self):
        """Show the about dialog."""
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
        """Show the help dialog."""
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
        """Set the AI model and update the configuration."""
        try:
            self.config_manager.set('OPENAI_MODEL', model)
            self.config_manager.save_config()
            logger.info(f"AI model set to: {model}")
            self._add_to_status(f"AIモデルを「{model}」に設定しました\n")
        except Exception as e:
            logger.error(f"Error setting AI model: {e}")
            messagebox.showerror("エラー", f"AIモデルの設定に失敗しました: {e}")


def main():
    """Application entry point."""
    try:
        # Set up root window
        root = tk.Tk()
        app = PDFRenamerApp(root)
        
        # Start the main loop
        root.mainloop()
        
    except Exception as e:
        logger.exception("Application error")
        messagebox.showerror("アプリケーションエラー", str(e))


if __name__ == "__main__":
    main()