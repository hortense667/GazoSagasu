"""
ローカルAI画像検索システム (Gemini + ローカル処理版)
(c) 2025 / Satoshi Endo @hortense667

Gemini 2.5 Flash-Liteを使用した超高速な画像キャプション生成と、
ローカルでのベクトル処理を組み合わせた画像検索システムです。

特徴:
- Gemini 2.5 Flash-Lite: 超高速な画像キャプション生成（2-4倍高速）
- ローカルベクトル処理: NPU/GPUを活用した高速処理
- バッチ処理: 1回で1000枚の画像を処理
- 差分更新: 変更されたファイルのみを再処理
- マルチプラットフォーム対応: Windows NPU、Mac Metal、Linux CUDA
"""

import os
import glob
import json
import base64
from datetime import datetime, time
from PIL import Image, ExifTags
import warnings
warnings.filterwarnings("ignore")

import time
try:
    from google.api_core import exceptions
except ImportError:
    print("google-api-coreライブラリが利用できません。レート制限の処理が制限されます。")
    print("インストール方法: pip install google-api-core")
    exceptions = None

# Intel MKLエラー回避のための設定（最優先で実行）
os.environ['MKL_THREADING_LAYER'] = 'GNU'
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['VECLIB_MAXIMUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'

# NumPy設定（MKL回避）
try:
    import numpy as np
    np.set_printoptions(threshold=1000)
    np.random.seed(42)
except ImportError as e:
    print(f"NumPy読み込みエラー: {e}")
    np = None

# sklearn設定
try:
    from sklearn.metrics.pairwise import cosine_similarity
except ImportError as e:
    print(f"sklearn読み込みエラー: {e}")
    cosine_similarity = None

# その他のライブラリ
try:
    import torch
    import google.generativeai as genai
    import onnxruntime as ort
except ImportError as e:
    print(f"ライブラリ読み込みエラー: {e}")
    torch = None
    genai = None
    ort = None

# 日本語形態素解析（janome）
try:
    from janome.tokenizer import Tokenizer
    japanese_tokenizer = Tokenizer()
    JAPANESE_TOKENIZER_AVAILABLE = True
except ImportError:
    print("janomeライブラリが利用できません。日本語の単語分割が制限されます。")
    print("インストール方法: pip install janome")
    japanese_tokenizer = None
    JAPANESE_TOKENIZER_AVAILABLE = False
except Exception as e:
    print(f"janomeの初期化に失敗しました: {e}")
    print("日本語の単語分割が制限されます。")
    japanese_tokenizer = None
    JAPANESE_TOKENIZER_AVAILABLE = False

import argparse
import calendar

class LocalAIImageSearcher:
    def __init__(self):
        """ローカルAI画像検索システムの初期化（Gemini + ローカル処理版）"""
        # MKLエラーチェック
        if np is None:
            print("❌ NumPyが読み込めません。MKLエラーが発生している可能性があります。")
            print("   解決方法:")
            print("   1. 管理者として実行")
            print("   2. アンチウイルスを一時的に無効化")
            print("   3. 別のPCで実行")
            return
        
        if cosine_similarity is None:
            print("❌ sklearnが読み込めません。")
            return
        
        self.device = self._get_best_device()
        self.gemini_model = None
        self.text_embedding_model = None
        self.text_embedding_tokenizer = None
        self.ort_session = None
        self.daily_limit = 1000  # Gemini API 1日の画像制限
        
        # TPM制限の設定
        self.tpm_limit = 250000  # 1分あたり250,000トークン
        self.rpm_limit = 15      # 1分あたり15リクエスト（公式ドキュメントに合わせる）
        self.tpm_usage = []      # 過去1分間のトークン使用量を記録
        self.rpm_usage = []      # 過去1分間のリクエスト数を記録
        
        # 有料契約の判定
        self.paid_plan = self._check_paid_plan()
        
        # 使用量の初期化
        self._load_daily_usage()
        
        # print(f"使用デバイス: {self.device}")
        # if self.paid_plan:
        #     print("✅ 有料契約モード: API制限なし")
        # else:
        #     print("📊 無料枠モード: 1日1000件制限")
        self._load_models()
    
    def _check_paid_plan(self):
        """有料契約の判定"""
        # 環境変数による判定
        if os.getenv('GEMINI_PAID_PLAN') == 'true':
            return True
        
        return False
    
    def _load_daily_usage(self):
        """日別使用量を読み込み"""
        from datetime import date
        today = date.today().isoformat()
        usage_file = "gemini_daily_usage.json"
        
        try:
            if os.path.exists(usage_file):
                with open(usage_file, 'r') as f:
                    usage_data = json.load(f)
                
                if usage_data.get('date') == today:
                    self.daily_usage_count = usage_data.get('count', 0)
                else:
                    # 日付が変わった場合はリセット
                    self.daily_usage_count = 0
            else:
                self.daily_usage_count = 0
        except:
            self.daily_usage_count = 0
        
        # 有料契約の場合は使用量を表示しない
        if not self.paid_plan:
            print(f"今日のGemini API使用量: {self.daily_usage_count}/{self.daily_limit}")
    
    def _save_daily_usage(self):
        """日別使用量を保存"""
        from datetime import date
        today = date.today().isoformat()
        usage_file = "gemini_daily_usage.json"
        
        usage_data = {
            'date': today,
            'count': self.daily_usage_count
        }
        
        try:
            with open(usage_file, 'w') as f:
                json.dump(usage_data, f)
        except Exception as e:
            print(f"使用量保存エラー: {e}")
    
    def _check_and_wait_for_rate_limits(self, estimated_tokens=1000):
        """TPM/RPM制限をチェックし、必要に応じて待機"""
        import time
        current_time = time.time()
        
        # 1分前のデータを削除
        self.tpm_usage = [t for t in self.tpm_usage if current_time - t['time'] < 60]
        self.rpm_usage = [t for t in self.rpm_usage if current_time - t['time'] < 60]
        
        # 現在の使用量を計算
        current_tpm = sum(t['tokens'] for t in self.tpm_usage)
        current_rpm = len(self.rpm_usage)
        
        # TPM制限チェック
        if current_tpm + estimated_tokens > self.tpm_limit:
            wait_time = 60 - (current_time - self.tpm_usage[0]['time']) if self.tpm_usage else 60
            if wait_time > 0:
                print(f"  ⏳ TPM制限に近づいています。{wait_time:.1f}秒待機します...")
                print(f"    現在のTPM使用量: {current_tpm}/{self.tpm_limit}")
                time.sleep(wait_time)
                return self._check_and_wait_for_rate_limits(estimated_tokens)  # 再帰的に再チェック
        
        # RPM制限チェック
        if current_rpm >= self.rpm_limit:
            wait_time = 60 - (current_time - self.rpm_usage[0]['time']) if self.rpm_usage else 60
            if wait_time > 0:
                print(f"  ⏳ RPM制限に達しました。{wait_time:.1f}秒待機します...")
                print(f"    現在のRPM使用量: {current_rpm}/{self.rpm_limit}")
                time.sleep(wait_time)
                return self._check_and_wait_for_rate_limits(estimated_tokens)  # 再帰的に再チェック
    
    def _record_api_usage(self, tokens_used):
        """API使用量を記録"""
        import time
        current_time = time.time()
        
        # TPM使用量を記録
        self.tpm_usage.append({
            'time': current_time,
            'tokens': tokens_used
        })
        
        # RPM使用量を記録
        self.rpm_usage.append({
            'time': current_time
        })
        
        # デバッグ情報（必要に応じてコメントアウト）
        # current_tpm = sum(t['tokens'] for t in self.tpm_usage)
        # current_rpm = len(self.rpm_usage)
        # print(f"    TPM使用量: {current_tpm}/{self.tpm_limit}, RPM使用量: {current_rpm}/{self.rpm_limit}")
    
    def _get_best_device(self):
        import platform
        
        providers = ort.get_available_providers()
        # print(f"利用可能なONNXプロバイダー: {providers}")
        
        # Macの場合はMPS（Metal）を優先
        if platform.system() == "Darwin":  # macOS
            if torch.backends.mps.is_available():
                print("✅ Mac MPS (Metal GPU) を使用します")
                return 'mps'
            else:
                print("✅ Mac CPU を使用します")
                return 'cpu'
        
        # Windowsの場合のみNPUを検出
        elif platform.system() == "Windows":
            # NPUの検出をより厳密に行う
            if 'DmlExecutionProvider' in providers:
            #    print("DirectMLプロバイダーが検出されました。NPUテストを実行中...")
                try:
                    # PyInstaller版でのパス処理
                    import sys
                    if getattr(sys, 'frozen', False):
                        # PyInstaller版の場合
                        base_path = os.path.dirname(sys.executable)
                        onnx_model_path = os.path.join(base_path, "clip_image_encoder.onnx")
                #         print(f"PyInstaller版: ONNXモデルパス = {onnx_model_path}")
                    else:
                        # Python実行の場合
                        onnx_model_path = "clip_image_encoder.onnx"
                #         print(f"Python実行版: ONNXモデルパス = {onnx_model_path}")
                    
                    # ONNXモデルファイルの存在チェック
                    if not os.path.exists(onnx_model_path):
                #         print(f"❌ ONNXモデルファイルが存在しません: {onnx_model_path}")
                #         print("🔄 CPUにフォールバックします")
                        return 'cpu'
                    
                #     print(f"✅ ONNXモデルファイルを確認: {onnx_model_path}")
                    
                    # DirectMLプロバイダーが実際に動作するかテスト
                    test_session = ort.InferenceSession(onnx_model_path, providers=['DmlExecutionProvider', 'CPUExecutionProvider'])
                #      print("✅ Ryzen AI NPU（DirectML）が利用可能です")
                    return 'npu'
                except Exception as e:
                    print(f"❌ DirectMLプロバイダーのテストに失敗: {e}")
                    print("🔄 CPUにフォールバックします")
                    return 'cpu'
            elif torch.cuda.is_available():
            #     print("✅ CUDA GPUを使用します")
                return 'cuda'
            else:
            #     print("✅ CPUを使用します")
                return 'cpu'
        
        # Linux/その他の場合
        else:
            if torch.cuda.is_available():
                print("✅ CUDA GPUを使用します")
                return 'cuda'
            else:
                print("✅ CPUを使用します")
                return 'cpu'

    
    def _load_models(self):
        """必要なモデルを読み込み（Gemini + ローカル処理版）"""
        # print("Gemini + ローカルモデルを読み込み開始...")
        
        try:
            # Gemini APIキーは環境変数から取得
            api_key = os.getenv('GEMINI_API_KEY')
            if not api_key:
                print("❌ GEMINI_API_KEY環境変数が設定されていません")
                print("Gemini APIキーを設定してください")
                return
            
            genai.configure(api_key=api_key)
            self.gemini_model = genai.GenerativeModel('gemini-2.5-flash-lite')
            print("Gemini 2.5 Flash-Lite モデルを初期化しました")
            
            # 2. ローカルテキスト埋め込みモデル
            # print("  ローカルテキスト埋め込みモデルを読み込み中...")
            from transformers import AutoTokenizer, AutoModel
            embedding_model_name = "sentence-transformers/all-MiniLM-L6-v2"
            self.text_embedding_tokenizer = AutoTokenizer.from_pretrained(embedding_model_name)
            self.text_embedding_model = AutoModel.from_pretrained(embedding_model_name)
            
            # デバイスに移動
            if self.device == 'cuda':
                self.text_embedding_model = self.text_embedding_model.to('cuda')
            elif self.device == 'mps':
                self.text_embedding_model = self.text_embedding_model.to('mps')
            
            # 3. ONNX Runtime セッション (NPU使用時のみ)
            if self.device == 'npu':
                self._setup_onnx_runtime()
            
            print("ローカルモデルの読み込み完了しました")
            # print(f"実際の処理デバイス: {self.device}")
            
        except Exception as e:
            print(f"モデル読み込みエラー: {e}")
            print("基本的な機能のみで動作します...")
            self.text_embedding_model = None
            self.text_embedding_tokenizer = None
            self.gemini_model = None
    
    def _setup_onnx_runtime(self):
        """ONNX Runtime (NPU) のセットアップ"""
        try:
            # NPU用プロバイダーの設定
            providers = [
                ('DmlExecutionProvider', {
                    'device_id': 0,
                }),
                'CPUExecutionProvider'
            ]
            
            # PyInstaller版でのパス処理
            import sys
            if getattr(sys, 'frozen', False):
                # PyInstaller版の場合
                base_path = os.path.dirname(sys.executable)
                onnx_model_path = os.path.join(base_path, "clip_image_encoder.onnx")
            #    print(f"PyInstaller版: ONNXモデルパス = {onnx_model_path}")
            else:
                # Python実行の場合
                onnx_model_path = "clip_image_encoder.onnx"
            #    print(f"Python実行版: ONNXモデルパス = {onnx_model_path}")
            
            # print(f"ONNXモデルパス: {onnx_model_path}")
            
            # 実際のONNXモデルファイルがある場合のみ
            if os.path.exists(onnx_model_path):
            #     print(f"✅ ONNXモデルファイルを確認: {onnx_model_path}")
                self.ort_session = ort.InferenceSession(onnx_model_path, providers=providers)
                # print("NPU用ONNXモデルを読み込みました")
                
                # 実際にNPUが使用されているかテスト
                try:
                    # 正しい入力名を取得
                    input_name = self.ort_session.get_inputs()[0].name
                    # print(f"ONNXモデルの入力名: {input_name}")
                    
                    # 入力の形状を取得
                    input_shape = self.ort_session.get_inputs()[0].shape
                    # print(f"ONNXモデルの入力形状: {input_shape}")
                    
                    # ダミー入力を正しい形状で作成
                    if len(input_shape) == 4:  # [batch, channels, height, width]
                        test_input = np.random.randn(1, 3, 224, 224).astype(np.float32)
                    else:
                        # 形状が不明な場合は適当なサイズでテスト
                        test_input = np.random.randn(1, 3, 224, 224).astype(np.float32)
                    
                    # 正しい入力名でテスト推論
                    test_output = self.ort_session.run(None, {input_name: test_input})
                    # print("✅ NPU推論テスト成功")
                except Exception as e:
                    print(f"❌ NPU推論テスト失敗: {e}")
                    print("🔄 CPUにフォールバックします")
                    self.device = 'cpu'
                    self.ort_session = None
            else:
                print(f"❌ ONNXモデルファイルが見つかりません: {onnx_model_path}")
                print("🔄 CPUにフォールバックします")
                self.device = 'cpu'
        except Exception as e:
            print(f"❌ ONNX Runtime NPU初期化失敗: {e}")
            print("🔄 CPUにフォールバックします")
            self.device = 'cpu'
    
    def analyze_image_with_gemini(self, image_path, unlimited_mode=False, max_size=512):
        """Gemini 2.5 Flash-Liteで画像を解析してキャプションを生成（最適化版）"""

        try:
            # 1日の制限チェック（有料契約でない場合のみ）
            if not self.paid_plan and not unlimited_mode and self.daily_usage_count >= self.daily_limit:
                print(f"⚠️  Gemini API 1日の制限 ({self.daily_limit}リクエスト) に達しました")
                print("   明日までお待ちください")
                return f"API制限によりスキップ: {os.path.basename(image_path)}"
            
            image = Image.open(image_path).convert('RGB')
            original_size = image.size
            
            # 画像サイズ最適化（API費用削減）
            if max(image.size) > max_size:
                # アスペクト比を保持してリサイズ
                ratio = max_size / max(image.size)
                new_size = (int(image.size[0] * ratio), int(image.size[1] * ratio))
                image = image.resize(new_size, Image.Resampling.LANCZOS)
                print(f"  画像最適化: {original_size} → {new_size} ")
            else:
                print(f"  画像サイズ: {original_size} (最適化不要)")

            if self.gemini_model:
                # 高速処理のための最適化プロンプト
                prompt = """
                この画像を簡潔に日本語で説明してください。
                内容、色合い、構図を含めてください。
                ファイル名: {filename}
                """.format(filename=os.path.basename(image_path))
                
                response = self._generate_content_with_retry([prompt, image])
                caption = response.text
                                
                # キャプションを日本語で拡張
                caption_ja = f"画像の説明: {caption}。ファイル名: {os.path.basename(image_path)}。"
                
                return caption_ja
            else:
                return f"画像ファイル: {os.path.basename(image_path)}"
                
        except Exception as e:
            return f"画像解析エラー ({os.path.basename(image_path)}): {str(e)}"
    
    def analyze_images_batch(self, image_paths, unlimited_mode=False, max_size=512, batch_size=5, max_files=None):
        """複数画像をバッチで処理（APIリクエスト数削減版）"""
        
        if not self.gemini_model:
            return [(path, f"画像ファイル: {os.path.basename(path)}") for path in image_paths]
        
        results = []
        current_batch_size = batch_size
        consecutive_errors = 0
        max_consecutive_errors = 3  # 連続エラーが3回を超えたらバッチサイズを削減
        processed_count = 0  # 処理済み件数を追跡
        
        # バッチサイズごとに処理
        i = 0
        while i < len(image_paths):
            # 件数制限チェック
            if max_files is not None and max_files > 0 and processed_count >= max_files:
                print(f"指定された処理件数 {max_files} 件に達しました。バッチ処理を停止します。")
                break
                
            
            batch = image_paths[i:i+current_batch_size]
            
            # このバッチで処理する実際のサイズを記録
            processed_batch_size = len(batch)
            
            # 件数制限に合わせてバッチサイズを調整
            if max_files is not None and max_files > 0:
                remaining_count = max_files - processed_count
                if remaining_count < len(batch):
                    batch = batch[:remaining_count]
                    processed_batch_size = len(batch) # 調整後のサイズを再記録
                    print(f"件数制限により、バッチサイズを {len(batch)} 件に調整しました")
            
            # 1日の制限チェック（有料契約でない場合のみ）
            if not self.paid_plan and not unlimited_mode and self.daily_usage_count >= self.daily_limit:
                print(f"⚠️  Gemini API 1日の制限 ({self.daily_limit}リクエスト) に達しました")
                print("   明日までお待ちください")
                # 残りの画像をスキップ
                for path in batch:
                    results.append((path, f"API制限によりスキップ: {os.path.basename(path)}"))
                break
            
            try:
                # 画像を読み込みと最適化
                images = []
                filenames = []
                valid_paths = []
                
                for path in batch:
                    # 件数制限チェック
                    if max_files is not None and max_files > 0 and processed_count >= max_files:
                        break
                        
                    try:
                        # ファイルの存在確認
                        if not os.path.exists(path):
                            print(f"  ファイルが存在しません: {path}")
                            results.append((path, f"画像解析エラー ({os.path.basename(path)}): ファイルが存在しません"))
                            continue
                        
                        # 画像ファイルの読み込み
                        img = Image.open(path).convert('RGB')
                        original_size = img.size
                        
                        # 画像サイズ最適化（API費用削減）
                        if max(img.size) > max_size:
                            ratio = max_size / max(img.size)
                            new_size = (int(img.size[0] * ratio), int(img.size[1] * ratio))
                            img = img.resize(new_size, Image.Resampling.LANCZOS)
                            print(f"  画像最適化: {os.path.basename(path)} {original_size} → {new_size}")
                        
                        images.append(img)
                        filenames.append(os.path.basename(path))
                        valid_paths.append(path)
                        
                    except Exception as e:
                        print(f"  画像読み込みエラー ({os.path.basename(path)}): {e}")
                        results.append((path, f"画像解析エラー ({os.path.basename(path)}): {e}"))
                        continue
                
                # 有効な画像がない場合はスキップ
                if not images:
                    print(f"  バッチ内に有効な画像がありません")
                    continue
                
                # バッチ処理用のプロンプトを作成
                prompt = f"""
                以下の{len(images)}枚の画像をそれぞれ簡潔に日本語で説明してください。
                各画像の説明は「画像1:」「画像2:」のように番号を付けて区切ってください。
                
                画像ファイル名:
                {chr(10).join([f"画像{i+1}: {filename}" for i, filename in enumerate(filenames)])}
                
                各画像について、内容、色合い、構図を含めて説明してください。
                """
                
                # トークン数の推定
                try:
                    prompt_tokens = self.gemini_model.count_tokens(prompt).total_tokens
                except Exception:
                    prompt_tokens = len(prompt) // 2 # 簡易計算
                
                tokens_per_image = 500  # 画像1枚あたりの推定トークン数（安全マージン込み）
                estimated_tokens = prompt_tokens + (len(images) * tokens_per_image)
                print(f"  推定トークン数: {estimated_tokens} (プロンプト: {prompt_tokens}, 画像: {len(images)}枚)")
                
                # 複数画像を一度に処理
                content = [prompt] + images
                response = self._generate_content_with_retry(content, estimated_tokens=estimated_tokens)
                
                
                # 結果を解析して個別の説明に分割
                descriptions = self._parse_batch_response(response.text, len(images), filenames)
                
                # エラー率をチェック
                error_count = sum(1 for desc in descriptions if desc == "説明を生成できませんでした")
                success_rate = (len(descriptions) - error_count) / len(descriptions)
                
                if success_rate < 0.8:  # 成功率が80%未満の場合
                    consecutive_errors += 1
                    print(f"  警告: バッチ処理の成功率が低い ({success_rate:.1%})")
                else:
                    consecutive_errors = 0  # リセット
                
                # 動的バッチサイズ調整
                if consecutive_errors >= max_consecutive_errors and current_batch_size > 3:
                    current_batch_size = max(3, current_batch_size - 2)
                    consecutive_errors = 0
                    print(f"  バッチサイズを {current_batch_size + 2} → {current_batch_size} に削減")
                elif consecutive_errors == 0 and current_batch_size < 10 and i > 0:
                    # 安定している場合はバッチサイズを増加
                    current_batch_size = min(10, current_batch_size + 1)
                    print(f"  バッチサイズを {current_batch_size - 1} → {current_batch_size} に増加")
                
                # 結果を保存（有効な画像のみ）
                for j, (path, desc) in enumerate(zip(valid_paths, descriptions)):
                    # 件数制限チェック
                    if max_files is not None and max_files > 0 and processed_count >= max_files:
                        break
                        
                    caption_ja = f"画像の説明: {desc}。ファイル名: {os.path.basename(path)}。"
                    results.append((path, caption_ja))
                    processed_count += 1

                # 使用量をカウント（有料契約でない場合のみ）
                if not self.paid_plan:
                    self.daily_usage_count += 1
                    # 使用量を保存
                    self._save_daily_usage()

                # TPM/RPM使用量の表示
                current_tpm = sum(t['tokens'] for t in self.tpm_usage)
                current_rpm = len(self.rpm_usage)
                print(f"  バッチ処理完了: {len(batch)}枚の画像を処理 (成功率: {success_rate:.1%}, 処理済み: {processed_count})")
                print(f"    現在の使用量 - TPM: {current_tpm}/{self.tpm_limit}, RPM: {current_rpm}/{self.rpm_limit}")
                
                # 件数制限に達した場合は処理を停止
                if max_files is not None and max_files > 0 and processed_count >= max_files:
                    print(f"指定された処理件数 {max_files} 件に達しました。バッチ処理を停止します。")
                    break
                
            except Exception as e:
                print(f"  バッチ処理エラー: {e}")
                consecutive_errors += 1
                
                # エラーの場合は個別処理にフォールバック
                for path in batch:
                    # 件数制限チェック
                    if max_files is not None and max_files > 0 and processed_count >= max_files:
                        break
                        
                    individual_result = self.analyze_image_with_gemini(path, unlimited_mode, max_size)
                    results.append((path, individual_result))
                    processed_count += 1
                
                # バッチサイズを削減
                if consecutive_errors >= max_consecutive_errors and current_batch_size > 3:
                    current_batch_size = max(3, current_batch_size - 2)
                    consecutive_errors = 0
                    print(f"  エラーによりバッチサイズを {current_batch_size + 2} → {current_batch_size} に削減")
            finally:
                i += processed_batch_size
        
        return results
    
    def _parse_batch_response(self, response_text, image_count, filenames):
        """バッチレスポンスを個別の説明に分割"""
        descriptions = []
        
        # 番号付きの説明を探す（複数のパターンを試行）
        import re
        
        # パターン1: 「画像1:」「画像2:」形式
        pattern1 = r'画像(\d+):\s*(.*?)(?=画像\d+:|$)'
        matches1 = re.findall(pattern1, response_text, re.DOTALL)
        
        # パターン2: 「1.」「2.」形式
        pattern2 = r'(\d+)\.\s*(.*?)(?=\d+\.|$)'
        matches2 = re.findall(pattern2, response_text, re.DOTALL)
        
        # パターン3: 「**1枚目の画像:**」形式
        pattern3 = r'\*\*(\d+)枚目の画像:\*\*\s*(.*?)(?=\*\*\d+枚目の画像:\*\*|$)'
        matches3 = re.findall(pattern3, response_text, re.DOTALL)
        
        # パターン4: 「**画像1（ファイル名）:**」形式
        pattern4 = r'\*\*画像(\d+)（.*?）:\*\*\s*(.*?)(?=\*\*画像\d+（.*?）:\*\*|$)'
        matches4 = re.findall(pattern4, response_text, re.DOTALL)
        
        # 最も適切なパターンを選択
        if len(matches1) == image_count:
            matches = matches1
        elif len(matches2) == image_count:
            matches = matches2
        elif len(matches3) == image_count:
            matches = matches3
        elif len(matches4) == image_count:
            matches = matches4
        else:
            # どのパターンも完全に一致しない場合は、最も近いものを使用
            all_matches = [matches1, matches2, matches3, matches4]
            best_matches = max(all_matches, key=len)
            if best_matches:
                matches = best_matches
            else:
                matches = []
        
        if matches:
            # 番号順にソート
            matches.sort(key=lambda x: int(x[0]))
            descriptions = [match[1].strip() for match in matches]
        else:
            # 正規表現で分割できない場合は手動分割
            lines = response_text.split('\n')
            current_desc = []
            current_num = 0
            
            for line in lines:
                line = line.strip()
                if line.startswith('画像') and ':' in line:
                    if current_desc and current_num > 0:
                        descriptions.append(' '.join(current_desc))
                        current_desc = []
                    current_num += 1
                elif line.startswith('**') and '**' in line and ':' in line:
                    if current_desc and current_num > 0:
                        descriptions.append(' '.join(current_desc))
                        current_desc = []
                    current_num += 1
                elif line and not line.startswith('**'):
                    current_desc.append(line)
            
            if current_desc and current_num > 0:
                descriptions.append(' '.join(current_desc))
        
        # 不足している場合は空文字で補完
        while len(descriptions) < image_count:
            descriptions.append("説明を生成できませんでした")
        
        return descriptions[:image_count]
    
    def get_text_embedding(self, text):
        """テキストの埋め込みベクトルを生成（ローカル処理）"""
        try:
            # モデルが読み込まれていない場合はエラー
            if self.text_embedding_model is None or self.text_embedding_tokenizer is None:
        #        print("  テキスト埋め込みモデルが読み込まれていません")
                return None
            
            # NPUではテキスト埋め込み用ONNXモデルがないため、PyTorchにフォールバック
            if self.device == 'npu':
                return self._get_embedding_pytorch(text)
            elif self.ort_session and self.device == 'npu':
                # NPU (ONNX Runtime) を使用
        #        print("  NPU (ONNX Runtime) でテキスト埋め込み生成中...")
                return self._get_embedding_onnx(text)
            else:
                # PyTorchモデルを使用
                if self.device == 'cuda':
                    device_info = "CUDA"
                elif self.device == 'mps':
                    device_info = "MPS (Mac Metal)"
                else:
                    device_info = "CPU"
        #        print(f"  PyTorch ({device_info}) でテキスト埋め込み生成中...")
                return self._get_embedding_pytorch(text)
                
        except Exception as e:
            print(f"埋め込み生成エラー: {e}")
            return None
    
    def _get_embedding_pytorch(self, text):
        """PyTorchモデルで埋め込み生成"""
        if self.text_embedding_model is None or self.text_embedding_tokenizer is None:
            print("  テキスト埋め込みモデルが初期化されていません")
            return None
            
        inputs = self.text_embedding_tokenizer(
            text, return_tensors="pt", 
            max_length=512, truncation=True, padding=True
        )
        
        # NPUの場合はCPUで処理
        if self.device == 'npu':
            device = 'cpu'
        elif self.device == 'cuda':
            device = 'cuda'
        elif self.device == 'mps':
            device = 'mps'
        else:
            device = 'cpu'
        
        if device != 'cpu':
            inputs = {k: v.to(device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.text_embedding_model(**inputs)
            # [CLS]トークンの埋め込みを使用
            embedding = outputs.last_hidden_state[:, 0, :].cpu().numpy()[0]
        
        return embedding.tolist()
    
    def _get_embedding_onnx(self, text):
        """ONNX Runtime (NPU) で埋め込み生成"""
        inputs = self.text_embedding_tokenizer(
            text, return_tensors="np", 
            max_length=512, truncation=True, padding=True
        )
        
        ort_inputs = {k: v for k, v in inputs.items()}
        ort_outputs = self.ort_session.run(None, ort_inputs)
        
        # 最初の出力 (last_hidden_state) の [CLS] トークン
        embedding = ort_outputs[0][0, 0, :]  
        return embedding.tolist()

    def _generate_content_with_retry(self, *args, **kwargs):
        """レート制限を考慮して、指数バックオフ付きでAPIを呼び出す"""
        if exceptions is None:
            # ライブラリがない場合は、そのまま呼び出す
            return self.gemini_model.generate_content(*args, **kwargs)

        estimated_tokens = kwargs.pop('estimated_tokens', 2000)
        max_retries = 5  # 最大リトライ回数
        delay = 2  # 初回待機時間（秒）

        for i in range(max_retries):
            try:
                # TPM/RPM制限の事前チェック
                self._check_and_wait_for_rate_limits(estimated_tokens)
                
                response = self.gemini_model.generate_content(*args, **kwargs)
                
                # 使用量を記録（実際のトークン数を取得できない場合は推定値を使用）
                actual_tokens = getattr(response, 'usage_metadata', None)
                if actual_tokens and hasattr(actual_tokens, 'total_token_count'):
                    tokens_used = actual_tokens.total_token_count
                else:
                    tokens_used = estimated_tokens
                
                self._record_api_usage(tokens_used)
                
                return response
            except exceptions.ResourceExhausted as e:
                # 429エラー (レート制限)
                print(f"  レート制限(RPM/TPM)に達しました。{delay}秒待機して再試行します... (試行 {i+1}/{max_retries})")
                time.sleep(delay)
                delay *= 2  # 次の待機時間を2倍に
            except exceptions.GoogleAPICallError as e:
                # その他のAPIエラー
                print(f"  API呼び出しエラー: {e}")
                if not hasattr(e, 'retryable') or not e.retryable:
                    raise e
                print(f"  {delay}秒待機して再試行します... (試行 {i+1}/{max_retries})")
                time.sleep(delay)
                delay *= 2
            except Exception as e:
                print(f"  予期せぬエラーが発生しました: {e}")
                raise e

        # 最大リトライ回数を超えた場合
        raise Exception(f"API呼び出しが{max_retries}回失敗しました。処理を中断します。")
    
    def generate_embeddings(self, max_files=None):
        """画像の説明文と埋め込みを生成（差分更新対応、1000枚ずつ処理）"""
        embeddings_file = "local_image_embeddings.json"
        embeddings_data = {}
        processed_count = 0
        new_files = []

        try:
            # 有料契約の判定
            unlimited_mode = self.paid_plan or (max_files == 0)
            paid_mode = self.paid_plan or (max_files is not None and max_files > 0)
            
            # 画像ファイルを取得（存在確認付き）
            image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.gif', '*.bmp', '*.webp']
            image_files = []
            for extension in image_extensions:
                found_files = glob.glob(f'**/{extension}', recursive=True)
                for file_path in found_files:
                    if os.path.exists(file_path) and os.path.isfile(file_path):
                        image_files.append(file_path)
            
            if not image_files:
                print("画像ファイルが見つかりませんでした。")
                return
            
            print(f"検出された画像ファイル数: {len(image_files)}")
            
            # 既存の埋め込みデータを読み込み（エラーハンドリング追加）
            if os.path.exists(embeddings_file):
                try:
                    with open(embeddings_file, 'r', encoding='utf-8') as f:
                        embeddings_data = json.load(f)
                    print(f"既存の埋め込みデータを読み込み: {len(embeddings_data)} 件")
                except (json.JSONDecodeError, UnicodeDecodeError) as e:
                    print(f"JSONファイルが破損しています: {e}")
                    print("破損したファイルを削除して新しく生成します...")
                    try:
                        os.remove(embeddings_file)
                        print("破損したファイルを削除しました")
                    except:
                        pass
                    embeddings_data = {}
            
            # 差分処理
            for image_file in image_files:
                file_stat = os.stat(image_file)
                current_mtime = file_stat.st_mtime

                # 埋め込みデータにファイルが存在しない場合、新しいファイルとして処理
                if image_file not in embeddings_data:
                    new_files.append(image_file)
                    continue
                
                # 既存のmtimeを取得し、2秒以上の差がある場合、またはmtimeが存在しない場合に再処理
                stored_mtime = embeddings_data[image_file].get('mtime')
                if stored_mtime is None or abs(current_mtime - stored_mtime) > 2:
                    new_files.append(image_file)
            
            if not new_files:
                print("処理が必要な新しいファイルはありません。")
                return
            
            # 最大件数制限
            if not paid_mode and max_files is not None and max_files > 0 and len(new_files) > max_files:
                new_files = new_files[:max_files]
            if paid_mode and max_files is not None and max_files > 0:
                if len(new_files) > max_files:
                    new_files = new_files[:max_files]
            
            print("-" * 50)
            
            target_count = len(new_files)
            batch_size = 1000
            total_batches = (len(new_files) + batch_size - 1) // batch_size
            
            for batch_idx in range(total_batches):
                if max_files is not None and max_files > 0 and processed_count >= max_files:
                    break
                    
                start_idx = batch_idx * batch_size
                end_idx = min(start_idx + batch_size, len(new_files))
                batch_files = new_files[start_idx:end_idx]

                if max_files is not None and max_files > 0:
                    remaining_count = max_files - processed_count
                    if remaining_count < len(batch_files):
                        batch_files = batch_files[:remaining_count]
                
                print("-" * 30)
                
                is_paid_mode = self.paid_plan or (max_files is not None and max_files > 0) or (max_files == 0)
                batch_results = self.analyze_images_batch(batch_files, unlimited_mode=is_paid_mode, batch_size=5, max_files=max_files)
                
                for i, (image_file, description) in enumerate(batch_results, 1):
                    if max_files is not None and max_files > 0 and processed_count >= max_files:
                        break
                        
                    print(f"処理中... {i}/{len(batch_results)}: {image_file}")
                    
                    # if "エラー:" in description or "API制限によりスキップ:" in description:
                    #     print(f"  スキップ: {description}")
                    #     continue
                    
                    embedding = self.get_text_embedding(description)
                    if embedding is None:
                        print(f"  埋め込み生成失敗: {image_file}")
                        continue
                    
                    image_creation_date = self._get_image_creation_date(image_file)

                    file_stat = os.stat(image_file)
                    embeddings_data[image_file] = {
                        'description': description,
                        'embedding': embedding,
                        'mtime': file_stat.st_mtime,
                        'size': file_stat.st_size,
                        'processed_at': datetime.now().isoformat(),
                        'image_creation_date': image_creation_date,
                        'device_used': self.device,
                        'batch': batch_idx + 1
                    }
                    processed_count += 1
                    print(f"  完了: 説明文 {len(description)} 文字、埋め込み次元 {len(embedding)} (処理済み: {processed_count}/{target_count})")

                skipped_count = sum(1 for _, desc in batch_results if "API制限によりスキップ:" in desc)
                if not is_paid_mode and skipped_count > 0:
                    print(f"  🛑 Gemini API制限に達しました。処理を停止します。")
                    print(f"  📊 現在の処理済み: {len(embeddings_data)} 件")
                    print(f"  🔄 明日までお待ちください")
                    break  # ループを抜けてfinallyブロックで保存処理を行う
                
                if max_files is not None and max_files > 0 and processed_count >= max_files:
                    break
        
        finally:
            if embeddings_data:
                try:
                    with open(embeddings_file, 'w', encoding='utf-8') as f:
                        json.dump(embeddings_data, f, ensure_ascii=False, indent=2)
                    print(f"\n埋め込みデータを最終保存しました。")
                except Exception as e:
                    print(f"\n最終データの保存中にエラーが発生しました: {e}")

            print(f"\n完了！ {processed_count} 件のファイルを処理しました。")
            print(f"総データ数: {len(embeddings_data)} 件")
            print(f"埋め込みファイル: {embeddings_file}")
            
            if new_files:
                original_requests = len(new_files)
                avg_batch_size = 6.5
                estimated_batch_requests = (len(new_files) + avg_batch_size - 1) // avg_batch_size
                reduction = original_requests - estimated_batch_requests
                if original_requests > 0:
                    reduction_rate = (reduction / original_requests) * 100
                    # print(f"\n🚀 バッチ処理の効果:")
                    # print(f"  個別処理の場合: {original_requests} 回のAPIリクエスト")
                    # print(f"  バッチ処理の場合: 約{estimated_batch_requests} 回のAPIリクエスト（推定）")
                    # print(f"  APIリクエスト削減: 約{reduction} 回 ({reduction_rate:.1f}%)")
                    # print(f"  処理効率向上: 約{original_requests/estimated_batch_requests:.1f}倍")


    
    def _expand_search_query(self, query):
        """検索クエリを拡張（同義語・関連語を追加）"""
        
        # クエリを小文字に変換して比較
        query_lower = query.lower()
        expanded_queries = [query]  # 元のクエリを含める
        
        # 1. 複合語を個別単語に分割して追加（日本語対応）
        individual_words = self._split_japanese_query(query)
        if len(individual_words) > 0:
            # 個別単語を追加（1個でも追加）
            expanded_queries.extend(individual_words)
        elif len(query.split()) > 1:
            # スペース区切りの複合語の場合
            space_words = query.split()
            expanded_queries.extend(space_words)
        
        # 2. 事前定義された同義語辞書による拡張
        found_in_dict = False
        
        # 4. 部分一致クエリの追加
        expanded_queries.extend(self._add_partial_matches(query))
        
        # 重複を除去
        expanded_queries = list(dict.fromkeys(expanded_queries))
        
        return expanded_queries
    
    
    def _split_japanese_query(self, query):
        """日本語クエリを適切に単語分割"""
        if JAPANESE_TOKENIZER_AVAILABLE and japanese_tokenizer:
            try:
                # janomeで形態素解析
                tokens = list(japanese_tokenizer.tokenize(query))  # ジェネレーターをリストに変換
                words = []
                
                for token in tokens:
                    # 品詞を取得
                    part_of_speech = token.part_of_speech.split(',')[0]
                    word = token.surface
                    
                    # 名詞、形容詞、動詞、副詞のみを抽出
                    if part_of_speech in ['名詞', '形容詞', '動詞', '副詞']:
                        if len(word) >= 1:  # 1文字以上の単語（「車」なども含める）
                            words.append(word)
                
                return words
            except Exception as e:
                print(f"日本語形態素解析エラー: {e}")
                # エラーの場合はスペース分割にフォールバック
                return query.split()
        else:
            # janomeが利用できない場合はスペース分割
            return query.split()
    
    def _add_partial_matches(self, query):
        """部分一致クエリの追加"""
        partial_matches = []
        
        # 日本語の単語分割を使用
        words = self._split_japanese_query(query)
        
        for word in words:
            # 3文字以上の単語の場合、部分一致も追加
            if len(word) >= 3:
                partial_matches.append(word)
                # 末尾の「ー」「ン」などを除去したバージョンも追加
                if word.endswith(('ー', 'ン', 'ン')):
                    partial_matches.append(word[:-1])
        
        return partial_matches
    
    def _calculate_flexible_match_score(self, description, expanded_queries):
        """より柔軟なテキストマッチングのスコアを計算"""
        score = 0
        for query in expanded_queries:
            query_lower = query.lower()
            if query_lower in description:
                score += 10 # 完全一致は高スコア
            elif any(word in description for word in query_lower.split()):
                score += 5  # 部分一致は中スコア
        return score

    def _hybrid_search(self, embeddings_data, original_query, expanded_queries, top_k, no_word=False, no_cos=False):
        """ハイブリッド検索（テキストマッチング + 埋め込み類似度）"""
        results = []
        
        if no_word and no_cos:
            print("エラー: -nowordと-nocosが両方指定されているため、検索を実行できません。")
            return []

        text_matches = []
        if not no_word:
            # 1. テキストマッチング（完全一致・部分一致）
            for image_file, data in embeddings_data.items():
                description = data['description'].lower()
                score = 0
                
                # 完全一致の重み付け
                for query in expanded_queries:
                    query_lower = query.lower()
                    if query_lower in description:
                        score += 10  # 完全一致は高スコア
                    elif any(word in description for word in query_lower.split()):
                        score += 5   # 部分一致は中スコア
                
                # 追加: より柔軟なマッチング
                score += self._calculate_flexible_match_score(description, expanded_queries)
                
                if score > 0:
                    text_matches.append({
                        'file': image_file,
                        'text_score': score,
                        'description': data['description'],
                        'processed_at': data.get('processed_at', '不明'),
                        'device_used': data.get('device_used', '不明')
                    })
            
            # テキストマッチング結果をスコアでソート
            text_matches.sort(key=lambda x: x['text_score'], reverse=True)

        embedding_matches = []
        if not no_cos:
            # 2. 埋め込み類似度検索
            try:
                # 元のクエリで埋め込み生成
                query_embedding = self.get_text_embedding(original_query)
                if query_embedding is not None:
                    for image_file, data in embeddings_data.items():
                        image_embedding = np.array(data['embedding']).reshape(1, -1)
                        query_embedding_array = np.array(query_embedding).reshape(1, -1)
                        
                        similarity = cosine_similarity(query_embedding_array, image_embedding)[0][0]
                        
                        embedding_matches.append({
                            'file': image_file,
                            'embedding_score': similarity,
                            'description': data['description'],
                            'processed_at': data.get('processed_at', '不明'),
                            'device_used': data.get('device_used', '不明')
                        })
                    
                    # 埋め込み類似度でソート
                    embedding_matches.sort(key=lambda x: x['embedding_score'], reverse=True)
            except Exception as e:
                print(f"埋め込み検索エラー: {e}")
        
        # 3. 結果の統合
        combined_results = {}
        
        # テキストマッチング結果を統合
        for i, match in enumerate(text_matches):
            file_key = match['file']
            combined_results[file_key] = {
                'file': file_key,
                'text_score': match['text_score'],
                'embedding_score': 0,
                'combined_score': match['text_score'] * (1.0 if no_cos else 0.7),  # テキストマッチングの重み
                'description': match['description'],
                'processed_at': match['processed_at'],
                'device_used': match['device_used'],
                'match_type': 'text'
            }
        
        # 埋め込み類似度結果を統合
        for i, match in enumerate(embedding_matches):
            file_key = match['file']
            embedding_score = match['embedding_score'] * 100  # スコアを0-100に正規化
            
            if file_key in combined_results:
                # 既存の結果と統合
                combined_results[file_key]['embedding_score'] = embedding_score
                combined_results[file_key]['combined_score'] += embedding_score * (0.0 if no_word else 0.3)  # 埋め込み類似度の重み
                combined_results[file_key]['match_type'] = 'hybrid'
            else:
                # 新しい結果として追加
                combined_results[file_key] = {
                    'file': file_key,
                    'text_score': 0,
                    'embedding_score': embedding_score,
                    'combined_score': embedding_score * (1.0 if no_word else 0.3),
                    'description': match['description'],
                    'processed_at': match['processed_at'],
                    'device_used': match['device_used'],
                    'match_type': 'embedding'
                }
        
        # 統合スコアでソート
        results = list(combined_results.values())
        results.sort(key=lambda x: x['combined_score'], reverse=True)
        
        # デバッグ情報を表示
        if not no_word:
            print(f"テキストマッチング結果: {len(text_matches)} 件")
        if not no_cos:
            print(f"埋め込み類似度結果: {len(embedding_matches)} 件")
        print(f"統合結果: {len(results)} 件")
             
        return results
    
    def _find_matched_queries(self, description, expanded_queries):
        """説明文でマッチしたクエリを特定"""
        matched = []
        description_lower = description.lower()
        
        for query in expanded_queries:
            query_lower = query.lower()
            if query_lower in description_lower:
                matched.append(query)
        
        return matched
    
    def search_with_embeddings(self, query, top_k=100, no_word=False, no_ex_word=False, no_cos=False, time_filter=None):
        """ハイブリッド検索（テキストマッチング + 埋め込み類似度）"""
        import os
        embeddings_file = "local_image_embeddings.json"
        
        if not os.path.exists(embeddings_file):
            print("埋め込みファイルが見つかりません。先に 'generate' を実行してください。")
            return
        
        try:
            with open(embeddings_file, 'r', encoding='utf-8') as f:
                embeddings_data = json.load(f)
        except (json.JSONDecodeError, UnicodeDecodeError) as e:
            print(f"JSONファイルが破損しています: {e}")
            print("破損したファイルを削除して新しく生成してください。")
            try:
                os.remove(embeddings_file)
                print("破損したファイルを削除しました")
            except:
                pass
            return
        
        if not embeddings_data:
            print("埋め込みデータがありません。")
            return
        
        # 日付フィルター処理
        if time_filter:
            filtered_data = {}
            start_date, end_date = time_filter
            for file, data in embeddings_data.items():
                try:
                    # image_creation_date のみを参照する
                    date_str = data.get('image_creation_date')
                    if date_str:
                        creation_date = datetime.fromisoformat(date_str)
                        if start_date <= creation_date <= end_date:
                            filtered_data[file] = data
                except (ValueError, TypeError):
                    continue # 日付が不正なデータは無視
            embeddings_data = filtered_data
            if not embeddings_data:
                print(f"指定された期間 {start_date.strftime('%Y-%m-%d')}～{end_date.strftime('%Y-%m-%d')} のデータが見つかりませんでした。")
                return

        print(f"検索対象: {len(embeddings_data)} 個の画像")
        print(f"検索クエリ: '{query}'")
        
        if no_ex_word:
            expanded_queries = [query]
            print("拡張クエリは無効です。")
        else:
            expanded_queries = self._expand_search_query(query)
            print(f"拡張クエリ: {expanded_queries}")

        search_results = self._hybrid_search(embeddings_data, query, expanded_queries, top_k, no_word, no_cos)
        
        if not search_results:
            print("検索結果が見つかりませんでした。")
            return
        
        if top_k == 0:
            top_results = search_results
        else:
            top_results = search_results[:top_k]

        import platform
        if platform.system() == "Darwin":
            self._serve_html_on_mac(top_results, query)
        else:
            self._generate_and_open_html(top_results, query)

    def _generate_html_content(self, results, query, base_url=""):
        html = [
            '<html><head><meta charset="utf-8"><title>ハイブリッド検索結果</title></head><body>',
            f'<h2>ハイブリッド検索結果: Top {len(results)} 件</h2>',
            f'<p><strong>検索クエリ:</strong> {query}</p>',
            '<div style="display:flex; flex-wrap:wrap;">'
        ]
        
        for i, result in enumerate(results, 1):
            text_score = result.get('text_score', 0)
            embedding_score = result.get('embedding_score', 0)
            combined_score = result.get('combined_score', 0)
            match_type = result.get('match_type', 'unknown')
            
            try:
                from PIL import Image
                import base64
                from io import BytesIO
                import os
                
                img = Image.open(result['file'])
                w, h = img.size
                new_w = 300
                new_h = int(h * (new_w / w))
                img = img.resize((new_w, new_h))
                
                file_ext = os.path.splitext(result['file'])[1].lower()
                buf = BytesIO()
                
                format_type = 'JPEG'
                mime_type = 'image/jpeg'
                if file_ext in ['.png']:
                    format_type, mime_type = 'PNG', 'image/png'
                elif file_ext in ['.gif']:
                    format_type, mime_type = 'GIF', 'image/gif'
                elif file_ext in ['.bmp']:
                    format_type, mime_type = 'BMP', 'image/bmp'
                elif file_ext in ['.webp']:
                    format_type, mime_type = 'WEBP', 'image/webp'
                
                img.save(buf, format=format_type)
                img_b64 = base64.b64encode(buf.getvalue()).decode('utf-8')
                
                absolute_path = os.path.abspath(result["file"])
                
                if base_url: # Mac (http server)
                    # Webサーバーのルートからの相対パスに変換
                    file_url = os.path.relpath(absolute_path, os.getcwd()).replace("\\", "/")
                    onclick_action = f"window.open('{base_url}/{file_url}', '_blank')"
                else: # Windows/Linux (file://)
                    file_url = absolute_path.replace("\\", "/")
                    onclick_action = f"window.open('file:///{file_url}', '_blank')"

                img_tag = f'<img src="data:{mime_type};base64,{img_b64}" width="{new_w}" style="display:block; margin-bottom:4px; cursor:pointer;" onclick="{onclick_action}" title="クリックでオリジナル画像を新しいタブで開く">'
            except Exception as e:
                img_tag = f'<div style="width:300px; height:200px; background:#ccc; display:flex; align-items:center; justify-content:center;">画像表示エラー: {e}</div>'
            
            html.append(f'''
            <div style="margin:10px; text-align:center; width:350px; border:1px solid #ddd; padding:10px; border-radius:5px;">
                {img_tag}
                <div style="font-size:12px; font-weight:bold; cursor:pointer;" onclick="{onclick_action}" title="クリックでオリジナル画像を新しいタブで開く">{i}. {result["file"]}</div>
                <div style="font-size:11px; color:#666; margin:5px 0;">
                    テキストスコア: {text_score:.1f} | 
                    埋め込みスコア: {embedding_score:.1f} | 
                    統合スコア: {combined_score:.1f}
                </div>
                <span style="font-size:10px; padding:2px 6px; border-radius:3px; color:white; background-color:{'#4CAF50' if match_type == 'text' else '#2196F3' if match_type == 'embedding' else '#FF9800'}">{match_type}</span>
                <div style="font-size:12px; word-break:break-all; margin-top:5px;">{result["description"][:80]}...</div>
            </div>
            ''')
        html.append('</div></body></html>')
        return '\n'.join(html)

    def _generate_and_open_html(self, results, query):
        import tempfile
        import webbrowser
        
        html_str = self._generate_html_content(results, query)
        with tempfile.NamedTemporaryFile('w', delete=False, suffix='.html', encoding='utf-8') as f:
            f.write(html_str)
            temp_html_path = f.name
        print(f"\nハイブリッド検索結果をサムネイルで表示します: {temp_html_path}")
        webbrowser.open(f'file://{temp_html_path}')

    def _serve_html_on_mac(self, results, query):
        import http.server
        import socketserver
        import webbrowser
        import threading
        import os
        import time

        PORT = 8000
        # ポートが使用中か確認し、空いているポートを探す
        while True:
            try:
                with socketserver.TCPServer(("", PORT), http.server.SimpleHTTPRequestHandler) as httpd:
                    break
            except OSError:
                print(f"ポート {PORT} は使用中です。別のポートを試します。")
                PORT += 1
        
        base_url = f"http://localhost:{PORT}"
        html_content = self._generate_html_content(results, query, base_url)
        
        # HTMLをカレントディレクトリに保存
        html_file_path = "search_results.html"
        with open(html_file_path, "w", encoding="utf-8") as f:
            f.write(html_content)

        Handler = http.server.SimpleHTTPRequestHandler
        httpd = socketserver.TCPServer(("", PORT), Handler)

        print(f"Macでローカルサーバーを起動します: {base_url}")
        
        # サーバーを別スレッドで起動
        server_thread = threading.Thread(target=httpd.serve_forever)
        server_thread.daemon = True
        server_thread.start()
        
        # 少し待ってからブラウザを開く
        time.sleep(1)
        webbrowser.open(f"{base_url}/{html_file_path}")
        
        print("ブラウザで結果を表示しました。スクリプトはサーバーが停止するまで待機します。")
        print("終了するには Ctrl+C を押してください。")
        
        try:
            # ユーザーが終了するまで待機
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            print("\nサーバーを停止しています...")
            httpd.shutdown()
            httpd.server_close()
            print("サーバーを停止しました。")
    
    def show_embedding_stats(self):
        """埋め込みデータの統計情報を表示"""
        embeddings_file = "local_image_embeddings.json"
        
        if not os.path.exists(embeddings_file):
            print("埋め込みファイルが見つかりません。")
            return
        
        try:
            with open(embeddings_file, 'r', encoding='utf-8') as f:
                embeddings_data = json.load(f)
        except (json.JSONDecodeError, UnicodeDecodeError) as e:
            print(f"JSONファイルが破損しています: {e}")
            return
        
        total_files = len(embeddings_data)
        if total_files == 0:
            print("埋め込みデータがありません。")
            return
        
        total_size = sum(data['size'] for data in embeddings_data.values())
        device_counts = {}
        batch_counts = {}
        
        for data in embeddings_data.values():
            device = data.get('device_used', '不明')
            device_counts[device] = device_counts.get(device, 0) + 1
            
            batch = data.get('batch', '不明')
            batch_counts[batch] = batch_counts.get(batch, 0) + 1
        
        print(f"ローカルAI埋め込み統計情報:")
        print(f"  総ファイル数: {total_files} 件")
        print(f"  総ファイルサイズ: {total_size / (1024*1024):.1f} MB")
        print(f"  平均ファイルサイズ: {total_size / total_files / 1024:.1f} KB")
        # print(f"  使用デバイス別:")
        # for device, count in device_counts.items():
        #     print(f"    {device}: {count} 件")
        # print(f"  バッチ処理別:")
        # for batch, count in batch_counts.items():
        #     print(f"    バッチ {batch}: {count} 件")
        
        if embeddings_data:
            sample_embedding = list(embeddings_data.values())[0]['embedding']
            embedding_size = len(sample_embedding) * 8 / 1024  # float64のサイズ
            print(f"  埋め込み次元数: {len(sample_embedding)}")
            print(f"  1件あたり埋め込みサイズ: {embedding_size:.1f} KB")
            print(f"  総埋め込みサイズ: {total_files * embedding_size:.1f} KB")

    def _get_image_creation_date(self, image_path):
        """画像の作成日時を取得（EXIF > ファイル名 > ファイル更新日時）"""
        # 1. EXIFから撮影日時を取得
        try:
            img = Image.open(image_path)
            exif_data = img._getexif()
            if exif_data:
                # 'DateTimeOriginal' (36867) を探す
                for tag, value in exif_data.items():
                    if tag in ExifTags.TAGS and ExifTags.TAGS[tag] == 'DateTimeOriginal':
                        # 'YYYY:MM:DD HH:MM:SS' 形式をパース
                        return datetime.strptime(value, '%Y:%m:%d %H:%M:%S').isoformat()
        except Exception:
            pass # EXIF読み取りエラーは無視

        # 2. ファイル名から日付らしき文字列を抽出
        import re
        filename = os.path.basename(image_path)
        # YYYYMMDD, YYYY-MM-DD, YYYY_MM_DD 形式などを探す
        match = re.search(r'(\d{4})[-_]?(\d{2})[-_]?(\d{2})', filename)
        if match:
            try:
                year, month, day = map(int, match.groups())
                return datetime(year, month, day).isoformat()
            except ValueError:
                pass # 不正な日付は無視

        # 3. ファイルの最終更新日時を使用
        try:
            mtime = os.path.getmtime(image_path)
            return datetime.fromtimestamp(mtime).isoformat()
        except Exception:
            return None

    def _parse_time_range(self, time_str):
        """
        時間文字列 (YYYY, YYYYMM, YYYYMMDD またはその範囲) を解釈し、
        (start_datetime, end_datetime) のタプルを返す。
        """
        def _parse_single_time(s):
            s = s.strip()
            if len(s) == 4: # YYYY
                year = int(s)
                start_date = datetime(year, 1, 1)
                end_date = datetime(year, 12, 31, 23, 59, 59, 999999)
                return start_date, end_date
            elif len(s) == 6: # YYYYMM
                year = int(s[:4])
                month = int(s[4:])
                _, last_day = calendar.monthrange(year, month)
                start_date = datetime(year, month, 1)
                end_date = datetime(year, month, last_day, 23, 59, 59, 999999)
                return start_date, end_date
            elif len(s) == 8: # YYYYMMDD
                year = int(s[:4])
                month = int(s[4:6])
                day = int(s[6:])
                start_date = datetime(year, month, day)
                end_date = datetime(year, month, day, 23, 59, 59, 999999)
                return start_date, end_date
            else:
                raise ValueError(f"無効な日付形式です: {s}。YYYY, YYYYMM, YYYYMMDD のいずれかで指定してください。")

        if '-' in time_str:
            start_str, end_str = time_str.split('-', 1)
            start_date, _ = _parse_single_time(start_str)
            _, end_date = _parse_single_time(end_str)
            return start_date, end_date
        else:
            return _parse_single_time(time_str)

def main():
    import sys
    
    # クレジット表示
    print("=" * 60)
    print("画像さがす君（ローカルAI画像検索システム = Gemini + ローカル処理) ver.1.0")
    print("(c) 2025 / Satoshi Endo @hortense667")
    print("=" * 60)
    
    parser = argparse.ArgumentParser(description="画像さがす君（ローカルAI画像検索システム）")
    subparsers = parser.add_subparsers(dest='command', required=True)

    # generate command
    generate_parser = subparsers.add_parser('generate', help='埋め込みを生成')
    generate_parser.add_argument('max_files', type=int, nargs='?', default=None, help='最大処理ファイル数（0で無制限）')

    # search command
    search_parser = subparsers.add_parser('search', help='画像を検索')
    search_parser.add_argument('query', type=str, help='検索クエリ')
    search_parser.add_argument('-noword', action='store_true', help='テキストマッチを行わない')
    search_parser.add_argument('-noexword', action='store_true', help='拡張クエリを行わない')
    search_parser.add_argument('-nocos', action='store_true', help='コサイン類似度評価を行わない')
    search_parser.add_argument('-n', type=int, default=100, help='結果の件数（0で無制限）')
    search_parser.add_argument('-t', type=str, help='時間でフィルタ (YYYY, YYYYMM, YYYYMMDD or 範囲)')

    # stats command
    stats_parser = subparsers.add_parser('stats', help='統計情報を表示')

    args = parser.parse_args()
    
    # AIシステムを初期化
    searcher = LocalAIImageSearcher()
    
    if args.command == "generate":
        if args.max_files is None:
            print("通常モード: 1日1000件のAPI無料枠内を前提として処理します")
        elif args.max_files == 0:
            print("完全無制限モード: 1日1000件のAPI無料枠を超えた場合は課金されます")
        else:
            print(f"有料契約モード: {args.max_files} 件を処理します（1日1000件のAPI無料枠を超えた場合は課金されます）")
        searcher.generate_embeddings(args.max_files)

    elif args.command == "search":
        time_filter = None
        if args.t:
            try:
                time_filter = searcher._parse_time_range(args.t)
            except ValueError as e:
                print(f"エラー: {e}")
                sys.exit(1)

        searcher.search_with_embeddings(
            args.query, 
            top_k=args.n, 
            no_word=args.noword, 
            no_ex_word=args.noexword, 
            no_cos=args.nocos,
            time_filter=time_filter
        )

    elif args.command == "stats":
        searcher.show_embedding_stats()
    else:
        print(f"不明なコマンド: {args.command}")
        parser.print_help()

if __name__ == "__main__":
    main() 