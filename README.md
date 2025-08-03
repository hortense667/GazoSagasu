# 画像さがす君（ローカルAI画像検索システム）
- (c)2025 / Satoshi Endo @hortense667

**Gemini + ローカル処理版**

> Gemini 2.5 Flash-Liteを使用した超高速な画像キャプション生成と、ローカルでのベクトル処理を組み合わせた画像検索システムです。

## 🚀 特徴

### 高速処理
- **Gemini 2.5 Flash-Lite**: 超高速な画像キャプション生成（2-4倍高速）
- **バッチ処理**: 1回で5枚の画像を同時処理（APIリクエスト数80%削減）
- **動的バッチサイズ**: 3-10枚の範囲で自動調整
- **差分更新**: 変更されたファイルのみを再処理

### マルチプラットフォーム対応
- **Windows**: Ryzen AI 9 PROのNPUを活用
- **Mac**: M1/M2のMetal GPUを活用
- **CPU**: 汎用CPUでも高速動作

### ローカル処理
- **完全ローカル動作**: ベクトル処理部分はローカルで実行
- **オフライン対応**: 埋め込み生成後はインターネット不要

### 日本語対応
- **日本語キャプション**: Geminiによる高品質な日本語説明
- **日本語検索**: 日本語クエリでの自然な検索
- **形態素解析**: janomeによる日本語単語分割

## 📋 システム要件

### 必須要件
- インターネット接続（画像処理時＝Gemini API）
- 最低4GB RAM
- 1GB以上の空き容量

### 推奨要件
- **Windows**: Ryzen AI 9 PRO搭載PC（非搭載でも納得できる高速性です）
- **Mac**: M1/M2搭載Mac

### 環境変数の設定
```bash
# Windows
set GEMINI_API_KEY=your_api_key_here

# Mac/Linux
export GEMINI_API_KEY=your_api_key_here
```

### Gemini APIキーの取得
1. [Google AI Studio](https://aistudio.google.com/)にアクセス
2. APIキーを生成
3. 環境変数に設定

### 実行形式のダウンロード
- 実行形式は本リポジトリのリリースとしてアップロードされています。
- Windows用、Mac用、それぞれ適切なものをダウンロードしてください。
- https://github.com/hortense667/GazoSagasu/releases/tag/GazoSagazu

## 🚀 使用方法

### 基本的な使用方法

#### 画像の埋め込み生成
- プログラム実行フォルダーより下のフォルダー内にある画像ファイルを処理対象とします。
- Windows版は「LocalAIImageSearch」、Mac版は「LocalAIImageSearch_Mac」です。Mac版は、それに準じて読み替えてください。
- Dropboxなどを使用している場合は、画像ファイルや関係するjsonファイル（local_image_embeddings.json、およびgemini_daily_usage.json）は、オフラインで実態が存在することが条件となります。
- Mac版の場合はセキュリティの関係でローカルWebサーバーを立てて検索結果のオリジナル画像を表示しています。スクリプトはその間動作しているので終了したい場合はCtrl-Cを押してください。

```bash
# 通常モード（1日1000件のAPI無料枠内＝1万枚弱の処理が可能）
LocalAIImageSearch generate

# 指定件数まで（1日1000件のAPI無料枠内を超えると有料）
LocalAIImageSearch generate 100

# 完全無制限モード（有料契約を想定）
LocalAIImageSearch generate 0
```

#### 画像検索
- 画像の埋め込み生成と同じく、プログラム実行フォルダーより下のフォルダー内にある画像ファイルを処理対象とします。
- Windows版は「LocalAIImageSearch」、Mac版は「LocalAIImageSearch_Mac」です。Mac版は、それに準じて読み替えてください。
- 結果はWeb画面を開いてサムネイルの形で表示されます。
- サムネイルかファイル名をクリックするとそのファイルが拡大表示されるので必要に応じてダウンロードなど行うことができます。
```bash
# 基本的な検索
LocalAIImageSearch search "カツカレー"

# 件数指定での検索（標準では100件出力されます）
LocalAIImageSearch search "風景" 300

# 使用方法の表示
LocalAIImageSearch 
```

#### 統計情報表示
```bash
LocalAIImageSearch stats
```

## 🔧 技術仕様

### アーキテクチャ

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Gemini API    │    │   ローカル処理   │    │   検索エンジン   │
│                 │    │                 │    │                 │
│ • 画像キャプション  │──▶│ • テキスト埋め込み │──▶│ • ハイブリッド検索 │
│ • バッチ処理      │    │ • ベクトル計算   │    │ • 類似度計算     │
│ • 高速化        │    │ • NPU/GPU活用   │    │ • 結果表示       │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

## 📁 ファイル構成

```
local-ai-image-search/
├── README.md                       # このファイル
├── LocalAIImageSearch              # 実行ファイル
├── LocalAIImageSearch_Mac          # 実行ファイル（Mac版）
├── LICENSE                         # ライセンス
├── local_image_embeddings.json     # 埋め込みデータ（自動生成）
├── gemini_daily_usage.json         # 使用量データ（自動生成）
└── images/                         # 画像フォルダ（例）
    ├── sample1.jpg
    ├── sample2.png
    └── ...
```

## 👨‍💻 作者

**Satoshi Endo @hortense667**

- GitHub: [@hortense667](https://github.com/hortense667)
- X: [@hortense667](https://x.com/hortense667)

## 🙏 謝辞

- **Google**: Gemini APIの提供
- **Hugging Face**: Transformersライブラリ
- **PyTorch**: 機械学習フレームワーク
- **ONNX Runtime**: 高速推論エンジン
- **janome**: 日本語形態素解析エンジン

## 免責事項
1. このソフトウェアの利用によって生じる損害について、一切の責任を負うものではありません。
2. このソフトウェアの使用方法などに関する質問に答えることや動作内容についてサポートを負いません。

---

**画像さがす君** - あなたの画像を賢く検索するAIアシスタント

*Made with ❤️ by Satoshi Endo* 

