# 画像さがす君（ローカルAI画像検索システム） ver.1.0.1
- (c)2025 / Satoshi Endo @hortense667

**Gemini + ローカル処理版**

> Gemini 2.5 Flash-Liteを使用した超高速な画像キャプション生成と、ローカルでのベクトル処理を組み合わせた画像検索システムです。

---

## 新しい使い方：GUIランチャー
このツールは、新しく追加されたGUIランチャーを使うのが最も簡単でオススメです。
コマンドラインを開く必要はありません。

### GUIランチャーの使い方 (Windows / Mac)

**⚠️ 重要**: GUIアプリケーションを使用する前に、必ず環境変数`GEMINI_API_KEY`を設定してください。設定方法は下記の「環境変数の設定」セクションを参照してください。

1.  配布されたzipファイルを解凍します。
2.  中にあるGUIランチャーを起動します。
    *   **Windowsの場合**: `Gazosk.exe` をダブルクリックします。
    *   **Macの場合**: `Gazosk` を「アプリケーション」フォルダにコピーし、そこからダブルクリックして起動します。
3.  **Generateタブ**:
    *   「処理対象パス」に、画像がたくさん入っているフォルダを指定します。
    *   「生成 開始」ボタンを押すと、画像の特徴分析（埋め込み生成）が始まります。ログが表示され、進捗を確認できます。
4.  **Searchタブ**:
    *   「処理対象パス」が正しいことを確認します。
    *   「クエリ」に探したい言葉（例：`青い空と白い雲`）を入力し、「検索 実行」ボタンを押します。
    *   検索が完了すると、自動的にWebブラウザが開き、結果がサムネイルで表示されます。

**トラブルシューティング**:
- GUIアプリケーションで「APIキーが設定されていません」というエラーが表示される場合は、環境変数の設定を確認してください。
- Macの場合、LaunchAgentsを使用した設定方法が最も確実です。
- 設定後は、GUIアプリケーションを再起動してください。

---

## 特徴

### 高速処理
- **Gemini 2.5 Flash-Lite**: 超高速な画像キャプション生成（2-4倍高速）
- **動的バッチ処理**: ネットワークの状態に合わせて、一度に5～10枚の画像を自動でまとめて処理し、APIリクエスト数を大幅に削減します。
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
- **Gemini APIキー**: 環境変数`GEMINI_API_KEY`として設定が必要

### 推奨要件
- **Windows**: Ryzen AI 9 PRO搭載PC（非搭載でも納得できる高速性です）
- **Mac**: M1/M2搭載Mac

### 環境変数の設定

#### Macでの設定方法

**方法1: ターミナルで一時的に設定（推奨）**
```bash
# ターミナルを開いて以下を実行
export GEMINI_API_KEY=your_api_key_here

# 設定を確認
echo $GEMINI_API_KEY
```

**方法2: 永続的に設定**
```bash
# ~/.zshrcファイルに追加（zshを使用している場合）
echo 'export GEMINI_API_KEY=your_api_key_here' >> ~/.zshrc
source ~/.zshrc

# または ~/.bash_profileファイルに追加（bashを使用している場合）
echo 'export GEMINI_API_KEY=your_api_key_here' >> ~/.bash_profile
source ~/.bash_profile
```

**方法3: GUIアプリケーション用の設定**
MacのGUIアプリケーション（Gazosk_Mac）が環境変数を認識するようにするには、以下のいずれかの方法を使用してください：

1. **LaunchAgentsを使用する方法（推奨）**:
   ```bash
   # LaunchAgentsディレクトリを作成
   mkdir -p ~/Library/LaunchAgents
   
   # plistファイルを作成
   cat > ~/Library/LaunchAgents/com.user.geminiapi.plist << EOF
   <?xml version="1.0" encoding="UTF-8"?>
   <!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
   <plist version="1.0">
   <dict>
       <key>Label</key>
       <string>com.user.geminiapi</string>
       <key>ProgramArguments</key>
       <array>
           <string>/bin/launchctl</string>
           <string>setenv</string>
           <string>GEMINI_API_KEY</string>
           <string>your_api_key_here</string>
       </array>
       <key>RunAtLoad</key>
       <true/>
   </dict>
   </plist>
   EOF
   
   # LaunchAgentを読み込み
   launchctl load ~/Library/LaunchAgents/com.user.geminiapi.plist
   ```

2. **アプリケーションをターミナルから起動する方法**:
   ```bash
   # 環境変数を設定してからアプリケーションを起動
   export GEMINI_API_KEY=your_api_key_here
   open /Applications/Gazosk.app
   ```

#### Windowsでの設定方法
```cmd
# コマンドプロンプトで一時的に設定
set GEMINI_API_KEY=your_api_key_here

# 永続的に設定（システム環境変数）
setx GEMINI_API_KEY "your_api_key_here"
```

#### Linuxでの設定方法
```bash
# 一時的に設定
export GEMINI_API_KEY=your_api_key_here

# 永続的に設定
echo 'export GEMINI_API_KEY=your_api_key_here' >> ~/.bashrc
source ~/.bashrc
```

### Gemini APIキーの取得
1. [Google AI Studio](https://aistudio.google.com/)にアクセス
2. アカウントにログイン（Googleアカウントが必要）
3. 「Get API key」をクリック
4. 「Create API key」を選択
5. 生成されたAPIキーをコピー
6. 上記の方法で環境変数に設定

### 環境変数設定の確認
設定が正しく行われているか確認するには：
```bash
# Mac/Linux
echo $GEMINI_API_KEY

# Windows
echo %GEMINI_API_KEY%
```

**注意**: GUIアプリケーション（Gazosk_Mac）を使用する場合は、アプリケーションを再起動する必要があります。

## 🔧 コマンドラインでの使用方法（上級者向け）
GUIランチャーを使わずに、従来通りコマンドラインから直接操作することも可能です。バッチ処理などを組みたい場合に便利です。

### 基本的な使用方法

#### 画像の埋め込み生成
- プログラム実行フォルダーより下のフォルダー内にある画像ファイルを処理対象とします。
- Windows版は「LocalAIImageSearch」、Mac版は「LocalAIImageSearch_Mac」です。Mac版は、それに準じて読み替えてください。
- Dropboxなどを使用している場合は、画像ファイルや関係するjsonファイル（local_image_embeddings.json、およびgemini_daily_usage.json）は、オフラインで実態が存在することが条件となります。

```bash
# 通常モード（1日1000件のAPI無料枠内＝1万枚弱の処理が可能）
LocalAIImageSearch generate

# 指定件数まで（1日1000件のAPI無料枠内を超えると有料）
LocalAIImageSearch generate 100

# 完全無制限モード（有料契約を想定）
LocalAIImageSearch generate 0
```

#### 画像検索

```bash
# 基本的な検索
LocalAIImageSearch search "カツカレー"

# 詳細な検索オプション
LocalAIImageSearch search -n 50 -t 2023 "カツカレー"
```
*   **オプション (順不同)**
    *   `-noword`: テキストマッチを行わない
    *   `-noexword`: キーワードを分解せず、指定した言葉通りに検索（固有名詞などに有効）
    *   `-nocos`: コサイン類似度評価を行わない
    *   `-n <件数>`: 結果の件数を指定 (例: -n 50, 0で無制限)
    *   `-t <時間|時間範囲>`: 画像の作成日でフィルタ (例: -t 2023, -t 202301, -t 20230115, -t 2020-2024, -t 202301-202303)


**日付フィルタリングについて**
- `-t`オプションは、以下の優先順位で取得された画像の日付情報に基づいてフィルタリングします。
    1. **EXIF情報**: 画像ファイルに記録された撮影日時
    2. **ファイル名**: `20230101`, `2023-01-01`のような日付パターン
    3. **ファイルの更新日時**: 上記2つがない場合の最終手段

#### 統計情報表示
```bash
LocalAIImageSearch stats
```

#### 既存データの移行（日付情報追加）
過去に生成した`local_image_embeddings.json`に画像の作成日を追加するには、`migrate_creation_date.exe`を実行します。

## ファイル構成
配布されるファイルは以下の通りです。

### 配布パッケージに含まれる主なファイル (ユーザー向け)
*   `Gazosk.exe` または `Gazosk.app`: **GUIランチャーです。通常はこのファイルを使ってください。**
*   `LocalAIImageSearch.exe`: (Windows用) コマンドライン版の本体です。
*   `LocalAIImageSearch_Mac`: (Mac用) コマンドライン版の本体です。
*   `migrate_creation_date.exe`:（Windows用）過去のデータに日付情報を追加するためのツールです。
*   `migrate_creation_date_Mac`:（Mac用）過去のデータに日付情報を追加するためのツールです。  
*   `clip_image_encoder.onnx`: 画像を分析するためのAIモデルです。NPU (Ryzen AIなど) の性能を最大限引き出すために使用します (このファイルがなくてもCPUで動作します)。
*   `LISENSE`: ライセンス情報です。
*   `README.md`: このファイルです。

### 実行すると自動で作られるファイル
*   `local_image_embeddings.json`: 画像の特徴を記録したデータベースです。
*   `gemini_daily_usage.json`: Gemini APIの使用量を記録するファイルです。
*   `gui_config.ini`: GUIランチャーの設定を保存するファイルです。

### ソースコード・開発者向けファイル
*   `local_image_super_search2.py`: 本体プログラムのPythonスクリプトです。
*   `Gazosk.py`: GUIランチャーのPythonスクリプトです。
*   `gazosk.ico` / `gazosk.icns`: アプリケーションのアイコンファイルです。
*   `requirements.txt`: 開発に必要なライブラリの一覧です。
*   `*.spec`: PyInstallerで実行ファイルを作成するための設定ファイルです。
*   `clip_image_encoder_export.py`: `clip_image_encoder.onnx`を生成するためのスクリプトです。
*   `migrate_add_creation_date.py`: データ移行ツールのPythonスクリプトです。

## 作者

**Satoshi Endo @hortense667**

- GitHub: [@hortense667](https://github.com/hortense667)
- X: [@hortense667](https://x.com/hortense667)

## 謝辞

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
