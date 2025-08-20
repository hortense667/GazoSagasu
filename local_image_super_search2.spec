# -*- mode: python ; coding: utf-8 -*-
"""
PyInstaller SPEC  (2025‑07‑27 改訂版)
local_image_super_search2.py を onedir 形式でビルド。
NPU対応: Ryzen AI 9 PRO (DirectML) 対応版
"""

import os                     # ★ 追加
from pathlib import Path
from PyInstaller.utils.hooks import (
    collect_submodules,
    collect_data_files,
    collect_dynamic_libs,
)

# --------------------------------------------------------------------
# 1. プロジェクトルートと必須ファイル
# --------------------------------------------------------------------
# ★ ここを __file__ ではなくカレントディレクトリ基準に変更
proj_dir = Path(os.getcwd()).resolve()

onnx_model = proj_dir / "clip_image_encoder.onnx"   # 必須モデル
if not onnx_model.exists():
    raise FileNotFoundError(
        f"clip_image_encoder.onnx が {proj_dir} に見つかりません"
    )

# janomeの辞書ファイルパスを取得
try:
    import janome
    janome_path = Path(janome.__file__).parent
    print(f"janomeパス: {janome_path}")
    
    # janomeの辞書ファイルを確認
    sysdic_path = janome_path / "sysdic"
    if sysdic_path.exists():
        print(f"janome辞書パス: {sysdic_path}")
        # 辞書ファイルの一覧を確認
        dict_files = list(sysdic_path.glob("*.py"))
        print(f"辞書ファイル数: {len(dict_files)}")
    else:
        print("janome辞書ディレクトリが見つかりません")
        
except ImportError:
    janome_path = None
    print("janomeが見つかりません")

# --------------------------------------------------------------------
# 2. 収集対象 (NPU対応版)
# --------------------------------------------------------------------
# NPU対応: Ryzen AI 9 PROのDirectMLプロバイダーをサポート
# - onnxruntime-directml 1.22.0以上が必要
# - DirectMLプロバイダーが自動的に含まれる
# - Windows環境でのみNPUが有効
hiddenimports = (
    collect_submodules("sklearn") +
    collect_submodules("google") +
    collect_submodules("google.generativeai") +
    collect_submodules("transformers") +
    collect_submodules("sentence_transformers") +
    collect_submodules("PIL") +
    ['PIL.ExifTags'] + # ExifTagsを明示的に追加
    collect_submodules("onnxruntime") +
    # NPU対応: DirectMLプロバイダー関連
    collect_submodules("onnxruntime.capi") +
    # 日本語形態素解析: janome
    collect_submodules("janome") +
    # janomeの具体的なモジュールを明示的に追加
    [
        "janome.tokenizer",
        "janome.charfilter",
        "janome.tokenfilter", 
        "janome.analyzer",
        "janome.sysdic",
        "janome.sysdic.entries_compact0",
        "janome.sysdic.entries_compact1",
        "janome.sysdic.entries_compact2",
        "janome.sysdic.entries_compact3",
        "janome.sysdic.entries_compact4",
        "janome.sysdic.entries_compact5",
        "janome.sysdic.entries_compact6",
        "janome.sysdic.entries_compact7",
        "janome.sysdic.entries_compact8",
        "janome.sysdic.entries_compact9",
        "janome.sysdic.entries_compact10",
        "janome.sysdic.entries_compact11",
        "janome.sysdic.entries_compact12",
        "janome.sysdic.entries_compact13",
        "janome.sysdic.entries_compact14",
        "janome.sysdic.entries_compact15"
    ]
)

# janomeの辞書ファイルを確実に含める
janome_data_files = []
if janome_path:
    sysdic_path = janome_path / "sysdic"
    if sysdic_path.exists():
        # 辞書ファイルをすべて含める
        for dict_file in sysdic_path.glob("*.py"):
            janome_data_files.append((str(dict_file), "janome/sysdic/"))
        print(f"janome辞書ファイル {len(janome_data_files)} 個を追加")
    else:
        print("janome辞書ディレクトリが見つかりません")

datas = (
    [(str(onnx_model), ".")] +                    # ONNX モデル
    collect_data_files("PIL") +                   # 画像プラグイン
    collect_data_files("sklearn") +
    collect_data_files("sentence_transformers") +
    collect_data_files("janome") +                # 日本語形態素解析辞書
    # janomeの辞書ファイルを明示的に追加
    collect_data_files("janome.sysdic") +         # システム辞書
    # janomeの辞書ファイルを直接指定
    janome_data_files
)

binaries = collect_dynamic_libs("onnxruntime")    # DLL / dylib (NPU対応)

# --------------------------------------------------------------------
# 3. PyInstaller ビルド定義
# --------------------------------------------------------------------
block_cipher = None

a = Analysis(
    ["local_image_super_search2.py"],
    pathex=[str(proj_dir)],                       # ★ proj_dir 使用
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.zipfiles,
    a.datas,
    [],
    name="LocalAIImageSearch",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=True,         # GUI 化したい場合は False
)

# onedir ビルド用の設定（ディレクトリ形式で配布）

# --------------------------------------------------------------------
# NPU対応ビルド時の注意事項
# --------------------------------------------------------------------
# 1. ビルド前に onnxruntime-directml がインストールされていることを確認
#    pip install onnxruntime-directml>=1.22.0
# 
# 2. ビルド前に janome がインストールされていることを確認
#    pip install janome
# 
# 3. ビルド後の動作確認
#    - Windows環境でNPUが認識されることを確認
#    - "使用デバイス: npu" と表示されることを確認
#    - 日本語形態素解析が正常に動作することを確認
# 
# 4. 配布時の注意
#    - LocalAIImageSearch ディレクトリ全体を配布
#    - clip_image_encoder.onnx ファイルが必須
#    - DirectML対応のWindows環境が必要
#    - Ryzen AI 9 PRO搭載PCで最適な性能を発揮
#    - janome辞書ファイルが含まれていることを確認
