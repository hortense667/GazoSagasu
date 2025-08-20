# -*- mode: python ; coding: utf-8 -*-

import os
import customtkinter

# customtkinterのテーマファイルなど、必要なアセットのパスを自動で探します
customtkinter_path = os.path.dirname(customtkinter.__file__)
assets_path = os.path.join(customtkinter_path, "assets")

block_cipher = None

a = Analysis(
    ['Gazosk.py'],
    pathex=[],
    binaries=[],
    datas=[(assets_path, 'customtkinter/assets'), ('gazosk.ico', '.')], # customtkinterのアセットとアイコンを同梱
    hiddenimports=[],
    hookspath=[],
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
    strip=False,
    upx=True,
    upx_exclude=[],
    name='Gazosk',
    debug=False,
    bootloader_ignore_signals=False,
    runtime_tmpdir=None,
    console=False,
    disable_windowed_traceback=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon='gazosk.ico'
)