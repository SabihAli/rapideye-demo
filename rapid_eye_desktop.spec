# -*- mode: python ; coding: utf-8 -*-
"""
PyInstaller Specification File (`rapid_eye_desktop.spec`)
Used to freeze the RapidEye Desktop application into a self-contained folder (`dist/RapidEye/`).

Usage:
    pyinstaller rapid_eye_desktop.spec --clean
"""

import os
from pathlib import Path

block_cipher = None

# Base directories to include
project_root = Path(__file__).resolve().parent

added_files = [
    # Static React Build
    (str(project_root / "web" / "dist"), "web/dist"),
    # Configuration
    (str(project_root / ".env"), "."),
    # Assets directory (bundles all test videos inside assets/ so client receives them on disk)
    (str(project_root / "assets"), "assets"),
    # ONNX Models & Sidecars
    (str(project_root / "data" / "models" / "onnx"), "data/models/onnx"),
]

hidden_imports = [
    "uvicorn.logging",
    "uvicorn.loops",
    "uvicorn.loops.auto",
    "uvicorn.protocols",
    "uvicorn.protocols.http",
    "uvicorn.protocols.http.auto",
    "uvicorn.protocols.websockets",
    "uvicorn.protocols.websockets.auto",
    "uvicorn.lifespan",
    "uvicorn.lifespan.on",
    "fastapi",
    "pydantic",
    "pydantic_settings",
    "dotenv",
    "onnxruntime",
    "cv2",
    "numpy",
    "lapx",
    "insightface",
    "webview",
]

a = Analysis(
    ['app_window.py'],
    pathex=[str(project_root)],
    binaries=[],
    datas=added_files,
    hiddenimports=hidden_imports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=['tkinter', 'matplotlib', 'notebook', 'jupyter', 'IPython'],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='RapidEye',
    icon='rapideye.ico',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='RapidEye',
)
