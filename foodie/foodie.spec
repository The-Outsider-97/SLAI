
# -*- mode: python ; coding: utf-8 -*-

import re

from pathlib import Path

block_cipher = None

# Automatically extract requirements
with open('requirements.txt', 'r') as f:
    requirements = f.read().splitlines()

hidden_imports = [
    re.split(r'[<>=!]', req)[0].strip() 
    for req in requirements 
    if req.strip() and not req.startswith('#')
]

a = Analysis(
    ['foodie/app.py'],  # Main entry point
    pathex=[str(Path.cwd())],  # Add root project path if needed
    binaries=[],
    datas=[
        ('foodie/static/*', 'static'),
        ('foodie/configs/*.yaml', 'configs'),
        ('foodie/data/*.json', 'data'),
        ('foodie/instance/*.db', 'instance'),
        ('foodie/utils/*.py', 'utils'),
        ('foodie/core/*.py', 'core'),
        ('src/agents/*.py', 'agents'),
        ('src/agents/adaptive/**/*', 'adaptive'),
        ('src/agents/base/**/*', 'base'),
        ('src/agents/collaborative/**/*', 'collaborative'),
        ('src/agents/evaluators/**/*', 'evaluators'),
        ('src/agents/execution/**/*', 'execution'),
        ('src/agents/factory/**/*', 'factory'),
        ('src/agents/knowledge/**/*', 'knowledge'),
        ('src/agents/planning/**/*', 'planning'),
        ('src/agents/reasoning/**/*', 'reasoning'),
        ('src/agents/safety/**/*', 'safety'),
        ('src/agents/language/**/*', 'language'),
        ('src/deployment/rollback/*.py', 'rollback'),
        ('src/utils/**/*', 'agent_utils'),
        ('src/tuning/**/*', 'hyperparam')
    ],
    hiddenimports=[
        'flask',
        'flask_cors',
        'flask_sqlalchemy',
        'flask_migrate',
        'werkzeug',
        'dotenv',
        'numpy',
        'yaml',
        'uuid'
    ],
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
    [],
    name='Foodie',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=True,  # Set to True for debugging
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon='foodie/static/img/logo.ico'
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='FoodieApp',
)