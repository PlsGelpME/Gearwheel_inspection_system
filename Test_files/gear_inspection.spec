# gear_inspection.spec
import os
from PyInstaller.utils.hooks import collect_data_files, collect_submodules

block_cipher = None

a = Analysis(
    ['run_dashboard.py'],
    pathex=['.'],
    binaries=[],
    datas=[
        ('pipeline', 'pipeline'),
        ('gui', 'gui'),
        ('results', 'results'),
        # Config file if it exists
        *([ ('gear_inspection_config.json', '.') ]
            if os.path.exists('gear_inspection_config.json') else []),
    ],
    hiddenimports=[
        'pipeline.gear_core',
        'pipeline.gear_mask',
        'pipeline.tooth_analysis',
        'pipeline.sideprofile',
        'gui.dashboard',
        'cv2',
        'numpy',
        'scipy',
        'scipy.signal',
        'scipy.signal._peak_finding',
        'PIL',
        'PIL.Image',
        'PIL.ImageTk',
        'tkinter',
        'tkinter.ttk',
        'tkinter.filedialog',
        'tkinter.messagebox',
        'urllib.request',
        'threading',
    ],
    hookspath=[],
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
    name='GearInspection',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=False,        # False = no black console window
    icon='None',            # add 'icon.ico' path here if you have one
)