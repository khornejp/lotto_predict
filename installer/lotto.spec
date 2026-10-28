# PyInstaller spec: CPU-only onedir build of the Streamlit UI.
# Build with installer/build.ps1 (uses the CPU-torch venv, not the CUDA Predict env).
from pathlib import Path

from PyInstaller.utils.hooks import collect_data_files, collect_submodules, copy_metadata

root = Path(SPECPATH).parent

datas = [
    (str(root / "app.py"), "."),        # Streamlit executes the script from disk
    (str(root / "lotto.xlsx"), "."),    # seed data, copied to %LOCALAPPDATA%\LottoML on first run
]
datas += collect_data_files("streamlit")
datas += copy_metadata("streamlit")

a = Analysis(
    [str(root / "launcher.py")],
    pathex=[str(root)],
    datas=datas,
    hiddenimports=collect_submodules("streamlit") + ["app"],
    excludes=["pytest", "tkinter", "matplotlib", "IPython"],
)
pyz = PYZ(a.pure)
exe = EXE(pyz, a.scripts, [], exclude_binaries=True, name="LottoML", console=True)
coll = COLLECT(exe, a.binaries, a.datas, name="LottoML")
