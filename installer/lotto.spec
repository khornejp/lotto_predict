# PyInstaller spec: CPU-only onedir build of the Streamlit UI.
# Build with installer/build.ps1 (uses the CPU-torch venv, not the CUDA Predict env).
import sys
from pathlib import Path

from PyInstaller.utils.hooks import collect_data_files, collect_submodules, copy_metadata

root = Path(SPECPATH).parent

datas = [
    (str(root / "app.py"), "."),        # Streamlit executes the script from disk
    (str(root / "lotto.xlsx"), "."),    # seed data, copied to %LOCALAPPDATA%\LottoML on first run
]
datas += collect_data_files("streamlit")
datas += copy_metadata("streamlit")

# pip torch on Windows depends on the `mkl` wheel, whose DLLs live in <venv>\Library\bin and are
# loaded at runtime (MKL picks CPU-specific kernels dynamically), so PyInstaller's analysis misses
# them and torch._C fails with 0xC06D007E. Ship the single-node CPU set; skip MPI/cluster variants.
mkl_bin = Path(sys.prefix) / "Library" / "bin"
mkl_patterns = ["mkl_core*.dll", "mkl_intel_thread*.dll", "mkl_def*.dll", "mkl_avx*.dll", "mkl_mc*.dll",
                "mkl_vml_*.dll", "mkl_rt*.dll", "mkl_msg.dll", "libiomp5md.dll", "libimalloc.dll"]
binaries = [(str(f), ".") for pat in mkl_patterns for f in mkl_bin.glob(pat)]
assert binaries, f"no MKL DLLs under {mkl_bin}: build with the CPU-torch venv"

a = Analysis(
    [str(root / "launcher.py")],
    pathex=[str(root)],
    binaries=binaries,
    datas=datas,
    hiddenimports=collect_submodules("streamlit") + ["app"],
    excludes=["pytest", "tkinter", "matplotlib", "IPython"],
)
pyz = PYZ(a.pure)
exe = EXE(pyz, a.scripts, [], exclude_binaries=True, name="LottoML", console=True)
coll = COLLECT(exe, a.binaries, a.datas, name="LottoML")
