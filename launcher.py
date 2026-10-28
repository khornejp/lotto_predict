"""Frozen-app entry point: put user data in a writable folder, then serve app.py with Streamlit.

The installed program folder may be read-only and PyInstaller's bundle folder is
replaced on upgrade, so lotto.xlsx lives in %LOCALAPPDATA%\LottoML. Checkpoints and train.log
go to <install folder>\checkpoints, the folder chosen in setup.
"""
from __future__ import annotations

import faulthandler
import os
import shutil
import sys
import threading
import webbrowser
from pathlib import Path


def ensure_user_data(bundled: Path, home: Path) -> Path:
    """Copy the bundled spreadsheet into ``home`` on first run; never overwrite it later.

    Args:
        bundled: Spreadsheet shipped inside the bundle.
        home: Writable per-user data folder.

    Returns:
        Path of the user's spreadsheet.
    """
    home.mkdir(parents=True, exist_ok=True)
    target = home / "lotto.xlsx"
    if not target.exists():
        shutil.copy2(bundled, target)
    return target


def work_dir(home: Path) -> Path:
    """Folder that ``checkpoints/`` (models, train.log) is created in.

    Args:
        home: Per-user data folder, used when running from source.

    Returns:
        The install folder (beside LottoML.exe) when frozen, else ``home``.
    """
    # Per-user install (PrivilegesRequired=lowest) defaults to a writable folder.
    return Path(sys.executable).parent if getattr(sys, "frozen", False) else home


def run() -> None:
    """Prepare user data, chdir to the work dir (checkpoints land there) and start the Streamlit server."""
    bundle = Path(getattr(sys, "_MEIPASS", Path(__file__).resolve().parent))
    home = Path(os.environ.get("LOCALAPPDATA", Path.home())) / "LottoML"
    os.environ["LOTTO_DATA"] = str(ensure_user_data(bundle / "lotto.xlsx", home))
    os.chdir(work_dir(home))
    import main  # noqa: F401  # pulls every project module into the PyInstaller analysis
    from streamlit.web import cli as stcli

    port = 8501
    threading.Timer(3.0, webbrowser.open, (f"http://localhost:{port}",)).start()
    sys.argv = streamlit_args(str(bundle / "app.py"), port)
    sys.exit(stcli.main())


def streamlit_args(app_path: str, port: int) -> list[str]:
    """Command line for serving ``app_path``: local-only and without interactive prompts.

    Headless mode skips Streamlit's first-run e-mail prompt (it blocks a console app forever);
    ``run`` opens the browser itself instead.

    Args:
        app_path: Absolute path of the Streamlit script.
        port: Local port.

    Returns:
        ``sys.argv`` for ``streamlit.web.cli.main``.
    """
    return ["streamlit", "run", app_path, "--global.developmentMode=false", "--server.headless=true",
            "--server.address=localhost", f"--server.port={port}", "--browser.gatherUsageStats=false"]


if __name__ == "__main__":
    faulthandler.enable()  # native crashes (e.g. a DLL failing to load) still print a Python traceback
    run()
