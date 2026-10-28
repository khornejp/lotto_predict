"""Frozen-app entry point: put user data in a writable folder, then serve app.py with Streamlit.

The installed program folder may be read-only and PyInstaller's bundle folder is
replaced on upgrade, so lotto.xlsx and checkpoints live in %LOCALAPPDATA%\LottoML.
"""
from __future__ import annotations

import os
import shutil
import sys
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


def run() -> None:
    """Prepare user data, chdir into it (checkpoints land there) and start the Streamlit server."""
    bundle = Path(getattr(sys, "_MEIPASS", Path(__file__).resolve().parent))
    home = Path(os.environ.get("LOCALAPPDATA", Path.home())) / "LottoML"
    os.environ["LOTTO_DATA"] = str(ensure_user_data(bundle / "lotto.xlsx", home))
    os.chdir(home)
    import main  # noqa: F401  # pulls every project module into the PyInstaller analysis
    from streamlit.web import cli as stcli

    sys.argv = ["streamlit", "run", str(bundle / "app.py"), "--global.developmentMode=false",
                "--server.headless=false", "--browser.gatherUsageStats=false"]
    sys.exit(stcli.main())


if __name__ == "__main__":
    run()
