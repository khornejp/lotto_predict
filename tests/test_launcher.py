from launcher import ensure_user_data


def test_ensure_user_data_copies_bundled_file_once(tmp_path):
    bundled = tmp_path / "bundle" / "lotto.xlsx"; bundled.parent.mkdir(); bundled.write_bytes(b"v1")
    home = tmp_path / "home"
    target = ensure_user_data(bundled, home)
    assert target == home / "lotto.xlsx" and target.read_bytes() == b"v1"
    target.write_bytes(b"user-updated")
    bundled.write_bytes(b"v2")
    assert ensure_user_data(bundled, home).read_bytes() == b"user-updated"   # never overwrite user data


def test_streamlit_args_local_only_and_non_interactive():
    from launcher import streamlit_args
    args = streamlit_args("C:/x/app.py", 8501)
    assert args[:3] == ["streamlit", "run", "C:/x/app.py"]
    assert "--server.headless=true" in args          # no first-run e-mail prompt blocking the server
    assert "--server.address=localhost" in args      # not reachable from other machines
    assert "--server.port=8501" in args


def test_work_dir_is_install_folder_when_frozen(tmp_path, monkeypatch):
    import launcher
    monkeypatch.setattr(launcher.sys, "frozen", True, raising=False)
    monkeypatch.setattr(launcher.sys, "executable", str(tmp_path / "LottoML.exe"))
    assert launcher.work_dir(tmp_path / "home") == tmp_path   # checkpoints + train.log beside LottoML.exe


def test_work_dir_falls_back_to_home_from_source(tmp_path):
    from launcher import work_dir
    assert work_dir(tmp_path / "home") == tmp_path / "home"
