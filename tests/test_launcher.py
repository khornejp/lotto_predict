from launcher import ensure_user_data


def test_ensure_user_data_copies_bundled_file_once(tmp_path):
    bundled = tmp_path / "bundle" / "lotto.xlsx"; bundled.parent.mkdir(); bundled.write_bytes(b"v1")
    home = tmp_path / "home"
    target = ensure_user_data(bundled, home)
    assert target == home / "lotto.xlsx" and target.read_bytes() == b"v1"
    target.write_bytes(b"user-updated")
    bundled.write_bytes(b"v2")
    assert ensure_user_data(bundled, home).read_bytes() == b"user-updated"   # never overwrite user data
