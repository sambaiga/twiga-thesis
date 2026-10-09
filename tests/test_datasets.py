"""Tests for twiga_thesis.datasets."""

import pytest

from twiga_thesis.datasets import SPS_UK_FILES, fetch_sps_uk_raw, load_mlvs_pt


def test_load_mlvs_pt_raises_clear_error_when_missing(tmp_path):
    missing = tmp_path / "MLVS-PT.parquet"
    with pytest.raises(FileNotFoundError, match="unpublished dataset"):
        load_mlvs_pt(missing)


def test_load_mlvs_pt_returns_path_when_present(tmp_path):
    present = tmp_path / "MLVS-PT.parquet"
    present.write_bytes(b"")
    assert load_mlvs_pt(present) == present


def test_fetch_sps_uk_raw_skips_existing_files(tmp_path, monkeypatch):
    destination_dir = tmp_path / "SPS-UK_raw"
    destination_dir.mkdir()
    for name in SPS_UK_FILES:
        (destination_dir / name).write_bytes(b"")

    def fail_if_called(*args, **kwargs):
        raise AssertionError("should not attempt a download when files already exist")

    monkeypatch.setattr("urllib.request.urlretrieve", fail_if_called)
    result = fetch_sps_uk_raw(destination_dir)
    assert result == destination_dir


def test_fetch_sps_uk_raw_downloads_missing_files(tmp_path, monkeypatch):
    destination_dir = tmp_path / "SPS-UK_raw"
    downloaded = []

    def fake_urlretrieve(url, target):
        downloaded.append(target.name)
        target.write_bytes(b"")

    monkeypatch.setattr("urllib.request.urlretrieve", fake_urlretrieve)
    fetch_sps_uk_raw(destination_dir)
    assert set(downloaded) == set(SPS_UK_FILES)
