"""Tests for twiga_thesis.paths."""

import pytest

from twiga_thesis.paths import external_data_dir, repo_root, thesis_data_dir


def test_repo_root_contains_pyproject():
    assert (repo_root() / "pyproject.toml").exists()


def test_thesis_data_dir_is_under_repo_root():
    assert thesis_data_dir() == repo_root() / "data"


def test_external_data_dir_raises_clear_error_when_unset(monkeypatch):
    monkeypatch.delenv("EXTERNAL_DATA_DIR", raising=False)
    with pytest.raises(FileNotFoundError, match="EXTERNAL_DATA_DIR is not set"):
        external_data_dir()


def test_external_data_dir_raises_when_path_missing(monkeypatch, tmp_path):
    monkeypatch.setenv("EXTERNAL_DATA_DIR", str(tmp_path / "does-not-exist"))
    with pytest.raises(FileNotFoundError, match="does not exist"):
        external_data_dir()


def test_external_data_dir_returns_path_when_set(monkeypatch, tmp_path):
    monkeypatch.setenv("EXTERNAL_DATA_DIR", str(tmp_path))
    assert external_data_dir() == tmp_path
