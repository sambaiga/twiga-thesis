"""Tests for twiga_thesis.datasets."""

import pandas as pd
import pytest

from twiga_thesis.datasets import SPS_UK_FILES, build_sps_uk_parquet, fetch_sps_uk_raw, load_mlvs_pt


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


def _write_sps_uk_raw_fixture(raw_dir):
    raw_dir.mkdir(parents=True, exist_ok=True)
    weather = pd.DataFrame(
        {
            "datetime": pd.date_range("2018-01-01", periods=3, freq="h"),
            "temp_location1": [5.0, 6.0, 7.0],
            "temp_location2": [7.0, 8.0, 9.0],
            "solar_location1": [0.0, 100.0, 200.0],
            "solar_location2": [0.0, 120.0, 220.0],
        }
    )
    weather.to_csv(raw_dir / "weather_train_set4.csv", index=False)

    half_hourly = pd.date_range("2018-01-01", periods=5, freq="30min")
    pd.DataFrame({"datetime": half_hourly[:3], "pv_power_mw": [0.0, 0.1, 0.2]}).to_csv(
        raw_dir / "pv_train_set4.csv", index=False
    )
    pd.DataFrame({"datetime": half_hourly[3:], "pv_power_mw": [0.3, 0.4]}).to_csv(
        raw_dir / "pv_test_set4.csv", index=False
    )
    pd.DataFrame({"datetime": half_hourly[:3], "demand_MW": [1.0, 1.1, 1.2]}).to_csv(
        raw_dir / "demand_train_set4.csv", index=False
    )
    pd.DataFrame({"datetime": half_hourly[3:], "demand_MW": [1.3, 1.4]}).to_csv(
        raw_dir / "demand_test_set4.csv", index=False
    )


class TestBuildSpsUkParquet:
    def test_raises_when_raw_files_missing(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="missing raw SPS-UK files"):
            build_sps_uk_parquet(tmp_path / "does-not-exist", tmp_path / "out.parquet")

    def test_produces_expected_schema(self, tmp_path):
        raw_dir = tmp_path / "raw"
        _write_sps_uk_raw_fixture(raw_dir)
        destination = tmp_path / "MLVS-UK.parquet"

        result = build_sps_uk_parquet(raw_dir, destination)

        assert result == destination
        df = pd.read_parquet(destination)
        assert list(df.columns) == ["timestamp", "NetLoad(MW)", "PV(MW)", "Ghi", "Temperature"]
        assert len(df) > 0
        assert not df["timestamp"].isna().any()

    def test_net_load_is_demand_minus_pv(self, tmp_path):
        raw_dir = tmp_path / "raw"
        _write_sps_uk_raw_fixture(raw_dir)
        destination = tmp_path / "MLVS-UK.parquet"

        build_sps_uk_parquet(raw_dir, destination)
        df = pd.read_parquet(destination).set_index("timestamp")

        # At the half-hourly timestamps covered by the fixture, demand - PV
        # should reproduce NetLoad exactly (both read straight from the raw
        # CSVs, no resampling involved on this side of the merge).
        first_ts = df.index[0]
        assert df.loc[first_ts, "NetLoad(MW)"] == pytest.approx(1.0 - 0.0)

    def test_weather_columns_are_averaged_across_stations(self, tmp_path):
        raw_dir = tmp_path / "raw"
        _write_sps_uk_raw_fixture(raw_dir)
        destination = tmp_path / "MLVS-UK.parquet"

        build_sps_uk_parquet(raw_dir, destination)
        df = pd.read_parquet(destination)

        # temp_location1=5.0, temp_location2=7.0 at the first weather timestamp -> mean 6.0.
        assert df["Temperature"].iloc[0] == pytest.approx(6.0)
