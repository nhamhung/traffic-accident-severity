"""Regression tests for hosted accident-dataset payload formats."""

import gzip
import zipfile

from traffic_accident_severity import config, data


def test_load_accidents_reads_zip_payload_saved_with_csv_suffix(tmp_path, monkeypatch):
    csv_path = tmp_path / "RTA Dataset.csv"
    payload = "Time,Accident_severity\n12:00:00,Slight Injury\n"
    with zipfile.ZipFile(csv_path, "w") as archive:
        archive.writestr("RTA Dataset.csv", payload)

    monkeypatch.setattr(config, "RAW_CSV", csv_path)
    monkeypatch.setattr(config, "SAMPLE_CSV", tmp_path / "missing-sample.csv.gz")

    loaded = data.load_accidents()
    assert loaded.loc[0, "Accident_severity"] == "Slight Injury"


def test_load_accidents_falls_back_to_cp1252(tmp_path, monkeypatch):
    csv_path = tmp_path / "RTA Dataset.csv"
    csv_path.write_bytes("Time,Area_accident_occured\n12:00:00,Café\n".encode("cp1252"))
    monkeypatch.setattr(config, "RAW_CSV", csv_path)
    monkeypatch.setattr(config, "SAMPLE_CSV", tmp_path / "missing-sample.csv.gz")

    loaded = data.load_accidents()
    assert loaded.loc[0, "Area_accident_occured"] == "Café"


def test_load_accidents_reads_bundled_gzip_sample(tmp_path, monkeypatch):
    sample_path = tmp_path / "accidents_sample.csv.gz"
    payload = "Time,Accident_severity\n12:00:00,Serious Injury\n".encode()
    sample_path.write_bytes(gzip.compress(payload))
    monkeypatch.setattr(config, "SAMPLE_CSV", sample_path)

    loaded = data.load_accidents()
    assert loaded.loc[0, "Accident_severity"] == "Serious Injury"
