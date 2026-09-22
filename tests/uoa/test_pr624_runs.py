# -*- coding: utf-8 -*-
"""
Tests for UoAReader on file sets that do not make one run, and on the
response chain arguments.

Fixtures are written here: one ASCII file per channel per stamp.
"""

from pathlib import Path

import numpy as np
import pytest

from mt_io.uoa import read_uoa

CHANNELS = ["BX", "BY", "BZ", "EX", "EY"]
SAMPLE_RATE = 10.0
N_SAMPLES = 60  # 6 s per file at 10 Hz


def write_edl(directory: Path, stamps, channels=CHANNELS, n_samples=N_SAMPLES):
    """One ASCII file per channel per stamp, values counting up from 1000."""
    directory.mkdir(parents=True, exist_ok=True)
    body = "\n".join(str(1000 + i) for i in range(n_samples)) + "\n"
    for stamp in stamps:
        for channel in channels:
            (directory / f"TEST01_{stamp}.{channel}").write_text(body)


def read(files, **kwargs):
    kwargs.setdefault("sample_rate", SAMPLE_RATE)
    return read_uoa(
        files,
        station_id="TEST01",
        dipole_length_ex=50.0,
        dipole_length_ey=50.0,
        **kwargs,
    )


class TestContiguity:
    def test_contiguous_list_reads(self, tmp_path):
        write_edl(tmp_path, ["240101000000", "240101000006", "240101000012"])
        run = read(sorted(tmp_path.glob("TEST01_*")))
        assert run.dataset.sizes["time"] == 3 * N_SAMPLES
        assert str(run.run_metadata.time_period.start).startswith("2024-01-01T00:00:00")

    def test_channel_file_missing_mid_run(self, tmp_path):
        """EX has no file at the second stamp: its later samples would be early"""
        write_edl(tmp_path, ["240101000000", "240101000006", "240101000012"])
        (tmp_path / "TEST01_240101000006.EX").unlink()
        with pytest.raises(ValueError) as error:
            read(sorted(tmp_path.glob("TEST01_*")))
        message = str(error.value)
        assert "EX: 6.0 s missing" in message
        assert "BX" not in message

    def test_short_files_stamped_apart(self, tmp_path):
        """Files of 6 s stamped 60 s apart are not one run"""
        write_edl(tmp_path, ["240101000000", "240101000100"])
        with pytest.raises(ValueError, match="54.0 s missing"):
            read(sorted(tmp_path.glob("TEST01_*")))

    def test_two_files_with_one_stamp(self, tmp_path):
        write_edl(tmp_path, ["240101000000", "240101000006"])
        write_edl(tmp_path / "old", ["240101000006"], channels=["BY"])
        with pytest.raises(ValueError, match="BY: 6.0 s overlap"):
            read(sorted(tmp_path.rglob("TEST01_*")))

    def test_channel_starting_late(self, tmp_path):
        write_edl(tmp_path, ["240101000000", "240101000006"])
        (tmp_path / "TEST01_240101000000.EY").unlink()
        with pytest.raises(ValueError, match="EY: starts at TEST01_240101000006.EY"):
            read(sorted(tmp_path.glob("TEST01_*")))

    def test_directory_with_a_gap(self, tmp_path):
        write_edl(tmp_path, ["240101000000", "240101000100"])
        with pytest.raises(ValueError, match="UoACollection"):
            read(tmp_path)
