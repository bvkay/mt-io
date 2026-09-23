# -*- coding: utf-8 -*-
"""
Tests for read_orange on several files and on its keyword arguments.

Files are written here to the documented layout: 40 byte header, 21 byte
records, then the end stamp.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from mt_io.uoa.orange import ADC_ZERO, read_orange
from tests.uoa.test_orange import encode_record

FILTER_POINT = 0x7A1  # 10.00064 Hz
STAMP = "%a %b %d %H:%M:%S %Y"


def write_file(path: Path, start: str, n: int = 50, end_stamp: bool = True) -> Path:
    """n records starting at `start`, the end stamp n / 10 s later"""
    t0 = pd.Timestamp(start)
    with open(path, "wb") as fid:
        fid.write(f" {n:08X} \n".encode("ascii"))
        fid.write(f"{t0:{STAMP}}\n".encode("ascii"))
        fid.write(f" {FILTER_POINT:03X}".encode("ascii"))
        for i in range(n):
            values = [ADC_ZERO + i, ADC_ZERO - i, ADC_ZERO + 2 * i, 0, 0, 20]
            values += [ADC_ZERO + 3 * i, ADC_ZERO - 3 * i]
            fid.write(encode_record(values))
        if end_stamp:
            t1 = t0 + pd.Timedelta(seconds=n / 10)
            fid.write(f" {t1:{STAMP}}\n".encode("ascii"))
    return path


def read(files, **kwargs):
    kwargs.setdefault("station_id", "ST61")
    kwargs.setdefault("dipole_length_ex", 20.0)
    kwargs.setdefault("dipole_length_ey", 20.0)
    return read_orange(files, **kwargs)


class TestJoin:
    def test_back_to_back(self, tmp_path):
        files = [
            write_file(tmp_path / "HFM1-000.BIN", "2009-06-16 02:01:04"),
            write_file(tmp_path / "HFM1-001.BIN", "2009-06-16 02:01:09"),
        ]
        run = read(files)
        assert run.dataset.sizes["time"] == 100

    def test_two_hours_apart(self, tmp_path):
        files = [
            write_file(tmp_path / "HFM1-000.BIN", "2009-06-16 02:01:04"),
            write_file(tmp_path / "HFM1-001.BIN", "2009-06-16 04:01:09"),
        ]
        with pytest.raises(ValueError, match="7200 s missing between HFM1-000.BIN"):
            read(files)

    def test_without_end_stamps(self, tmp_path):
        """The start plus samples over the rate stands in for the end stamp"""
        a = write_file(tmp_path / "A.BIN", "2009-06-16 02:01:04", end_stamp=False)
        b = write_file(tmp_path / "B.BIN", "2009-06-16 02:01:09", end_stamp=False)
        c = write_file(tmp_path / "C.BIN", "2009-06-16 03:01:09", end_stamp=False)
        assert read([a, b]).dataset.sizes["time"] == 100
        with pytest.raises(ValueError, match="3595 s missing between B.BIN and C.BIN"):
            read([a, b, c])

    def test_end_stamp_read(self, tmp_path):
        from mt_io.uoa.orange import OrangeDataReader

        reader = OrangeDataReader(
            write_file(tmp_path / "A.BIN", "2009-06-16 02:01:04", n=36000)
        )
        reader.read()
        assert reader.end_time == pd.Timestamp("2009-06-16 03:01:04", tz="UTC")
