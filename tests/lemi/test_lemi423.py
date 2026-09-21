# -*- coding: utf-8 -*-
"""
Tests for the LEMI-423 reader on synthetic B423 files.

The files are written here in the documented layout: a 1024 byte ASCII
header, then 30 byte little endian records.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from mt_io.lemi.lemi423 import Read_Lemi_Data, Read_Lemi_Header, read_lemi423

RECORD = Read_Lemi_Data.binary_format


def write_b423(
    path: Path,
    epoch: int = 1624510579,
    n: int = 2000,
    rate: int = 1000,
    alt_line: str = "%Alt 119.9,m 12 2",
) -> Path:
    """Write a B423 file of `n` records at `rate` Hz starting at `epoch`."""
    when = pd.Timestamp(epoch, unit="s", tz="UTC")
    lines = [
        "%LEMI423 #0036",
        "%FIRMWARE Ver.2.1",
        "%MADE in UKRAINE",
        " ",
        f"%Date {when:%Y/%m/%d}",
        f"%Time {when:%H:%M:%S}",
        "%Ubat 13.16V",
        "%Current 101.7mA",
        "%Free 30424MB",
        "%Lat 2944.90064,S",
        "%Lon 13900.11658,E",
        alt_line,
        " ",
        "%Kmx = 2.909985e-06",
        "%Kmy = 2.909481e-06",
        "%Kmz = 2.908610e-06",
        "%Ax = -5.002100e+01",
        "%Ay = -4.990500e+01",
        "%Az = -4.994700e+01",
        "%Ke1 = 2.910737e-04",
        "%Ke2 = 2.909547e-04",
        "%Ae1 = -5.004800e+03",
        "%Ae2 = -4.958000e+03",
    ]
    header = ("\r\n".join(lines) + "\r\n").encode("ascii").ljust(1024, b" ")
    records = np.zeros(n, dtype=RECORD)
    index = np.arange(n)
    records["time"] = epoch + index // rate
    records["tick"] = (index % rate) * (1000 // rate)
    for k, name in enumerate(("Bx", "By", "Bz", "Ex", "Ey")):
        records[name] = index * (k + 1) - 1000
    path.write_bytes(header + records.tobytes())
    return path


class TestHeader:
    @pytest.mark.parametrize(
        "alt_line, elevation",
        [
            ("%Alt  27.1,m 12 2", 27.1),
            ("%Alt 125.2,m 12 1", 125.2),
            ("%Alt1060.0,m 12 1", 1060.0),
        ],
    )
    def test_altitude_field_widths(self, tmp_path, alt_line, elevation):
        """Four digit altitudes have no space after the tag"""
        fn = write_b423(tmp_path / "1624510579.B423", alt_line=alt_line)
        header = Read_Lemi_Header(fn).read()
        assert header["elevation"] == pytest.approx(elevation)
        assert header["latitude"] == pytest.approx(-(29 + 44.90064 / 60))

    def test_read_with_four_digit_altitude(self, tmp_path):
        fn = write_b423(tmp_path / "1624510579.B423", alt_line="%Alt1060.0,m 12 1")
        run = read_lemi423(fn)
        assert run.station_metadata.location.elevation == pytest.approx(1060.0)
