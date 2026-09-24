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


class TestCollection:
    def test_b423_listed_by_default(self, tmp_path):
        """A LEMI-423 folder lists its files without naming the rate"""
        from mt_io.lemi import LEMICollection

        for k in range(2):
            write_b423(
                tmp_path / f"{1624510579 + 2 * k}.B423", epoch=1624510579 + 2 * k
            )
        lc = LEMICollection(tmp_path, file_ext=["B423"])
        df = lc.to_dataframe()
        assert len(df) == 2
        assert set(df.sample_rate) == {1000.0}
        assert df.run.nunique() == 1

    def test_other_rates_are_counted(self, tmp_path):
        from mt_io.lemi import LEMICollection

        write_b423(tmp_path / "1624510579.B423")
        lc = LEMICollection(tmp_path, file_ext=["B423"])
        messages = []
        handler = lc.logger.add(lambda m: messages.append(str(m)), level="WARNING")
        try:
            df = lc.to_dataframe(sample_rates=[1])
        finally:
            lc.logger.remove(handler)
        assert len(df) == 0
        assert any("1 at 1000.0 Hz" in m for m in messages), messages


class TestSampleRate:
    def _warnings(self, action):
        from loguru import logger

        messages = []
        handler = logger.add(lambda m: messages.append(str(m)), level="WARNING")
        try:
            result = action()
        finally:
            logger.remove(handler)
        return result, messages

    def test_rate_from_tick(self, tmp_path):
        fn = write_b423(tmp_path / "1624510579.B423")
        reader = Read_Lemi_Data(fn, {})
        reader.read_dataframe()
        assert reader.sample_rate == 1000.0
        assert reader.read_summary()["sample_rate"] == 1000.0

    def test_tick_always_zero_is_reported(self, tmp_path):
        """One record a second, tick 0: no rate, and a warning naming the file"""
        fn = write_b423(tmp_path / "1624510579.B423", n=30, rate=1)
        reader = Read_Lemi_Data(fn, {})
        summary, messages = self._warnings(reader.read_summary)
        assert summary["sample_rate"] is None
        assert any(
            "1624510579.B423" in m and "1 records per second" in m for m in messages
        )
        _, messages = self._warnings(reader.read_dataframe)
        assert reader.sample_rate is None
        assert any("1624510579.B423" in m for m in messages)

    def test_reader_says_where_the_rate_came_from(self, tmp_path):
        fn = write_b423(tmp_path / "1624510579.B423", n=30, rate=1)
        run, messages = self._warnings(lambda: read_lemi423(fn))
        assert run.run_metadata.sample_rate == 1.0
        assert any("taken from the time stamps" in m for m in messages)


def write_rsp(path: Path) -> Path:
    """A normalized coil response: amplitude 1 in the pass band"""
    freqs = np.logspace(-3, 3, 25)
    amps = freqs / np.sqrt(freqs**2 + 0.01)
    phases = np.degrees(np.arctan2(0.1, freqs))
    body = "\n".join(f"{f:.6e} {a:.6e} {p:.6e}" for f, a, p in zip(freqs, amps, phases))
    path.write_text("B\nfreq amp phas\n" + body + "\n")
    return path


class TestCoilResponse:
    def test_units_pass_validation(self, tmp_path):
        from mt_io.lemi.lemi423 import read_lemi_coil_response

        fap = read_lemi_coil_response(write_rsp(tmp_path / "l120n.rsp"), "36")
        assert fap.units_in == "nanoTesla"
        assert fap.units_out == "nanoTesla"

    def test_chain_with_coil_response(self, tmp_path):
        """calibration_fn gives hx a valid chain: coil, then linear"""
        fn = write_b423(tmp_path / "1624510579.B423")
        rsp = write_rsp(tmp_path / "l120n.rsp")
        run = read_lemi423(fn, calibration_fn=rsp, dipole_length_ex=50.0)

        hx = run.hx
        response = hx.channel_response
        assert response.names == ["lemi_120_36_response", "lemi423_linear_hx"]
        assert response.units_in == "nanoTesla"
        assert response.units_out == "digital counts"
        stages = {f.name: f.stage for f in hx.channel_metadata.filters}
        assert stages == {"lemi_120_36_response": 1, "lemi423_linear_hx": 2}

        # the product is the coil table times 1/K whatever the order
        from mt_io.lemi.lemi423 import read_lemi_coil_response

        f = np.array([0.5, 5.0, 50.0])
        coil = read_lemi_coil_response(rsp).complex_response(f)
        np.testing.assert_allclose(
            response.complex_response(f), coil / 2.909985e-06, rtol=1e-12
        )
        np.testing.assert_allclose(np.abs(coil[1:]), 1.0, rtol=1e-3)

        assert run.ex.channel_response.names == [
            "lemi423_dipole_ex_50.0m",
            "lemi423_linear_ex",
        ]


def pandas_frame(fn: Path) -> pd.DataFrame:
    """The records as the reader built them with pandas arithmetic"""
    with open(fn, "rb") as f:
        f.read(1024)
        df = pd.DataFrame(np.fromfile(f, dtype=RECORD))
    df["time"] = pd.to_datetime(df["time"], unit="s", utc=True) + pd.to_timedelta(
        df["tick"], unit="ms"
    )
    df.set_index("time", inplace=True)
    return df[["Bx", "By", "Bz", "Ex", "Ey"]].sort_index()


class TestVectorisedRead:
    def _shuffle(self, fn, seed=3):
        n = (fn.stat().st_size - 1024) // RECORD.itemsize
        arr = np.memmap(fn, dtype=RECORD, mode="r+", offset=1024, shape=(n,))
        arr[:] = arr[np.random.default_rng(seed).permutation(n)]
        arr.flush()
        del arr

    @pytest.mark.parametrize("shuffle", [False, True])
    def test_same_frame_as_pandas(self, tmp_path, shuffle):
        fn = write_b423(tmp_path / "1624510579.B423", n=3000)
        if shuffle:
            self._shuffle(fn)
        got = Read_Lemi_Data(fn, {}).read_dataframe()
        want = pandas_frame(fn)
        assert got.index.dtype == want.index.dtype
        pd.testing.assert_frame_equal(got, want, check_exact=True)

    def test_files_out_of_order(self, tmp_path):
        a = write_b423(tmp_path / "1624510579.B423", n=3000)
        b = write_b423(tmp_path / "1624510582.B423", epoch=1624510582, n=3000)
        forward = read_lemi423([a, b])
        backward = read_lemi423([b, a])
        assert forward.dataset.equals(backward.dataset)
        assert forward.dataset.sizes["time"] == 6000


class TestGPSStatus:
    def _file(self, tmp_path):
        fn = write_b423(tmp_path / "1624510579.B423", n=2000)
        arr = np.memmap(fn, dtype=RECORD, mode="r+", offset=1024, shape=(2000,))
        arr["sync"] = np.arange(2000) % 7 - 3
        arr["stage"] = np.arange(2000) % 4
        arr.flush()
        del arr
        return fn

    def test_frame_columns(self, tmp_path):
        fn = self._file(tmp_path)
        df = Read_Lemi_Data(fn, {}).read_dataframe(gps=True)
        assert list(df.columns) == ["Bx", "By", "Bz", "Ex", "Ey", "sync", "stage"]
        np.testing.assert_array_equal(df["sync"], np.arange(2000) % 7 - 3)
        assert df["sync"].dtype == np.int8 and df["stage"].dtype == np.uint8
        assert list(Read_Lemi_Data(fn, {}).read_dataframe().columns) == [
            "Bx",
            "By",
            "Bz",
            "Ex",
            "Ey",
        ]

    def test_auxiliary_channels(self, tmp_path):
        fn = self._file(tmp_path)
        run = read_lemi423(fn, gps_status=True)
        assert sorted(run.channels) == sorted(
            ["hx", "hy", "hz", "ex", "ey", "gps_sync", "gps_stage"]
        )
        np.testing.assert_array_equal(
            run.dataset["gps_stage"].values, np.arange(2000) % 4
        )
        assert run.gps_sync.channel_metadata.type == "auxiliary"
        plain = read_lemi423(fn)
        assert sorted(plain.channels) == ["ex", "ey", "hx", "hy", "hz"]
        assert plain.dataset.equals(run.dataset[["hx", "hy", "hz", "ex", "ey"]])


def test_blank_header_raises_value_error(tmp_path):
    """A B423 file whose 1024-byte header block is all zero raises a ValueError naming it."""
    from mt_io.lemi.lemi423 import Read_Lemi_Header

    path = tmp_path / "1678734241.B423"
    path.write_bytes(bytes(1024) + bytes(4096))
    with pytest.raises(ValueError, match="1678734241.B423.*header unreadable"):
        Read_Lemi_Header(path).read()
