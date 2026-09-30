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

from mt_io.lemi.lemi423 import read_lemi423, Read_Lemi_Data, Read_Lemi_Header

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
        """read_lemi423 puts a four digit altitude in the station elevation"""
        fn = write_b423(tmp_path / "1624510579.B423", alt_line="%Alt1060.0,m 12 1")
        run = read_lemi423(fn)
        assert run.station_metadata.location.elevation == pytest.approx(1060.0)

    def test_blank_header_raises_value_error(self, tmp_path):
        """A header block of zero bytes raises a ValueError naming the file"""
        from mt_io.lemi.lemi423 import Read_Lemi_Header

        path = tmp_path / "1678734241.B423"
        path.write_bytes(bytes(1024) + bytes(4096))
        with pytest.raises(ValueError, match="1678734241.B423.*header unreadable"):
            Read_Lemi_Header(path).read()


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
        """Files dropped by an explicit sample_rates are counted in one warning"""
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

    def test_blank_header_skipped(self, tmp_path):
        """A zero header or a 0 byte file is left out with a warning"""
        from mt_io.lemi import LEMICollection

        for k in range(2):
            write_b423(
                tmp_path / f"{1624510579 + 2 * k}.B423", epoch=1624510579 + 2 * k
            )
        (tmp_path / "1624510583.B423").write_bytes(bytes(1024 + 2000 * 30))
        (tmp_path / "1624510585.B423").write_bytes(b"")
        lc = LEMICollection(tmp_path, file_ext=["B423"])
        messages = []
        handler = lc.logger.add(lambda m: messages.append(str(m)), level="WARNING")
        try:
            df = lc.to_dataframe()
        finally:
            lc.logger.remove(handler)
        assert len(df) == 2
        skipped = [m for m in messages if "header unreadable" in m and "skipping" in m]
        assert len(skipped) == 2, messages
        assert any("1624510583.B423" in m for m in skipped)
        assert any("1624510585.B423" in m for m in skipped)

    def test_garbled_header_raises(self, tmp_path):
        """A header that is there but does not parse is not skipped"""
        from mt_io.lemi import LEMICollection
        from mt_io.lemi.lemi423 import B423HeaderError

        write_b423(tmp_path / "1624510579.B423")
        fn = write_b423(tmp_path / "1624510581.B423", epoch=1624510581)
        raw = fn.read_bytes()
        assert raw.count(b"%Lat 2944.90064,S") == 1
        fn.write_bytes(raw.replace(b"%Lat 2944.90064,S", b"%Lat 29x4.9OO64,S"))
        lc = LEMICollection(tmp_path, file_ext=["B423"])
        with pytest.raises(ValueError, match="could not convert") as error:
            lc.to_dataframe()
        assert not isinstance(error.value, B423HeaderError)


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
        """A counting tick gives the rate in read_dataframe and read_summary"""
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
        """read_lemi423 warns when it takes the rate from the time stamps"""
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
        """The coil table carries units mt_metadata knows, nanoTesla in and out"""
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
        """In order or shuffled, the frame equals the one pandas arithmetic gave"""
        fn = write_b423(tmp_path / "1624510579.B423", n=3000)
        if shuffle:
            self._shuffle(fn)
        got = Read_Lemi_Data(fn, {}).read_dataframe()
        want = pandas_frame(fn)
        assert got.index.dtype == want.index.dtype
        pd.testing.assert_frame_equal(got, want, check_exact=True)

    def test_files_out_of_order(self, tmp_path):
        """Files given last first read the same as files given in order"""
        a = write_b423(tmp_path / "1624510579.B423", n=3000)
        b = write_b423(tmp_path / "1624510582.B423", epoch=1624510582, n=3000)
        forward = read_lemi423([a, b])
        backward = read_lemi423([b, a])
        assert forward.dataset.equals(backward.dataset)
        assert forward.dataset.sizes["time"] == 6000

    def _split(self, tmp_path, first, second):
        """Two files holding the record ranges first and second of one 5 s run"""
        whole = write_b423(tmp_path / "1624510579.B423", n=5000)
        data = whole.read_bytes()
        files = []
        for k, (i0, i1) in enumerate((first, second)):
            fn = tmp_path / f"part{k}.B423"
            fn.write_bytes(
                data[:1024]
                + data[1024 + i0 * RECORD.itemsize : 1024 + i1 * RECORD.itemsize]
            )
            files.append(fn)
        return whole, files

    def test_overlapping_files(self, tmp_path):
        """Records in both files are kept once"""
        whole, files = self._split(tmp_path, (0, 3000), (2000, 5000))
        run = read_lemi423(files)
        assert run.dataset.sizes["time"] == 5000
        assert run.dataset.equals(read_lemi423(whole).dataset)

        # what the reader did before the integer index: concat, sort, drop
        frames = [Read_Lemi_Data(fn, {}).read_dataframe() for fn in files]
        want = pd.concat(frames).sort_index()
        want = want[~want.index.duplicated(keep="first")]
        channels = {"Bx": "hx", "By": "hy", "Bz": "hz", "Ex": "ex", "Ey": "ey"}
        for name, code in channels.items():
            np.testing.assert_array_equal(run.dataset[code].values, want[name])

    def test_same_file_twice(self, tmp_path):
        """Every record of a file given twice is kept once"""
        fn = write_b423(tmp_path / "1624510579.B423", n=3000)
        run = read_lemi423([fn, fn])
        assert run.dataset.sizes["time"] == 3000
        assert run.dataset.equals(read_lemi423(fn).dataset)


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
        """gps=True adds the sync and stage columns; the default leaves them out"""
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
        """gps_status=True adds gps_sync and gps_stage as auxiliary channels"""
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


class TestPartialRecord:
    """A file that ends part way through a record, as a cut copy leaves it"""

    def _cut(self, tmp_path, n_whole=1000, extra=17):
        fn = write_b423(tmp_path / "1695393386.B423", epoch=1695393386, n=2000)
        intact = Read_Lemi_Data(fn, {}).read_dataframe()
        fn.write_bytes(fn.read_bytes()[: 1024 + n_whole * RECORD.itemsize + extra])
        return fn, intact

    def _warnings(self, action):
        from loguru import logger

        messages = []
        handler = logger.add(lambda m: messages.append(str(m)), level="WARNING")
        try:
            result = action()
        finally:
            logger.remove(handler)
        return result, [m for m in messages if "part way through a record" in m]

    def test_whole_records_read(self, tmp_path):
        """Whole records are read and one warning gives the bytes left over"""
        fn, intact = self._cut(tmp_path)
        df, messages = self._warnings(Read_Lemi_Data(fn, {}).read_dataframe)
        pd.testing.assert_frame_equal(df, intact.iloc[:1000], check_exact=True)
        assert len(messages) == 1, messages
        assert "1695393386.B423" in messages[0]
        assert "1000 whole records read and 17 bytes left over" in messages[0]

    def test_summary(self, tmp_path):
        """read_summary counts the whole records only and warns once"""
        fn, intact = self._cut(tmp_path)
        summary, messages = self._warnings(Read_Lemi_Data(fn, {}).read_summary)
        assert summary["n_samples"] == 1000
        assert summary["end"] == intact.index[999]
        assert len(messages) == 1, messages

    def test_reader(self, tmp_path):
        """read_lemi423 reads the whole records and the warning names the file"""
        fn, _ = self._cut(tmp_path)
        run, messages = self._warnings(lambda: read_lemi423(fn))
        assert run.dataset.sizes["time"] == 1000
        assert any("1695393386.B423" in m for m in messages), messages

    def test_whole_file_is_quiet(self, tmp_path):
        """A file of whole records gives no warning"""
        fn = write_b423(tmp_path / "1695393386.B423", epoch=1695393386, n=2000)
        _, messages = self._warnings(Read_Lemi_Data(fn, {}).read_dataframe)
        assert messages == []


class TestGapBetweenFiles:
    def _read(self, files, **kwargs):
        from loguru import logger

        messages = []
        handler = logger.add(lambda m: messages.append(str(m)), level="WARNING")
        try:
            run = read_lemi423(files, **kwargs)
        finally:
            logger.remove(handler)
        return run, [m for m in messages if "gap between the files" in m]

    def _refused(self, files):
        with pytest.raises(ValueError, match="do not make one contiguous run") as error:
            read_lemi423(files)
        return str(error.value)

    def _inside(self, tmp_path):
        """A long file, a file whose records lie inside it, and one after a gap"""
        a = write_b423(tmp_path / "1624510579.B423", epoch=1624510579, n=10000)
        size, data = RECORD.itemsize, a.read_bytes()
        inner = tmp_path / "1624510581.B423"
        inner.write_bytes(data[:1024] + data[1024 + 2000 * size : 1024 + 4000 * size])
        c = write_b423(tmp_path / "1624510592.B423", epoch=1624510592, n=2000)
        return [a, inner, c]

    def test_gap_refused(self, tmp_path):
        """Files with a gap between them raise, naming both files and the gap"""
        a = write_b423(tmp_path / "1624510579.B423", epoch=1624510579, n=2000)
        b = write_b423(tmp_path / "1624510584.B423", epoch=1624510584, n=2000)
        message = self._refused([a, b])
        assert "1624510579.B423 to 1624510584.B423: 3.001 s gap" in message
        assert "LEMICollection" in message
        assert "fill_gaps=True" in message

    def test_gap_refused_out_of_order(self, tmp_path):
        """The gap is found when the later file is given first"""
        a = write_b423(tmp_path / "1624510579.B423", epoch=1624510579, n=2000)
        b = write_b423(tmp_path / "1624510584.B423", epoch=1624510584, n=2000)
        assert "1624510579.B423 to 1624510584.B423" in self._refused([b, a])

    def test_gap_refused_after_files_that_join(self, tmp_path):
        """The error names the file the gap follows, not the first of a chain"""
        a = write_b423(tmp_path / "1624510579.B423", epoch=1624510579, n=2000)
        b = write_b423(tmp_path / "1624510581.B423", epoch=1624510581, n=2000)
        c = write_b423(tmp_path / "1624510586.B423", epoch=1624510586, n=2000)
        assert "1624510581.B423 to 1624510586.B423" in self._refused([a, b, c])

    def test_gap_refused_after_file_inside_another(self, tmp_path):
        """A gap after a file inside another is measured from the outer one"""
        message = self._refused(self._inside(tmp_path))
        assert "1624510579.B423 to 1624510592.B423: 3.001 s gap" in message

    def test_files_that_join_are_quiet(self, tmp_path):
        """Files that join read as one run with no warning and no NaN"""
        a = write_b423(tmp_path / "1624510579.B423", epoch=1624510579, n=2000)
        b = write_b423(tmp_path / "1624510581.B423", epoch=1624510581, n=2000)
        run, gaps = self._read([a, b])
        assert gaps == []
        assert run.dataset.time.size == 4000
        assert not np.isnan(run.dataset.hx.values).any()

    def test_gap_left_as_nan(self, tmp_path):
        """With fill_gaps the samples after a gap keep their own time stamps"""
        a = write_b423(tmp_path / "1624510579.B423", epoch=1624510579, n=2000)
        b = write_b423(tmp_path / "1624510584.B423", epoch=1624510584, n=2000)
        run, gaps = self._read([a, b], fill_gaps=True)
        assert len(gaps) == 1, gaps
        assert "1624510579.B423 to 1624510584.B423: 3.001 s gap" in gaps[0]
        assert "3000 samples left as NaN" in gaps[0]
        times = pd.DatetimeIndex(run.dataset.time.values)
        assert times.size == 7000
        assert times[5000] == pd.Timestamp(1624510584, unit="s")
        assert times[-1] == pd.Timestamp(1624510584, unit="s") + pd.Timedelta("1999ms")
        hx = run.dataset.hx.values
        assert np.isnan(hx[2000:5000]).all()
        np.testing.assert_array_equal(hx[:2000], np.arange(2000) - 1000)
        np.testing.assert_array_equal(hx[5000:], np.arange(2000) - 1000)

    def test_gap_filled_out_of_order(self, tmp_path):
        """With fill_gaps the gap is found when the later file is given first"""
        a = write_b423(tmp_path / "1624510579.B423", epoch=1624510579, n=2000)
        b = write_b423(tmp_path / "1624510584.B423", epoch=1624510584, n=2000)
        run, gaps = self._read([b, a], fill_gaps=True)
        assert len(gaps) == 1, gaps
        assert "1624510579.B423 to 1624510584.B423: 3.001 s gap" in gaps[0]
        times = pd.DatetimeIndex(run.dataset.time.values)
        assert times.size == 7000
        assert times[5000] == pd.Timestamp(1624510584, unit="s")

    def test_gap_filled_after_file_inside_another(self, tmp_path):
        """A filled gap after a file inside another starts where the outer one ends"""
        run, gaps = self._read(self._inside(tmp_path), fill_gaps=True)
        assert len(gaps) == 1, gaps
        assert "1624510579.B423 to 1624510592.B423: 3.001 s gap" in gaps[0]
        assert "3000 samples left as NaN" in gaps[0]
        times = pd.DatetimeIndex(run.dataset.time.values)
        assert times.size == 15000
        assert times.is_unique
        assert times[13000] == pd.Timestamp(1624510592, unit="s")
        hx = run.dataset.hx.values
        np.testing.assert_array_equal(hx[:10000], np.arange(10000) - 1000)
        assert np.isnan(hx[10000:13000]).all()
        np.testing.assert_array_equal(hx[13000:], np.arange(2000) - 1000)

    def test_gap_filled_after_files_that_join(self, tmp_path):
        """The fill warning names the file the gap follows, not the first of a chain"""
        a = write_b423(tmp_path / "1624510579.B423", epoch=1624510579, n=2000)
        b = write_b423(tmp_path / "1624510581.B423", epoch=1624510581, n=2000)
        c = write_b423(tmp_path / "1624510586.B423", epoch=1624510586, n=2000)
        run, gaps = self._read([a, b, c], fill_gaps=True)
        assert len(gaps) == 1, gaps
        assert "1624510581.B423 to 1624510586.B423: 3.001 s gap" in gaps[0]
        assert "3000 samples left as NaN" in gaps[0]
        times = pd.DatetimeIndex(run.dataset.time.values)
        assert times.size == 9000
        assert times[7000] == pd.Timestamp(1624510586, unit="s")
