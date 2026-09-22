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


def write_rsp(path: Path) -> Path:
    """A normalized coil response table, amplitude about 1 in the pass band"""
    freqs = np.logspace(-3, 3, 25)
    amps = freqs / np.sqrt(freqs**2 + 0.01)
    phases = np.degrees(np.arctan2(0.1, freqs))
    rows = "\n".join(f"{f:.6e} {a:.6e} {p:.6e}" for f, a, p in zip(freqs, amps, phases))
    path.write_text("B\nfreq amp phas\n" + rows + "\n")
    return path


class TestLEMI120Chain:
    def test_sensitivity_note(self):
        from mt_io.uoa.pr624 import create_lemi120_dc_gain_filter

        f = create_lemi120_dc_gain_filter("hx")
        assert f.gain == 400000.0
        assert "400 mV/nT" in f.comments.value

    def _warnings(self, action):
        from loguru import logger

        messages = []
        handler = logger.add(lambda m: messages.append(str(m)), level="WARNING")
        try:
            result = action()
        finally:
            logger.remove(handler)
        return result, messages

    def test_default_chain_is_named(self, tmp_path):
        """No sensor_type at 1000 Hz: fluxgate chain, and two warnings say so"""
        write_edl(tmp_path, ["240101000000"], n_samples=100)
        files = sorted(tmp_path.glob("TEST01_*"))
        run, messages = self._warnings(lambda: read(files, sample_rate=1000.0))
        names = [f.name for f in run.hx.channel_metadata.filters]
        assert names == ["uoa_bartington_hx"]
        assert any("sensor_type not given" in m for m in messages)
        assert any("1000 Hz read with the Bartington fluxgate chain" in m for m in messages)

        _, messages = self._warnings(
            lambda: read(files, sample_rate=10.0, sensor_type="bartington")
        )
        assert not any("sensor_type" in m or "fluxgate chain" in m for m in messages)

    def test_lemi120_chain(self, tmp_path):
        write_edl(tmp_path, ["240101000000"], n_samples=100)
        rsp = write_rsp(tmp_path / "l120n.rsp")
        run = read(
            sorted(tmp_path.glob("TEST01_*")),
            sample_rate=1000.0,
            sensor_type="lemi120",
            **{f"calibration_fn_{c}": rsp for c in ("bx", "by", "bz")},
        )
        response = run.hx.channel_response
        assert response.names == ["lemi_120_hx_response", "lemi120_dc_gain_hx"]
        assert response.units_in == "nanoTesla"
        assert response.units_out == "microVolt"

    def test_unreadable_coil_file_raises(self, tmp_path):
        write_edl(tmp_path, ["240101000000"], n_samples=100)
        bad = tmp_path / "broken.rsp"
        bad.write_text("B\nfreq amp phas\nnot a number\n")
        with pytest.raises(ValueError, match="hx coil response"):
            read(
                sorted(tmp_path.glob("TEST01_*")),
                sample_rate=1000.0,
                sensor_type="lemi120",
                calibration_fn_bx=bad,
                calibration_fn_by=bad,
                calibration_fn_bz=bad,
            )


class TestChannelGain:
    def _run(self, tmp_path, **kwargs):
        write_edl(tmp_path, ["240101000000"])
        return read(sorted(tmp_path.glob("TEST01_*")), sensor_type="bartington", **kwargs)

    def test_default_chain_unchanged(self, tmp_path):
        run = self._run(tmp_path)
        assert run.ex.channel_response.names == [
            "uoa_dipole_ex_50.0m",
            "uoa_efield_terminal_box_gain",
        ]
        assert run.ex.channel_response.filters_list[-1].gain == 10.0
        assert run.hx.channel_response.names == ["uoa_bartington_hx"]

    def test_declared_gain_reads_smaller(self, tmp_path):
        """ex declared at x100 instead of the x10 box calibrates 10 times smaller"""
        default = self._run(tmp_path / "a")
        run = self._run(tmp_path / "b", channel_gain={"EX": 100.0, "hx": 10.0})
        assert run.ex.channel_response.names == ["uoa_dipole_ex_50.0m", "uoa_gain_ex_x100"]
        assert run.ey.channel_response.names == default.ey.channel_response.names
        assert run.hx.channel_response.names == ["uoa_bartington_hx", "uoa_gain_hx_x10"]
        f = np.array([0.1, 1.0])
        for comp in ("ex", "hx"):
            ratio = np.abs(
                getattr(run, comp).channel_response.complex_response(f)
                / getattr(default, comp).channel_response.complex_response(f)
            )
            np.testing.assert_allclose(ratio, 10.0)
        np.testing.assert_array_equal(run.ex.ts, default.ex.ts)

    def test_efield_gain_still_sets_the_box(self, tmp_path):
        run = self._run(tmp_path, efield_gain=1.0)
        assert run.ex.channel_response.names[-1] == "uoa_efield_terminal_box_gain"
        assert run.ex.channel_response.filters_list[-1].gain == 1.0

    def test_unknown_channel(self, tmp_path):
        with pytest.raises(ValueError, match="channel_gain names 'tp'"):
            self._run(tmp_path, channel_gain={"tp": 10.0})


def test_recorder_ini_flags_kept(tmp_path):
    from mt_io.uoa import UoACollection

    (tmp_path / "config").mkdir()
    lines = ["[recorder]", "station_long_identifier=TEST01_"]
    for n, ident in enumerate(["BX", "BY", "BZ", "EX", "EY", "TP"]):
        lines += [
            f"channel_{n}_samplerate=10",
            f"channel_{n}_long_id={ident}",
            f"channel_{n}_high_gain={int(ident in ('EX', 'EY'))}",
        ]
    (tmp_path / "config" / "recorder.ini").write_text("\n".join(lines) + "\n")
    info = UoACollection(tmp_path).read_recorder_ini()
    assert info["channel_high_gain"] == {0: "0", 1: "0", 2: "0", 3: "1", 4: "1", 5: "0"}
    assert info["channel_long_id"][3] == "EX"
    assert info["high_gain"] is True
    assert info["sample_rate"] == 10.0
