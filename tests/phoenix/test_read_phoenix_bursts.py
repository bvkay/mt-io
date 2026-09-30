# -*- coding: utf-8 -*-
"""
Tests for reading every burst of a segmented Phoenix file (td_24k).

The files are written here in the MTU-5C/8A layout: a 128 byte header, then
per burst a 32 byte sub-header (GPS stamp, sample count, counts, min, max,
mean) followed by its float32 samples.
"""

import json
import struct
from pathlib import Path

import numpy as np
import pytest
from mt_metadata.common.mttime import MTime

from mt_io.phoenix import open_phoenix, PhoenixCollection, read_phoenix
from mt_io.phoenix.readers.segmented.decimated_segmented_reader import (
    DecimatedSegmentCollection,
    Segment,
)

RECORDING_ID = 1619493876
FIRST_BURST = 1619493900
N_SAMPLES = 480


def burst_values(k: int, n: int = N_SAMPLES) -> np.ndarray:
    """The samples written for burst `k`, different in every burst."""
    return np.arange(n, dtype=np.float32) + 1000.0 * (k + 1)


def phoenix_header(sample_rate: int, channel: int = 0) -> bytes:
    """The 128 byte header of a file at `sample_rate` Hz."""
    header = bytearray(128)
    struct.pack_into("<BBH", header, 0, 2, 2, 128)
    header[4:12] = b"MTU-5C\x00\x00"
    header[12:20] = b"10128\x00\x00\x00"
    struct.pack_into("<IBIH", header, 20, RECORDING_ID, channel, 1, 360)
    header[52] = 0x08  # electric channel
    struct.pack_into("<HbBIH", header, 59, sample_rate, 0, 4, 64, 0)
    struct.pack_into("<fff", header, 71, 139.0, -29.7, 120.0)
    return bytes(header)


def write_td24k(path: Path, stamps: list, n: int = N_SAMPLES, channel: int = 0):
    """Write a td_24k file with one burst of `n` samples per GPS stamp."""
    body = b""
    for k, stamp in enumerate(stamps):
        values = burst_values(k, n)
        sub = struct.pack(
            "<IIHHfff", stamp, n, 0, 0, values.min(), values.max(), values.mean()
        )
        body += sub.ljust(32, b"\x00") + values.tobytes()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(phoenix_header(24000, channel) + body)
    return path


def write_td150(path: Path, n: int = 1500, channel: int = 0) -> Path:
    """Write a continuous td_150 file of `n` samples."""
    path.parent.mkdir(parents=True, exist_ok=True)
    samples = np.arange(n, dtype=np.float32)
    path.write_bytes(phoenix_header(150, channel) + samples.tobytes())
    return path


def write_station(folder: Path, station: str = "MT001") -> Path:
    """Write the recmeta.json a station folder needs."""
    folder.mkdir(parents=True, exist_ok=True)
    channel = {
        "ty": "E",
        "ga": 1,
        "sampleRate": 24000,
        "lp": 10000,
        "length1": 50,
        "length2": 50,
        "pot_p": 1,
        "pot_n": 1,
    }
    recmeta = {
        "instid": "10128",
        "start": "2021-04-27T03:24:36+00:00",
        "stop": "2021-04-27T04:24:36+00:00",
        "receiver_commercial_name": "MTU-5C",
        "receiver_model": "MTU-5C",
        "motherboard": {"mb_fw_ver": "1"},
        "layout": {
            "Station_Name": station,
            "Survey_Name": "TEST",
            "Company_Name": "",
            "Notes": "",
            "Operator": "",
        },
        "channel_map": {"mapping": [{"idx": 0, "tag": "E1"}, {"idx": 7, "tag": "E2"}]},
        "chconfig": {"chans": [dict(channel, tag="E1"), dict(channel, tag="E2")]},
        "timing": {
            "gps_lat": -29.7,
            "gps_lon": 139.0,
            "gps_alt": 120.0,
            "tm_drift": 0.0,
        },
    }
    (folder / "recmeta.json").write_text(json.dumps(recmeta))
    return folder


@pytest.fixture
def two_bursts(tmp_path):
    """A station with one td_24k file of two bursts 30 s apart."""
    station = write_station(tmp_path / "10128_2021-04-27-032436")
    return write_td24k(
        station / "0" / f"10128_{RECORDING_ID:08X}_0_00000001.td_24k",
        [FIRST_BURST, FIRST_BURST + 30],
    )


class TestEveryBurst:
    def test_read_phoenix_returns_every_burst(self, two_bursts):
        bursts = read_phoenix(two_bursts)
        assert isinstance(bursts, list)
        assert len(bursts) == 2
        for k, ch in enumerate(bursts):
            assert ch.n_samples == N_SAMPLES
            assert ch.sample_rate == 24000
            np.testing.assert_array_equal(ch.ts, burst_values(k))
            assert ch.start == MTime(time_stamp=FIRST_BURST + 30 * k, gps_time=True)
        assert bursts[1].start - bursts[0].start == 30

    def test_to_channel_ts_twice(self, two_bursts):
        """The walk starts at the first burst each time"""
        reader = open_phoenix(two_bursts)
        first = reader.to_channel_ts()
        second = reader.to_channel_ts()
        assert [c.start for c in first] == [c.start for c in second]
        assert len(second) == 2
        collection = DecimatedSegmentCollection(two_bursts)
        assert len(collection.to_channel_ts()) == 2
        assert len(collection.to_channel_ts()) == 2

    def test_metadata_only_steps_over_the_samples(self, two_bursts):
        with open(two_bursts, "rb") as stream:
            stream.seek(128)
            segment = Segment(stream)
            segment.read_segment(metadata_only=True)
            assert segment.data is None
            assert stream.tell() == 128 + 32 + 4 * N_SAMPLES

        for reader in (
            open_phoenix(two_bursts),
            DecimatedSegmentCollection(two_bursts),
        ):
            segments = reader.read_segments(metadata_only=True)
            assert [s.gps_time_stamp for s in segments] == [
                MTime(time_stamp=FIRST_BURST + 30 * k, gps_time=True) for k in (0, 1)
            ]
            assert [s.n_samples for s in segments] == [N_SAMPLES, N_SAMPLES]
            assert [s.segment for s in segments] == [0, 1]
            assert all(s.data is None for s in segments)

    def test_short_last_burst_warns(self, two_bursts):
        """A file cut inside a burst gives what is there, with a warning"""
        from loguru import logger

        raw = two_bursts.read_bytes()
        two_bursts.write_bytes(raw[: len(raw) - 4 * 100])
        messages = []
        handler = logger.add(lambda m: messages.append(str(m)), level="WARNING")
        try:
            bursts = read_phoenix(two_bursts)
        finally:
            logger.remove(handler)
        assert [c.n_samples for c in bursts] == [N_SAMPLES, N_SAMPLES - 100]
        assert any("segment 1 holds 380 of 480 samples" in m for m in messages)

    def test_partial_sub_header_at_the_end(self, two_bursts):
        """Bytes after the last burst too few for a sub-header end the walk"""
        two_bursts.write_bytes(two_bursts.read_bytes() + b"\x01" * 10)
        bursts = read_phoenix(two_bursts)
        assert len(bursts) == 2
        for k, ch in enumerate(bursts):
            np.testing.assert_array_equal(ch.ts, burst_values(k))
        for reader in (
            open_phoenix(two_bursts),
            DecimatedSegmentCollection(two_bursts),
        ):
            assert len(reader.read_segments(metadata_only=True)) == 2


def burst_file(station: Path, channel: int, sequence: int) -> Path:
    """Path of a td_24k file in a station folder."""
    return (
        station
        / str(channel)
        / f"10128_{RECORDING_ID:08X}_{channel}_{sequence:08X}.td_24k"
    )


class TestCollectionBursts:
    def test_one_run_per_burst(self, tmp_path):
        """Every burst of every file is a run, with its file and position"""
        station = write_station(tmp_path / "10128_2021-04-27-032436")
        write_td24k(burst_file(station, 0, 1), [FIRST_BURST, FIRST_BURST + 30])
        write_td24k(burst_file(station, 0, 2), [FIRST_BURST + 60])
        collection = PhoenixCollection(tmp_path)
        assert len(collection.to_dataframe([24000])) == 3
        runs = collection.get_runs([24000])["MT001"]
        assert list(runs) == ["sr24k_0001", "sr24k_0002", "sr24k_0003"]
        for k, run_df in enumerate(runs.values()):
            stamp = MTime(time_stamp=FIRST_BURST + 30 * k, gps_time=True)
            assert list(run_df.component) == ["e1"]
            assert list(run_df.n_samples) == [N_SAMPLES]
            for row in run_df.itertuples():
                assert MTime(time_stamp=row.start) == stamp
                ch = read_phoenix(row.fn)[row.segment]
                assert ch.start == stamp
                np.testing.assert_array_equal(ch.ts, burst_values(row.segment))

    def test_continuous_and_segmented_files(self, tmp_path):
        """A td_150 row leaves the segment column an integer one"""
        station = write_station(tmp_path / "10128_2021-04-27-032436")
        write_td150(station / "0" / f"10128_{RECORDING_ID:08X}_0_00000001.td_150")
        write_td24k(burst_file(station, 0, 1), [FIRST_BURST, FIRST_BURST + 30])
        collection = PhoenixCollection(tmp_path)
        runs = collection.get_runs([150, 24000])["MT001"]
        assert sorted(runs) == ["sr150_0001", "sr24k_0001", "sr24k_0002"]
        for k, run_id in enumerate(("sr24k_0001", "sr24k_0002")):
            stamp = MTime(time_stamp=FIRST_BURST + 30 * k, gps_time=True)
            for row in runs[run_id].itertuples():
                ch = read_phoenix(row.fn)[row.segment]
                assert ch.start == stamp
                np.testing.assert_array_equal(ch.ts, burst_values(k))
        df = collection.to_dataframe([150, 24000])
        assert str(df.segment.dtype) == "Int64"
        assert df[df.sample_rate == 150].segment.isna().all()

    def test_stations_sharing_burst_stamps(self, tmp_path):
        """Run names are given per station, as two stations share slots"""
        for name, stamps in (
            ("MT001", [FIRST_BURST, FIRST_BURST + 30]),
            ("MT002", [FIRST_BURST + 30, FIRST_BURST + 60]),
        ):
            station = write_station(tmp_path / name, station=name)
            for sequence, stamp in enumerate(stamps, 1):
                write_td24k(burst_file(station, 0, sequence), [stamp])
        runs = PhoenixCollection(tmp_path).get_runs([24000])
        for name, first in (("MT001", FIRST_BURST), ("MT002", FIRST_BURST + 30)):
            assert list(runs[name]) == ["sr24k_0001", "sr24k_0002"]
            for k, run_df in enumerate(runs[name].values()):
                assert len(run_df) == 1
                start = MTime(time_stamp=run_df.start.iloc[0])
                assert start == MTime(time_stamp=first + 30 * k, gps_time=True)
