# -*- coding: utf-8 -*-
"""
Module to read and parse native Phoenix Geophysics data formats of the
MTU-5C Family

This module implements Streamed readers for segmented-decimated  time series
 formats of the MTU-5C family.

:author: Jorge Torres-Solis

Revised 2022 by J. Peacock
"""

from __future__ import annotations

from pathlib import Path
from struct import unpack_from
from typing import Any, BinaryIO

# =============================================================================
# Imports
# =============================================================================
import numpy as np
from mt_metadata.common.mttime import MTime
from mt_timeseries import ChannelTS

from mt_io.phoenix.readers import TSReaderBase


# =============================================================================
class SubHeader:
    """
    Class for subheader of segmented files.

    This class handles the parsing and access to subheader information in
    Phoenix Geophysics segmented time series files. The subheader contains
    metadata about each segment including timing, sample counts, and statistics.

    Parameters
    ----------
    **kwargs
        Arbitrary keyword arguments that are set as attributes

    Attributes
    ----------
    header_length : int
        Length of the subheader in bytes (32 bytes)
    _header : bytes or None
        Raw header bytes from the file
    _unpack_dict : dict
        Dictionary defining how to unpack different header fields
    """

    def __init__(self, **kwargs) -> None:
        self.header_length = 32
        self._header = None

        for key, value in kwargs.items():
            setattr(self, key, value)
        self._unpack_dict = {
            "gps_time_stamp": {"dtype": "I", "index": 0},
            "n_samples": {"dtype": "I", "index": 4},
            "saturation_count": {"dtype": "H", "index": 8},
            "missing_count": {"dtype": "H", "index": 10},
            "value_min": {"dtype": "f", "index": 12},
            "value_max": {"dtype": "f", "index": 16},
            "value_mean": {"dtype": "f", "index": 20},
        }

    def __str__(self) -> str:
        """String representation of the subheader information."""
        lines = ["subheader information:", "-" * 30]
        for key in [
            "gps_time_stamp",
            "n_samples",
            "saturation_count",
            "missing_count",
            "value_min",
            "value_max",
            "value_mean",
        ]:
            lines.append(f"\t{key:<25}: {getattr(self, key)}")
        return "\n".join(lines)

    def __repr__(self) -> str:
        """String representation of the subheader."""
        return self.__str__()

    def _has_header(self) -> bool:
        """
        Check if header data has been loaded.

        Returns
        -------
        bool
            True if header is loaded, False otherwise
        """
        if self._header is not None:
            return True
        return False

    def _unpack_value(self, key: str) -> tuple[Any, ...] | None:
        """
        Unpack a value from the header bytes.

        Parameters
        ----------
        key : str
            Key name for the value to unpack

        Returns
        -------
        tuple or None
            Unpacked value tuple, or None if header not available

        Raises
        ------
        IOError
            If unpacking fails
        """
        if self._has_header() and self._header is not None:
            try:
                return unpack_from(
                    self._unpack_dict[key]["dtype"],
                    self._header,
                    self._unpack_dict[key]["index"],
                )
            except Exception as error:
                raise IOError(error)
        return None

    @property
    def gps_time_stamp(self) -> MTime | None:
        """
        GPS time stamp in UTC.

        Returns
        -------
        MTime or None
            GPS timestamp if header is available, None otherwise
        """
        if self._has_header():
            value = self._unpack_value("gps_time_stamp")
            if value is not None:
                return MTime(time_stamp=value[0], gps_time=True)
        return None

    @property
    def n_samples(self) -> int | None:
        """
        Number of samples in the segment.

        Returns
        -------
        int or None
            Number of samples if header is available, None otherwise
        """
        if self._has_header():
            value = self._unpack_value("n_samples")
            if value is not None:
                return value[0]
        return None

    @property
    def saturation_count(self) -> int | None:
        """
        Number of saturated samples.

        Returns
        -------
        int or None
            Saturation count if header is available, None otherwise
        """
        if self._has_header():
            value = self._unpack_value("saturation_count")
            if value is not None:
                return value[0]
        return None

    @property
    def missing_count(self) -> int | None:
        """
        Number of missing samples.

        Returns
        -------
        int or None
            Missing sample count if header is available, None otherwise
        """
        if self._has_header():
            value = self._unpack_value("missing_count")
            if value is not None:
                return value[0]
        return None

    @property
    def value_min(self) -> float | None:
        """
        Minimum value in the segment.

        Returns
        -------
        float or None
            Minimum value if header is available, None otherwise
        """
        if self._has_header():
            value = self._unpack_value("value_min")
            if value is not None:
                return value[0]
        return None

    @property
    def value_max(self) -> float | None:
        """
        Maximum value in the segment.

        Returns
        -------
        float or None
            Maximum value if header is available, None otherwise
        """
        if self._has_header():
            value = self._unpack_value("value_max")
            if value is not None:
                return value[0]
        return None

    @property
    def value_mean(self) -> float | None:
        """
        Mean value in the segment.

        Returns
        -------
        float or None
            Mean value if header is available, None otherwise
        """
        if self._has_header():
            value = self._unpack_value("value_mean")
            if value is not None:
                return value[0]
        return None

    def unpack_header(self, stream: BinaryIO) -> None:
        """
        Unpack the header from a binary stream.

        Parameters
        ----------
        stream : BinaryIO
            Binary stream to read header from
        """
        if self.header_length > 0:
            # be sure to read from the beginning of the file
            self._header = stream.read(self.header_length)
        else:
            return


class Segment(SubHeader):
    """
    A segment class to hold a single segment.

    This class represents a single time series segment with its associated
    metadata and data. It inherits from SubHeader to provide access to
    segment-specific header information.

    Parameters
    ----------
    stream : BinaryIO
        Binary file stream to read from
    **kwargs
        Additional keyword arguments passed to SubHeader

    Attributes
    ----------
    stream : BinaryIO
        The file stream for reading data
    data : np.ndarray or None
        Time series data for this segment
    """

    def __init__(self, stream: BinaryIO, **kwargs) -> None:
        super().__init__(**kwargs)
        self.stream = stream
        self.data: np.ndarray | None = None

    def read_segment(self, metadata_only: bool = False) -> None:
        """
        Read the segment data from the file stream.

        With ``metadata_only`` the samples are stepped over, so the stream
        is left at the next sub-header either way. At the end of the file,
        where no whole sub-header is left, the segment has no header.

        Parameters
        ----------
        metadata_only : bool, optional
            If True, only read metadata without loading data, by default False
        """
        self.unpack_header(self.stream)
        if self._header is None or len(self._header) < self.header_length:
            self._header = None
            return
        if metadata_only:
            self.stream.seek(4 * self.n_samples, 1)
        else:
            self.data = np.fromfile(self.stream, dtype=np.float32, count=self.n_samples)

    @property
    def segment_start_time(self) -> MTime | None:
        """
        Get the segment start time.

        Returns
        -------
        MTime or None
            GPS timestamp of segment start, or None if not available
        """
        return self.gps_time_stamp

    @property
    def segment_end_time(self) -> MTime | None:
        """
        Calculate the segment end time.

        Returns
        -------
        MTime or None
            Estimated end time based on start time, sample count and sample rate,
            or None if required information is not available
        """
        start_time = self.segment_start_time
        if (
            start_time is not None
            and self.n_samples is not None
            and hasattr(self, "sample_rate")
        ):
            return start_time + (self.n_samples / self.sample_rate)
        return None


def _read_segments(
    reader: DecimatedSegmentedReader | DecimatedSegmentCollection,
    metadata_only: bool = False,
) -> list[Segment]:
    """
    Read every segment of a segmented file, from the first sub-header.

    The walk starts after the file header each time, so it can be repeated
    on the same reader, and stops where no whole sub-header is left.

    Parameters
    ----------
    reader : DecimatedSegmentedReader or DecimatedSegmentCollection
        Reader with an open stream
    metadata_only : bool, optional
        If True, read the sub-headers and step over the samples, by default
        False

    Returns
    -------
    list[Segment]
        One Segment per burst, in file order; ``segment`` is its position,
        0 for the first

    Raises
    ------
    ValueError
        If stream is not available
    """
    if reader.stream is None:
        raise ValueError("Stream is not available")
    kwargs = {
        "instrument_type": reader.instrument_type,
        "instrument_serial_number": reader.instrument_serial_number,
        "latitude": reader.gps_lat,
        "longitude": reader.gps_long,
        "elevation": reader.gps_elevation,
        "sample_rate": reader.sample_rate,
        "channel_id": reader.channel_id,
        "channel_type": reader.channel_type,
    }
    reader.stream.seek(reader.header_length)
    segments = []
    while True:
        segment = Segment(reader.stream, segment=len(segments), **kwargs)
        segment.read_segment(metadata_only=metadata_only)
        if not segment._has_header():
            break
        if segment.data is not None and segment.data.size < segment.n_samples:
            reader.logger.warning(
                f"{reader.file_name}: segment {segment.segment} holds "
                f"{segment.data.size} of {segment.n_samples} samples, the file "
                "ends part way through it"
            )
        segments.append(segment)
    reader.logger.debug(f"Read {len(segments)} segments from {reader.file_name}")
    return segments


def _segments_to_channel_ts(
    reader: DecimatedSegmentedReader | DecimatedSegmentCollection,
    rxcal_fn: str | Path | None = None,
    scal_fn: str | Path | None = None,
) -> list[ChannelTS]:
    """
    Make one ChannelTS per segment, each starting at its own GPS stamp.

    Parameters
    ----------
    reader : DecimatedSegmentedReader or DecimatedSegmentCollection
        Reader with an open stream
    rxcal_fn : str, Path or None, optional
        Path to receiver calibration file, by default None
    scal_fn : str, Path or None, optional
        Path to sensor calibration file, by default None

    Returns
    -------
    list[ChannelTS]
        One ChannelTS per segment, in file order
    """
    seq_list = []
    for segment in _read_segments(reader):
        ch_metadata = reader._update_channel_metadata_from_recmeta()
        if segment.segment_start_time is not None:
            ch_metadata.time_period.start = segment.segment_start_time.isoformat()
        if segment.segment_end_time is not None:
            ch_metadata.time_period.end = segment.segment_end_time.isoformat()

        seq_list.append(
            ChannelTS(
                channel_type=ch_metadata.type,
                data=segment.data,
                channel_metadata=ch_metadata,
                run_metadata=reader.run_metadata,
                station_metadata=reader.station_metadata,
                channel_response=reader.get_channel_response(
                    rxcal_fn=rxcal_fn, scal_fn=scal_fn
                ),
            )
        )
    return seq_list


class DecimatedSegmentedReader(TSReaderBase):
    """
    Class to create a streamer for segmented decimated time series.

    This reader handles segmented decimated time series files such as 'td_24k'.
    These files hold a burst per segment, each with a sub header giving its
    GPS stamp and sample count; ``to_channel_ts`` returns every burst.

    Parameters
    ----------
    path : str or Path
        Path to the time series file
    num_files : int, optional
        Number of files in the sequence, by default 1
    report_hw_sat : bool, optional
        Whether to report hardware saturation, by default False
    **kwargs
        Additional keyword arguments passed to parent TSReaderBase class

    Attributes
    ----------
    sub_header : SubHeader
        SubHeader instance for parsing segment headers
    subheader : dict
        Dictionary for additional subheader information
    """

    def __init__(
        self,
        path: str | Path,
        num_files: int = 1,
        report_hw_sat: bool = False,
        **kwargs,
    ) -> None:
        # Init the base class
        super().__init__(
            path,
            num_files=num_files,
            header_length=128,
            report_hw_sat=report_hw_sat,
            **kwargs,
        )

        self._channel_metadata = self._update_channel_metadata_from_recmeta()
        self.sub_header = SubHeader()
        self.subheader = {}

    def read_segment(self, metadata_only: bool = False) -> Segment:
        """
        Read a single segment from the file.

        Parameters
        ----------
        metadata_only : bool, optional
            If True, only read metadata without loading data, by default False

        Returns
        -------
        Segment
            Segment object containing data and metadata

        Raises
        ------
        ValueError
            If stream is not available
        """
        kwargs = {
            "instrument_type": self.instrument_type,
            "instrument_serial_number": self.instrument_serial_number,
            "latitude": self.gps_lat,
            "longitude": self.gps_long,
            "elevation": self.gps_elevation,
            "sample_rate": self.sample_rate,
            "channel_id": self.channel_id,
            "channel_type": self.channel_type,
            "segment": 0,
        }

        if self.stream is None:
            raise ValueError("Stream is not available")

        segment = Segment(self.stream, **kwargs)
        segment.read_segment(metadata_only=metadata_only)

        return segment

    def read_segments(self, metadata_only: bool = False) -> list[Segment]:
        """
        Read every segment of the file, from the first sub-header.

        Parameters
        ----------
        metadata_only : bool, optional
            If True, read the sub-headers and step over the samples, by
            default False

        Returns
        -------
        list[Segment]
            One Segment per burst, in file order
        """
        return _read_segments(self, metadata_only=metadata_only)

    def to_channel_ts(
        self, rxcal_fn: str | Path | None = None, scal_fn: str | Path | None = None
    ) -> list[ChannelTS]:
        """
        Convert every segment of the file to a ChannelTS.

        A td_24k file holds a burst per segment (12 bursts of 2 s, 30 s
        apart, in a 6 minute MTU-8A file), so each becomes its own
        ChannelTS starting at its GPS stamp.

        Parameters
        ----------
        rxcal_fn : str, Path or None, optional
            Path to receiver calibration file, by default None
        scal_fn : str, Path or None, optional
            Path to sensor calibration file, by default None

        Returns
        -------
        list[ChannelTS]
            One ChannelTS per segment, in file order
        """
        return _segments_to_channel_ts(self, rxcal_fn=rxcal_fn, scal_fn=scal_fn)


class DecimatedSegmentCollection(TSReaderBase):
    """
    Class to read multiple segments from a segmented decimated time series file.

    This reader handles files containing multiple segments of decimated time series
    data such as 'td_24k'. Each segment has its own sub header with metadata.

    Parameters
    ----------
    path : str or Path
        Path to the time series file
    num_files : int, optional
        Number of files in the sequence, by default 1
    report_hw_sat : bool, optional
        Whether to report hardware saturation, by default False
    **kwargs
        Additional keyword arguments passed to parent TSReaderBase class

    Attributes
    ----------
    sub_header : SubHeader
        SubHeader instance for parsing segment headers
    subheader : dict
        Dictionary for additional subheader information
    """

    def __init__(
        self,
        path: str | Path,
        num_files: int = 1,
        report_hw_sat: bool = False,
        **kwargs,
    ) -> None:
        # Init the base class
        super().__init__(
            path,
            num_files=num_files,
            header_length=128,
            report_hw_sat=report_hw_sat,
            **kwargs,
        )

        if self.stream is not None:
            self.unpack_header(self.stream)
        self.sub_header = SubHeader()
        self.subheader = {}

    def read_segments(self, metadata_only: bool = False) -> list[Segment]:
        """
        Read all segments from the file, from the first sub-header.

        Parameters
        ----------
        metadata_only : bool, optional
            If True, read the sub-headers and step over the samples, by
            default False

        Returns
        -------
        list[Segment]
            List of Segment objects containing data and metadata

        Raises
        ------
        ValueError
            If stream is not available
        """
        return _read_segments(self, metadata_only=metadata_only)

    def to_channel_ts(
        self, rxcal_fn: str | Path | None = None, scal_fn: str | Path | None = None
    ) -> list[ChannelTS]:
        """
        Convert all segments to ChannelTS objects.

        Parameters
        ----------
        rxcal_fn : str, Path or None, optional
            Path to receiver calibration file, by default None
        scal_fn : str, Path or None, optional
            Path to sensor calibration file, by default None

        Returns
        -------
        list[ChannelTS]
            List of ChannelTS objects, one for each segment
        """
        return _segments_to_channel_ts(self, rxcal_fn=rxcal_fn, scal_fn=scal_fn)
