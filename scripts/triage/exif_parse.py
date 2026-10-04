#!/usr/bin/env python3
"""Pure-python parsers for the metadata that can hide in a MyceliumSeg JPEG header.

Everything here is I/O free and stdlib only: the scan in ``exif_scan.py`` feeds
these helpers the leading bytes of a JPEG (read straight out of a zip entry) or
the bytes of a macOS ``AppleDouble`` sidecar and gets plain dicts back.

Covered sources of capture metadata:
  * EXIF  (APP1, TIFF/IFD0 + Exif IFD + GPS IFD + IFD1)
  * XMP   (APP1 ``http://ns.adobe.com/xap/1.0/``, APP13 ``Adobe_CM`` XMP)
  * IPTC  (APP13 Photoshop IRB, 8BIM 0x0404 IIM records)
  * JPEG  COM comments
  * JFIF  APP0 density fields
  * AppleDouble extended attributes (``com.apple.quarantine`` -> download time)

Every helper is unit-tested in ``tests/test_exif_parse.py``.
"""

from __future__ import annotations

import re
import struct
import zlib
from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from itertools import pairwise

# --------------------------------------------------------------------------- #
# JPEG markers
# --------------------------------------------------------------------------- #

SOI = 0xD8
EOI = 0xD9
SOS = 0xDA
COM = 0xFE
APP0 = 0xE0
APP1 = 0xE1
APP2 = 0xE2
APP13 = 0xED

_STANDALONE_MARKERS = frozenset({0x01, SOI, EOI} | set(range(0xD0, 0xD8)))

MARKER_NAMES = {
    0x01: "TEM",
    SOI: "SOI",
    0xC0: "SOF0",
    0xC1: "SOF1",
    0xC2: "SOF2",
    0xC3: "SOF3",
    0xC5: "SOF5",
    0xC6: "SOF6",
    0xC7: "SOF7",
    0xC9: "SOF9",
    0xCA: "SOF10",
    0xCB: "SOF11",
    0xCD: "SOF13",
    0xCE: "SOF14",
    0xCF: "SOF15",
    0xC4: "DHT",
    0xC8: "JPG",
    0xCC: "DAC",
    0xD9: "EOI",
    0xDA: "SOS",
    0xDB: "DQT",
    0xDC: "DNL",
    0xDD: "DRI",
    0xDE: "DHP",
    0xDF: "EXP",
    0xFE: "COM",
}
MARKER_NAMES.update({APP0 + i: f"APP{i}" for i in range(16)})

SOF_MARKERS = frozenset({0xC0, 0xC1, 0xC2, 0xC3, 0xC5, 0xC6, 0xC7, 0xC9,
                         0xCA, 0xCB, 0xCD, 0xCE, 0xCF})

EXIF_SIGNATURE = b"Exif\x00\x00"
XMP_SIGNATURES = (
    b"http://ns.adobe.com/xap/1.0/\x00",
    b"http://ns.adobe.com/xmp/extension/\x00",
    b"http://ns.adobe.com/exif/1.0/\x00",
)
PHOTOSHOP_IRB_SIGNATURE = b"Photoshop 3.0\x00"
JFIF_SIGNATURE = b"JFIF\x00"


def marker_name(marker: int) -> str:
    """Human readable name for a JPEG marker byte (``0xC4`` -> ``"DHT"``)."""
    return MARKER_NAMES.get(marker, f"0x{marker:02X}")


class JpegParseError(ValueError):
    """The byte stream is not a well formed JPEG header."""


class NotAJpegError(JpegParseError):
    """The bytes do not start with an SOI marker."""


class JpegTruncated(JpegParseError):
    """The header continues past the end of the supplied bytes."""


# --------------------------------------------------------------------------- #
# small text helpers
# --------------------------------------------------------------------------- #

_CONTROL = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")


def clean_text(raw: bytes | str, limit: int = 200) -> str:
    """Decode metadata bytes into a single-line, CSV-safe string.

    Non-printable bytes become spaces, whitespace runs collapse, and anything
    longer than ``limit`` characters is cut with an explicit ``...`` marker so a
    truncated value is never mistaken for a complete one.
    """
    if isinstance(raw, bytes):
        text = raw.decode("utf-8", errors="replace")
    else:
        text = raw
    text = text.lstrip("﻿")
    text = _CONTROL.sub(" ", text).replace("\r", " ").replace("\n", " ")
    text = re.sub(r"\s+", " ", text).strip()
    if len(text) > limit:
        text = text[: max(limit - 3, 0)] + "..."
    return text


def binary_preview(raw: bytes, head: int = 24) -> str:
    """Describe an undecodable value as ``<binary N bytes> hexhead``."""
    return f"<binary {len(raw)} bytes> {raw[:head].hex()}"


def median(values: Sequence[float]) -> float | None:
    """Median of a sequence, or ``None`` when empty (even-length safe)."""
    if not values:
        return None
    ordered = sorted(values)
    mid = len(ordered) // 2
    if len(ordered) % 2:
        return float(ordered[mid])
    return (ordered[mid - 1] + ordered[mid]) / 2.0


def hex_epoch_to_datetime(value: str) -> datetime | None:
    """Decode a macOS quarantine hex epoch (``68e5cb7f``) as an aware datetime."""
    try:
        return datetime.fromtimestamp(int(value, 16), tz=timezone.utc)
    except (ValueError, OSError, OverflowError):
        return None


def decode_escaped_utf8(text: str) -> str:
    """Turn ``\\xe7\\x99\\xbe`` style escapes back into the text they encode.

    Baidu Netdisk stores the downloader's name in the quarantine attribute with
    every non-ASCII byte written as a literal ``\\xNN`` escape, so the raw value
    has to be reassembled before it is readable.
    """
    if "\\x" not in text:
        return text
    out = bytearray()
    index = 0
    while index < len(text):
        escape = text[index : index + 4]
        if len(escape) == 4 and escape[0] == "\\" and escape[1] == "x" and \
                all(char in "0123456789abcdefABCDEF" for char in escape[2:]):
            out.append(int(escape.removeprefix("\\x"), 16))
            index += 4
        else:
            out += text[index].encode("utf-8")
            index += 1
    return out.decode("utf-8", errors="replace")


# --------------------------------------------------------------------------- #
# JFIF (APP0)
# --------------------------------------------------------------------------- #


def parse_jfif(payload: bytes) -> dict[str, object]:
    """Decode a JFIF APP0 payload.  Returns ``{}`` for a non-JFIF APP0."""
    if not payload.startswith(JFIF_SIGNATURE) or len(payload) < 14:
        return {}
    version = payload[5:7]
    units = payload[7]
    x_density, y_density = struct.unpack_from(">HH", payload, 8)
    x_thumb, y_thumb = payload[12], payload[13]
    return {
        "version": f"{version[0]}.{version[1]:02d}",
        "units": units,  # 0 = aspect ratio, 1 = dots/inch, 2 = dots/cm
        "x_density": x_density,
        "y_density": y_density,
        "x_thumbnail": x_thumb,
        "y_thumbnail": y_thumb,
    }


# --------------------------------------------------------------------------- #
# EXIF / TIFF
# --------------------------------------------------------------------------- #

TAG_NAMES: dict[int, str] = {
    0x0100: "ImageWidth",
    0x0101: "ImageLength",
    0x0102: "BitsPerSample",
    0x0103: "Compression",
    0x0106: "PhotometricInterpretation",
    0x010E: "ImageDescription",
    0x010F: "Make",
    0x0110: "Model",
    0x0112: "Orientation",
    0x0115: "SamplesPerPixel",
    0x011A: "XResolution",
    0x011B: "YResolution",
    0x0128: "ResolutionUnit",
    0x0131: "Software",
    0x0132: "DateTime",
    0x013B: "Artist",
    0x8298: "Copyright",
    0x829A: "ExposureTime",
    0x829D: "FNumber",
    0x8822: "ExposureProgram",
    0x8827: "ISOSpeedRatings",
    0x9000: "ExifVersion",
    0x9003: "DateTimeOriginal",
    0x9004: "DateTimeDigitized",
    0x9010: "OffsetTime",
    0x9011: "OffsetTimeOriginal",
    0x9012: "OffsetTimeDigitized",
    0x9101: "ComponentsConfiguration",
    0x9201: "ShutterSpeedValue",
    0x9202: "ApertureValue",
    0x9204: "ExposureBiasValue",
    0x9207: "MeteringMode",
    0x9286: "UserComment",
    0x9290: "SubSecTime",
    0x9291: "SubSecTimeOriginal",
    0x9292: "SubSecTimeDigitized",
    0xA000: "FlashpixVersion",
    0xA001: "ColorSpace",
    0xA002: "PixelXDimension",
    0xA003: "PixelYDimension",
    0xA005: "InteroperabilityIFDPointer",
    0xA20E: "FocalPlaneXResolution",
    0xA210: "FocalPlaneYResolution",
    0xA217: "SensingMethod",
    0xA300: "FileSource",
    0xA301: "SceneType",
    0xA402: "ExposureMode",
    0xA403: "WhiteBalance",
    0xA404: "DigitalZoomRatio",
    0xA405: "FocalLengthIn35mmFilm",
    0xA406: "SceneCaptureType",
    0xA430: "CameraOwnerName",
    0xA431: "BodySerialNumber",
    0xA432: "LensSpecification",
    0xA433: "LensMake",
    0xA434: "LensModel",
    0xA435: "LensSerialNumber",
}
_TYPE_SIZES = {1: 1, 2: 1, 3: 2, 4: 4, 5: 8, 6: 1, 7: 1, 8: 2, 9: 4, 10: 8, 11: 4, 12: 8, 13: 4}
_IFD_POINTERS = {0x8769: "ExifIFD", 0x8825: "GPSIFD", 0xA005: "InteropIFD"}


def _decode_value(raw: bytes, typ: int, count: int) -> object:
    """Decode one IFD entry value according to its TIFF field type."""
    # UserComment and friends carry an 8-byte charset code ahead of the text.
    if raw[:8] == b"ASCII\x00\x00\x00":
        return clean_text(raw[8:])
    if raw[:8] == b"JIS\x00\x00\x00\x00\x00":
        return clean_text(raw[8:])
    if raw[:8] == b"UNICODE\x00":
        return clean_text(raw[8:].decode("utf-16-le", errors="replace"))
    if typ == 2:  # ASCII
        return clean_text(raw.split(b"\x00", 1)[0])
    if typ in (1, 6, 7):  # BYTE / SBYTE / UNDEFINED
        if all(32 <= b < 127 or b in (9, 10, 13) for b in raw[:64]):
            return clean_text(raw)
        return binary_preview(raw)
    fmt = {3: "H", 8: "h", 4: "I", 9: "i"}.get(typ)
    if fmt:
        try:
            values = list(struct.unpack(f"<{count}{fmt}", raw[: count * struct.calcsize(fmt)]))
        except struct.error:
            return binary_preview(raw)
        return values[0] if len(values) == 1 else values
    if typ in (5, 10):  # (S)RATIONAL
        fmt = "I" if typ == 5 else "i"
        size = struct.calcsize(fmt)
        out = []
        for i in range(count):
            num, den = struct.unpack_from(f"<{fmt}{fmt}", raw, i * 2 * size)
            out.append(round(num / den, 6) if den else None)
        return out[0] if len(out) == 1 else out
    if typ == 11:  # FLOAT
        out = list(struct.unpack(f"<{count}f", raw[: 4 * count]))
        return out[0] if len(out) == 1 else out
    if typ == 12:  # DOUBLE
        out = list(struct.unpack(f"<{count}d", raw[: 8 * count]))
        return out[0] if len(out) == 1 else out
    return binary_preview(raw)


def _iter_ifd(
    buf: bytes,
    endian: str,
    offset: int,
    prefix: str,
    out: dict[str, object],
    *,
    depth: int = 0,
    max_entries: int = 4096,
) -> None:
    """Walk one TIFF IFD at ``offset`` and merge its tags into ``out``.

    Sub-IFDs (Exif, GPS, Interoperability) are followed recursively, bounded by
    ``depth`` and ``max_entries`` so a malformed pointer cannot loop forever.
    """
    if depth > 3 or offset < 0 or offset + 2 > len(buf):
        return
    (count,) = struct.unpack_from(f"{endian}H", buf, offset)
    count = min(count, max_entries)
    pos = offset + 2
    for _ in range(count):
        if pos + 12 > len(buf):
            return
        tag, typ, n = struct.unpack_from(f"{endian}HHI", buf, pos)
        size = _TYPE_SIZES.get(typ, 0) * n
        if size <= 0 or n > max_entries:
            pos += 12
            continue
        if size <= 4:
            raw = buf[pos + 8 : pos + 8 + size]
        else:
            (data_offset,) = struct.unpack_from(f"{endian}I", buf, pos + 8)
            raw = buf[data_offset : data_offset + size] if data_offset < len(buf) else b""
        pos += 12
        if tag in _IFD_POINTERS and typ in (4, 13) and raw:
            (sub,) = struct.unpack_from(f"{endian}I", raw, 0)
            _iter_ifd(buf, endian, sub, f"{prefix}{_IFD_POINTERS[tag]}.", out,
                      depth=depth + 1, max_entries=max_entries)
            continue
        name = TAG_NAMES.get(tag)
        key = f"{prefix}{name}" if name else f"{prefix}Tag_0x{tag:04X}"
        value = _decode_value(raw, typ, n)
        if key not in out or out[key] in ("", None, binary_preview(b"")):
            out[key] = value


def parse_tiff_exif(payload: bytes) -> dict[str, object]:
    """Parse an EXIF APP1 payload (the TIFF block starting after ``Exif\\x00\\x00``).

    Returns a flat ``{"ExifIFD.DateTimeOriginal": "2024:05:06 07:08:09", ...}``
    mapping.  Unknown tags are kept as ``Tag_0x....`` so vendor-private data is
    visible rather than silently dropped.
    """
    if payload.startswith(EXIF_SIGNATURE):
        payload = payload.removeprefix(EXIF_SIGNATURE)
    if len(payload) < 8:
        raise JpegParseError("EXIF block shorter than a TIFF header")
    order = payload[:2]
    if order == b"II":
        endian = "<"
    elif order == b"MM":
        endian = ">"
    else:
        raise JpegParseError(f"unknown TIFF byte order {order!r}")
    magic, ifd0 = struct.unpack_from(f"{endian}HI", payload, 2)
    if magic != 42:
        raise JpegParseError(f"bad TIFF magic {magic}")
    out: dict[str, object] = {}
    _iter_ifd(payload, endian, ifd0, "", out)
    # IFD1 (thumbnail) can repeat tags; keep it under its own prefix.
    if ifd0 + 2 <= len(payload):
        (count,) = struct.unpack_from(f"{endian}H", payload, ifd0)
        nxt = ifd0 + 2 + 12 * min(count, 4096)
        if nxt + 4 <= len(payload):
            (ifd1,) = struct.unpack_from(f"{endian}I", payload, nxt)
            if ifd1:
                _iter_ifd(payload, endian, ifd1, "IFD1.", out)
    return out


def vendor_tags(exif: dict[str, object]) -> dict[str, object]:
    """Subset of ``exif`` holding tags with no standard name (vendor/private)."""
    return {k: v for k, v in exif.items() if "Tag_0x" in k}# --------------------------------------------------------------------------- #
# XMP
# --------------------------------------------------------------------------- #

XMP_KEYS = frozenset({
    "xmp:CreateDate", "xmp:ModifyDate", "xmp:MetadataDate", "xmp:CreatorTool",
    "xmp:CreatorWork", "xmp:Label", "xmp:Rating", "xmp:Nickname",
    "photoshop:DateCreated", "photoshop:City", "photoshop:State",
    "photoshop:Country", "photoshop:Credit", "photoshop:Instructions",
    "tiff:Make", "tiff:Model", "tiff:Software", "tiff:DateTime",
    "tiff:ImageDescription", "tiff:DocumentName",
    "exif:DateTimeOriginal", "exif:DateTimeDigitized",
    "exifEX:DateTimeOriginal", "exifEX:BodySerialNumber",
    "aux:SerialNumber", "aux:Lens", "aux:LensSerialNumber", "aux:OwnerName",
    "aux:Firmware", "aux:ImageNumber",
    "Iptc4xmpCore:DateCreated", "Iptc4xmpCore:Location", "Iptc4xmpCore:CreatorContactInfo",
    "MicrosoftPhoto:DateTaken", "MicrosoftPhoto:LastKeywordXMP",
    "crs:Version", "crs:ProcessVersion", "crs:ExposureTime", "crs:DateTimeOriginal",
})

_XMP_ATTR = re.compile(r"""([A-Za-z][\w.\-]*:[A-Za-z]\w*)\s*=\s*("[^"]{0,600}"|'[^']{0,600}')""")
_XMP_XMLNS = re.compile(r'xmlns:([\w.\-]+)\s*=\s*"([^"]{0,300})"')


def parse_xmp(payload: bytes, *, vendor_keywords: Sequence[str] = ()) -> dict[str, str]:
    """Pull acquisition-relevant attributes out of an XMP packet.

    Returns the well-known keys in :data:`XMP_KEYS` that are present, plus any
    attribute whose value contains one of ``vendor_keywords`` (so a private
    rig name such as ``FPheno2000`` is still caught).
    """
    text = payload.decode("utf-8", errors="replace")
    keywords = [k.lower() for k in vendor_keywords]
    out: dict[str, str] = {}
    for prefix, uri in _XMP_XMLNS.findall(text):
        if any(k in uri.lower() for k in keywords):
            out[f"xmlns:{prefix}"] = clean_text(uri, 200)
    for name, quoted in _XMP_ATTR.findall(text):
        value = clean_text(quoted[1:-1], 200)
        if not value:
            continue
        if name in XMP_KEYS or any(k in value.lower() for k in keywords):
            out[name] = value
    return out


# --------------------------------------------------------------------------- #
# IPTC IIM (APP13 Photoshop IRB)
# --------------------------------------------------------------------------- #

IPTC_DATASET_NAMES = {
    (2, 0): "RecordVersion",
    (2, 5): "ObjectName",
    (2, 7): "EditStatus",
    (2, 10): "Urgency",
    (2, 15): "Category",
    (2, 20): "SupplementalCategories",
    (2, 25): "Keywords",
    (2, 40): "SpecialInstructions",
    (2, 55): "DateCreated",
    (2, 60): "TimeCreated",
    (2, 62): "DigitalCreationDate",
    (2, 63): "DigitalCreationTime",
    (2, 65): "OriginatorProgram",
    (2, 80): "Byline",
    (2, 85): "BylineTitle",
    (2, 90): "City",
    (2, 92): "Sublocation",
    (2, 95): "ProvinceState",
    (2, 100): "CountryCode",
    (2, 101): "CountryName",
    (2, 105): "Headline",
    (2, 110): "Credit",
    (2, 115): "Source",
    (2, 116): "Copyright",
    (2, 118): "Contact",
    (2, 120): "Caption",
    (2, 122): "CaptionWriter",
    (2, 200): "CustomField",
}


def _iter_8bim(payload: bytes) -> Iterator[tuple[int, bytes]]:
    """Yield ``(resource_id, data)`` for every 8BIM block of a Photoshop IRB."""
    pos = 0
    n = len(payload)
    while pos + 12 <= n:
        if payload[pos : pos + 4] != b"8BIM":
            break
        (res_id,) = struct.unpack_from(">H", payload, pos + 4)
        name_len = payload[pos + 6]
        name_start = pos + 7
        data_start = name_start + name_len + 1
        data_start += data_start % 2  # pad to even
        if data_start + 4 > n:
            return
        (size,) = struct.unpack_from(">I", payload, data_start)
        start = data_start + 4
        end = min(start + size, n)
        yield res_id, payload[start:end]
        pos = end + (end % 2)


def parse_iptc(payload: bytes) -> dict[str, str]:
    """Decode IPTC/IIM datasets from an APP13 Photoshop IRB payload."""
    if payload.startswith(PHOTOSHOP_IRB_SIGNATURE):
        payload = payload.removeprefix(PHOTOSHOP_IRB_SIGNATURE)
    out: dict[str, str] = {}
    for res_id, data in _iter_8bim(payload):
        if res_id != 0x0404:  # 0x0404 = IPTC-NAA (IIM)
            continue
        pos = 0
        while pos + 5 <= len(data):
            if data[pos] != 0x1C:
                pos += 1
                continue
            record, dataset = data[pos + 1], data[pos + 2]
            size = int.from_bytes(data[pos + 3 : pos + 5], "big")
            pos += 5
            if size & 0x8000:  # extended dataset (size spans further bytes)
                length_of_length = size & 0x7FFF
                size = int.from_bytes(data[pos : pos + length_of_length], "big")
                pos += length_of_length
            value = data[pos : pos + size]
            pos += size
            name = IPTC_DATASET_NAMES.get((record, dataset), f"{record}:{dataset}")
            key = f"iptc:{name}"
            text = clean_text(value, 200)
            out[key] = f"{out[key]} | {text}" if key in out else text
    return out


# --------------------------------------------------------------------------- #
# JPEG header
# --------------------------------------------------------------------------- #


@dataclass
class JpegHeader:
    """Everything the scan needs from one JPEG's leading marker segments."""

    marker_names: list[str] = field(default_factory=list)
    segments: list[tuple[int, bytes]] = field(default_factory=list)
    reached_sos: bool = False
    truncated: bool = False
    header_bytes: int = 0
    sof_marker: int | None = None
    width: int | None = None
    height: int | None = None
    precision: int | None = None
    components: int | None = None
    jfif: dict[str, object] = field(default_factory=dict)
    exif: dict[str, object] = field(default_factory=dict)
    xmp: dict[str, str] = field(default_factory=dict)
    iptc: dict[str, str] = field(default_factory=dict)
    comments: list[str] = field(default_factory=list)
    icc_profile_bytes: int = 0
    embedded_jpegs: int = 0
    adobe_app13: bool = False

    @property
    def signature(self) -> str:
        """Marker chain up to SOS, e.g. ``"SOI|APP0|DQT|DQT|SOF7|DHTx4|SOS"``."""
        counts: dict[str, int] = {}
        for name in self.marker_names:
            counts[name] = counts.get(name, 0) + 1
        return "|".join(
            name if count == 1 else f"{name}x{count}" for name, count in counts.items()
        )

    @property
    def has_exif(self) -> bool:
        return bool(self.exif)

    @property
    def has_any_metadata(self) -> bool:
        return bool(self.exif or self.xmp or self.iptc or self.comments)

    def all_text_values(self) -> list[str]:
        """Every decoded string value, for the vendor-keyword sweep."""
        values: list[str] = []
        for source in (self.exif, self.jfif):
            values += [str(v) for v in source.values() if isinstance(v, (str, int, float))]
        values += list(self.xmp.values()) + list(self.iptc.values()) + list(self.comments)
        return values


def iter_segments(data: bytes) -> Iterator[tuple[int, bytes]]:
    """Yield ``(marker, payload)`` for every marker segment before the first SOS.

    Raises :class:`NotAJpegError` when ``data`` is not a JPEG and
    :class:`JpegTruncated` when a segment runs past the end of ``data``.
    """
    if not data.startswith(b"\xff\xd8"):
        raise NotAJpegError(f"missing SOI marker (got {data[:4]!r})")
    yield SOI, b""
    pos, n = 2, len(data)
    while True:
        if pos + 2 > n:
            raise JpegTruncated(f"header ends after {n} bytes without a marker")
        if data[pos] != 0xFF:
            raise JpegParseError(f"expected 0xFF marker at byte {pos}, got 0x{data[pos]:02X}")
        start = pos
        while pos < n and data[pos] == 0xFF:  # marker padding
            pos += 1
        if pos >= n:
            raise JpegTruncated("header ends inside a marker")
        marker = data[pos]
        pos += 1
        if marker in _STANDALONE_MARKERS:
            yield marker, b""
            if marker == EOI:
                return
            continue
        if marker == 0x00:
            raise JpegParseError("stuffed byte outside of entropy-coded data")
        if pos + 2 > n:
            raise JpegTruncated(f"truncated length field for {marker_name(marker)}")
        length = int.from_bytes(data[pos : pos + 2], "big")
        if length < 2:
            raise JpegParseError(f"invalid segment length {length} for {marker_name(marker)}")
        if pos + length > n:
            raise JpegTruncated(
                f"{marker_name(marker)} needs {length} bytes, {n - pos} available")
        yield marker, data[pos + 2 : pos + length]
        pos += length
        if marker == SOS:
            return
        if pos <= start:
            raise JpegParseError("marker walk made no progress")


def parse_jpeg_header(
    data: bytes,
    *,
    vendor_keywords: Sequence[str] = (),
    allow_truncation: bool = False,
) -> JpegHeader:
    """Parse the marker segments at the start of a JPEG.

    With ``allow_truncation`` the parse stops at the end of ``data`` and sets
    :attr:`JpegHeader.truncated` instead of raising, so a caller with a byte
    budget can record "we did not reach SOS" instead of crashing.
    """
    head = JpegHeader()
    try:
        for marker, payload in iter_segments(data):
            head.segments.append((marker, payload))
            head.marker_names.append(marker_name(marker))
            if marker == SOI:
                continue
            if marker in (APP0, APP1, APP2, APP13):
                if marker == APP0 and payload.startswith(JFIF_SIGNATURE):
                    head.jfif = parse_jfif(payload)
                elif marker == APP1 and payload.startswith(EXIF_SIGNATURE):
                    head.exif = parse_tiff_exif(payload)
                elif marker == APP1 and payload.startswith(XMP_SIGNATURES):
                    head.xmp = parse_xmp(payload, vendor_keywords=vendor_keywords)
                elif marker == APP13 and payload.startswith(PHOTOSHOP_IRB_SIGNATURE):
                    head.iptc = parse_iptc(payload)
                    head.adobe_app13 = True
                elif marker == APP2 and payload.startswith(b"ICC_PROFILE\x00"):
                    head.icc_profile_bytes = len(payload)
            elif marker == COM:
                head.comments.append(clean_text(payload, 300))
            elif marker in SOF_MARKERS:
                head.sof_marker = marker
                if len(payload) >= 6:
                    head.precision = payload[0]
                    head.height = int.from_bytes(payload[1:3], "big")
                    head.width = int.from_bytes(payload[3:5], "big")
                    head.components = payload[5]
            elif marker == SOS:
                head.reached_sos = True
                head.header_bytes = _consumed(data, head)
    except JpegTruncated:
        if not allow_truncation:
            raise
        head.truncated = True
    if not head.reached_sos:
        head.header_bytes = min(len(data), head.header_bytes or len(data))
    return head


def _consumed(data: bytes, head: JpegHeader) -> int:
    """Byte offset just past the last complete segment we managed to read."""
    total = 0
    for marker, payload in head.segments:
        total += 2 if marker in _STANDALONE_MARKERS else 4 + len(payload)
    return min(total, len(data))


def read_header(read, *, chunk: int = 8192, budget: int = 1 << 20) -> JpegHeader:
    """Read just enough of a JPEG stream to parse its header.

    ``read`` is a callable ``n -> bytes`` over a *stream* (a zip entry, a file
    handle); nothing is ever seeked, so no archive is extracted to disk.  At
    most ``budget`` bytes are consumed, which for these 4608x3456 images is a
    few hundred bytes rather than the multi-megabyte image body.
    """
    buf = bytearray(read(min(chunk, budget)))
    while True:
        try:
            return parse_jpeg_header(bytes(buf))
        except JpegTruncated:
            if len(buf) >= budget:
                return parse_jpeg_header(bytes(buf), allow_truncation=True)
            more = read(min(chunk, budget - len(buf)))
            if not more:
                return parse_jpeg_header(bytes(buf), allow_truncation=True)
            buf += more


# --------------------------------------------------------------------------- #
# zip entry naming
# --------------------------------------------------------------------------- #

ENTRY_IMAGE = "image"
ENTRY_MASK = "mask"
ENTRY_APPLEDOUBLE = "appledouble"
ENTRY_DS_STORE = "ds_store"
ENTRY_DIR = "dir"
ENTRY_OTHER = "other"

_ID_RE = re.compile(r"(\d{4,8})(?=\.[^.]+$)")
_CONDITION_TAGS = ("trainset", "testset", "MYG", "PDA", "TEMP15", "GS", "PO", "TS")


def classify_entry(name: str) -> str:
    """Bucket a zip entry into image / mask / sidecar / directory / other."""
    base = name.rsplit("/", 1)[-1]
    if name.endswith("/"):
        return ENTRY_DIR
    if "__MACOSX/" in name or base.startswith("._"):
        return ENTRY_APPLEDOUBLE
    if base == ".DS_Store":
        return ENTRY_DS_STORE
    lowered = name.lower()
    if lowered.endswith((".jpg", ".jpeg")):
        return ENTRY_IMAGE
    if lowered.endswith((".png", ".bmp", ".tif", ".tiff")):
        return ENTRY_MASK
    return ENTRY_OTHER


def image_id_from_entry(name: str) -> str:
    """The zero-padded numeric id in an entry name, or ``""`` if there is none."""
    match = _ID_RE.search(name.rsplit("/", 1)[-1])
    return match.group(1) if match else ""


def condition_tag_from_entry(name: str) -> str:
    """Condition / split label carried by the path (``TEMP15``, ``trainset``, ...)."""
    parts = name.split("/")[:-1]
    for part in reversed(parts):
        if part in _CONDITION_TAGS:
            return part
    return ""


def find_vendor_hits(values: Iterable[str], keywords: Sequence[str]) -> list[str]:
    """Return ``keyword=value`` strings for every value matching a vendor keyword."""
    hits: set[str] = set()
    lowered = [k.lower() for k in keywords]
    for value in values:
        for key in lowered:
            if key and key in value.lower():
                hits.add(f"{key}={clean_text(value, 60)}")
    return sorted(hits)


# --------------------------------------------------------------------------- #
# PNG text chunks (the labeled release ships masks next to every image)
# --------------------------------------------------------------------------- #

PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"
_PNG_TEXT_CHUNKS = {b"tEXt", b"zTXt", b"iTXt", b"eXIf"}


def parse_png_text(data: bytes) -> dict[str, object]:
    """List the metadata chunks of a PNG header prefix.

    Returns the chunk types seen before the first ``IDAT`` plus any decodable
    ``tIME`` / ``tEXt`` / ``zTXt`` / ``iTXt`` / ``eXIf`` content.  PNGs can hold
    a timestamp just as JPEGs can, so the masks are checked too.
    """
    out: dict[str, object] = {"chunks": [], "text": {}, "time": "", "exif_bytes": 0}
    if not data.startswith(PNG_SIGNATURE):
        raise JpegParseError("not a PNG")
    pos, n = len(PNG_SIGNATURE), len(data)
    while pos + 8 <= n:
        (size,) = struct.unpack_from(">I", data, pos)
        ctype = data[pos + 4 : pos + 8]
        start = pos + 8
        end = start + size
        name = ctype.decode("ascii", errors="replace")
        out["chunks"].append(name)  # type: ignore[union-attr]
        payload = data[start : min(end, n)] if size <= n else b""
        if ctype == b"tIME" and len(payload) >= 7:
            # PNG 1.2 spec: tIME is "Universal Time" with no zone suffix.
            year, month, day, hour, minute, second = struct.unpack(">HBBBBB", payload[:7])
            try:
                out["time"] = datetime(
                    year, month, day, hour, minute, second, tzinfo=timezone.utc
                ).isoformat()
            except ValueError:
                out["time"] = ""
        elif ctype == b"eXIf":
            out["exif_bytes"] = len(payload)
        elif ctype in _PNG_TEXT_CHUNKS:
            out["text"].update(_png_text(ctype, payload))
        if ctype == b"IDAT" or ctype == b"IEND":
            break
        pos = end + 4  # skip CRC
        if pos <= start - 8:
            break
    return out


def _png_text(ctype: bytes, payload: bytes) -> dict[str, str]:
    """Decode one PNG text chunk into ``{keyword: value}``."""
    keyword, _, rest = payload.partition(b"\x00")
    key = clean_text(keyword, 64)
    try:
        if ctype == b"tEXt":
            return {key: clean_text(rest, 200)}
        if ctype == b"zTXt":
            method, _, blob = rest.partition(b"\x00")
            if method == b"\x00":
                return {key: clean_text(zlib.decompress(blob), 200)}
            return {}
        if ctype == b"iTXt":
            # keyword \0 flag compression_method language \0 translated \0 text
            flag, _, blob = rest.partition(b"\x00")
            _, _, blob = blob.partition(b"\x00")
            _, _, blob = blob.partition(b"\x00")
            if flag == b"\x01" and blob[:1] == b"\x00":
                blob = zlib.decompress(blob[1:])
            return {key: clean_text(blob, 200)}
    except (zlib.error, UnicodeDecodeError):
        return {key: "<undecodable>"}
    return {}


# --------------------------------------------------------------------------- #
# zip extra fields (extended timestamps)
# --------------------------------------------------------------------------- #

EXTRA_NTFS = 0x000A
EXTRA_UT = 0x5455
# The uid/gid field is written under three different ids depending on the
# archiver; all three share the layout version, uidSize, uid, gidSize, gid.
EXTRA_UNIX_IDS = (0x7875, 0x5855, 0x7578)
_FILETIME_EPOCH = 116444736000000000   # 100 ns ticks between 1601 and 1970


@dataclass(frozen=True)
class ZipExtra:
    """Extended metadata stored in a zip entry's extra field.

    The DOS mtime in the zip header has 2-second resolution and no timezone;
    the ``UT`` (Info-ZIP) and ``NTFS`` extra fields carry the same instant at
    1-second (NTFS: sub-second) resolution *in UTC*, which is both sharper and
    enough to recover the packing machine's UTC offset.
    """

    mtime_utc: datetime | None = None
    source: str = ""                    # "ntfs", "ut" or ""
    uid: int | None = None
    gid: int | None = None
    field_ids: tuple[int, ...] = ()


def _filetime_to_datetime(value: int) -> datetime | None:
    """Convert a Windows FILETIME (100 ns ticks since 1601) to UTC."""
    if value <= _FILETIME_EPOCH:
        return None
    return datetime(1970, 1, 1, tzinfo=timezone.utc) + timedelta(
        microseconds=(value - _FILETIME_EPOCH) // 10
    )


def parse_zip_extra(extra: bytes) -> ZipExtra:
    """Decode the extra field of a zip entry.

    Understands ``NTFS`` (0x000a, sub-second mtime), ``UT`` (0x5455, 1 s) and
    the uid/gid field under any of :data:`EXTRA_UNIX_IDS`.  Unknown or truncated
    fields are skipped rather than raising.
    """
    mtime: datetime | None = None
    source = ""
    uid = gid = None
    seen: list[int] = []
    pos, size = 0, len(extra)
    while pos + 4 <= size:
        field_id, length = struct.unpack_from("<HH", extra, pos)
        if pos + 4 + length > size:      # truncated trailing field
            break
        payload = extra[pos + 4 : pos + 4 + length]
        seen.append(field_id)
        pos += 4 + length

        if field_id == EXTRA_NTFS and len(payload) >= 4:
            inner = 4                       # reserved bytes
            while inner + 4 <= len(payload):
                tag, tag_size = struct.unpack_from("<HH", payload, inner)
                value = payload[inner + 4 : inner + 4 + tag_size]
                if tag == 1 and len(value) >= 8:
                    mtime = _filetime_to_datetime(struct.unpack_from("<Q", value)[0])
                    source = "ntfs"
                inner += 4 + tag_size
        elif field_id == EXTRA_UT and payload:
            flags = payload[0]
            cursor = 1
            for bit, name in ((1, "ut"), (2, "atime"), (3, "ctime")):
                if flags & bit and cursor + 4 <= len(payload):
                    (seconds,) = struct.unpack_from("<i", payload, cursor)
                    if name == "ut" and mtime is None and seconds > 0:
                        mtime = datetime.fromtimestamp(seconds, tz=timezone.utc)
                        source = "ut"
                    cursor += 4
        elif field_id in EXTRA_UNIX_IDS and len(payload) >= 2:
            cursor = 1                                    # version byte
            uid_size = payload[cursor]
            cursor += 1
            if 1 <= uid_size <= 8 and cursor + uid_size <= len(payload):
                uid = int.from_bytes(payload[cursor : cursor + uid_size], "little")
                cursor += uid_size
            if cursor < len(payload):
                gid_size = payload[cursor]
                cursor += 1
                if 1 <= gid_size <= 8 and cursor + gid_size <= len(payload):
                    gid = int.from_bytes(payload[cursor : cursor + gid_size], "little")
    return ZipExtra(mtime_utc=mtime, source=source, uid=uid, gid=gid,
                    field_ids=tuple(seen))


# --------------------------------------------------------------------------- #
# macOS AppleDouble sidecars
# --------------------------------------------------------------------------- #

APPLEDOUBLE_MAGIC = 0x00051607
_QUARANTINE_NAME = b"com.apple.quarantine"


def parse_appledouble(data: bytes) -> list[dict[str, object]]:
    """List the entries of a macOS ``AppleDouble`` sidecar (``._name``).

    AppleDouble files carry no image data -- only macOS file metadata such as
    extended attributes -- which is why a ``._x.jpg`` must never be mistaken for
    an image.  Unknown entry ids are reported by number with a hex preview.
    """
    if len(data) < 26:
        raise JpegParseError("AppleDouble file shorter than 26 bytes")
    magic, _version = struct.unpack_from(">II", data, 0)
    if magic != APPLEDOUBLE_MAGIC:
        raise JpegParseError(f"bad AppleDouble magic 0x{magic:08x}")
    (count,) = struct.unpack_from(">H", data, 24)
    entries: list[dict[str, object]] = []
    for i in range(count):
        head = 26 + i * 12
        if head + 12 > len(data):
            break
        entry_id, offset, length = struct.unpack_from(">III", data, head)
        value = data[offset : offset + length] if offset < len(data) else b""
        entries.append({
            "id": entry_id,
            "length": length,
            "quarantine": parse_quarantine(value),
        })
    return entries


def parse_quarantine(attr: bytes) -> dict[str, str] | None:
    """Decode a ``com.apple.quarantine`` xattr value.

    The value has the form ``q/0081;68e5cb7f;<downloader>;<uuid>``: the second
    field is a hex epoch timestamp of when the *download* happened (a browser or
    mail client set the flag), not when the picture was taken.
    """
    pos = attr.find(_QUARANTINE_NAME)
    if pos < 0:
        return None
    pos += len(_QUARANTINE_NAME)
    if attr[pos : pos + 1] == b"\x00":   # xattr names are NUL terminated
        pos += 1
    end = attr.find(b"\x00", pos)
    value = attr[pos : end if end >= 0 else None]
    if not value:
        return None
    fields = value.decode("utf-8", errors="replace").split(";")
    out = {"value": clean_text(value, 120), "type": fields[0]}
    if len(fields) > 1:
        stamp = hex_epoch_to_datetime(fields[1])
        out["downloaded"] = stamp.isoformat() if stamp else ""
    if len(fields) > 2:
        out["agent"] = clean_text(decode_escaped_utf8(fields[2]), 80)
    return out


# --------------------------------------------------------------------------- #
# timestamp sessionisation (are zip mtimes acquisition times?)
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class Session:
    """A run of files whose stored mtimes are no more than ``gap_s`` apart."""

    start: datetime
    end: datetime
    count: int

    @property
    def span_seconds(self) -> float:
        return (self.end - self.start).total_seconds()


def group_sessions(times: Iterable[datetime], gap_s: float = 60.0) -> list[Session]:
    """Split ``times`` into sessions separated by gaps larger than ``gap_s``.

    Imaging a dish over weeks produces evenly spaced, far-apart timestamps; a
    copy or export job produces dense bursts.  The burst count is what separates
    the two, so this is the core of the zip-mtime verdict.
    """
    ordered = sorted(t for t in times if t is not None)
    if not ordered:
        return []
    sessions: list[Session] = []
    start = previous = ordered[0]
    count = 1
    for stamp in ordered[1:]:
        if (stamp - previous).total_seconds() > gap_s:
            sessions.append(Session(start, previous, count))
            start, count = stamp, 0
        previous = stamp
        count += 1
    sessions.append(Session(start, previous, count))
    return sessions


def forward_step_fraction(values: Sequence) -> float | None:
    """Fraction of consecutive pairs where ``values`` does not decrease.

    A shuffled / renamed collection lands near 0.5; a genuine time series is
    near 1.0.
    """
    if len(values) < 2:
        return None
    steps = sum(1 for a, b in pairwise(values) if b >= a)
    return steps / (len(values) - 1)
