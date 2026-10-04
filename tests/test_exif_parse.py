"""Unit tests for the MyceliumSeg triage metadata parsers.

The parsers must find real metadata when it is there and report its absence
cleanly when it is not, so most tests build a synthetic JPEG that *does* carry
EXIF/XMP/IPTC/COM data and assert on the decoded values.  Without that, a
parser that silently returns nothing would still "pass" on this dataset.
"""

# Naive datetimes are deliberate throughout: a zip's DOS mtime field is local
# time with no timezone attached, so the fixtures that model it must be naive.
# ruff: noqa: DTZ001

from __future__ import annotations

import io
import struct
import zipfile
from datetime import datetime, timedelta, timezone

import exif_parse as ep
import pytest
from PIL import Image

# --------------------------------------------------------------------------- #
# helpers: build synthetic JPEG segments
# --------------------------------------------------------------------------- #


def segment(marker: int, payload: bytes) -> bytes:
    """A JPEG marker segment (length field included)."""
    return bytes([0xFF, marker]) + struct.pack(">H", len(payload) + 2) + payload


def jfif_app0(density: int = 1, units: int = 0) -> bytes:
    return segment(0xE0, b"JFIF\x00\x01\x01" + bytes([units]) +
                   struct.pack(">HH", density, density) + b"\x00\x00")


def sof0(width: int, height: int) -> bytes:
    return segment(0xC0, bytes([8]) + struct.pack(">HH", height, width) + b"\x03")


def tiff_block(entries: dict[int, tuple[int, object]], *, endian: str = "<") -> bytes:
    """Hand-build a flat, single-IFD TIFF/EXIF block (no sub-IFDs).

    ``entries`` maps a tag id to ``(type, value)``; ASCII (2), SHORT (3),
    LONG (4) and UNDEFINED (7) are supported, which is enough to exercise the
    decoder without pulling in a real camera file.  The entry *count* field
    holds the number of values (bytes for ASCII/UNDEFINED), as TIFF requires.
    """
    order = b"II" if endian == "<" else b"MM"
    count = len(entries)
    ifd_size = 2 + 12 * count + 4
    blob_start = 8 + ifd_size
    ifd = bytearray(struct.pack(f"{endian}H", count))
    blobs = bytearray()
    for tag, (typ, value) in entries.items():
        if typ in (2, 7):
            raw = value.encode() + b"\x00" if isinstance(value, str) else bytes(value)
            n_values = len(raw)
        else:
            fmt = {1: "B", 3: "H", 4: "I"}[typ]
            values = (value,) if isinstance(value, int) else tuple(value)
            raw = struct.pack(f"{endian}{fmt * len(values)}", *values)
            n_values = len(values)
        size = len(raw)
        if size <= 4:
            value_field = raw + b"\x00" * (4 - size)
        else:
            value_field = struct.pack(f"{endian}I", blob_start + len(blobs))
            blobs += raw
        ifd += struct.pack(f"{endian}HHI", tag, typ, n_values) + value_field
    ifd += struct.pack(f"{endian}I", 0)  # no IFD1
    body = order + struct.pack(f"{endian}HI", 42, 8) + bytes(ifd) + bytes(blobs)
    return b"Exif\x00\x00" + body


def jpeg_with(*, exif: bytes | None = None, xmp: bytes | None = None,
              iptc: bytes | None = None, comment: bytes | None = None,
              width: int = 8, height: int = 6) -> bytes:
    """Assemble a JPEG header prefix; SOS is followed by a few entropy bytes."""
    parts = [b"\xff\xd8"]
    if exif is None and xmp is None and iptc is None and comment is None:
        parts.append(jfif_app0())
    if exif:
        parts.append(segment(0xE1, exif))
    if xmp:
        parts.append(segment(0xE1, b"http://ns.adobe.com/xap/1.0/\x00" + xmp))
    if iptc:
        parts.append(segment(0xED, b"Photoshop 3.0\x00" + iptc))
    if comment:
        parts.append(segment(0xFE, comment))
    parts += [sof0(width, height), segment(0xDA, b"\x03\x01\x00\x02\x11\x03\x11\x00\x3f\x00")]
    parts.append(b"\x00" * 64)  # entropy-coded data
    return b"".join(parts)


def iptc_iim(*datasets: tuple[int, int, bytes]) -> bytes:
    """Build an IPTC/IIM block wrapped in an 8BIM resource."""
    iim = bytearray()
    for record, dataset, value in datasets:
        iim += bytes([0x1C, record, dataset]) + struct.pack(">H", len(value)) + value
    block = b"8BIM" + struct.pack(">H", 0x0404) + b"\x00\x00" + \
        struct.pack(">I", len(iim)) + bytes(iim)
    return block + b"\x00" * (len(block) % 2)


def appledouble(attrs: dict[str, bytes]) -> bytes:
    """Build a macOS AppleDouble sidecar holding the given xattrs.

    Same shape as the sidecars in the real release: magic + version + 16-byte
    filler + entry count, then one entry (id 9) whose payload is an ``ATTR``
    extended-attribute block of ``name\\0value\\0`` records.
    """
    records = b"".join(
        struct.pack(">I", len(name.encode()) + 1) + name.encode() + b"\x00" + value + b"\x00"
        for name, value in attrs.items()
    )
    attr_block = b"\x00" * 36 + b"ATTR" + struct.pack(">II", 44 + len(records), 44) + records
    entry = struct.pack(">III", 9, 26 + 12, len(attr_block)) + attr_block
    header = (b"\x00\x05\x16\x07\x00\x02\x00\x00" + b"Mac OS X".ljust(16, b" ")
              + struct.pack(">H", 1))
    return header + entry


# --------------------------------------------------------------------------- #
# EXIF / TIFF
# --------------------------------------------------------------------------- #


def test_tiff_exif_decodes_datetime_make_and_model() -> None:
    block = tiff_block({
        0x010F: (2, "Canon"),
        0x0110: (2, "EOS R5"),
        0x0132: (2, "2024:05:06 07:08:09"),
        0x9003: (2, "2024:05:06 07:08:09"),
        0x9004: (2, "2024:05:06 07:08:09"),
    })
    exif = ep.parse_tiff_exif(block)
    assert exif["Make"] == "Canon"
    assert exif["Model"] == "EOS R5"
    assert exif["DateTime"] == "2024:05:06 07:08:09"
    assert exif["DateTimeOriginal"] == "2024:05:06 07:08:09"
    assert exif["DateTimeDigitized"] == "2024:05:06 07:08:09"


def test_tiff_exif_handles_big_endian() -> None:
    block = tiff_block({0x9003: (2, "2025:01:02 03:04:05")}, endian=">")
    assert ep.parse_tiff_exif(block)["DateTimeOriginal"] == "2025:01:02 03:04:05"


def test_tiff_exif_keeps_unknown_tags_as_vendor_tags() -> None:
    block = tiff_block({0x010F: (2, "FPheno2000"), 0xC640: (2, "dish-17-plate-2")})
    parsed = ep.parse_tiff_exif(block)
    assert ep.vendor_tags(parsed) == {"Tag_0xC640": "dish-17-plate-2"}
    assert "Tag_0xC640" not in [k for k in parsed if k in ep.TAG_NAMES.values()]


def test_tiff_exif_rational_and_short_values() -> None:
    block = tiff_block({0x0112: (3, 1), 0x8827: (3, [400, 800, 1600])})
    parsed = ep.parse_tiff_exif(block)
    assert parsed["Orientation"] == 1
    assert parsed["ISOSpeedRatings"] == [400, 800, 1600]


def test_tiff_exif_usercomment_with_charset_prefix() -> None:
    block = tiff_block({0x9286: (7, b"ASCII\x00\x00\x00plate B dish 12")})
    assert ep.parse_tiff_exif(block)["UserComment"] == "plate B dish 12"


def test_tiff_exif_rejects_bad_byte_order() -> None:
    with pytest.raises(ep.JpegParseError):
        ep.parse_tiff_exif(b"Exif\x00\x00XX\x2a\x00\x00\x00\x08")


def test_tiff_exif_rejects_short_block() -> None:
    with pytest.raises(ep.JpegParseError):
        ep.parse_tiff_exif(b"Exif\x00\x00II")


def test_pillow_exif_block_round_trip() -> None:
    """A real encoder (Pillow) must be readable, not just hand-built bytes."""
    exif = Image.Exif()
    exif[0x010F] = "Fuji"
    exif[0x0110] = "FPheno2000"
    exif[0x0131] = "FPheno2000 1.2.3"
    exif[0x9003] = "2024:09:20 11:24:36"
    exif[0x9286] = b"ASCII\x00\x00\x00dish 12"
    buffer = io.BytesIO()
    Image.new("RGB", (16, 12)).save(buffer, "JPEG", exif=exif.tobytes())
    head = ep.parse_jpeg_header(buffer.getvalue(), vendor_keywords=("fpheno",))
    assert head.has_exif
    assert head.exif["DateTimeOriginal"] == "2024:09:20 11:24:36"
    assert head.exif["Make"] == "Fuji"
    assert head.exif["UserComment"] == "dish 12"
    assert ep.find_vendor_hits(head.all_text_values(), ("fpheno",)) == [
        "fpheno=FPheno2000", "fpheno=FPheno2000 1.2.3",
    ]


# --------------------------------------------------------------------------- #
# JPEG container
# --------------------------------------------------------------------------- #


def test_jpeg_header_reports_geometry_and_marker_chain() -> None:
    head = ep.parse_jpeg_header(jpeg_with(width=64, height=48))
    assert (head.width, head.height) == (64, 48)
    assert head.sof_marker == 0xC0
    assert head.reached_sos is True
    assert head.truncated is False
    assert head.signature == "SOI|APP0|SOF0|SOS"
    assert head.has_any_metadata is False


def test_jpeg_header_reads_exif_xmp_iptc_and_comment_together() -> None:
    data = jpeg_with(
        exif=tiff_block({0x9003: (2, "2024:09:20 11:24:36")}),
        xmp=b'<x:xmpmeta xmlns:x="adobe:ns:meta/"><rdf:RDF>'
            b'<rdf:Description xmp:CreateDate="2024-09-20T11:24:36" '
            b'xmp:CreatorTool="FPheno2000"/>'
            b"</rdf:RDF></x:xmpmeta>",
        iptc=iptc_iim((2, 120, b"dish 7, plate C"), (2, 55, b"20240920")),
        comment=b"\xef\xbb\xbfshot by rig 3",
    )
    head = ep.parse_jpeg_header(data, vendor_keywords=("fpheno",))
    assert head.exif["DateTimeOriginal"] == "2024:09:20 11:24:36"
    assert head.xmp["xmp:CreateDate"] == "2024-09-20T11:24:36"
    assert head.xmp["xmp:CreatorTool"] == "FPheno2000"
    assert head.iptc["iptc:Caption"] == "dish 7, plate C"
    assert head.iptc["iptc:DateCreated"] == "20240920"
    assert head.comments == ["shot by rig 3"]
    assert head.has_any_metadata is True


def test_jpeg_header_rejects_non_jpeg_bytes() -> None:
    with pytest.raises(ep.NotAJpegError):
        ep.parse_jpeg_header(appledouble({"com.apple.quarantine": b"q/0081;68e5cb7f;X;Y"}))


def test_jpeg_header_raises_when_header_is_cut_short() -> None:
    data = jpeg_with(exif=tiff_block({0x9003: (2, "2024:09:20 11:24:36")}))
    with pytest.raises(ep.JpegTruncated):
        ep.parse_jpeg_header(data[:len(data) - 70])


def test_jpeg_header_can_tolerate_truncation_when_allowed() -> None:
    exif = tiff_block({0x9003: (2, "2024:09:20 11:24:36")})
    # Cut in the middle of the SOF0 segment: the geometry never arrives.
    cut = 2 + len(segment(0xE1, exif)) + 4
    head = ep.parse_jpeg_header(jpeg_with(exif=exif)[:cut], allow_truncation=True)
    assert head.truncated is True
    assert head.reached_sos is False
    assert head.width is None
    assert head.exif["DateTimeOriginal"] == "2024:09:20 11:24:36"


def test_jpeg_header_rejects_bogus_segment_length() -> None:
    data = b"\xff\xd8" + b"\xff\xe1" + b"\x00\x01" + b"junk"
    with pytest.raises(ep.JpegParseError):
        ep.parse_jpeg_header(data)


class _Stream:
    """Minimal non-seekable stream, like the handle a zip entry provides."""

    def __init__(self, data: bytes, chunk: int = 7) -> None:
        self._data = data
        self._pos = 0
        self._chunk = chunk
        self.bytes_read = 0

    def read(self, size: int = -1) -> bytes:
        size = self._chunk if size is None or size < 0 else min(size, self._chunk)
        out = self._data[self._pos : self._pos + size]
        self._pos += len(out)
        self.bytes_read += len(out)
        return out


def test_read_header_stops_at_sos_without_reading_the_image() -> None:
    body = jpeg_with(width=4608, height=3456) + b"\x5a" * 3_000_000
    stream = _Stream(body)
    head = ep.read_header(stream.read, chunk=8192, budget=1 << 20)
    assert head.reached_sos is True
    assert (head.width, head.height) == (4608, 3456)
    assert stream.bytes_read < 4096          # nothing like the 3 MB image body
    assert head.header_bytes < stream.bytes_read


def test_read_header_gives_up_at_the_budget() -> None:
    # A JPEG whose first segment claims 60 kB, with only 100 bytes available.
    data = b"\xff\xd8" + b"\xff\xe1" + struct.pack(">H", 60000) + b"\x00" * 100
    head = ep.read_header(_Stream(data, chunk=64).read, chunk=64, budget=128)
    assert head.truncated is True
    assert head.reached_sos is False


# --------------------------------------------------------------------------- #
# JFIF / PNG / AppleDouble
# --------------------------------------------------------------------------- #


def test_parse_jfif_density() -> None:
    assert ep.parse_jfif(b"JFIF\x00\x01\x02\x01" + struct.pack(">HH", 300, 300)
                          + b"\x00\x00") == {
        "version": "1.02", "units": 1, "x_density": 300, "y_density": 300,
        "x_thumbnail": 0, "y_thumbnail": 0,
    }
    assert ep.parse_jfif(b"JFXX\x00\x00") == {}


def test_parse_png_text_finds_tEXt_and_tIME() -> None:
    def chunk(ctype: bytes, payload: bytes) -> bytes:
        return struct.pack(">I", len(payload)) + ctype + payload + b"\x00\x00\x00\x00"

    png = ep.PNG_SIGNATURE
    png += chunk(b"tIME", struct.pack(">HBBBBB", 2025, 10, 8, 9, 56, 38))
    png += chunk(b"tEXt", b"Software\x00FPheno2000")
    png += chunk(b"IDAT", b"\x00" * 8)
    png += chunk(b"tEXt", b"Comment\x00after IDAT, must be ignored")
    parsed = ep.parse_png_text(png)
    assert parsed["time"] == "2025-10-08T09:56:38+00:00"   # PNG tIME is UTC
    assert parsed["text"] == {"Software": "FPheno2000"}
    assert parsed["chunks"] == ["tIME", "tEXt", "IDAT"]


def test_parse_png_text_rejects_non_png() -> None:
    with pytest.raises(ep.JpegParseError):
        ep.parse_png_text(b"\xff\xd8 not a png")


def test_parse_appledouble_decodes_quarantine_timestamp() -> None:
    # Baidu Netdisk escapes its name and stamps a single download moment.
    blob = appledouble({"com.apple.quarantine":
                        b"q/0081;68e5cb7f;"
                        b"\\xe7\\x99\\xbe\\xe5\\xba\\xa6\\xe7\\xbd\\x91\\xe7\\x9b\\x98;"})
    entries = ep.parse_appledouble(blob)
    assert len(entries) == 1
    quarantine = entries[0]["quarantine"]
    assert quarantine is not None
    assert quarantine["type"] == "q/0081"
    assert quarantine["agent"] == "百度网盘"
    assert quarantine["downloaded"] == "2025-10-08T02:25:03+00:00"


def test_parse_appledouble_reads_plain_downloader_names() -> None:
    blob = appledouble({"com.apple.quarantine": b"q/0081;68e5cb7f;Chrome;ABC-123"})
    quarantine = ep.parse_appledouble(blob)[0]["quarantine"]
    assert quarantine is not None
    assert quarantine["agent"] == "Chrome"
    assert quarantine["downloaded"] == "2025-10-08T02:25:03+00:00"


def test_parse_appledouble_without_quarantine() -> None:
    entries = ep.parse_appledouble(appledouble({"com.apple.FinderInfo": b"\x00" * 32}))
    assert entries[0]["quarantine"] is None


def test_parse_appledouble_rejects_bad_magic() -> None:
    with pytest.raises(ep.JpegParseError):
        ep.parse_appledouble(b"\x00" * 40)


def test_hex_epoch_to_datetime_handles_garbage() -> None:
    # The stamp carried by the sidecars in the real release (2025-10-08).
    assert ep.hex_epoch_to_datetime("68e5cb7f") == datetime(
        2025, 10, 8, 2, 25, 3, tzinfo=timezone.utc)
    assert ep.hex_epoch_to_datetime("not-hex") is None


def test_decode_escaped_utf8() -> None:
    # Exactly how Baidu Netdisk writes its name in com.apple.quarantine.
    assert ep.decode_escaped_utf8("\\xe7\\x99\\xbe\\xe5\\xba\\xa6\\xe7\\xbd\\x91\\xe7\\x9b\\x98") \
        == "百度网盘"
    assert ep.decode_escaped_utf8("Safari") == "Safari"
    assert ep.decode_escaped_utf8("mixed \\x41 text") == "mixed A text"
    assert ep.decode_escaped_utf8("trailing backslash \\x") == "trailing backslash \\x"


# --------------------------------------------------------------------------- #
# zip entry naming and sessionisation
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "name,kind",
    [
        ("unlabeled_data2/00004988.jpg", ep.ENTRY_IMAGE),
        ("labeled-GL/testset/image/00000458.JPG", ep.ENTRY_IMAGE),
        ("__MACOSX/MYG/image/._00018461.jpg", ep.ENTRY_APPLEDOUBLE),
        ("labeled-GL/trainset/image/._00000456.jpg", ep.ENTRY_APPLEDOUBLE),
        ("labeled-GL/trainset/mask/00000456.png", ep.ENTRY_MASK),
        ("MYG/.DS_Store", ep.ENTRY_DS_STORE),
        ("MYG/image/", ep.ENTRY_DIR),
        ("MYG/notes.txt", ep.ENTRY_OTHER),
    ],
)
def test_classify_entry(name: str, kind: str) -> None:
    assert ep.classify_entry(name) == kind


def test_image_id_and_condition_tag() -> None:
    assert ep.image_id_from_entry("unlabeled_data2/00004988.jpg") == "00004988"
    assert ep.image_id_from_entry("__MACOSX/MYG/image/._00018461.jpg") == "00018461"
    assert ep.image_id_from_entry("MYG/.DS_Store") == ""
    assert ep.condition_tag_from_entry("TEMP15/image/00018478.jpg") == "TEMP15"
    assert ep.condition_tag_from_entry("labeled-GL/trainset/image/1.jpg") == "trainset"
    assert ep.condition_tag_from_entry("unlabeled_data7/00016188.jpg") == ""


def test_group_sessions_splits_export_bursts() -> None:
    base = datetime(2024, 9, 20, 11, 24, 20)
    times = [base + timedelta(seconds=6 * i) for i in range(5)]
    times += [datetime(2024, 11, 14, 10, 0, 0), datetime(2024, 11, 14, 10, 0, 3)]
    sessions = ep.group_sessions(times, gap_s=60)
    assert [(s.start, s.count) for s in sessions] == [
        (base, 5), (datetime(2024, 11, 14, 10, 0, 0), 2),
    ]
    assert sessions[0].span_seconds == 24.0
    assert sessions[1].span_seconds == 3.0


def test_group_sessions_keeps_a_daily_series_together_at_a_large_gap() -> None:
    """An hourly imaging interval must stay one session; a 60 s gap must not."""
    times = [datetime(2024, 5, 1, 8) + timedelta(hours=6 * i) for i in range(10)]
    assert len(ep.group_sessions(times, gap_s=60)) == 10
    assert len(ep.group_sessions(times, gap_s=6 * 3600)) == 1


def test_group_sessions_ignores_out_of_order_input() -> None:
    times = [datetime(2024, 5, 1, 12), datetime(2024, 5, 1, 8), datetime(2024, 5, 1, 9)]
    sessions = ep.group_sessions(times, gap_s=60)
    assert [s.start for s in sessions] == [datetime(2024, 5, 1, 8),
                                            datetime(2024, 5, 1, 9),
                                            datetime(2024, 5, 1, 12)]
    assert all(s.count == 1 for s in sessions)


def test_group_sessions_empty() -> None:
    assert ep.group_sessions([], gap_s=60) == []


def test_forward_step_fraction() -> None:
    assert ep.forward_step_fraction([1, 2, 3]) == 1.0
    assert ep.forward_step_fraction([1, 2, 1, 2, 1]) == 0.5   # shuffled
    assert ep.forward_step_fraction([3, 3, 1]) == 0.5         # ties count as forward
    assert ep.forward_step_fraction([5]) is None


def test_median() -> None:
    assert ep.median([3, 1, 2]) == 2
    assert ep.median([1, 2, 3, 4]) == 2.5
    assert ep.median([]) is None


def test_clean_text_is_csv_safe() -> None:
    assert ep.clean_text(b"a\x00b\nc  d") == "a b c d"
    assert ep.clean_text("x" * 300, limit=10) == "xxxxxxx..."


# --------------------------------------------------------------------------- #
# end-to-end: scan_archive over a synthetic zip
# --------------------------------------------------------------------------- #


def test_scan_archive_reads_images_masks_and_sidecars(tmp_path) -> None:
    import exif_scan

    image = jpeg_with(exif=tiff_block({0x9003: (2, "2024:09:20 11:24:36")}))
    buffer = io.BytesIO()
    Image.new("L", (20, 10), 255).save(buffer, "PNG")
    archive_path = tmp_path / "synthetic.zip"
    with zipfile.ZipFile(archive_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for index, name in ((1, "unlabeled_data9/00000001.jpg"),
                            (2, "unlabeled_data9/00000002.jpg")):
            info = zipfile.ZipInfo(name, date_time=(2024, 9, 20, 11, 24, 20 + index))
            zf.writestr(info, image)
            if index == 1:
                sidecar = zipfile.ZipInfo(
                    "__MACOSX/unlabeled_data9/image/._00000001.jpg",
                    date_time=(2024, 9, 20, 11, 24, 20),
                )
                zf.writestr(sidecar,
                            appledouble({"com.apple.quarantine":
                                         b"q/0081;68e5cb7f;Safari;UUID-1"}))
            mask = zipfile.ZipInfo(f"unlabeled_data9/mask/{index:08d}.png",
                                   date_time=(2024, 9, 20, 11, 24, 20))
            zf.writestr(mask, buffer.getvalue())

    archive, rows, summary = exif_scan.scan_archive((str(archive_path), exif_scan.ScanConfig()))
    assert archive == "synthetic.zip"
    assert summary["images"] == 2
    assert summary["masks"] == 2
    assert summary["appledouble"] == 1
    assert summary["images_with_exif"] == 2
    assert summary["images_with_datetime"] == 2
    assert summary["n_sessions"] == 1
    assert summary["quarantine_files"] == 1
    assert summary["quarantine_sidecars"] == 1
    assert summary["quarantine_agents"] == "Safari=1"
    assert summary["image_id_min"] == 1
    assert summary["image_id_max"] == 2
    assert summary["id_mtime_forward_fraction"] == 1.0
    by_id = {row["image_id"]: row for row in rows}
    assert by_id["00000001"]["appledouble_downloaded"] == "2025-10-08T02:25:03+00:00"
    assert by_id["00000001"]["exif_datetimeoriginal"] == "2024:09:20 11:24:36"
    assert by_id["00000001"]["has_exif"] == 1
    assert by_id["00000002"]["appledouble_downloaded"] == ""
    assert all(row["reached_sos"] == 1 for row in rows)


def test_parse_zip_extra_ntfs_sub_second_mtime() -> None:
    delta = (datetime(2025, 10, 12, 6, 44, 12, 388455, tzinfo=timezone.utc)
             - datetime(1601, 1, 1, tzinfo=timezone.utc))
    ticks = (delta.days * 86400 + delta.seconds) * 10_000_000 + delta.microseconds * 10
    payload = struct.pack("<I", 0) + struct.pack("<HH", 1, 24) + struct.pack("<Q", ticks)
    extra = struct.pack("<HH", 0x000A, len(payload)) + payload
    parsed = ep.parse_zip_extra(extra)
    assert parsed.source == "ntfs"
    assert parsed.field_ids == (0x000A,)
    assert parsed.mtime_utc == datetime(2025, 10, 12, 6, 44, 12, 388455,
                                        tzinfo=timezone.utc)


def test_parse_zip_extra_ut_mtime_and_unix_ids() -> None:
    stamp = int(datetime(2025, 10, 11, 3, 42, 6, tzinfo=timezone.utc).timestamp())
    ut = struct.pack("<HH", 0x5455, 5) + struct.pack("<Bi", 1, stamp)
    unix = b"\x01\x04" + struct.pack("<I", 502) + b"\x04" + struct.pack("<I", 20)
    parsed = ep.parse_zip_extra(ut + struct.pack("<HH", 0x7578, len(unix)) + unix)
    assert parsed.source == "ut"
    assert parsed.mtime_utc == datetime(2025, 10, 11, 3, 42, 6, tzinfo=timezone.utc)
    assert (parsed.uid, parsed.gid) == (502, 20)
    assert parsed.field_ids == (0x5455, 0x7578)


def test_parse_zip_extra_handles_junk() -> None:
    assert ep.parse_zip_extra(b"").mtime_utc is None
    assert ep.parse_zip_extra(b"\x0a\x00").field_ids == ()          # truncated header
    truncated = struct.pack("<HH", 0x5455, 40) + b"\x01\x00\x00\x00"
    assert ep.parse_zip_extra(truncated).mtime_utc is None


def test_deep_scan_markers_finds_stray_metadata() -> None:
    import exif_scan

    # Metadata smuggled after the SOS marker is not a segment, so the header
    # walk cannot see it; the raw-byte grep must.
    smuggled = jpeg_with() + b"\x00" * 32 + b"Exif\x00\x00trailer"
    assert exif_scan.deep_scan_markers(smuggled) == ["Exif\x00\x00"]
    assert exif_scan.deep_scan_markers(jpeg_with()) == []


def test_deep_scan_markers_ignores_coincidental_8bim() -> None:
    import exif_scan

    # `8BIM` appears by chance in entropy-coded data (3 of the 20,176 real
    # images contain it mid-scan); only a well-formed resource block counts.
    coincidental = jpeg_with() + b"\x00" * 16 + b"\xe1$?\xbc]\xd4" + b"8BIM" + \
        b"GO>\xff\x00\xa0\xe5x\xc5Y"
    assert exif_scan.deep_scan_markers(coincidental) == []

    resource = (b"8BIM" + struct.pack(">H", 0x0404) + b"\x00\x00"
                + struct.pack(">I", 4) + b"data")
    assert exif_scan.deep_scan_markers(jpeg_with() + b"\x00" * 8 + resource) == ["8BIM"]


def test_deep_scan_ignores_disabled_config(tmp_path) -> None:
    import exif_scan

    archive_path = tmp_path / "nodeep.zip"
    with zipfile.ZipFile(archive_path, "w") as zf:
        zf.writestr(zipfile.ZipInfo("unlabeled_data9/00000001.jpg",
                                    date_time=(2024, 9, 20, 11, 24, 20)),
                    jpeg_with() + b"Exif\x00\x00trailer")
    _, rows, summary = exif_scan.scan_archive(
        (str(archive_path), exif_scan.ScanConfig(deep_scan=False))
    )
    assert rows[0]["deep_markers"] == ""
    assert summary["deep_marker_hits"] == 0


def test_scan_archive_flags_deep_scan_hits(tmp_path) -> None:
    import exif_scan

    image = jpeg_with() + b"\x00" * 16 + b"Photoshop 3.0\x00trailer"
    archive_path = tmp_path / "sneaky.zip"
    with zipfile.ZipFile(archive_path, "w") as zf:
        zf.writestr(zipfile.ZipInfo("unlabeled_data9/00000001.jpg",
                                    date_time=(2024, 9, 20, 11, 24, 20)), image)
    _, rows, summary = exif_scan.scan_archive(
        (str(archive_path), exif_scan.ScanConfig(deep_scan=True))
    )
    assert rows[0]["has_exif"] == 0          # no EXIF segment declared
    assert rows[0]["deep_markers"] == "Photoshop 3.0"
    assert summary["deep_marker_hits"] == 1


def test_scan_archive_ignores_sidecars_that_shadow_non_images(tmp_path) -> None:
    """A `._.DS_Store` sidecar must not be counted as a quarantined image."""
    import exif_scan

    archive_path = tmp_path / "dsstore.zip"
    with zipfile.ZipFile(archive_path, "w") as zf:
        zf.writestr(zipfile.ZipInfo("MYG/.DS_Store", date_time=(2025, 10, 8, 8, 55, 52)),
                    b"\x00" * 32)
        zf.writestr(zipfile.ZipInfo("__MACOSX/MYG/._.DS_Store",
                                    date_time=(2025, 10, 8, 8, 55, 52)),
                    appledouble({"com.apple.quarantine": b"q/0081;68e5cb7f;Safari;U"}))
        zf.writestr(zipfile.ZipInfo("MYG/image/00018461.jpg",
                                    date_time=(2025, 10, 8, 8, 55, 52)), jpeg_with())
    _, rows, summary = exif_scan.scan_archive(
        (str(archive_path), exif_scan.ScanConfig())
    )
    assert summary["appledouble"] == 1
    assert summary["quarantine_sidecars"] == 1        # the sidecar exists...
    assert summary["quarantine_files"] == 0           # ...but shadows no image
    assert rows[0]["appledouble_downloaded"] == ""


def test_scan_archive_honours_the_limit_and_reports_it(tmp_path) -> None:
    import exif_scan

    archive_path = tmp_path / "many.zip"
    with zipfile.ZipFile(archive_path, "w") as zf:
        for index in range(1, 6):
            zf.writestr(zipfile.ZipInfo(f"unlabeled_data9/{index:08d}.jpg",
                                        date_time=(2024, 9, 20, 11, 24, 20)),
                        jpeg_with())
    _, rows, summary = exif_scan.scan_archive(
        (str(archive_path), exif_scan.ScanConfig(limit=2))
    )
    assert len(rows) == 2
    assert summary["images"] == 2


# --------------------------------------------------------------------------- #
# numbering.py helpers
# --------------------------------------------------------------------------- #


def test_forward_step_fraction_monotonic() -> None:
    from exif_parse import forward_step_fraction
    assert forward_step_fraction([1, 2, 3, 4, 5]) == 1.0
    assert forward_step_fraction([5, 4, 3, 2, 1]) == 0.0
    assert forward_step_fraction([1, 2, 1, 2, 1]) == 0.5
    assert forward_step_fraction([1, 1, 1, 1]) == 1.0  # ties count as forward
    assert forward_step_fraction([1]) is None
    assert forward_step_fraction([]) is None


def test_forward_step_fraction_datetime() -> None:
    from datetime import datetime, timedelta
    from exif_parse import forward_step_fraction
    base = datetime(2024, 1, 1, 12, 0, 0)
    times = [base + timedelta(hours=i) for i in range(5)]
    assert forward_step_fraction(times) == 1.0
    # Shuffled
    shuffled = [times[2], times[0], times[4], times[1], times[3]]
    assert forward_step_fraction(shuffled) == 0.5


def test_run_lengths() -> None:
    import numbering
    assert numbering.run_lengths([1, 1, 1, 2, 2, 3]) == [3, 2, 1]
    assert numbering.run_lengths([1, 2, 3]) == [1, 1, 1]
    assert numbering.run_lengths([1]) == [1]
    assert numbering.run_lengths([]) == []


def test_numbering_bright_fraction_synthetic() -> None:
    import numbering
    from PIL import Image
    import io
    
    # Create a synthetic image with a bright center (simulating grown colony)
    img = Image.new("RGB", (4608, 3456), (50, 50, 50))  # dark background
    # Add orange rim (simulating dish)
    for y in range(100, 200):
        for x in range(100, 4500):
            img.putpixel((x, y), (255, 128, 0))  # orange
    for y in range(3256, 3356):
        for x in range(100, 4500):
            img.putpixel((x, y), (255, 128, 0))
    for x in range(100, 200):
        for y in range(100, 3356):
            img.putpixel((x, y), (255, 128, 0))
    for x in range(4408, 4508):
        for y in range(100, 3356):
            img.putpixel((x, y), (255, 128, 0))
    # Bright center (colony)
    for y in range(1000, 2000):
        for x in range(1500, 3000):
            img.putpixel((x, y), (220, 220, 220))
    
    buf = io.BytesIO()
    img.save(buf, "JPEG", quality=90)
    img_data = buf.getvalue()
    
    frac = numbering._bright_fraction(img_data, downscale=16, threshold=200)
    assert frac is not None
    assert 0 < frac < 1


def test_numbering_bright_fraction_dark() -> None:
    import numbering
    from PIL import Image
    import io
    
    # Dark image (early growth)
    img = Image.new("RGB", (4608, 3456), (30, 30, 30))
    buf = io.BytesIO()
    img.save(buf, "JPEG", quality=90)
    img_data = buf.getvalue()
    
    frac = numbering._bright_fraction(img_data, downscale=16, threshold=200)
    assert frac is not None
    assert frac < 0.1  # very few bright pixels


def test_numbering_bright_fraction_bright() -> None:
    import numbering
    from PIL import Image
    import io
    
    # Bright image (late growth)
    img = Image.new("RGB", (4608, 3456), (230, 230, 230))
    buf = io.BytesIO()
    img.save(buf, "JPEG", quality=90)
    img_data = buf.getvalue()
    
    frac = numbering._bright_fraction(img_data, downscale=16, threshold=200)
    assert frac is not None
    assert frac > 0.9  # mostly bright


def test_analyze_archive_synthetic(tmp_path) -> None:
    import numbering
    import zipfile
    from datetime import datetime, timezone
    
    # Create synthetic JPEGs with increasing brightness
    images = []
    for i in range(5):
        img = Image.new("RGB", (4608, 3456), (50 + i * 40, 50 + i * 40, 50 + i * 40))
        buf = io.BytesIO()
        img.save(buf, "JPEG", quality=90)
        images.append(buf.getvalue())
    
    archive_path = tmp_path / "test_numbering.zip"
    with zipfile.ZipFile(archive_path, "w") as zf:
        for i, img_data in enumerate(images):
            info = zipfile.ZipInfo(f"unlabeled/test/{i+1:08d}.jpg",
                                   date_time=(2024, 1, 1, 12, 0, i * 10))
            zf.writestr(info, img_data)
    
    cfg = numbering.ScanConfig(limit=5, downscale=16, brightness_threshold=200)
    rows = numbering.analyze_archive((archive_path, cfg))
    
    assert len(rows) == 5
    # IDs should be in order
    assert [r["id"] for r in rows] == ["00000001", "00000002", "00000003", "00000004", "00000005"]
    # Bright fractions should increase (approximately)
    bright_vals = [float(r["bright_frac"]) for r in rows if r["bright_frac"]]
    assert len(bright_vals) == 5
    # Generally increasing
    assert bright_vals[-1] > bright_vals[0]
