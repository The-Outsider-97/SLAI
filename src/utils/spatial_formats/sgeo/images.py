"""Embedded texture images: PNG and JPEG only, validated without decoding pixels.

An SGEO file may carry encoded images (textures) byte for byte. This module
decides *whether the bytes are a well-formed image of an allowed size* so the
rest of the library -- and any renderer that later opens the file -- never has
to trust them:

* the container structure is walked completely (every PNG chunk and its CRC,
  every JPEG marker segment up to ``EOI``), so truncated, padded or spliced
  files are rejected;
* the declared dimensions are read from the header and capped
  (:data:`MAX_IMAGE_SIDE`, :data:`MAX_IMAGE_PIXELS`), which bounds the memory a
  viewer needs to decode the image -- pixel data is **never** decompressed
  here, so a decompression bomb costs nothing;
* only coding schemes every glTF viewer supports are accepted (PNG; baseline,
  extended-sequential and progressive Huffman JPEG, 8 bit, 1 or 3 components).

Nothing here executes, decodes or interprets pixel data.
"""
from __future__ import annotations

from dataclasses import dataclass
from zlib import crc32

from ..common.errors import ResourceLimitError, UnsupportedFeatureError, ValidationError

__all__ = ["ImageInfo", "inspect_image", "texture_path", "MIME_EXTENSIONS", "MAX_IMAGES", "MAX_IMAGE_BYTES",
           "MAX_TOTAL_IMAGE_BYTES", "MAX_IMAGE_SIDE", "MAX_IMAGE_PIXELS"]

MIME_EXTENSIONS = {"image/png": "png", "image/jpeg": "jpg"}
MAX_IMAGES = 1000
MAX_IMAGE_BYTES = 32 * 1024 * 1024          # one encoded image
MAX_TOTAL_IMAGE_BYTES = 64 * 1024 * 1024    # all encoded images of one scene
MAX_IMAGE_SIDE = 16384
MAX_IMAGE_PIXELS = 8192 * 8192
_MAX_PNG_CHUNKS = 100_000

_PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"
_PNG_DEPTHS = {0: (1, 2, 4, 8, 16), 2: (8, 16), 3: (1, 2, 4, 8), 4: (8, 16), 6: (8, 16)}
_PNG_CHANNELS = {0: 1, 2: 3, 3: 1, 4: 2, 6: 4}


@dataclass(frozen=True)
class ImageInfo:
    mime: str
    width: int
    height: int
    channels: int
    bit_depth: int


def texture_path(sha256: str, mime: str) -> str:
    """Archive member name of an image: content addressed, so identical images share one member."""
    try:
        extension = MIME_EXTENSIONS[mime]
    except (KeyError, TypeError):
        raise ValidationError("unsupported image type (only image/png and image/jpeg are allowed)") from None
    return "textures/sha256-%s.%s" % (sha256, extension)


def _check_size(width: int, height: int) -> None:
    if width < 1 or height < 1:
        raise ValidationError("image dimensions must be positive")
    if width > MAX_IMAGE_SIDE or height > MAX_IMAGE_SIDE or width * height > MAX_IMAGE_PIXELS:
        raise ResourceLimitError("image exceeds the size limit (at most %d pixels per side and %d pixels in all)"
                                 % (MAX_IMAGE_SIDE, MAX_IMAGE_PIXELS),
                                 context={"width": width, "height": height})


def _inspect_png(data: bytes) -> ImageInfo:
    size = len(data)
    pos = len(_PNG_SIGNATURE)
    width = height = depth = color = None
    seen_plte = seen_idat = ended = False
    idat_closed = False
    chunks = 0
    while pos < size:
        if ended:
            raise ValidationError("PNG has data after the IEND chunk")
        if size - pos < 12:
            raise ValidationError("truncated PNG chunk")
        length = int.from_bytes(data[pos:pos + 4], "big")
        kind = data[pos + 4:pos + 8]
        if length > 0x7FFFFFFF or pos + 12 + length > size:
            raise ValidationError("PNG chunk exceeds the file")
        if not all(65 <= b <= 90 or 97 <= b <= 122 for b in kind):
            raise ValidationError("invalid PNG chunk type")
        body = data[pos + 8:pos + 8 + length]
        if int.from_bytes(data[pos + 8 + length:pos + 12 + length], "big") != crc32(kind + body):
            raise ValidationError("PNG chunk checksum mismatch")
        chunks += 1
        if chunks > _MAX_PNG_CHUNKS:
            raise ResourceLimitError("PNG has too many chunks")
        if chunks == 1:
            if kind != b"IHDR" or length != 13:
                raise ValidationError("PNG must start with a 13 byte IHDR chunk")
            width = int.from_bytes(body[0:4], "big")
            height = int.from_bytes(body[4:8], "big")
            depth, color, compression, filtering, interlace = body[8], body[9], body[10], body[11], body[12]
            if color not in _PNG_DEPTHS or depth not in _PNG_DEPTHS[color]:
                raise ValidationError("invalid PNG colour type / bit depth combination")
            if compression != 0 or filtering != 0 or interlace not in (0, 1):
                raise ValidationError("invalid PNG compression, filter or interlace method")
            _check_size(width, height)
        elif kind == b"IHDR":
            raise ValidationError("duplicate PNG IHDR chunk")
        elif kind == b"PLTE":
            if seen_idat or seen_plte or length == 0 or length % 3 or length > 768 or color in (0, 4):
                raise ValidationError("invalid PNG palette")
            seen_plte = True
        elif kind == b"IDAT":
            if idat_closed:
                raise ValidationError("PNG image data chunks are not contiguous")
            seen_idat = True
        elif kind == b"IEND":
            if length:
                raise ValidationError("PNG IEND chunk must be empty")
            ended = True
        if seen_idat and kind != b"IDAT":
            idat_closed = True
        pos += 12 + length
    if not ended:
        raise ValidationError("PNG is missing its IEND chunk")
    if not seen_idat:
        raise ValidationError("PNG has no image data")
    if color == 3 and not seen_plte:
        raise ValidationError("indexed PNG has no palette")
    return ImageInfo("image/png", width, height, _PNG_CHANNELS[color], depth)


_JPEG_SOF = {0xC0, 0xC1, 0xC2}                       # Huffman sequential / extended / progressive
_JPEG_UNSUPPORTED_SOF = {0xC3, 0xC5, 0xC6, 0xC7, 0xC9, 0xCA, 0xCB, 0xCD, 0xCE, 0xCF}


def _inspect_jpeg(data: bytes) -> ImageInfo:
    size = len(data)
    pos = 2
    frame = None
    scans = 0
    while True:
        # find the next marker (any run of 0xFF fill bytes may precede it)
        if pos >= size or data[pos] != 0xFF:
            raise ValidationError("malformed JPEG marker")
        while pos < size and data[pos] == 0xFF:
            pos += 1
        if pos >= size:
            raise ValidationError("truncated JPEG")
        marker = data[pos]
        pos += 1
        if marker == 0xD9:                            # EOI
            if pos != size:
                raise ValidationError("JPEG has data after the EOI marker")
            break
        if marker in (0x00, 0x01, 0xD8) or 0xD0 <= marker <= 0xD7:
            raise ValidationError("unexpected JPEG marker")
        if pos + 2 > size:
            raise ValidationError("truncated JPEG")
        length = int.from_bytes(data[pos:pos + 2], "big")
        if length < 2 or pos + length > size:
            raise ValidationError("JPEG segment exceeds the file")
        segment = data[pos + 2:pos + length]
        if marker in _JPEG_UNSUPPORTED_SOF:
            raise UnsupportedFeatureError("unsupported JPEG coding process (arithmetic, lossless or hierarchical)")
        if marker in _JPEG_SOF:
            if frame is not None:
                raise ValidationError("JPEG has more than one frame header")
            if len(segment) < 6:
                raise ValidationError("truncated JPEG frame header")
            precision = segment[0]
            height = int.from_bytes(segment[1:3], "big")
            width = int.from_bytes(segment[3:5], "big")
            components = segment[5]
            if precision != 8:
                raise UnsupportedFeatureError("only 8 bit JPEG is supported")
            if components not in (1, 3) or len(segment) != 6 + 3 * components:
                raise UnsupportedFeatureError("only grey or three component JPEG is supported")
            if height == 0:
                raise UnsupportedFeatureError("JPEG with a DNL-defined height is not supported")
            _check_size(width, height)
            frame = (width, height, components)
        pos += length
        if marker == 0xDA:                            # SOS: entropy-coded data follows
            if frame is None:
                raise ValidationError("JPEG scan before its frame header")
            scans += 1
            while True:
                nxt = data.find(b"\xff", pos)
                if nxt < 0 or nxt + 1 >= size:
                    raise ValidationError("truncated JPEG scan")
                code = data[nxt + 1]
                if code == 0x00 or 0xD0 <= code <= 0xD7 or code == 0xFF:
                    pos = nxt + 1 if code == 0xFF else nxt + 2
                    continue
                pos = nxt
                break
    if frame is None or not scans:
        raise ValidationError("JPEG has no image data")
    return ImageInfo("image/jpeg", frame[0], frame[1], frame[2], 8)


def inspect_image(data) -> ImageInfo:
    """Validate ``data`` as a PNG or JPEG and return its header facts.

    Raises ``ValidationError`` for a malformed file, ``ResourceLimitError`` when it is too large and
    ``UnsupportedFeatureError`` for a JPEG coding process viewers cannot be expected to decode.
    """
    if not isinstance(data, (bytes, bytearray)):
        raise ValidationError("image data must be bytes")
    data = bytes(data)
    if not data:
        raise ValidationError("image data is empty")
    if len(data) > MAX_IMAGE_BYTES:
        raise ResourceLimitError("image exceeds the %d byte limit" % MAX_IMAGE_BYTES)
    if data.startswith(_PNG_SIGNATURE):
        return _inspect_png(data)
    if data.startswith(b"\xff\xd8\xff"):
        return _inspect_jpeg(data)
    raise ValidationError("unrecognised image data (only PNG and JPEG are supported)")
