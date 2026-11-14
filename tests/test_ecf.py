import struct

import numpy as np

from boxprop.ecf import decode_chunk


def _ts_stream(times: list[int]) -> bytes:
    out = bytearray(struct.pack("<Q", times[0]))
    cur = times[0]
    i = 0
    while i < len(times):
        t = times[i]
        count = 1
        while i + count < len(times) and times[i + count] == t and count < 14:
            count += 1
        delta = t - cur
        if delta >= 15:
            nib = []
            while delta:
                nib.append(delta & 0xF)
                delta >>= 4
            out += bytes(0xF0 | n for n in nib)
            delta = 0
        out.append((delta << 4) | count)
        cur = t
        i += count
    return bytes(out)


def test_decode_runs_masks_and_polarity_runs():
    times = [1000, 1000, 1003, 1003, 1003, 1100, 1100]
    ys = [5, 5, 5, 9, 9, 9, 9]
    xs = [10, 11, 13, 40, 41, 42, 700]
    ps = [1, 1, 1, 0, 0, 0, 1]
    n = len(times)
    buf = bytearray(struct.pack("<I", n << 2))
    buf += _ts_stream(times)
    buf += struct.pack("<HH", (5 << 5) | 3, (9 << 5) | 4)
    buf += struct.pack("<HHH", (10 << 5) | 0b10100, (40 << 5) | 0b11000, (700 << 5))
    buf += bytes([(1 << 7) | 3, 3, (1 << 7) | 1])
    ev = decode_chunk(bytes(buf))
    assert ev["t"].tolist() == times
    assert ev["y"].tolist() == ys
    assert ev["x"].tolist() == xs
    assert ev["p"].tolist() == ps


def test_decode_packed_xyp():
    times = [50, 51, 52, 53, 54]
    xyp = [(100, 7, 1), (1279, 719, 0), (0, 0, 1), (640, 360, 0), (5, 6, 1)]
    vals = [(y << 12) | (x << 1) | p for x, y, p in xyp] + [0, 0, 0]
    words = []
    for k in range(0, len(vals) - 3, 4):
        v0, v1, v2, v3 = vals[k:k + 4]
        words += [(v0 << 8) | (v1 & 0xFF), ((v1 >> 8) << 16) | (v2 & 0xFFFF), ((v2 >> 16) << 24) | v3]
    buf = bytearray(struct.pack("<I", (len(times) << 2) | 0b10))
    buf += _ts_stream(times)
    buf += struct.pack(f"<{len(words)}I", *words)
    ev = decode_chunk(bytes(buf))
    assert ev["t"].tolist() == times
    assert list(zip(ev["x"].tolist(), ev["y"].tolist(), ev["p"].tolist())) == xyp


def test_large_time_jump_uses_nibble_prefix():
    times = [0, 70000, 70000]
    buf = bytearray(struct.pack("<I", 3 << 2))
    buf += _ts_stream(times)
    buf += struct.pack("<H", (1 << 5) | 3)
    buf += struct.pack("<HH", (3 << 5) | 0b10000, (9 << 5))
    buf += bytes([3])
    ev = decode_chunk(bytes(buf))
    assert ev["t"].tolist() == times
    assert ev["x"].tolist() == [3, 4, 9]
    assert np.all(ev["p"] == 0)
