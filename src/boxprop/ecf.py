"""Decoder for Prophesee's ECF-compressed HDF5 event files, ported from prophesee-ai/hdf5_ecf (Apache-2.0)."""
from pathlib import Path

import numpy as np

EVENT_DTYPE = np.dtype([("x", "<u2"), ("y", "<u2"), ("p", "<i2"), ("t", "<i8")])

try:
    from numba import njit
except ImportError:
    def njit(*args, **kwargs):
        if args and callable(args[0]):
            return args[0]
        return lambda f: f


@njit(cache=True)
def _decode(buf, ts, ys, xs, ps):
    header = buf[0] | (buf[1] << 8) | (buf[2] << 16) | (buf[3] << 24)
    n = header >> 2
    yxp_packed = (header >> 1) & 1
    xp_packed = header & 1
    pos = 4
    t0 = np.int64(0)
    for k in range(8):
        t0 |= buf[pos + k] << (8 * k)
    pos += 8
    cur = t0
    i = 0
    while i < n:
        v = buf[pos]
        t = v >> 4
        c = v & 0xF
        if t != 0xF:
            cur += t
            if c == 0xF:
                c = buf[pos + 1] | (buf[pos + 2] << 8)
                pos += 2
            for _ in range(c):
                ts[i] = cur
                i += 1
            pos += 1
        else:
            dt = np.int64(0)
            j = 0
            while t == 0xF:
                dt |= c << (4 * j)
                j += 1
                pos += 1
                v = buf[pos]
                t = v >> 4
                c = v & 0xF
            cur += dt
    if yxp_packed:
        i = 0
        while i < n:
            w0 = buf[pos] | (buf[pos + 1] << 8) | (buf[pos + 2] << 16) | (buf[pos + 3] << 24)
            w1 = buf[pos + 4] | (buf[pos + 5] << 8) | (buf[pos + 6] << 16) | (buf[pos + 7] << 24)
            w2 = buf[pos + 8] | (buf[pos + 9] << 8) | (buf[pos + 10] << 16) | (buf[pos + 11] << 24)
            vs = (w0 >> 8, (w0 & 0xFF) | ((w1 >> 16) << 8), (w1 & 0xFFFF) | ((w2 >> 24) << 16), w2 & 0xFFFFFF)
            for j in range(4):
                if i + j < n:
                    ys[i + j] = (vs[j] >> 12) & 0x7FF
                    xs[i + j] = (vs[j] >> 1) & 0x7FF
                    ps[i + j] = vs[j] & 1
            i += 4
            pos += 12
        return n
    i = 0
    while i < n:
        v = buf[pos] | (buf[pos + 1] << 8)
        y = v >> 5
        c = v & 0x1F
        if c == 0x1F:
            c = buf[pos + 2] | (buf[pos + 3] << 8)
            pos += 2
        for _ in range(c):
            ys[i] = y
            i += 1
        pos += 2
    if xp_packed:
        i = 0
        while i < n:
            a = buf[pos] | (buf[pos + 1] << 8)
            b = buf[pos + 2] | (buf[pos + 3] << 8)
            c = buf[pos + 4] | (buf[pos + 5] << 8)
            vs = (a >> 4, (a & 0xF) | ((b >> 8) << 4), (b & 0xFF) | ((c >> 12) << 8), c & 0xFFF)
            for j in range(4):
                if i + j < n:
                    xs[i + j] = vs[j] >> 1
                    ps[i + j] = vs[j] & 1
            i += 4
            pos += 6
        return n
    i = 0
    while i < n:
        v = buf[pos] | (buf[pos + 1] << 8)
        x = v >> 5
        m = v & 0x1F
        xs[i] = x
        i += 1
        for j in range(5):
            if m & (1 << (4 - j)):
                if i < n:
                    xs[i] = x + j + 1
                i += 1
        pos += 2
    i = 0
    while i < n:
        v = buf[pos]
        p = v >> 7
        c = v & 0x7F
        if c == 0x7F:
            c = buf[pos + 1] | (buf[pos + 2] << 8)
            pos += 2
        for _ in range(c):
            ps[i] = p
            i += 1
        pos += 1
    return n


def decode_chunk(raw: bytes) -> np.ndarray:
    buf = np.frombuffer(raw, dtype=np.uint8)
    n = int(np.frombuffer(raw[:4], "<u4")[0] >> 2)
    pad = np.concatenate([buf, np.zeros(16, np.uint8)]).astype(np.int64)
    ts = np.zeros(n + 5, np.int64)
    ys = np.zeros(n + 5, np.uint16)
    xs = np.zeros(n + 5, np.uint16)
    ps = np.zeros(n + 5, np.int16)
    _decode(pad, ts, ys, xs, ps)
    out = np.empty(n, EVENT_DTYPE)
    out["t"], out["x"], out["y"], out["p"] = ts[:n], xs[:n], ys[:n], ps[:n]
    return out


def read_events(path: str | Path, t_start: int | None = None, t_end: int | None = None,
                dataset: str = "CD/events") -> np.ndarray:
    import h5py

    with h5py.File(path, "r") as f:
        d = f[dataset]
        chunk = d.chunks[0]
        n_chunks = (d.shape[0] + chunk - 1) // chunk
        filters = list(d._filters.keys())
        first, last = 0, n_chunks
        if (t_start is not None or t_end is not None) and "CD/indexes" in f:
            ix = f["CD/indexes"]
            first, last = _chunk_range(ix["ts"].astype(np.int64) - int(ix.attrs.get("offset", 0)),
                                       ix["id"].astype(np.int64), d.shape[0], chunk, n_chunks, t_start, t_end)
        parts = []
        for c in range(first, last):
            if filters:
                _, raw = d.id.read_direct_chunk((c * chunk,))
                parts.append(decode_chunk(raw))
            else:
                parts.append(d[c * chunk:(c + 1) * chunk].astype(EVENT_DTYPE))
    ev = np.concatenate(parts) if parts else np.empty(0, EVENT_DTYPE)
    if t_start is not None:
        ev = ev[ev["t"] >= t_start]
    if t_end is not None:
        ev = ev[ev["t"] < t_end]
    return ev


def _chunk_range(ts: np.ndarray, ids: np.ndarray, n_events: int, chunk: int, n_chunks: int,
                 t_start: int | None, t_end: int | None) -> tuple[int, int]:
    valid = ids >= 0
    ts, ids = ts[valid], ids[valid]
    if len(ts) == 0:
        return 0, n_chunks
    lo = ids[max(np.searchsorted(ts, t_start, "right") - 1, 0)] if t_start is not None else 0
    hi_i = np.searchsorted(ts, t_end, "left") if t_end is not None else len(ts)
    hi = ids[hi_i] if hi_i < len(ids) else n_events
    return int(lo // chunk), int(min(n_chunks, hi // chunk + 1))
