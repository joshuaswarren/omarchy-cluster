"""Observe paging while a model decodes; changes nothing.

    python3 -m omarchy_cluster.paging --out FILE [--model SHARD1 --layers A-B] [--interval S]

Every interval, one JSON line: the node's page-in and fault counters since the last line
(Linux /proc/vmstat pgmajfault, pgpgin, pswpin, pswpout; macOS vm_stat pageins, swapins,
swapouts), and with --model the share of each layer's bytes that sits in the page cache
(mincore on the GGUF file). On the node that mmaps layers it cannot hold, the layers with
the lowest share are the ones the kernel evicts and re-reads. Only that node pages; the other
ranks hold their layers in device memory, so run it there for the counters alone."""
import argparse
import ctypes
import json
import mmap
import signal
import subprocess
import sys
import time

_LINUX_KEYS = ("pgmajfault", "pgpgin", "pswpin", "pswpout")
_MAC_KEYS = {"Pageins": "pageins", "Swapins": "swapins", "Swapouts": "swapouts"}


def read_counters():
    """Cumulative paging counters of this node."""
    if sys.platform == "darwin":
        out = subprocess.run(["vm_stat"], capture_output=True, text=True, check=True).stdout
        res = {}
        for line in out.splitlines():
            key, _, val = line.partition(":")
            if key in _MAC_KEYS:
                res[_MAC_KEYS[key]] = int(val.strip().rstrip("."))
        return res
    with open("/proc/vmstat") as f:
        vm = dict(line.split() for line in f)
    return {k: int(vm[k]) for k in _LINUX_KEYS if k in vm}


def delta(a, b):
    return {k: b[k] - a[k] for k in b if k in a}


def _resident_pages(path, off, size):
    """(pages of [off, off+size) in the page cache, pages in the range), through mincore."""
    gran = mmap.ALLOCATIONGRANULARITY
    start = off // gran * gran
    length = off + size - start
    npages = -(-length // mmap.PAGESIZE)
    libc = ctypes.CDLL(None, use_errno=True)
    libc.mincore.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_void_p]
    vec = (ctypes.c_ubyte * npages)()
    with open(path, "rb") as f, mmap.mmap(f.fileno(), length, access=mmap.ACCESS_COPY, offset=start) as mm:
        buf = ctypes.c_char.from_buffer(mm)
        try:
            if libc.mincore(ctypes.addressof(buf), length, vec) != 0:
                raise OSError(ctypes.get_errno(), "mincore " + path)
        finally:
            del buf  # an exported buffer would keep the mmap from closing
    return sum(v & 1 for v in vec), npages


def resident_fraction(extents):
    """Share of a layer's pages that are in the page cache, from its (path, offset, size)
    extents. Pages shared with a neighbouring tensor count once per extent."""
    hit = total = 0
    for path, off, size in extents:
        h, n = _resident_pages(path, off, size)
        hit, total = hit + h, total + n
    return hit / total if total else 1.0


def main(argv=None):
    ap = argparse.ArgumentParser(description=(__doc__ or "").split("\n")[0])
    ap.add_argument("--out", required=True)
    ap.add_argument("--interval", type=float, default=1.0)
    ap.add_argument("--model", help="first GGUF shard, for per-layer residency")
    ap.add_argument("--layers", help="A-B: the layers this node mmaps (with --model)")
    args = ap.parse_args(argv)
    layers, ext = [], None
    if args.model:
        from . import llamacpp_engine
        a, _, b = (args.layers or "").partition("-")
        if not a.isdigit() or not b.isdigit():
            sys.exit("--model needs --layers A-B")
        ext = llamacpp_engine.layer_extents(args.model)
        layers = list(range(int(a), int(b) + 1))
    stop = []
    signal.signal(signal.SIGTERM, lambda *_: stop.append(1))
    prev = read_counters()
    with open(args.out, "a") as out:
        while not stop:
            try:
                time.sleep(args.interval)
            except KeyboardInterrupt:
                break
            cur = read_counters()
            row = {"t": round(time.time(), 3), "d": delta(prev, cur), "ts": args.interval}
            if ext:
                row["layers"] = {str(il): round(resident_fraction(ext[il]), 4) for il in layers}
            prev = cur
            out.write(json.dumps(row) + "\n")
            out.flush()


if __name__ == "__main__":
    main()
