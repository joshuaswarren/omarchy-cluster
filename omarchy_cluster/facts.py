"""Node facts: OS, chip, unified memory, GPU backend, free memory, interfaces."""
from __future__ import annotations

import json
import os
import platform
import re
import shutil
import socket
import subprocess

from . import AGENT_VERSION

# Apple Silicon device-tree target codes -> marketing names (Asahi SoC naming).
CHIP_NAMES = {
    "t8103": "Apple M1",
    "t6000": "Apple M1 Pro",
    "t6001": "Apple M1 Max",
    "t6002": "Apple M1 Ultra",
    "t6020": "Apple M2 Pro",
    "t6021": "Apple M2 Max",
    "t6022": "Apple M2 Ultra",
    "t6030": "Apple M3 Pro",
    "t6031": "Apple M3 Max",
    "t6034": "Apple M3 Max",
}

# MCDMA (RDMA transport, github.com/ashhart/MCDMA) detection inputs.
IB_ROOT = "/sys/class/infiniband"
MCDMA_PEER = "~/.local/libexec/mcdma"


def _run(cmd, timeout=6):
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        return p.stdout
    except (OSError, subprocess.SubprocessError):
        return ""


def _apple_chip_code():
    try:
        raw = open("/proc/device-tree/compatible", "rb").read().decode("utf-8", "replace")
    except OSError:
        return None
    for part in raw.split("\x00"):
        m = re.fullmatch(r"apple,t(\d+)", part.strip())
        if m:
            return "t" + m.group(1)
    return None


def _is_apple_silicon_linux():
    return platform.system() == "Linux" and _apple_chip_code() is not None


def perf_cores(cpu_root="/sys/devices/system/cpu"):
    """Number of the fastest CPU cores, the default llama.cpp thread count. One slow core in an
    OpenMP team holds every op back: on an M1 Max, decode with 2 threads ran 37.7 tok/s on two
    performance cores and 24.0 with one efficiency core in the pair. Linux: CPUs at the highest
    cpu_capacity. macOS: hw.perflevel0.physicalcpu. Falls back to every CPU."""
    if platform.system() == "Darwin":
        out = _run(["sysctl", "-n", "hw.perflevel0.physicalcpu"]).strip()
        return int(out) if out.isdigit() and int(out) > 0 else (os.cpu_count() or 4)
    caps = []
    try:
        for name in os.listdir(cpu_root):
            if re.fullmatch(r"cpu\d+", name):
                with open(os.path.join(cpu_root, name, "cpu_capacity")) as f:
                    caps.append(int(f.read()))
    except (OSError, ValueError):
        caps = []
    return caps.count(max(caps)) if caps else (os.cpu_count() or 4)


def chip():
    if platform.system() == "Darwin":
        out = _run(["sysctl", "-n", "machdep.cpu.brand_string"]).strip()
        return out or platform.machine()
    code = _apple_chip_code()
    if code:
        return CHIP_NAMES.get(code, "Apple " + code.upper())
    for line in _run(["lscpu"]).splitlines():
        if line.startswith("Model name:"):
            return line.split(":", 1)[1].strip()
    return platform.machine()


def os_name():
    if platform.system() == "Darwin":
        return "macOS " + (platform.mac_ver()[0] or platform.release())
    pretty = ""
    try:
        for line in open("/etc/os-release"):
            if line.startswith("PRETTY_NAME="):
                pretty = line.split("=", 1)[1].strip().strip('"')
    except OSError:
        pass
    return pretty or platform.system() + " " + platform.release()


def memory_total_bytes():
    if platform.system() == "Darwin":
        out = _run(["sysctl", "-n", "hw.memsize"]).strip()
        return int(out) if out.isdigit() else 0
    try:
        for line in open("/proc/meminfo"):
            if line.startswith("MemTotal:"):
                return int(line.split()[1]) * 1024
    except OSError:
        pass
    return 0


def memory_free_bytes():
    if platform.system() == "Darwin":
        pagesize = int(_run(["sysctl", "-n", "hw.pagesize"]).strip() or "16384")
        free = inactive = 0
        for line in _run(["vm_stat"]).splitlines():
            if line.startswith("Pages free:"):
                free = int(line.split()[-1].rstrip("."))
            elif line.startswith("Pages inactive:"):
                inactive = int(line.split()[-1].rstrip("."))
        return (free + inactive) * pagesize
    try:
        for line in open("/proc/meminfo"):
            if line.startswith("MemAvailable:"):
                return int(line.split()[1]) * 1024
    except OSError:
        pass
    return 0


def gpu_backend():
    """metal on macOS; on Linux: cuda (NVIDIA), vulkan (any ICD), else none."""
    if platform.system() == "Darwin":
        return "metal"
    if shutil.which("nvidia-smi"):
        return "cuda"
    try:
        if any(f.endswith(".json") for f in os.listdir("/usr/share/vulkan/icd.d")):
            return "vulkan"
    except OSError:
        pass
    return "none"


def boot_id():
    if platform.system() == "Darwin":
        return _run(["sysctl", "-n", "kern.boottime"]).strip()
    try:
        return open("/proc/sys/kernel/random/boot_id").read().strip()
    except OSError:
        return ""


def _prefix_from_netmask_hex(hexstr):
    try:
        return bin(int(hexstr, 16)).count("1")
    except ValueError:
        return 0


def _linux_interfaces():
    out = _run(["ip", "-j", "addr"])
    try:
        data = json.loads(out)
    except ValueError:
        return []
    result = []
    for dev in data:
        name = dev.get("ifname", "")
        if name in ("lo", ""):
            continue
        ips = [
            {"ip": a["local"], "prefix": int(a.get("prefixlen", 0))}
            for a in dev.get("addr_info", [])
            if a.get("family") == "inet"
        ]
        if not ips:
            continue
        speed = None
        try:
            s = open("/sys/class/net/%s/speed" % name).read().strip()
            if s.isdigit():
                speed = int(s)
        except OSError:
            pass
        result.append({
            "name": name,
            "ips": ips,
            "speed_mbps": speed,
            "wireless": os.path.isdir("/sys/class/net/%s/wireless" % name) or name.startswith("wl"),
        })
    return result


def _mac_interfaces():
    ports = {}
    out = _run(["networksetup", "-listallhardwareports"])
    portname = ""
    for line in out.splitlines():
        if line.startswith("Hardware Port:"):
            portname = line.split(":", 1)[1].strip()
        elif line.startswith("Device:"):
            ports[line.split(":", 1)[1].strip()] = portname

    out = _run(["ifconfig", "-a"])
    result = []
    cur = None
    info = None
    for line in out.splitlines() + ["\0"]:
        m = re.match(r"^(\S+?): flags=", line)
        if m or line == "\0":
            if info and info["ips"]:
                result.append(info)
            cur = m.group(1) if m else None
            info = {"name": cur, "ips": [], "speed_mbps": None,
                    "wireless": ports.get(cur) in ("Wi-Fi", "AirPort")} if cur else None
            if cur == "lo0":
                cur, info = None, None
            continue
        if info is None:
            continue
        m = re.match(r"\s+inet (\d+\.\d+\.\d+\.\d+) netmask 0x([0-9a-fA-F]+)", line)
        if m:
            info["ips"].append({"ip": m.group(1), "prefix": _prefix_from_netmask_hex(m.group(2))})
            continue
        m = re.search(r"media: \S+ \((\d+)(\S*)base", line) or re.search(r"media: (\d+)(\S*)base", line)
        if m and info["speed_mbps"] is None:
            mult = 1000 if "g" in (m.group(2) or "").lower() else 1
            info["speed_mbps"] = int(m.group(1)) * mult
    return result


def interfaces():
    if platform.system() == "Darwin":
        return _mac_interfaces()
    return _linux_interfaces()


def mcdma():
    """MCDMA (RDMA) on this node: {"available", "reason", "devices"}. Detection only.
    macOS: Apple Thunderbolt RDMA enabled (`rdma_ctl status`) with an rdma_en* interface.
    Linux: an ACTIVE port under /sys/class/infiniband and the MCDMA peer tool."""
    if platform.system() == "Darwin":
        if not shutil.which("rdma_ctl"):
            return {"available": False, "reason": "no rdma_ctl", "devices": []}
        status = _run(["rdma_ctl", "status"]).strip() or "unknown"
        devs = [i for i in _run(["ifconfig", "-l"]).split() if i.startswith("rdma_en")]
        if status != "enabled":
            return {"available": False, "reason": "rdma_ctl status: %s" % status, "devices": devs}
        if not devs:
            return {"available": False, "reason": "rdma enabled but no rdma_en interface", "devices": []}
        return {"available": True, "reason": "rdma_ctl enabled", "devices": devs}
    try:
        names = sorted(os.listdir(IB_ROOT))
    except OSError:
        names = []
    if not names:
        return {"available": False, "reason": "no /sys/class/infiniband devices", "devices": []}
    active = []
    for d in names:
        ports = os.path.join(IB_ROOT, d, "ports")
        try:
            plist = sorted(os.listdir(ports))
        except OSError:
            continue
        for port in plist:
            try:
                with open(os.path.join(ports, port, "state")) as f:
                    if "ACTIVE" in f.read():
                        active.append("%s/%s" % (d, port))
            except OSError:
                pass
    if not active:
        return {"available": False, "reason": "no ACTIVE infiniband port", "devices": []}
    if not os.access(os.path.expanduser(MCDMA_PEER), os.X_OK):
        return {"available": False, "reason": "no MCDMA peer tool at %s" % MCDMA_PEER, "devices": active}
    return {"available": True, "reason": "ACTIVE port and peer tool", "devices": active}


_DATA_VOLUME_JXA = ('ObjC.import("Foundation"); var k = "NSURLVolumeAvailableCapacityForImportantUsageKey"; '
                    'ObjC.unwrap($.NSURL.fileURLWithPath("/System/Volumes/Data")'
                    '.resourceValuesForKeysError($.NSArray.arrayWithObject($(k)), null).objectForKey(k));')


def data_volume_free():
    """(bytes, source) free on the macOS Data volume, the one that holds swap. "important": the space macOS
    itself offers apps, which counts purgeable space (content caches, local snapshots) that df leaves out
    (85 GB by df, 435 GB usable on the Mac Studio). "statvfs": df's number, used when that read fails; it
    is a floor, not a fact. (None, None) when neither works."""
    out = _run(["/usr/bin/osascript", "-l", "JavaScript", "-e", _DATA_VOLUME_JXA], timeout=10).strip()
    if out.isdigit():
        return int(out), "important"
    try:
        st = os.statvfs("/System/Volumes/Data")
        return st.f_bavail * st.f_frsize, "statvfs"
    except OSError:
        return None, None


def macos_memory():
    """macOS only: {"memory_free_pages_bytes": free pages, "memory_purgeable_bytes",
    "memory_file_backed_bytes": cached file pages (dropped without swapping),
    "memory_reclaimable_bytes": inactive + speculative pages (anonymous ones swap),
    "data_volume_free_bytes": free space on the volume that holds swap}."""
    pagesize = int(_run(["sysctl", "-n", "hw.pagesize"]).strip() or "16384")
    pages = {"Pages free": 0, "Pages inactive": 0, "Pages speculative": 0, "Pages purgeable": 0,
             "File-backed pages": 0}
    for line in _run(["vm_stat"]).splitlines():
        for k in pages:
            if line.startswith(k + ":"):
                pages[k] = int(line.split()[-1].rstrip("."))
    data_free, data_source = data_volume_free()
    m = re.search(r"percentage:\s*(\d+)%", _run(["memory_pressure", "-Q"]))
    return {"memory_pressure_free_pct": int(m.group(1)) if m else None,
            "memory_free_pages_bytes": pages["Pages free"] * pagesize,
            "memory_purgeable_bytes": pages["Pages purgeable"] * pagesize,
            "memory_file_backed_bytes": pages["File-backed pages"] * pagesize,
            "memory_reclaimable_bytes": (pages["Pages inactive"] + pages["Pages speculative"]) * pagesize,
            "data_volume_free_bytes": data_free, "data_volume_free_source": data_source}


def collect_facts():
    facts = {
        "agent_version": AGENT_VERSION,
        "name": socket.gethostname().split(".")[0],
        "os": os_name(),
        "arch": platform.machine(),
        "chip": chip(),
        "gpu_backend": gpu_backend(),
        "unified_memory": platform.system() == "Darwin" or _is_apple_silicon_linux(),
        "memory_total_bytes": memory_total_bytes(),
        "memory_free_bytes": memory_free_bytes(),
        "interfaces": interfaces(),
        "boot_id": boot_id(),
        "tools": {"iperf3": bool(shutil.which("iperf3"))},
        "transports": {"mcdma": mcdma()},
    }
    if platform.system() == "Darwin":
        facts.update(macos_memory())
    return facts
