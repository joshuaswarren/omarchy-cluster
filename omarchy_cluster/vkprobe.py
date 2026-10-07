"""How much GPU memory one process can really allocate through Vulkan.

Honeykrisp (Mesa's Vulkan driver for Apple GPUs on Linux) reports a heap from
HK_SYSMEM, but its user GPU address space ends first: on an M2 Max with 94 GiB it
stopped at 63 GiB while the heap said 80. llama.cpp then fails with
"Failed to allocate BO VMA". This allocates 1 GiB DEVICE_LOCAL blocks until the
driver refuses, frees them, and prints JSON. Stdlib only (ctypes + libvulkan).

Run with HK_SYSMEM set to the machine's RAM so the heap is not what stops it:
    HK_SYSMEM=100000000000 python3 -m omarchy_cluster.vkprobe
"""
from __future__ import annotations

import ctypes
import json
import sys

GIB = 1 << 30


def count_blocks(alloc, free, max_blocks):
    """Allocate blocks with alloc() (a handle, or None when refused) until refused or
    max_blocks, free every one, and return how many succeeded."""
    got = []
    try:
        while len(got) < max_blocks:
            h = alloc()
            if h is None:
                break
            got.append(h)
    finally:
        for h in got:
            free(h)
    return len(got)


class _AppInfo(ctypes.Structure):
    _fields_ = [("sType", ctypes.c_uint32), ("pNext", ctypes.c_void_p), ("pApplicationName", ctypes.c_char_p),
                ("applicationVersion", ctypes.c_uint32), ("pEngineName", ctypes.c_char_p),
                ("engineVersion", ctypes.c_uint32), ("apiVersion", ctypes.c_uint32)]


class _InstanceInfo(ctypes.Structure):
    _fields_ = [("sType", ctypes.c_uint32), ("pNext", ctypes.c_void_p), ("flags", ctypes.c_uint32),
                ("pApplicationInfo", ctypes.POINTER(_AppInfo)), ("enabledLayerCount", ctypes.c_uint32),
                ("ppEnabledLayerNames", ctypes.c_void_p), ("enabledExtensionCount", ctypes.c_uint32),
                ("ppEnabledExtensionNames", ctypes.c_void_p)]


class _MemType(ctypes.Structure):
    _fields_ = [("propertyFlags", ctypes.c_uint32), ("heapIndex", ctypes.c_uint32)]


class _MemHeap(ctypes.Structure):
    _fields_ = [("size", ctypes.c_uint64), ("flags", ctypes.c_uint32)]


class _MemProps(ctypes.Structure):
    _fields_ = [("memoryTypeCount", ctypes.c_uint32), ("memoryTypes", _MemType * 32),
                ("memoryHeapCount", ctypes.c_uint32), ("memoryHeaps", _MemHeap * 16)]


class _QueueInfo(ctypes.Structure):
    _fields_ = [("sType", ctypes.c_uint32), ("pNext", ctypes.c_void_p), ("flags", ctypes.c_uint32),
                ("queueFamilyIndex", ctypes.c_uint32), ("queueCount", ctypes.c_uint32),
                ("pQueuePriorities", ctypes.POINTER(ctypes.c_float))]


class _DeviceInfo(ctypes.Structure):
    _fields_ = [("sType", ctypes.c_uint32), ("pNext", ctypes.c_void_p), ("flags", ctypes.c_uint32),
                ("queueCreateInfoCount", ctypes.c_uint32), ("pQueueCreateInfos", ctypes.POINTER(_QueueInfo)),
                ("enabledLayerCount", ctypes.c_uint32), ("ppEnabledLayerNames", ctypes.c_void_p),
                ("enabledExtensionCount", ctypes.c_uint32), ("ppEnabledExtensionNames", ctypes.c_void_p),
                ("pEnabledFeatures", ctypes.c_void_p)]


class _AllocInfo(ctypes.Structure):
    _fields_ = [("sType", ctypes.c_uint32), ("pNext", ctypes.c_void_p), ("allocationSize", ctypes.c_uint64),
                ("memoryTypeIndex", ctypes.c_uint32)]


_DEVICE_LOCAL = 0x1
_GPU_TYPES = (1, 2)  # VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU, DISCRETE_GPU


def probe(max_gib=256, lib="libvulkan.so.1"):
    """{"device", "heap_bytes", "alloc_cap_bytes"} for the first GPU (not a CPU
    implementation such as lavapipe)."""
    vk = ctypes.CDLL(lib)
    app = _AppInfo(0, None, b"omarchy-cluster-vkprobe", 1, None, 0, (1 << 22) | (2 << 12))
    ici = _InstanceInfo(1, None, 0, ctypes.pointer(app), 0, None, 0, None)
    inst = ctypes.c_void_p()
    if vk.vkCreateInstance(ctypes.byref(ici), None, ctypes.byref(inst)):
        raise RuntimeError("vkCreateInstance failed")
    try:
        n = ctypes.c_uint32(0)
        vk.vkEnumeratePhysicalDevices(inst, ctypes.byref(n), None)
        pds = (ctypes.c_void_p * n.value)()
        vk.vkEnumeratePhysicalDevices(inst, ctypes.byref(n), pds)
        props = ctypes.create_string_buffer(1024)  # VkPhysicalDeviceProperties is 824 bytes
        for pd in pds:
            vk.vkGetPhysicalDeviceProperties(ctypes.c_void_p(pd), props)
            if int.from_bytes(props.raw[16:20], "little") in _GPU_TYPES:
                break
        else:
            raise RuntimeError("no Vulkan GPU")
        name = props.raw[20:276].split(b"\0", 1)[0].decode(errors="replace")
        mp = _MemProps()
        vk.vkGetPhysicalDeviceMemoryProperties(ctypes.c_void_p(pd), ctypes.byref(mp))
        ti = next(i for i in range(mp.memoryTypeCount) if mp.memoryTypes[i].propertyFlags & _DEVICE_LOCAL)
        heap = mp.memoryHeaps[mp.memoryTypes[ti].heapIndex].size
        prio = ctypes.c_float(1.0)
        qi = _QueueInfo(2, None, 0, 0, 1, ctypes.pointer(prio))
        dci = _DeviceInfo(3, None, 0, 1, ctypes.pointer(qi), 0, None, 0, None, None)
        dev = ctypes.c_void_p()
        if vk.vkCreateDevice(ctypes.c_void_p(pd), ctypes.byref(dci), None, ctypes.byref(dev)):
            raise RuntimeError("vkCreateDevice failed")
        vk.vkFreeMemory.argtypes = [ctypes.c_void_p, ctypes.c_uint64, ctypes.c_void_p]
        try:
            def alloc():
                mem = ctypes.c_uint64(0)
                ai = _AllocInfo(5, None, GIB, ti)
                return None if vk.vkAllocateMemory(dev, ctypes.byref(ai), None, ctypes.byref(mem)) else mem.value

            blocks = count_blocks(alloc, lambda h: vk.vkFreeMemory(dev, h, None), max_gib)
        finally:
            vk.vkDestroyDevice(dev, None)
    finally:
        vk.vkDestroyInstance(inst, None)
    return {"device": name, "heap_bytes": heap, "alloc_cap_bytes": blocks * GIB}


if __name__ == "__main__":
    json.dump(probe(), sys.stdout)
    print()
