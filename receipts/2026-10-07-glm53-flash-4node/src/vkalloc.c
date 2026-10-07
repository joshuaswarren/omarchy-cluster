// Allocate 1 GiB DEVICE_LOCAL blocks until the driver refuses or MAXGIB is reached; print the count.
#include <vulkan/vulkan.h>
#include <stdio.h>
#include <stdlib.h>
int main(int argc, char **argv) {
  int maxg = argc > 1 ? atoi(argv[1]) : 64;
  VkApplicationInfo app = {VK_STRUCTURE_TYPE_APPLICATION_INFO, 0, "vkalloc", 1, 0, 0, VK_API_VERSION_1_2};
  VkInstanceCreateInfo ici = {VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO, 0, 0, &app};
  VkInstance inst; if (vkCreateInstance(&ici, 0, &inst)) return 2;
  uint32_t n = 1; VkPhysicalDevice pd; vkEnumeratePhysicalDevices(inst, &n, &pd); if (!n) return 3;
  VkPhysicalDeviceMemoryProperties mp; vkGetPhysicalDeviceMemoryProperties(pd, &mp);
  uint32_t ti = 0; for (; ti < mp.memoryTypeCount; ti++) if (mp.memoryTypes[ti].propertyFlags & VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT) break;
  printf("heap0 %llu MiB\n", (unsigned long long)(mp.memoryHeaps[0].size >> 20));
  float pr = 1; VkDeviceQueueCreateInfo q = {VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO, 0, 0, 0, 1, &pr};
  VkDeviceCreateInfo dci = {VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO, 0, 0, 1, &q};
  VkDevice dev; if (vkCreateDevice(pd, &dci, 0, &dev)) return 4;
  VkDeviceMemory *m = calloc(maxg, sizeof *m); int i = 0;
  for (; i < maxg; i++) {
    VkMemoryAllocateInfo a = {VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO, 0, 1ull << 30, ti};
    VkResult r = vkAllocateMemory(dev, &a, 0, &m[i]);
    if (r) { printf("alloc %d failed: %d\n", i + 1, r); break; }
  }
  printf("allocated %d GiB\n", i);
  for (int j = 0; j < i; j++) vkFreeMemory(dev, m[j], 0);
  vkDestroyDevice(dev, 0); vkDestroyInstance(inst, 0); return 0;
}
