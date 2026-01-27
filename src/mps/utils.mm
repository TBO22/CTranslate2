#ifdef __APPLE__

#import <Metal/Metal.h>
#import <Foundation/Foundation.h>

#include <mutex>

#include "mps/utils.h"
#include "ctranslate2/utils.h"

namespace ctranslate2 {
  namespace mps {

    static id<MTLDevice> g_device = nil;
    static id<MTLCommandQueue> g_queue = nil;
    static int g_device_index = 0;

    static void ensure_initialized() {
      static std::once_flag once;
      std::call_once(once, []() {
        g_device = MTLCreateSystemDefaultDevice();
        if (g_device) {
          g_queue = [g_device newCommandQueue];
        }
      });
    }

    bool has_mps() {
      ensure_initialized();
      return g_device != nil;
    }

    int get_device_count() {
      if (!has_mps())
        return 0;
      return 1;
    }

    int get_device_index() {
      return g_device_index;
    }

    void set_device_index(int index) {
      if (index != 0)
        THROW_INVALID_ARGUMENT("MPS device index must be 0 (single device only)");
      g_device_index = index;
    }

    void synchronize() {
      if (!has_mps())
        return;
      id<MTLCommandBuffer> buf = [g_queue commandBuffer];
      [buf commit];
      [buf waitUntilCompleted];
    }

    void* allocate_buffer(size_t size) {
      if (!has_mps())
        throw std::runtime_error("MPS device not available");
      id<MTLBuffer> buf = [g_device newBufferWithLength:size
                                               options:MTLResourceStorageModeShared];
      if (!buf)
        throw std::runtime_error("MPS buffer allocation failed");
      return (__bridge void*)buf;
    }

    void free_buffer(void* ptr) {
      if (ptr) {
        id<MTLBuffer> buf = (__bridge_transfer id<MTLBuffer>)ptr;
        (void)buf;
      }
    }

    void* get_command_queue() {
      if (!has_mps())
        return nullptr;
      return (__bridge void*)g_queue;
    }

    void* get_device() {
      if (!has_mps())
        return nullptr;
      return (__bridge void*)g_device;
    }

  }
}

#endif  // __APPLE__
