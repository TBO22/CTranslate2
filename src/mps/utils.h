#pragma once

#ifdef __APPLE__

#include <cstddef>

namespace ctranslate2 {
  namespace mps {

    bool has_mps();
    int get_device_count();
    int get_device_index();
    void set_device_index(int index);
    void synchronize();

    void* allocate_buffer(size_t size);
    void free_buffer(void* ptr);

    void* get_command_queue();
    void* get_device();

  }
}

#endif  // __APPLE__
