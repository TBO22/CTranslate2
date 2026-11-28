// Linux ELF diagnostics only. No CTranslate2 source or numerical code changes.
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <dlfcn.h>

#include <ctranslate2/storage_view.h>

#include "symbols.h"

struct Counters {
  uint64_t encoder_ns, decoder_ns, encoder_calls, decoder_calls;
  uint64_t decoder_row_steps, max_step_plus_one;
};

static std::atomic<uint64_t> encoder_ns{0}, decoder_ns{0};
static std::atomic<uint64_t> encoder_calls{0}, decoder_calls{0};
static std::atomic<uint64_t> decoder_row_steps{0}, max_step_plus_one{0};

template <typename T> static T original(const char* name) {
  static void* handle = [] {
    const char* path = std::getenv("CT2_DIAG_LIBRARY");
    void* library = path ? dlopen(path, RTLD_NOW | RTLD_LOCAL) : nullptr;
    if (!library) {
      std::fprintf(stderr, "Cannot open diagnostic CT2 library: %s\n", dlerror());
      std::abort();
    }
    return library;
  }();
  void* symbol = dlsym(handle, name);
  if (!symbol) {
    std::fprintf(stderr, "Missing diagnostic symbol %s: %s\n", name, dlerror());
    std::abort();
  }
  return reinterpret_cast<T>(symbol);
}

extern "C" void diag_reset() {
  encoder_ns = decoder_ns = encoder_calls = decoder_calls = 0;
  decoder_row_steps = max_step_plus_one = 0;
}

extern "C" void diag_read(Counters* result) {
  *result = {encoder_ns.load(), decoder_ns.load(), encoder_calls.load(),
             decoder_calls.load(), decoder_row_steps.load(), max_step_plus_one.load()};
}

using Clock = std::chrono::steady_clock;
static uint64_t elapsed(Clock::time_point start) {
  return std::chrono::duration_cast<std::chrono::nanoseconds>(Clock::now() - start).count();
}

// Opaque pointers preserve the x86-64 Itanium C++ ABI's reference/this arguments.
// Symbol names are extracted and verified against BOTH exact wheel libraries.
extern "C" void encoder_hook(void*, const void*, const void*, void*) asm(ENCODER_SYMBOL);
extern "C" void encoder_hook(void* self, const void* ids, const void* lengths,
                              void* output) {
  using Function = void (*)(void*, const void*, const void*, void*);
  static const Function call = original<Function>(ENCODER_SYMBOL);
  const auto start = Clock::now();
  call(self, ids, lengths, output);
  encoder_ns += elapsed(start);
  ++encoder_calls;
}

extern "C" void decoder_hook(void*, const void*, const void*, int64_t, void*, void*,
                              void*, bool) asm(DECODER_SYMBOL);
extern "C" void decoder_hook(void* self, const void* ids, const void* lengths,
                              int64_t step, void* state, void* output,
                              void* attention, bool return_logits) {
  using Function = void (*)(void*, const void*, const void*, int64_t, void*, void*, void*, bool);
  static const Function call = original<Function>(DECODER_SYMBOL);
  const uint64_t rows = static_cast<const ctranslate2::StorageView*>(ids)->dim(0);
  const auto start = Clock::now();
  call(self, ids, lengths, step, state, output, attention, return_logits);
  decoder_ns += elapsed(start);
  ++decoder_calls;
  decoder_row_steps += rows;
  const uint64_t position = step >= 0 ? step + 1 : 0;
  auto previous = max_step_plus_one.load();
  while (previous < position && !max_step_plus_one.compare_exchange_weak(previous, position)) {}
}
