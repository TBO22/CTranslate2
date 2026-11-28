// Linux ELF diagnostics only. No CTranslate2 source or numerical code changes.
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <dlfcn.h>
#include <sys/resource.h>

#include <ctranslate2/storage_view.h>

#include "symbols.h"

struct Counters {
  uint64_t encoder_ns, decoder_ns, encoder_calls, decoder_calls;
  uint64_t decoder_row_steps, max_step_plus_one;
};

static std::atomic<uint64_t> encoder_ns{0}, decoder_ns{0};
static std::atomic<uint64_t> encoder_calls{0}, decoder_calls{0};
static std::atomic<uint64_t> decoder_row_steps{0}, max_step_plus_one{0};
static constexpr int categories = 6;
static std::atomic<uint64_t> operator_ns[2 * categories]{}, operator_calls[2 * categories]{};
static std::atomic<uint64_t> operator_minor_faults[2 * categories]{}, operator_major_faults[2 * categories]{};
static thread_local int active_phase = 0;
static const bool operators_enabled = std::getenv("CT2_DIAG_OPERATORS") != nullptr;

struct Phase {
  int previous;
  explicit Phase(int phase) : previous(active_phase) { active_phase = phase; }
  ~Phase() { active_phase = previous; }
};

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
  for (int i = 0; i < 2 * categories; ++i) {
    operator_ns[i] = 0;
    operator_calls[i] = 0;
    operator_minor_faults[i] = 0;
    operator_major_faults[i] = 0;
  }
}

extern "C" void diag_read_operators(uint64_t* result) {
  for (int i = 0; i < 2 * categories; ++i) {
    result[i] = operator_ns[i].load();
    result[i + 2 * categories] = operator_calls[i].load();
    result[i + 4 * categories] = operator_minor_faults[i].load();
    result[i + 6 * categories] = operator_major_faults[i].load();
  }
}

extern "C" void diag_read(Counters* result) {
  *result = {encoder_ns.load(), decoder_ns.load(), encoder_calls.load(),
             decoder_calls.load(), decoder_row_steps.load(), max_step_plus_one.load()};
}

using Clock = std::chrono::steady_clock;
static uint64_t elapsed(Clock::time_point start) {
  return std::chrono::duration_cast<std::chrono::nanoseconds>(Clock::now() - start).count();
}

template <typename Function> static void time_operator(int category, Function call) {
  if (!operators_enabled || !active_phase) {
    call();
    return;
  }
  const int index = (active_phase - 1) * categories + category;
  rusage before{}, after{};
  if (getrusage(RUSAGE_SELF, &before) != 0) std::abort();
  const auto start = Clock::now();
  call();
  operator_ns[index] += elapsed(start);
  if (getrusage(RUSAGE_SELF, &after) != 0) std::abort();
  operator_minor_faults[index] += static_cast<uint64_t>(after.ru_minflt - before.ru_minflt);
  operator_major_faults[index] += static_cast<uint64_t>(after.ru_majflt - before.ru_majflt);
  ++operator_calls[index];
}

// Opaque pointers preserve the x86-64 Itanium C++ ABI's reference/this arguments.
// Symbol names are extracted and verified against BOTH exact wheel libraries.
extern "C" void encoder_hook(void*, const void*, const void*, void*) asm(ENCODER_SYMBOL);
extern "C" void encoder_hook(void* self, const void* ids, const void* lengths,
                              void* output) {
  using Function = void (*)(void*, const void*, const void*, void*);
  static const Function call = original<Function>(ENCODER_SYMBOL);
  Phase phase(1);
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
  Phase phase(2);
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

extern "C" void dense_hook(void*, const void*, void*, const void*) asm(DENSE_SYMBOL);
extern "C" void dense_hook(void* self, const void* input, void* output, const void* residual) {
  using Function = void (*)(void*, const void*, void*, const void*);
  static const Function call = original<Function>(DENSE_SYMBOL);
  time_operator(0, [&] { call(self, input, output, residual); });
}

extern "C" void matmul_hook(void*, const void*, const void*, void*) asm(MATMUL_SYMBOL);
extern "C" void matmul_hook(void* self, const void* a, const void* b, void* c) {
  using Function = void (*)(void*, const void*, const void*, void*);
  static const Function call = original<Function>(MATMUL_SYMBOL);
  time_operator(1, [&] { call(self, a, b, c); });
}

extern "C" void softmax_hook(void*, const void*, const void*, void*) asm(SOFTMAX_SYMBOL);
extern "C" void softmax_hook(void* self, const void* input, const void* lengths, void* output) {
  using Function = void (*)(void*, const void*, const void*, void*);
  static const Function call = original<Function>(SOFTMAX_SYMBOL);
  time_operator(2, [&] { call(self, input, lengths, output); });
}

extern "C" void mask_hook(float*, const int32_t*, const int32_t*, int64_t, int64_t,
                           int64_t, int64_t, int64_t, int64_t, int64_t) asm(MASK_SYMBOL);
extern "C" void mask_hook(float* scores, const int32_t* self, const int32_t* memory,
                           int64_t rows, int64_t self_size, int64_t memory_size,
                           int64_t query_size, int64_t offset, int64_t window, int64_t right_window) {
  using Function = void (*)(float*, const int32_t*, const int32_t*, int64_t, int64_t,
                            int64_t, int64_t, int64_t, int64_t, int64_t);
  static const Function call = original<Function>(MASK_SYMBOL);
  time_operator(3, [&] { call(scores, self, memory, rows, self_size, memory_size,
                              query_size, offset, window, right_window); });
}

extern "C" void concat_hook(void*, const void*, void*) asm(CONCAT_SYMBOL);
extern "C" void concat_hook(void* self, const void* inputs, void* output) {
  using Function = void (*)(void*, const void*, void*);
  static const Function call = original<Function>(CONCAT_SYMBOL);
  time_operator(4, [&] { call(self, inputs, output); });
}

extern "C" void tile_hook(void*, const void*, void*) asm(TILE_SYMBOL);
extern "C" void tile_hook(void* self, const void* input, void* output) {
  using Function = void (*)(void*, const void*, void*);
  static const Function call = original<Function>(TILE_SYMBOL);
  time_operator(5, [&] { call(self, input, output); });
}
