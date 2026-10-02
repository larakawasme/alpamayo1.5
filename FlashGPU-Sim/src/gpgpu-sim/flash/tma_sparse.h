#ifndef FLASH_GPGPU_SIM_TMA_SPARSE_H
#define FLASH_GPGPU_SIM_TMA_SPARSE_H

#include <bitset>
#include <cstdint>

// Sparse state captured for one TMA tile request.
struct tma_sparse_window_t {
  bool enabled = false;
  uint32_t dimension = 0;
  std::bitset<256> active;
};

#endif
