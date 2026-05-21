#include "makoveeva_matmul_double_omp/omp/include/ops_omp.hpp"

#include <omp.h>

#include <cmath>
#include <cstddef>
#include <vector>

#include "makoveeva_matmul_double_omp/common/include/common.hpp"

namespace makoveeva_matmul_double_omp {

namespace {

constexpr size_t kDefaultBlockSize = 64;

// Получить оптимальный размер блока для алгоритма Фокса
size_t GetOptimalBlockSize(size_t n) {
  // Используем степени двойки для лучшей локальности кэша
  if (n <= 64) {
    return n;
  }
  if (n <= 128) {
    return 64;
  }
  if (n <= 256) {
    return 64;
  }
  if (n <= 512) {
    return 128;
  }
  return 256;
}

// Умножение блока матрицы A на блок матрицы B и добавление к блоку C
void MultiplyBlocksAdd(const std::vector<double> &a, const std::vector<double> &b, std::vector<double> &c,
                       size_t block_size, size_t block_row_a, size_t block_col_a, size_t block_row_b,
                       size_t block_col_b, size_t block_row_c, size_t block_col_c, size_t n) {
#pragma omp parallel for collapse(2) default(none) \
    shared(a, b, c, n, block_size, block_row_a, block_col_a, block_row_b, block_col_b, block_row_c, block_col_c)
  for (size_t i = 0; i < block_size; ++i) {
    for (size_t j = 0; j < block_size; ++j) {
      double sum = 0.0;
      for (size_t k = 0; k < block_size; ++k) {
        const size_t idx_a = (block_row_a * block_size + i) * n + (block_col_a * block_size + k);
        const size_t idx_b = (block_row_b * block_size + k) * n + (block_col_b * block_size + j);
        sum += a[idx_a] * b[idx_b];
      }
      const size_t idx_c = (block_row_c * block_size + i) * n + (block_col_c * block_size + j);
      c[idx_c] += sum;
    }
  }
}

}  // namespace

// Убираем : n_(0) - инициализация будет в классе
MatmulDoubleOMPTask::MatmulDoubleOMPTask(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput() = std::vector<double>();
}

bool MatmulDoubleOMPTask::ValidationImpl() {
  const auto &input = GetInput();
  const size_t n = std::get<0>(input);
  const auto &a = std::get<1>(input);
  const auto &b = std::get<2>(input);

  return n > 0 && a.size() == n * n && b.size() == n * n;
}

bool MatmulDoubleOMPTask::PreProcessingImpl() {
  const auto &input = GetInput();
  n_ = std::get<0>(input);
  a_ = std::get<1>(input);
  b_ = std::get<2>(input);
  c_.assign(n_ * n_, 0.0);

  return true;
}

bool MatmulDoubleOMPTask::RunImpl() {
  if (n_ <= 0) {
    return false;
  }

  const size_t n = n_;
  const auto &a = a_;
  const auto &b = b_;
  auto &c = c_;

  // Получаем оптимальный размер блока
  const size_t block_size = GetOptimalBlockSize(n);

  // Проверяем, что матрица может быть разбита на блоки
  if (n % block_size != 0) {
    // Если не делится нацело, используем блоки максимального размера
    return RunSimpleMultiply();
  }

  const size_t num_blocks = n / block_size;

  // Алгоритм Фокса для умножения матриц
  // Фаза 1: Циклический сдвиг блоков матрицы B
  // Фаза 2: Умножение и аккумуляция результатов

  // Для каждого блочного ряда в матрице C
  for (size_t i_block = 0; i_block < num_blocks; ++i_block) {
    // Для каждой фазы сдвига
    for (size_t phase = 0; phase < num_blocks; ++phase) {
      // Для каждого столбца блоков в текущем ряду
      for (size_t j_block = 0; j_block < num_blocks; ++j_block) {
        // Блок A из позиции (i_block, (j_block + phase) % num_blocks)
        // Блок B из позиции ((j_block + phase) % num_blocks, j_block)
        const size_t block_col_a = (j_block + phase) % num_blocks;
        const size_t block_row_b = (j_block + phase) % num_blocks;

        MultiplyBlocksAdd(a, b, c, block_size, i_block, block_col_a, block_row_b, j_block, i_block, j_block, n);
      }
    }
  }

  GetOutput() = c_;
  return true;
}

bool MatmulDoubleOMPTask::RunSimpleMultiply() {
  const size_t n = n_;
  const auto &a = a_;
  const auto &b = b_;
  auto &c = c_;

#pragma omp parallel for collapse(2) default(none) shared(a, b, c, n)
  for (size_t i = 0; i < n; ++i) {
    for (size_t j = 0; j < n; ++j) {
      double sum = 0.0;
      for (size_t k = 0; k < n; ++k) {
        sum += a[(i * n) + k] * b[(k * n) + j];
      }
      c[(i * n) + j] = sum;
    }
  }

  return true;
}

bool MatmulDoubleOMPTask::PostProcessingImpl() {
  return true;
}

}  // namespace makoveeva_matmul_double_omp
