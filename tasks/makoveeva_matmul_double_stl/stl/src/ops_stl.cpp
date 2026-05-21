#include "makoveeva_matmul_double_stl/stl/include/ops_stl.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <mutex>
#include <thread>
#include <vector>

#include "makoveeva_matmul_double_stl/common/include/common.hpp"

namespace makoveeva_matmul_double_stl {

namespace {

// Выбирает размер блока в зависимости от размера матрицы
[[nodiscard]] size_t SelectBlockSize(size_t n) {
  if (n <= 64) {
    return n;
  }
  if (n <= 256) {
    return 64;
  }
  if (n <= 1024) {
    return 128;
  }
  return 256;
}

}  // namespace

MatmulDoubleSTLTask::MatmulDoubleSTLTask(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput() = std::vector<double>();
}

bool MatmulDoubleSTLTask::ValidationImpl() {
  const auto &input = GetInput();
  const size_t n = std::get<0>(input);
  const auto &a = std::get<1>(input);
  const auto &b = std::get<2>(input);

  return n > 0 && a.size() == n * n && b.size() == n * n;
}

bool MatmulDoubleSTLTask::PreProcessingImpl() {
  const auto &input = GetInput();
  n_ = std::get<0>(input);
  A_ = std::get<1>(input);
  B_ = std::get<2>(input);
  C_.assign(n_ * n_, 0.0);

  return true;
}

bool MatmulDoubleSTLTask::RunImpl() {
  if (n_ <= 0) {
    return false;
  }

  const size_t n = n_;

  // Выбираем размер блока для оптимальной локальности кэша
  const size_t block_size = SelectBlockSize(n);

  // Проверяем что матрица делится нацело на размер блока
  if (n % block_size != 0) {
    return RunSimpleMultiply();
  }

  const size_t grid_size = n / block_size;

  // Получаем количество потоков
  const size_t num_threads = std::thread::hardware_concurrency();

  // Создаём mutex для синхронизации доступа к матрице C
  std::mutex write_mutex;

  // Общее количество итераций
  const size_t total_iterations = grid_size * grid_size * grid_size;

  // ========================================================================
  // ОСНОВНОЙ ЦИКЛ - АЛГОРИТМ ФОКСА СО STL ПОТОКАМИ
  // ========================================================================

  // Проверяем есть ли смысл создавать потоки
  if (total_iterations >= num_threads) {
    // Создаём вектор потоков
    std::vector<std::thread> threads;
    threads.reserve(num_threads);

    // Вычисляем количество итераций на один поток
    const size_t iterations_per_thread = total_iterations / num_threads;

    // Создаём потоки
    for (size_t t = 0; t < num_threads; ++t) {
      const size_t start_step = t * iterations_per_thread;
      const size_t end_step = (t == num_threads - 1) ? total_iterations : start_step + iterations_per_thread;

      threads.emplace_back(&MatmulDoubleSTLTask::Worker, this, start_step, end_step, grid_size, block_size,
                           std::ref(write_mutex));
    }

    // Ждём завершения всех потоков
    for (auto &thread : threads) {
      thread.join();
    }
  } else {
    // Количество итераций мало, выполняем в одном потоке
    Worker(0, total_iterations, grid_size, block_size, write_mutex);
  }

  GetOutput() = C_;
  return true;
}

void MatmulDoubleSTLTask::Worker(size_t start_step, size_t end_step, size_t grid_size, size_t block_size,
                                 std::mutex &write_mutex) {
  // Обрабатываем диапазон итераций [start_step, end_step)
  for (size_t step_i_j = start_step; step_i_j < end_step; ++step_i_j) {
    // ЭТАП 1: Декодирование одномерного индекса в трёхмерный (step, i, j)
    const size_t step = step_i_j / (grid_size * grid_size);
    const size_t i = (step_i_j % (grid_size * grid_size)) / grid_size;
    const size_t j = step_i_j % grid_size;

    // ЭТАП 2: Вычисление root блока для алгоритма Фокса
    // root = (i + step) % grid_size - КЛЮЧЕВАЯ ЛИНИЯ!
    const size_t root = (i + step) % grid_size;

    // ЭТАП 3: Создание локального буфера (не общего!)
    std::vector<double> local_block(block_size * block_size, 0.0);

    // ЭТАП 4: Умножение блока A[i][root] на блок B[root][j]
    for (size_t bi = 0; bi < block_size; ++bi) {
      for (size_t bj = 0; bj < block_size; ++bj) {
        double sum = 0.0;
        for (size_t bk = 0; bk < block_size; ++bk) {
          const size_t idx_a = ((i * block_size + bi) * n_) + (root * block_size + bk);
          const size_t idx_b = ((root * block_size + bk) * n_) + (j * block_size + bj);
          sum += A_[idx_a] * B_[idx_b];
        }
        local_block[(bi * block_size) + bj] += sum;
      }
    }

    // ЭТАП 5: Безопасное добавление результата в матрицу C
    {
      std::lock_guard<std::mutex> lock(write_mutex);
      for (size_t bi = 0; bi < block_size; ++bi) {
        for (size_t bj = 0; bj < block_size; ++bj) {
          const size_t idx_c = ((i * block_size + bi) * n_) + (j * block_size + bj);
          C_[idx_c] += local_block[(bi * block_size) + bj];
        }
      }
    }
  }
}

bool MatmulDoubleSTLTask::RunSimpleMultiply() {
  const size_t n = n_;
  const auto &a = A_;
  const auto &b = B_;
  auto &c = C_;

  // Получаем количество потоков
  const size_t num_threads = std::thread::hardware_concurrency();

  // Проверяем есть ли смысл создавать потоки
  if (n >= num_threads) {
    std::vector<std::thread> threads;
    threads.reserve(num_threads);

    const size_t rows_per_thread = n / num_threads;

    for (size_t t = 0; t < num_threads; ++t) {
      const size_t start_row = t * rows_per_thread;
      const size_t end_row = (t == num_threads - 1) ? n : start_row + rows_per_thread;

      threads.emplace_back([this, start_row, end_row, &a, &b, &c, n]() {
        for (size_t i = start_row; i < end_row; ++i) {
          for (size_t j = 0; j < n; ++j) {
            double sum = 0.0;
            for (size_t k = 0; k < n; ++k) {
              sum += a[(i * n) + k] * b[(k * n) + j];
            }
            c[(i * n) + j] = sum;
          }
        }
      });
    }

    for (auto &thread : threads) {
      thread.join();
    }
  } else {
    // Простое однопоточное умножение
    for (size_t i = 0; i < n; ++i) {
      for (size_t j = 0; j < n; ++j) {
        double sum = 0.0;
        for (size_t k = 0; k < n; ++k) {
          sum += a[(i * n) + k] * b[(k * n) + j];
        }
        c[(i * n) + j] = sum;
      }
    }
  }

  return true;
}

bool MatmulDoubleSTLTask::PostProcessingImpl() {
  return true;
}

}  // namespace makoveeva_matmul_double_stl
