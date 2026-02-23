#include <cuda_runtime.h>

#pragma once

// Функция для проверки ошибок CUDA
#define CHECK_CUDA(call) { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error in %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
        exit(EXIT_FAILURE); \
    } \
}

// Функция бинаризации пикселя для каждого ядра
// Получает на вход 3 канала, выдает на выходе 1 канал
// Сравнивает среднее значение по каналам с пороговым значением (threshold)
// Заменяет на 1 значение больше или равное пороговому (threshold)
// Заменяет на 0 в противном случае
__global__ void binarize_kernel(const unsigned char *input,
                                uint8_t *binary, unsigned char *interim,
                                int width, int height, int threshold) {
    // Определяем индексы
    int x = blockIdx.x * blockDim.x + threadIdx.x;      // Индекс столбца
    int y = blockIdx.y * blockDim.y + threadIdx.y;      // Индекс строки

    // Проверка выхода за границы изображения
    if (x < width && y < height) {
        int idx = y * width + x;
        uint8_t r = input[idx * 3 + 0];             // Red (красный)
        uint8_t g = input[idx * 3 + 1];             // Green (зеленый)
        uint8_t b = input[idx * 3 + 2];             // Blue (синий)
        int intensity = (r + g + b) / 3;            // Интенсивность (среднее по каналам)

        // Сравнение интенсивности с порогом, перевод в бинарный формат (0 и 1)
        // 1 для true, 0 для false
        binary[idx] = (intensity >= threshold);
        uint8_t colorValue = binary[idx] * 255;
        interim[idx * 3 + 0] = colorValue;
        interim[idx * 3 + 1] = colorValue;
        interim[idx * 3 + 2] = colorValue;
    }
}

// Функция эрозии пикселя для каждого ядра GPU
// Проводит для всех соседних пикселей логическое умножение (AND)
// Заменяет значение пикселя на полученный результат
// Радиус поиска соседей определяется шагом эрозии (step)
__global__ void erode_kernel(const uint8_t *binary,
                             unsigned char *output,
                             int width, int height, int step) {
    // Определяем индексы
    int x = blockIdx.x * blockDim.x + threadIdx.x;      // Индекс столбца
    int y = blockIdx.y * blockDim.y + threadIdx.y;      // Индекс строки

    // Проверка выхода за границы изображения
    if (x < width && y < height) {
        // Определяем соседей с учетом граничных пикселей
        int up = max(0, y - step);                  // Строка сверху
        int left = max(0, x - step);                // Столбец слева
        int down = min(height - 1, y + step);       // Строка снизу
        int right = min(width - 1, x + step);       // Столбец справа

        // Переменная для хранения результата
        // Имеет начальное значение 1 для операции AND
        uint8_t result = 1;

        // Проводим операцию AND для всех соседей
        for(int row = up; row <= down; row++) {
            for(int col = left; col <= right; col++) {
                result &= binary[row * width + col];
            }
        }

        // Возвращаем результат
        int idx = y * width + x;
        uint8_t colorValue = result * 255;
        output[idx * 3 + 0] = colorValue;
        output[idx * 3 + 1] = colorValue;
        output[idx * 3 + 2] = colorValue;
    }
}

// Функция диляции пикселя для каждого ядра GPU
// Проводит для всех соседних пикселей логическое сложение (OR)
// Заменяет значение пикселя на полученный результат
// Радиус поиска соседей определяется шагом диляции (step)
__global__ void dilate_kernel(const uint8_t *binary,
                              unsigned char *output,
                              int width, int height, int step) {
    // Определяем индексы
    int x = blockIdx.x * blockDim.x + threadIdx.x;      // Индекс столбца
    int y = blockIdx.y * blockDim.y + threadIdx.y;      // Индекс строки

    // Проверка выхода за границы изображения
    if (x < width && y < height) {
        // Определяем соседей с учетом граничных пикселей
        int up = max(0, y - step);                  // Строка сверху
        int left = max(0, x - step);                // Столбец слева
        int down = min(height - 1, y + step);       // Строка снизу
        int right = min(width - 1, x + step);       // Столбец справа

        // Переменная для хранения результата
        // Имеет начальное значение 0 для операции OR
        uint8_t result = 0;

        // Проводим операцию OR для всех соседей
        for(int row = up; row <= down; row++) {
            for(int col = left; col <= right; col++) {
                result |= binary[row * width + col];
            }
        }

        // Возвращаем результат
        int idx = y * width + x;
        uint8_t colorValue = result * 255;
        output[idx * 3 + 0] = colorValue;
        output[idx * 3 + 1] = colorValue;
        output[idx * 3 + 2] = colorValue;
    }
}