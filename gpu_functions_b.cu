#include <cuda_runtime.h>
#include <iostream>

#pragma once

// Функция для проверки ошибок CUDA
#define CHECK_CUDA(call) { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error in %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
        exit(EXIT_FAILURE); \
    } \
}

// Гауссова матрица 3x3
__device__ const uint8_t gaussMatrix[3][3] = {
    {1, 2, 1},
    {2, 4, 2},
    {1, 2, 1}
};

// Функция уменьшения размера изображения для каждого ядра GPU
// Принимает на вход height * width * 3 значений unsigned char (пикселей)
// Преобразует в (height / mult) * (width / mult) * 3 значений
// d_input - исходный массив каналов пикселей
// d_output - полученный массив каналов пикселей
// width - ширина исходного изображения (без учета каналов)
// height - высота исходного изображения (без учета каналов)
// mult - коэффициент уменьшения (например, 2 для уменьшения в 2 раза)
__global__ void downscale_kernel(const unsigned char *d_input,
                            unsigned char *d_output,
                            int width, int height,
                            int mult = 2) {
    int radius = 1;    

    // Определяем индексы
    int x = blockIdx.x * blockDim.x + threadIdx.x;      // Индекс столбца
    int y = blockIdx.y * blockDim.y + threadIdx.y;      // Индекс строки

    // Проверка выхода за границы изображения
    // Проверка остатка от деления
    // Нужна для того, чтобы несколько потоки не записывали результат в одну ячейку
    if (x < width && y < height &&
        x % mult == 0 && y % mult == 0) {
        // Определяем соседей с учетом граничных пикселей
        int up = max(0, y - radius);                    // Строка сверху
        int left = max(0, x - radius);                  // Столбец слева
        int down = min(height - 1, y + radius);         // Строка снизу
        int right = min(width - 1, x + radius);         // Столбец справа

        // Диаметр округи, в которой происходит поиск соседей
        int diam = 2 * radius + 1;

        // Значения каналов в результате Гауссова размытия
        float valueRed = 0.0f;      // Гауссово значение для канала Red
        float valueGreen = 0.0f;    // Гауссово значение для канала Green
        float valueBlue = 0.0f;     // Гауссово значение для канала Blue

        // Поиск соседей с проверкой выхода за границы
        // Вместо отсутствующих соседей дублируются исходные значения
        // Поиск соседей в строках
        for(int row = 0; row < diam; row++) {
            int i = y - radius + row;
            // Если выходим за границы, подставляем значения ближайших строк
            if (i < up) {
                i = up;
            } else if (i > down) {
                i = down;
            }

            // Поиск соседей в столбцах
            for(int col = 0; col < diam; col++) {
                int j = x - radius + col;
                // Если выходим за границы, подставляем значения ближайших столбцов
                if (j < left) {
                    j = left;
                } else if (j > right) {
                    j = right;
                }
                
                int idx = i * width + j;
                // Записываем значения после Гауссова размытия
                valueRed += d_input[3 * idx + 0] * gaussMatrix[row][col];
                valueGreen += d_input[3 * idx + 1] * gaussMatrix[row][col];
                valueBlue += d_input[3 * idx + 2] * gaussMatrix[row][col];
            }
        }

        int idx = (y / mult) * (width / mult) + (x / mult);
        // Записываем результат для вывода
        d_output[idx * 3 + 0] = valueRed / 16;
        d_output[idx * 3 + 1] = valueGreen / 16;
        d_output[idx * 3 + 2] = valueBlue / 16;
    }
}