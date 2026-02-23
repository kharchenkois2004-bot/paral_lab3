#include <iostream>
#include <direct.h>
#include <opencv2/opencv.hpp>

#include <cuda_runtime.h>

#include "logging.cpp"
#include "logrecord_a.cpp"
#include "gpu_functions_a.cu"

#pragma once

using namespace std;

// Столько комментариев нужно потому,
// Что я и сам потом забуду,
// Что сотворил в ночном бреду.

int main() {
    // Постоянные
    const int bufSize = 64;                         // Размер буфера для файла изображения

    const char imageFiles[][bufSize] = {            // Список изображений
        "Spider_10x.png",
        "BlueScreen_10x.png",
        "Linux_10x.png",
        "Game_10x.png",
        "Livesey_10x.jpg",
        "PTSR_10x.jpg"
    };

    const char *inputImage = imageFiles[0];         // Имя изображения

    const char *inputDir = "input";                 // Директория с изображения
    const char *outputDir = "output";               // Директория для результатов обработки
    const char *logDir = "logs";                    // Директория для логов

    const char *outputDir_a = "output/output_a";    // Директория с результатами обработки программы A
    const char *logDir_a = "logs/logs_a";           // Директория с логами программы A
    const char *processMethod = "erosion";          // Способ обработки (для имени выходного файла)

    const int threshold = 128;                      // Пороговое значение для интенсивности
    const int step = 2;                             // Шаг эрозии
    const int xThreads = 16;                        // Размерность блока по X
    const int yThreads = 16;                        // Размерность блока по Y

    const bool DEBUG = false;

    // Вывод информации о GPU
    if (DEBUG) {
        int deviceId;
        cudaGetDevice(&deviceId);
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, deviceId);

        printf("Maximum threads per block: %d\n", prop.maxThreadsPerBlock);
        printf("Max threads dim X: %d\n", prop.maxThreadsDim[0]);
        printf("Max threads dim Y: %d\n", prop.maxThreadsDim[1]);
        printf("Max threads dim Z: %d\n", prop.maxThreadsDim[2]);
        printf("Multiprocessor count: %d\n", prop.multiProcessorCount);
    }    

    // Создаем необходимые директории
    _mkdir(inputDir);
    _mkdir(outputDir);
    _mkdir(outputDir_a);
    _mkdir(logDir);
    _mkdir(logDir_a);
    
    // Получение пути до изображения
    char inputFile[bufSize];
    sprintf_s(inputFile, bufSize, "%s/%s", inputDir, inputImage);

    // Загруженное изображение
    cv::Mat image = cv::imread(inputFile, cv::IMREAD_UNCHANGED);
    if (image.empty()) {        
        printf("Failed to read the image: %s\n"
               "Ensure that image is located in %s directory and the name is correct\n",
                inputFile, inputDir);
        return -1;
    }

    // Проверка, что изображение трёхканальное
    if (image.type() != CV_8UC3) {
        cerr << "Image is not 3-channel 8-bit. Please convert." << endl;
        return -1;
    } // Проверка изображения на непрерывность (для упрощения копирования)
    if (!image.isContinuous()) {
        image = image.clone();   // делаем непрерывным
    }

    printf("Image \"%s\" loaded\n"
            "Resolution: %dx%d Pixels: %d\n",
            inputFile,
            image.cols, image.rows, image.cols * image.rows);

    // Исходные размеры изображения
    int width = image.cols;
    int height = image.rows;

    // Вычисляем размеры массивов для загрузки на GPU
    size_t imageSize = height * width * sizeof(uchar) * 3;  // Размер исходного изображения в байтах
    size_t binSize = height * width * sizeof(uint8_t);          // Размер черно-белого изображения в байтах

    // d - device
    // Массивы для переноса данных на GPU
    unsigned char *d_input;         // Входной массив до бинаризации
    unsigned char *d_interim;       // Промежуточный массив после бинаризации (для вывода)
    unsigned char *d_output;        // Итоговый массив после эрозии (результат)
    uint8_t *d_binary;              // Промежуточный массив после бинаризации (для эрозии)

    // Определяем память в байтах для массивов
    CHECK_CUDA(cudaMalloc(&d_input, imageSize));
    CHECK_CUDA(cudaMalloc(&d_interim, imageSize));
    CHECK_CUDA(cudaMalloc(&d_output, imageSize));    
    CHECK_CUDA(cudaMalloc(&d_binary, binSize));

    // Копируем imageSize байтов из image.data в d_input (с Host на Device)
    CHECK_CUDA(cudaMemcpy(d_input, image.data, imageSize, cudaMemcpyHostToDevice));

    // Кол-во потоков на каждый блок
    dim3 threadsPerBlock(xThreads, yThreads);
    // Кол-во блоков в каждой сетке
    dim3 blocksPerGrid((width + threadsPerBlock.x - 1) / threadsPerBlock.x,
                        (height + threadsPerBlock.y - 1) / threadsPerBlock.y);

    // Используется 2 таймера
    // Между операцией бинаризации и эрозии происходит сохранение изображения
    // Сохранение изображение происходит на CPU, что затрудняет расчет времени

    // Старт таймера для подсчета времени бинаризации
    auto timerBegin_binary = chrono::high_resolution_clock::now();

    // Операция бинаризации (черно-белый формат) изображения на GPU
    binarize_kernel<<<blocksPerGrid, threadsPerBlock>>>(d_input, d_binary, d_interim, width, height, threshold);
    // Проверка на возникновение ошибок в ходе операции (Ы)
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());

    // h - host
    // Промежуточный массив для сохранения бинарного изображения
    unsigned char *h_interim = new unsigned char[width * height * 3];
    // Копируем imageSize байтов из d_interim в h_interim (с Device на Host)
    CHECK_CUDA(cudaMemcpy(h_interim, d_interim, imageSize, cudaMemcpyDeviceToHost));

    // Переносим изменения в изображение
    image.data = h_interim;

    // Финиш таймера для подсчета времени бинаризации
    auto timerEnd_binary = chrono::high_resolution_clock::now();

    // Получение пути до обработанного изображения
    char outputFile[bufSize];
    sprintf_s(outputFile, bufSize, "%s/binary_%s", outputDir_a, inputImage);

    // Сохраняем бинарное изображение
    cv::imwrite(outputFile, image);
    printf("Image \"%s\" saved\n", outputFile);   
    
    // Старт таймера для подсчета времени эрозии
    auto timerBegin_erosion = chrono::high_resolution_clock::now();

    // Промежуточный бинарный массив для операции эрозии
    uint8_t *h_binary = new uint8_t[width * height];
    // Копируем binSize байтов из d_binary в h_binary (с Device на Host)
    CHECK_CUDA(cudaMemcpy(h_binary, d_binary, binSize, cudaMemcpyDeviceToHost));

    // Операция эрозии бинарного изображения на GPU
    erode_kernel<<<blocksPerGrid, threadsPerBlock>>>(d_binary, d_output, width, height, step);
    // Проверка на возникновение ошибок в ходе операции (Ы)
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());

    // Итоговый массив после эрозии
    unsigned char *h_output = new unsigned char[width * height * 3];
    // Копируем imageSize байтов из d_output в h_output (с Device на Host)
    CHECK_CUDA(cudaMemcpy(h_output, d_output, imageSize, cudaMemcpyDeviceToHost));

    // Переносим изменения в изображение
    image.data = h_output;

    // Финиш таймера для подсчета времени эрозии
    auto timerEnd_erosion = chrono::high_resolution_clock::now();
    // длительность выполнения операция на GPU
    chrono::duration<double, milli> timerDuration = (timerEnd_binary - timerBegin_binary) +
                                                    (timerEnd_erosion - timerBegin_erosion);

    printf("Exectution time = %.4f milliseconds\n", timerDuration.count());

    // Структура для хранения записи лога программы A
    LogRecord_a logRecord(inputImage,
                          image.cols, image.rows,
                          threshold, step,
                          xThreads, yThreads,
                          timerDuration.count());

    // Сохраняем лог
    saveLog(logRecord, logDir_a);

    // Получение пути до обработанного изображения с эрозией
    sprintf_s(outputFile, bufSize,
              "%s/binary_%s_%s",
              outputDir_a, processMethod, inputImage);
    
    // Сохраняем полученное изображение
    cv::imwrite(outputFile, image);    
    printf("Image \"%s\" saved\n", outputFile);

    // Удаление массивов, чтобы не занимали память (ибо дорогая)
    delete[] h_binary;
    delete[] h_interim;
    delete[] h_output;
    CHECK_CUDA(cudaFree(d_input));
    CHECK_CUDA(cudaFree(d_binary));
    CHECK_CUDA(cudaFree(d_interim));
    CHECK_CUDA(cudaFree(d_output));

    return 0;
}