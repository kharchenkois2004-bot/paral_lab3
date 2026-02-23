#include <iostream>
#include <chrono>
#include <direct.h>
#include <opencv2/opencv.hpp>

#include <cuda_runtime.h>

#include "logging.cpp"
#include "logrecord_b.cpp"
#include "gpu_functions_b.cu"

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

    const char *inputImage = imageFiles[5];         // Имя изображения

    const char *inputDir = "input";                 // Директория с изображения
    const char *outputDir = "output";               // Директория для результатов обработки
    const char *logDir = "logs";                    // Директория для логов

    const char *outputDir_b = "output/output_b";    // Директория с результатами обработки программы B
    const char *logDir_b = "logs/logs_b";           // Директория с логами программы B
    const char *processMethod = "gauss";            // Способ обработки (для имени выходного файла)

    const int mult = 2;                             // Коэффициент уменьшения
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
    _mkdir(outputDir_b);
    _mkdir(logDir);
    _mkdir(logDir_b);
    
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
    int width = image.cols;     // Ширина в пикселях
    int height = image.rows;    // Высота в пикселях

    // Размер исходного изображения в байтах
    size_t imageSize = height * width * sizeof(uchar) * 3;
    // Размер уменьшенного изображения в байтах              
    size_t downSize = (height / mult) * (width / mult) * sizeof(uchar) * 3;

    // d - device
    // Массивы для переноса данных на GPU
    unsigned char *d_input;         // Входной массив с исходным изображением
    unsigned char *d_output;        // Выходной массив с уменьшенным изображениемc

    // Определяем память в байтах для массивов
    CHECK_CUDA(cudaMalloc(&d_input, imageSize));
    CHECK_CUDA(cudaMalloc(&d_output, downSize));

    // Копируем imageSize байтов из image.data в d_input (с Host на Device)
    CHECK_CUDA(cudaMemcpy(d_input, image.data, imageSize, cudaMemcpyHostToDevice));

    // Кол-во потоков на каждый блок
    dim3 threadsPerBlock(xThreads, yThreads);
    // Кол-во блоков в каждой сетке
    dim3 blocksPerGrid((width + threadsPerBlock.x - 1) / threadsPerBlock.x,
                        (height + threadsPerBlock.y - 1) / threadsPerBlock.y);

    // Старт таймера для подсчета времени
    auto timerBegin = chrono::high_resolution_clock::now();    

    // Операция уменьшения изображения в mult раз на GPU
    downscale_kernel<<<blocksPerGrid, threadsPerBlock>>>(d_input, d_output, width, height, mult);
    // Проверка на возникновение ошибок в ходе операции (Ы)
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());

    // h - host
    // Выходной массив с уменьшенным изображением
    unsigned char *h_output = new unsigned char[(width / mult) * (height / mult) * 3];    
    // Копируем downSize байтов из d_output в h_output (с Device на Host)
    CHECK_CUDA(cudaMemcpy(h_output, d_output, downSize, cudaMemcpyDeviceToHost));

    // Была проведена попытка заменить исходный image.data на новый уменьшенный массив
    // Результат: фиаско, разочарование, грусть, тоски и недосып, а также ERROR
    // Итог: смириться с поражением и создать новый класс для уменьшенного изображения

    // Класс Mat для уменьшенного изображения
    cv::Mat resultImage(height / mult, width / mult, CV_8UC3, h_output);

    // Финиш таймера для подсчета времени
    auto timerEnd = chrono::high_resolution_clock::now();
    // длительность выполнения операция на GPU
    chrono::duration<double, milli> timerDuration = (timerEnd - timerBegin);

    printf("Exectution time = %.4f milliseconds\n", timerDuration.count());

    // Размеры уменьшенного изображения
    int newWidth = resultImage.cols;    // Ширина в пикселях
    int newHeight = resultImage.rows;   // Высота в пикселях

    // Структура для хранения записи лога программы B
    LogRecord_b logRecord(inputImage,
                          width, height,
                          newWidth, newHeight, mult,
                          xThreads, yThreads,
                          timerDuration.count());

    // Сохраняем лог
    saveLog(logRecord, logDir_b);

    // Получение пути до обработанного изображения с эрозией
    char outputImage[bufSize];
    sprintf_s(outputImage, bufSize,
              "%s/%s_%s",
              outputDir_b, processMethod, inputImage);
    
    // Сохраняем полученное изображение
    cv::imwrite(outputImage, resultImage);    
    printf("Image \"%s\" saved\n", outputImage);

    // Удаление массивов, чтобы не занимали память (ибо дорогая)
    delete[] h_output;
    CHECK_CUDA(cudaFree(d_input));
    CHECK_CUDA(cudaFree(d_output));

    return 0;
}