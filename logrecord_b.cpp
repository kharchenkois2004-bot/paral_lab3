#include "logging.cpp"

#pragma once

// Запись лога для программы A
struct LogRecord_b : public ILogRecord {  
    char *date;
    const char *imageName;

    int oldWidth;
    int oldHeight;
    int newWidth;
    int newHeight;

    int mult;
    int xThreads;
    int yThreads;

    double milliseconds;

    LogRecord_b(const char *imageName,
                int oldWidth, int oldHeight,
                int newWidth, int newHeight,
                int mult,
                int xThreads, int yThreads,
                double milliseconds
               ) {
        time_t now = time(0);
        tm* local_time = std::localtime(&now);
        this->date = asctime(local_time);
        
        this->imageName = imageName;

        this->oldWidth = oldWidth;
        this->oldHeight = oldHeight;
        this->newWidth = newWidth;
        this->newHeight = newHeight;

        this->mult = mult;
        this->xThreads = xThreads;
        this->yThreads = yThreads;
        this->milliseconds = milliseconds;
    }

    // Функция для составления имени файла лога
    // Имя файла состоит из имени иозбражения и кол-ва потоков 
    char* logFilename() override {
        const int bufSize = 128;
        char *filename = new char[bufSize];
        sprintf_s(filename, bufSize,
                  "%s.%s",
                  imageName, "txt");
        return filename;
    }

    // Функция для составления содержимого лога
    // Сохраняет в текстовом виде все поля структуры
    char* logContent() override {
        const int bufSize = 256;
        char *content = new char[bufSize];
        sprintf_s(content, bufSize,
                  "%sImage: \"%s\"; Resolution (source): %dx%d;\n"
                  "Resolution (result): %dx%d; Coefficient: %d;\n"
                  "Threads: %dx%d; Execution time: %.4f ms;",
                  date, imageName, oldWidth, oldHeight,
                  newWidth, newHeight, mult,
                  xThreads, yThreads,
                  milliseconds);
        return content;
    }
};