#include <iostream>
#include <fstream>

using namespace std;

#pragma once

// Интерфейс для логов
// Нужен для того, чтобы использовать одну функцию для сохранения логов в файл
struct ILogRecord {
    // Функция для составления имени файла лога
    // Имя файла состоит из имени иозбражения и кол-ва потоков    
    virtual char* logFilename() = 0;

    // Функция для составления содержимого лога
    // Сохраняет в текстовом виде все поля лога
    virtual char* logContent() = 0;

    // Деструктор по умолчанию
    virtual ~ILogRecord() = default;
};

/* Функция для сохранения логов файл
- logRecord - запись лога (наследник ILogRecord)
- directory - путь для сохранения логов */
int saveLog(ILogRecord &logRecord, const char *directory) {
    // Получаем имя и содержимое логов
    const char *filename = logRecord.logFilename();
    const char *content = logRecord.logContent();

    // Составляем путь до файла лога
    const int bufSize = 256;
    char logFile[bufSize];

    sprintf_s(logFile, bufSize,
              "%s/%s",
              directory, filename);
    
    // Создаем файл для логов
    ofstream outfile (logFile, ios::out | ios::app);
    if (!outfile.is_open()) {
        cerr << "Error opening file!" << std::endl;
        return 1;
    }
    // Записываем в файл
    outfile << content << endl;
    // Закрываем файл
    outfile.close();
    delete[] filename;
    delete[] content;

    return 0;
}