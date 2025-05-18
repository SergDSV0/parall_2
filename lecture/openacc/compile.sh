#!/bin/bash

# Создаем директорию для скомпилированных файлов
mkdir -p compile

# Компилируем каждый .cpp файл в текущей директории
for file in *.cpp; do
    if [ -f "$file" ]; then
        output_name="compile/${file%.*}"
        echo "Компиляция $file -> $output_name"
        nvc++ -acc -gpu=mem:managed "$file" -o "$output_name"
    fi
done

echo "Компиляция завершена!"
