import os

def collect_python_files(output_filename='combined_code.py'):
    # Получаем путь к текущему скрипту и определяем выходной файл
    current_script = os.path.abspath(__file__)
    output_path = os.path.abspath(output_filename)
    
    with open(output_path, 'w', encoding='utf-8') as outfile:
        # Рекурсивно обходим директории
        for root, _, files in os.walk('.'):
            for file in files:
                if file.endswith('.py'):
                    file_path = os.path.join(root, file)
                    abs_path = os.path.abspath(file_path)
                    
                    # Пропускаем текущий скрипт и выходной файл
                    if abs_path in [current_script, output_path]:
                        continue
                    
                    # Записываем заголовок с именем файла
                    outfile.write(f'\n\n# {"-" * 60}\n')
                    outfile.write(f'# FILE: {file_path}\n')
                    outfile.write(f'# {"-" * 60}\n\n')
                    
                    # Читаем и записываем содержимое файла
                    try:
                        with open(file_path, 'r', encoding='utf-8') as infile:
                            outfile.write(infile.read())
                    except Exception as e:
                        outfile.write(f'# ERROR reading file: {e}\n')

if __name__ == '__main__':
    collect_python_files()
    print("All Python files have been combined into 'combined_code.py'")
