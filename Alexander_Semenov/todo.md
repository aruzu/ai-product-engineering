Отлично! Вот подробный чеклист в формате todo.md, основанный на 15 шагах, которые мы разработали.

# TODO - Автоматизированный Анализ Отзывов и Симуляция Фокус-группы

Этот файл отслеживает прогресс разработки проекта согласно плану.

## Фаза 1: Фундамент Проекта и Загрузка Данных

### Шаг 1: Настройка Структуры Проекта и Зависимостей
- [x] Создать папку `src/`
- [x] Создать папку `data/`
- [x] Добавить `.gitkeep` в `data/`
- [x] Создать папку `output/`
- [x] Добавить `.gitkeep` в `output/`
- [x] Создать папку `tests/`
- [x] Добавить `.gitkeep` в `tests/`
- [x] Создать файл `requirements.txt`
- [x] Добавить `python-dotenv`, `pandas`, `openai` в `requirements.txt`
- [x] Создать файл `.gitignore`
- [x] Добавить стандартные исключения Python (.pyc, __pycache__, venv/, .env) в `.gitignore`
- [x] Создать файл `.env`
- [x] Добавить `OPENAI_API_KEY=ВАШ_КЛЮЧ_СЮДА` в `.env`
- [x] Создать пустой файл `README.md`
- [x] *Действие пользователя:* Создать виртуальное окружение (`venv`)
- [x] *Действие пользователя:* Установить зависимости (`pip install -r requirements.txt`)
- [x] *Действие пользователя:* Создать файл `.env` из `.env.example` и добавить реальный API ключ

### Шаг 2: Базовый Скрипт, CLI и Загрузка .env
- [x] Создать файл `src/main.py`
- [x] Импортировать `argparse`, `os`, `load_dotenv` в `src/main.py`
- [x] Определить функцию `main()` в `src/main.py`
- [x] Вызвать `load_dotenv()` в начале `main()`
- [x] Реализовать парсинг аргументов с помощью `argparse`
- [x] Добавить обязательный аргумент `--csv-path` (или `-c`)
- [x] Получить значение `csv_path` из аргументов
- [x] Получить значение `OPENAI_API_KEY` из `os.getenv()`
- [x] Добавить проверку наличия `OPENAI_API_KEY`, вывести ошибку и выйти (`exit(1)`) если отсутствует
- [x] Добавить (временную) проверку существования файла `csv_path` (`os.path.exists`), вывести ошибку и выйти (`exit(1)`) если отсутствует
- [x] Добавить placeholder `print` для `csv_path` и статуса API ключа
- [x] Добавить блок `if __name__ == "__main__":` для вызова `main()`
- [x] Добавить docstring для функции `main()`

### Шаг 3: Модуль Загрузки и Валидации CSV
- [x] Создать файл `src/data_loader.py`
- [x] Импортировать `pandas as pd` в `src/data_loader.py`
- [x] Определить константу `REQUIRED_COLUMNS` со списком колонок
- [x] Определить функцию `load_reviews(csv_path: str) -> pd.DataFrame:`
- [x] Добавить docstring для `load_reviews` (параметры, возврат, исключения)
- [x] Использовать `try-except` для `pd.read_csv(csv_path)`
- [x] Обработать `FileNotFoundError` (перевыбросить или обернуть)
- [x] Обработать ошибки парсинга (`pd.errors.ParserError`, `Exception`), поднять `ValueError`
- [x] Реализовать проверку наличия всех колонок из `REQUIRED_COLUMNS`
- [x] Поднять `ValueError` с сообщением об отсутствующих колонках, если проверка не пройдена
- [x] Вернуть DataFrame в случае успеха
- [x] **Модификация `src/main.py`:**
    - [x] Импортировать `load_reviews` из `src.data_loader`
    - [x] Удалить проверку `os.path.exists(csv_path)`
    - [x] Обернуть вызов `load_reviews(csv_path)` в `try-except`
    - [x] Обработать `FileNotFoundError` и `ValueError` от `load_reviews`, вывести ошибку и выйти (`exit(1)`)
    - [x] Вывести сообщение об успешной загрузке и количестве строк (`len(df)`)
    - [x] Сохранить результат в переменную `reviews_df`

### Шаг 4: Базовая Настройка Логирования
- [x] Создать файл `src/logger_config.py`
- [x] Импортировать `logging` в `src/logger_config.py`
- [x] Определить функцию `setup_logger()`
- [x] Настроить `logging.basicConfig` в `setup_logger` (уровень INFO, формат, StreamHandler)
- [x] Добавить docstring для `setup_logger`
- [x] **Модификация `src/main.py`:**
    - [x] Импортировать `logging` и `setup_logger`
    - [x] Вызвать `setup_logger()` в начале `main()`
    - [x] Заменить все `print` на `logging.info` или `logging.error`
    - [x] Добавить лог `logging.info("Запуск пайплайна...")` в начале
    - [x] Добавить лог `logging.info("Пайплайн успешно завершил загрузку данных.")` (или аналогичный) в конце (пока что после загрузки данных)
- [x] **Модификация `src/data_loader.py`:**
    - [x] Импортировать `logging`
    - [x] Добавить `logging.info` перед чтением CSV
    - [x] Добавить `logging.error` при `FileNotFoundError`
    - [x] Добавить `logging.error` при `ValueError` (ошибка парсинга/колонок)
    - [x] Добавить `logging.info` после успешной загрузки и валидации

---
*Чеклист завершен. Пройдитесь по пунктам для отслеживания прогресса.*