import os

from settings.config import RESULTS_DIR, logger

def _get_next_result_dir() -> str:
    """
        Создаёт новую директорию для сохранения результатов.

        return:
            str: Полный путь к созданной директории результатов.
    """

    os.makedirs(RESULTS_DIR, exist_ok=True)

    existing = [
        name for name in os.listdir(RESULTS_DIR)
        if os.path.isdir(os.path.join(RESULTS_DIR, name)) and name.startswith("interview_")
    ]

    indexes = []
    for name in existing:
        try:
            index = int(name.split("_")[1])
            indexes.append(index)
        except (IndexError, ValueError):
            continue

    next_index = max(indexes, default=0) + 1
    folder_name = f"interview_{next_index:03d}"
    full_path = os.path.join(RESULTS_DIR, folder_name)

    os.makedirs(full_path, exist_ok=False)
    return full_path

def save_transcript(transcript: str) -> str:
    """
       Сохраняет текст расшифровки (транскрипта) в файл "transcript.txt" в новой директории результатов.

       params:
           transcript (str): Текст стенограммы аудио (результат распознавания речи).

       return:
           str: Путь к директории, где был сохранён транскрипт.
    """

    save_dir = _get_next_result_dir()
    path = os.path.join(save_dir, "transcript.txt")

    with open(path, "w", encoding="utf-8") as f:
        f.write(transcript)

    return save_dir

def load_prompt_files(folder_path="prompts", exclude=None):
    """
    Загружает все .txt файлы из указанной папки и объединяет в одну строку.
    exclude — список имён файлов, которые нужно исключить.
    """
    prompt_parts = []
    exclude = exclude or []

    for filename in sorted(os.listdir(folder_path)):
        if filename.endswith(".txt") and filename not in exclude:
            with open(os.path.join(folder_path, filename), "r", encoding="utf-8") as f:
                prompt_parts.append(f.read().strip())

    return "\n\n".join(prompt_parts)

def save_results_file(gpt_response: str, save_dir: str) -> str:
    """
    Сохраняет результат работы GPT (техническое задание) в ту же папку, где и transcript.
    """
    path = os.path.join(save_dir, "technical_specification.txt")
    with open(path, "w", encoding="utf-8") as f:
        f.write(gpt_response)

    logger.info(f"Техническое задание сохранено в: {path}")
    return path
