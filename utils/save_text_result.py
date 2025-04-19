import os

from settings.config import RESULTS_DIR

def _get_next_result_dir() -> str:
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
    save_dir = _get_next_result_dir()
    path = os.path.join(save_dir, "transcript.txt")

    with open(path, "w", encoding="utf-8") as f:
        f.write(transcript)

    return save_dir
