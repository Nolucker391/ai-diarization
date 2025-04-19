import os
import sys

from settings.config import logger
from utils.extract_audio import accept_file
from utils.save_text_result import save_transcript
from settings.config import WHISPER_MODEL_NAME, DEVICE, BATCH_SIZE, SUPPRESS_NUMERALS, LANGUAGE
from models.diarize import transcribe_audio


def main():
    if len(sys.argv) < 2:
        logger.error("❌ Укажите путь к видеофайлу.\nПример: python main.py video.mp4")
        sys.exit(1)

    video_path = sys.argv[1]
    if not os.path.isfile(video_path):
        logger.error("❌ Указанный файл не найден.")
        sys.exit(1)

    # 1. Извлечение аудио
    audio_path = accept_file(video_path)
    # audio_path = "audio_output/.wav"
    # 2. Передаем аудио в модуль диаризации

    transcript = transcribe_audio(audio_path, whisper_model_name=WHISPER_MODEL_NAME, device=DEVICE,
                                  batch_size=BATCH_SIZE, suppress_numerals=SUPPRESS_NUMERALS, language=LANGUAGE)

    return save_transcript(transcript)


if __name__ == "__main__":
    logger.info("Приложение запущено.")
    main()