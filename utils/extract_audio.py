import os
import ffmpeg
import whisper

from settings.config import logger

def extract_audio_ffmpeg(video_path, audio_path="audio_output/audio.wav"):
    """
    Извлекает аудиодорожку из видеофайла с помощью ffmpeg и сохраняет её в формате WAV.

    params:
        video_path (str): Путь к исходному видеофайлу.
        audio_path (str): Путь, по которому будет сохранён извлечённый аудиофайл (по умолчанию: "audio_output/audio.wav").

    return:
        str: Путь к сохранённому аудиофайлу.

    """
    os.makedirs(os.path.dirname(audio_path), exist_ok=True)

    logger.info(f"Извлечение аудио из {video_path} с помощью ffmpeg...")
    (
        ffmpeg
        .input(video_path)
        .output(audio_path, ac=1, ar='16000', format='wav')
        .run(overwrite_output=True, quiet=True)
    )
    logger.info(f"Аудио сохранено в {audio_path}")
    return audio_path


def transcribe_audio(audio_path):
    """
       Выполняет распознавание речи из аудиофайла с помощью модели Whisper.

       params:
           audio_path (str): Путь к аудиофайлу в формате WAV.

       return:
           dict: Результат распознавания речи, включая текст и сегменты.
    """

    logger.info(f"Загрузка модели Whisper...")
    model = whisper.load_model("base")  # можно заменить на medium/large для большей точности

    logger.info(f"Распознавание речи...")
    result = model.transcribe(audio_path, language='ru', task="transcribe")
    logger.info(f"Распознавание завершено.")
    return result



def accept_file(input_file):
    """
        Принимает путь к видеофайлу, проверяет его существование, извлекает аудио
        и выполняет транскрибацию.

        params:
            input_file (str): Путь к видеофайлу.

        return:
            str: Путь к извлечённому аудиофайлу.

    """

    if len(input_file) < 2:
        logger.error("❌ Укажите путь к видеофайлу.\nПример: python extract_audio.py video.mp4")

    video_path = input_file
    if not os.path.isfile(video_path):
        logger.error("❌ Указанный файл не найден.")

    audio_path = extract_audio_ffmpeg(video_path)
    transcribe_audio(audio_path)

    return audio_path

