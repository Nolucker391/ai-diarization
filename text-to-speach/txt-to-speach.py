import asyncio
import edge_tts
import os
import shutil

from moviepy import AudioFileClip, ColorClip
from pydub import AudioSegment

# Настройки
OUTPUT_FOLDER = "output" # путь для сохранения аудио-звуки
FINAL_MP4 = "dialog_output.mp4" # итог-файл
DIALOG_FILE = "example.txt" # читаемый txt-файл

# Голоса для озвучки
voices = {
    "Speaker 1": "en-US-GuyNeural",
    "Speaker 2": "en-US-JennyNeural"
}

def parse_dialog(file_path):
    """
    Читает txt-файл
    """
    dialog_lines = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            if ":" in line:
                speaker, text = line.strip().split(":", 1)
                dialog_lines.append((speaker.strip(), text.strip()))
    return dialog_lines


async def generate_audio(dialog):
    """
    Функция, для генерации mp3-аудио с инструментом edge-tts
    """
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)
    for i, (speaker, text) in enumerate(dialog):
        voice = voices.get(speaker, "en-US-GuyNeural")
        filename = os.path.join(OUTPUT_FOLDER, f"line_{i+1:02d}.mp3")
        communicate = edge_tts.Communicate(text, voice)
        print(f"🎤 Generating: {filename}")
        await communicate.save(filename)


def merge_audio_to_single_mp3():
    """
    Функция, для объединения всех созданных аудио-файлов в один аудиофайл
    """
    files = sorted([f for f in os.listdir(OUTPUT_FOLDER) if f.endswith(".mp3")])
    combined = AudioSegment.empty()
    for file in files:
        audio = AudioSegment.from_file(os.path.join(OUTPUT_FOLDER, file))
        combined += audio + AudioSegment.silent(duration=500)
    final_audio_path = os.path.join(OUTPUT_FOLDER, "full_dialog.mp3")
    combined.export(final_audio_path, format="mp3")
    return final_audio_path


def create_mp4_from_audio(audio_path, output_path):
    """
    Функция, для создание mp4-формата видео.
    """
    audio_clip = AudioFileClip(audio_path)
    duration = audio_clip.duration

    # Заглушка-видео с чёрным фоном
    video = ColorClip(size=(1280, 720), color=(0, 0, 0), duration=duration).with_fps(24)

    # Добавляем звук к видео
    video = video.with_audio(audio_clip)

    # сохраняем
    video.write_videofile(output_path, codec="libx264", audio_codec="aac")


if __name__ == "__main__":
    dialog = parse_dialog(DIALOG_FILE)
    asyncio.run(generate_audio(dialog))
    audio_path = merge_audio_to_single_mp3()
    create_mp4_from_audio(audio_path, FINAL_MP4)
    print(f"Готово! Файл сохранён как: {FINAL_MP4}")
    shutil.rmtree(OUTPUT_FOLDER)