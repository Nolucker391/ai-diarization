## 🔧 Установка

1. Клонируйте репозиторий:
````
git clone https://github.com/Nolucker391/ai-diarization.git
````

2. Установите зависимости:

````
pip install -r requirements.txt
````

⚠️ если вдруг не корректно работает с ru-языком или ошибки при сборе пакетов, поробуйте
выполнить очистку зависимостей и подтяните с флагом --no-cache-dir.

````
pip install --no-cache-dir -r requirements.txt
````

3. 🔑 Добавьте ключ от OpenAI в переменные окружения `.env`:

````
OPENAI_API_KEY = str
````

4. ▶️ Запуск программы:

````
python main.py "файл_видео.mp4"
````

## 📁 Зависимости системы

в системе должен быть установлен:

- [python 3.10+](https://www.python.org/downloads/release/python-3106/)
- [ffmpeg](https://github.com/BtbN/FFmpeg-Builds/releases) — для извлечения аудио из видео (и закинуть их желательно в переменную среду)
- [C++ SDK](https://visualstudio.microsoft.com/ru/visual-cpp-build-tools/) (на случай если вылезут ошибки с файлами c++)
- [Perl](https://strawberryperl.com/)




Больших объемов видео-файлов могут занять некоторое время



Если возникнет ошибка с регионом на запрос к GPT: 
1. Простой вариант Использовать в системе VPN
2. Попробовать export в переменные окружения прокси сервера
````
set HTTP_PROXY=http://your-proxy:port
set HTTPS_PROXY=http://your-proxy:port
````
3. Установить [proxifer](https://www.proxifier.com/)



источник: https://github.com/MahmoudAshraf97/whisper-diarization