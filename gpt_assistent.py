import os
import openai
from dotenv import load_dotenv

class GPTAssistant:
    """
    GPT-ассистент для генерации ТЗ их транскрита.

    Используется модель: GPT-4-Turbo
    """
    @staticmethod
    def _load_api_key():
        """
        Загружает переменную OPENAI_API_KEY из .env-файла и устанавливает её в openai.api_key.
        Вызывается при каждом обращении к API (чтобы не держать состояние в классе).
        """
        load_dotenv()
        openai.api_key = os.getenv("OPENAI_API_KEY")
        if not openai.api_key:
            raise ValueError("OpenAI API key not found in .env")

    @staticmethod
    def chat(
        message: str,
        model: str = "gpt-4-turbo",
        temperature: float = 0.7,
        max_tokens: int = 1024,
        top_p: float = 1.0,
        frequency_penalty: float = 0.0,
        presence_penalty: float = 0.0,
        stop: list[str] = None
    ) -> str:
        """
        Отправляет одно сообщение в Chat Completions API (GPT-3.5 / GPT-4).

        Аргументы:
            message (str): Текст сообщения, который будет отправлен в качестве user-промта.
            role (str): Роль отправителя — "user", "system", "assistant" (по умолчанию "user").
            model (str): Название модели (gpt-3.5-turbo, gpt-4 и т.д.).
            temperature (float): Контроль креативности/детерминированности (0.0 — жёсткий, 1.0 — креативный).
            max_tokens (int): Максимальное количество токенов в ответе.
            top_p (float): Альтернатива temperature, работает как сэмплирование из вероятностного распределения.
            frequency_penalty (float): Штраф за повторение одинаковых фраз.
            presence_penalty (float): Штраф за появление новых тем.
            stop (list[str] | None): Список токенов, при которых генерация остановится.

        Возвращает:
            str: Содержимое ответа от модели, без изменения (response["choices"][0]["message"]["content"]).
            Роль фиксирована: system.
        """
        GPTAssistant._load_api_key()

        messages = [
            {"role": "system", "content": message},
        ]
        client = openai.OpenAI()
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p,
            frequency_penalty=frequency_penalty,
            presence_penalty=presence_penalty,
            stop=stop
        )

        return response.choices[0].message.content

    @staticmethod
    def complete(
        prompt: str,
        model: str = "text-davinci-003",
        temperature: float = 0.7,
        max_tokens: int = 1024,
        top_p: float = 1.0,
        frequency_penalty: float = 0.0,
        presence_penalty: float = 0.0,
        stop: list[str] = None
    ) -> str:
        """
        Отправляет запрос в Completions API (устаревший режим, только text-* модели).

        Аргументы:
            prompt (str): Готовая строка-промпт для генерации.
            model (str): Название модели (например, "text-davinci-003").
            temperature, max_tokens, top_p, frequency_penalty, presence_penalty, stop:
                см. описание выше — параметры аналогичны методу chat().

        Возвращает:
            str: Текст ответа от модели, без изменений (response["choices"][0]["text"]).
        """
        GPTAssistant._load_api_key()

        response = openai.Completion.create(
            model=model,
            prompt=prompt,
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p,
            frequency_penalty=frequency_penalty,
            presence_penalty=presence_penalty,
            stop=stop
        )

        return response["choices"][0]["text"]

    @staticmethod
    def moderate(text: str) -> dict:
        """
        Отправляет текст на проверку в Moderation API.

        Аргументы:
            text (str): Текст, который нужно проверить.

        Возвращает:
            dict: Словарь с результатами анализа (ключевые поля: "flagged", "categories", "category_scores").
        """
        GPTAssistant._load_api_key()

        response = openai.Moderation.create(input=text)
        return response["results"][0]
