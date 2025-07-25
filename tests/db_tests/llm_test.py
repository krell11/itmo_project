from openai import AsyncOpenAI
import asyncio
import time
from colorama import Fore, Style  # Для цветного вывода

from tqdm.asyncio import tqdm
# Настройка цветов для разных запросов
COLORS = [Fore.RED, Fore.GREEN, Fore.YELLOW, Fore.BLUE, Fore.MAGENTA, Fore.CYAN]

client = AsyncOpenAI(
    base_url="http://localhost:8000/v1",
    api_key="dummy_value"
)


async def async_request(prompt, request_id):
    color = COLORS[request_id % len(COLORS)]
    start_time = time.time()

    print(f"{color}🚀 Запрос {request_id} СТАРТ ({time.strftime('%H:%M:%S', time.localtime())})"
          f"\nПромпт: {prompt[:40]}...{Style.RESET_ALL}")

    try:
        response = await client.chat.completions.create(
            model="/models/models--deepseek-ai--DeepSeek-R1-Distill-Qwen-1.5B",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            max_tokens=1000
        )

        duration = time.time() - start_time

        print(f"{color}✅ Запрос {request_id} ЗАВЕРШЕН [Длительность: {duration:.2f}s]"
              f"\nОтвет: {response.choices[0].message.content[:60]}...{Style.RESET_ALL}\n")

        return {
            "id": request_id,
            "start": start_time,
            "end": time.time(),
            "duration": duration,
            "response": response
        }

    except Exception as e:
        print(f"{color}❌ Запрос {request_id} ОШИБКА: {str(e)}{Style.RESET_ALL}")
        return None


async def main():

    prompts = [
        "Explain quantum computing in simple terms",
        "Describe neural networks like I'm five years old",
        "What is artificial intelligence?",
        "How does GPS work?",
        "Explain blockchain technology"
    ]
    tasks = [async_request(prompt, i + 1) for i, prompt in enumerate(prompts)]
    results = []
    for f in tqdm(asyncio.as_completed(tasks), total=len(tasks)):
        results.append(await f)

    tasks = [async_request(prompt, i + 1) for i, prompt in enumerate(prompts)]
    results = await asyncio.gather(*tasks)

    print("\n📊 Статистика выполнения:")
    print(f"{'-' * 40}")
    for result in filter(None, results):
        print(f"Запрос {result['id']}:")
        print(f"  Начало:   {time.strftime('%H:%M:%S', time.localtime(result['start']))}")
        print(f"  Конец:    {time.strftime('%H:%M:%S', time.localtime(result['end']))}")
        print(f"  Длит.:    {result['duration']:.2f}s")
        print(f"{'-' * 40}")



if __name__ == "__main__":
    asyncio.run(main())