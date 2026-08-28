import asyncio
import aiohttp
from loguru import logger
from tqdm.asyncio import tqdm

from evaluation.eval.config import API_BASE, API_KEY, MODEL_NAME, CONCURRENT_REQUESTS


async def api_worker(session, queue, results_list):
    url = f"{API_BASE}/chat/completions"

    while True:
        try:
            task = queue.get_nowait()
        except asyncio.QueueEmpty:
            break

        payload = {
            "model": MODEL_NAME,
            "messages": task['payload'],
            "temperature": 0.0,
            "max_tokens": 1024,
        }

        response_text = None
        try:
            async with session.post(url, json=payload) as response:
                if response.status == 200:
                    result = await response.json()
                    response_text = result['choices'][0]['message']['content']
                else:
                    err_text = await response.text()
                    logger.error(f"API Error {response.status}: {err_text[:200]}")
        except Exception as e:
            logger.error(f"Request failed: {e}")

        results_list.append((task, response_text))
        queue.task_done()


async def progress_monitor(total, results_list):
    with tqdm(total=total, mininterval=1.0) as pbar:
        last_len = 0
        while last_len < total:
            current_len = len(results_list)
            diff = current_len - last_len
            if diff > 0:
                pbar.update(diff)
                last_len = current_len

            if current_len >= total:
                break

            await asyncio.sleep(0.5)


async def run_massive_inference(prepared_tasks):
    logger.info(f"PHASE 2: Blasting API with {CONCURRENT_REQUESTS} concurrent workers via Queue...")

    connector = aiohttp.TCPConnector(limit=0, ttl_dns_cache=300)
    timeout = aiohttp.ClientTimeout(total=1200)

    queue = asyncio.Queue()
    for t in prepared_tasks:
        queue.put_nowait(t)

    results_list = []

    async with aiohttp.ClientSession(connector=connector, timeout=timeout, headers={"Authorization": f"Bearer {API_KEY}"}) as session:
        monitor = asyncio.create_task(progress_monitor(len(prepared_tasks), results_list))

        workers = [
            asyncio.create_task(api_worker(session, queue, results_list))
            for _ in range(CONCURRENT_REQUESTS)
        ]

        await asyncio.gather(*workers)
        await monitor

    return results_list
