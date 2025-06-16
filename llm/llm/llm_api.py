'''
作用：使用异步编程和API交互来处理聊天消息，并计算成本。
'''


import asyncio
import queue
import threading
import time
from typing import List, Dict, Union, Generator, Optional

from tqdm import tqdm
from openai import AsyncOpenAI, OpenAI

from utils.llm import prompts_to_chat_messages, prompt_to_chat_message

MODEL_PRICES = {
    "default": (5, 15),
    "gpt-3.5-turbo-1106": (1, 2),
    "gpt-3.5-turbo": (0.5, 1.5),
    "gpt-3.5-turbo-0125": (0.375, 1.125),
    "gpt-4o": (1.875, 7.5),
    "gpt-4o-2024-05-13": (3.75, 11.25),
    "gpt-4o-2024-08-06": (1.875, 7.5),
    "gpt-4-turbo": (10, 30),
    "claude-3-haiku-20240307": (0.5, 2.5),
    'gpt-4o-mini-2024-07-18': (0.15, 0.6),
    'gpt-4o-mini': (0.1125, 0.45),
    'Qwen2.5-72B-Instruct': (1, 1),
    'Qwen2-72B-Instruct': (1, 1),
    'gemini-1.5-pro-latest': (0.63, 2.5),
    'Vendor-A/Qwen/Qwen2.5-72B-Instruct': (1, 1),
    'Vendor-A/Qwen/Qwen2-72B-Instruct': (1, 1),
}

EMB_MODEL_PRICES = {
    "default": (5, 15),
    "text-embedding-3-small": (0.02,),
    "text-embedding-3-large": (0.13,)
}


class APIChat:
    def __init__(self, model: str, api_key: str, base_url: Optional[str] = None, num_workers: int = 16, timeout: int = 180,
                 max_retries: int = 3, request_interval: float = 0.02):
        # 包括模型名称、客户端实例、工作线程数、请求间隔、上次请求时间、请求锁、使用情况统计和信号量。
        self.model = model
        self.client = AsyncOpenAI(api_key=api_key, base_url=base_url, timeout=timeout, max_retries=max_retries)
        self.num_workers = num_workers
        self.request_interval = request_interval
        self.last_request_time = 0
        self.request_lock = asyncio.Lock()
        self.usage = {"prompt_tokens": 0, "completion_tokens": 0, "cost": 0}
        self.semaphore = asyncio.Semaphore(num_workers)

    async def wait_for_interval(self):
        # 异步方法wait_for_interval，用于确保请求之间有固定的间隔。
        # 使用async with语句获取请求锁，计算等待时间，如果需要则等待，最后更新上次请求时间。
        async with self.request_lock:
            current_time = time.time()
            wait_time = self.last_request_time + self.request_interval - current_time
            if wait_time > 0:
                await asyncio.sleep(wait_time)
            self.last_request_time = time.time()

    def calculate_cost(self, prompt_tokens, completion_tokens):
        # 计算调用api多少钱
        prompt_price, completion_price = MODEL_PRICES.get(self.model, MODEL_PRICES["default"])
        return prompt_tokens * prompt_price * 1e-6 + completion_tokens * completion_price * 1e-6

    def __call__(self, prompts: Union[str, List[Union[str, List[str], None]]], temperature: float = 0.95,
                 top_p: float = 0.7, max_tokens: int = 2048) -> List[str]:
        # 将提示转换为聊天消息列表，并初始化当前提示和完成令牌计数器。
        messages_list = prompts_to_chat_messages(prompts)
        current_prompt_tokens = 0
        current_completion_tokens = 0

        # # 定义了一个异步方法process_message，用于处理单个消息。
        # async def process_message(messages, index):
        #     # 使用nonlocal关键字来修改外部作用域的变量，并检查消息是否为None。
        #     nonlocal current_prompt_tokens, current_completion_tokens
        #     if messages is None:
        #         return index, None
            
        #     # 使用信号量来限制并发请求，等待请求间隔，发送请求，并更新令牌计数器。
        #     async with self.semaphore:
        #         await self.wait_for_interval()
        #         response = await self.client.chat.completions.create(
        #             messages=messages,
        #             model=self.model,
        #             temperature=temperature,
        #             top_p=top_p,
        #             max_tokens=max_tokens
        #         )
        #         if response.usage:
        #             current_prompt_tokens += response.usage.prompt_tokens
        #             current_completion_tokens += response.usage.completion_tokens
        #         return index, response.choices[0].message.content
        
        async def process_message(messages, index):
            # 使用nonlocal关键字来修改外部作用域的变量，并检查消息是否为None。
            nonlocal current_prompt_tokens, current_completion_tokens
            if messages is None:
                return index, None
            
            # 使用信号量来限制并发请求，等待请求间隔，发送请求，并更新令牌计数器。
            async with self.semaphore:
                await self.wait_for_interval()
                response = await self.client.chat.completions.create(
                    messages=messages,
                    model=self.model,
                    temperature=temperature,
                    top_p=top_p,
                    max_tokens=max_tokens
                )
                
                # 检查response.usage是否存在，并且completion_tokens是否为None
                if response.usage and response.usage.completion_tokens is not None:
                    current_prompt_tokens += response.usage.prompt_tokens
                    current_completion_tokens += response.usage.completion_tokens
                else:
                    # 如果response.usage或completion_tokens为None，可以选择记录日志或采取其他措施
                    print(f"Warning: response.usage or completion_tokens is None for index {index}")
                
                return index, response.choices[0].message.content
            
        # 定义了一个异步方法process_messages，用于处理多个消息。
        async def process_messages():
            # 创建一个任务列表，并将每个消息的处理任务添加到列表中。
            tasks = []
            for index, messages in enumerate(messages_list):
                task = asyncio.ensure_future(process_message(messages, index))
                tasks.append(task)
            
            # 使用asyncio.as_completed来等待所有任务完成，并收集响应。
            responses = []
            for future in tqdm(asyncio.as_completed(tasks), total=len(tasks), desc="Processing messages"):
                index, response = await future
                responses.append((index, response))

            # 按索引排序，并返回响应内容列表。
            responses = sorted(responses, key=lambda x: x[0])

            return [response for _, response in responses]
        
        # 获取事件循环，并运行process_messages方法来处理消息。
        loop = asyncio.get_event_loop()
        responses = loop.run_until_complete(process_messages())

        # 更新使用情况统计，并计算成本。
        self.usage["prompt_tokens"] += current_prompt_tokens
        self.usage["completion_tokens"] += current_completion_tokens
        cost = self.calculate_cost(current_prompt_tokens, current_completion_tokens)
        self.usage["cost"] += cost

        print(f"cost: {cost}, prompt_tokens: {current_prompt_tokens}, completion_tokens: {current_completion_tokens}")
        return responses


