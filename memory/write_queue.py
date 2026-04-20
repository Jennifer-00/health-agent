import asyncio
from concurrent.futures import ThreadPoolExecutor

_executors: dict[str, ThreadPoolExecutor] = {}
_pending_futures: set = set()


def _get_executor(user_id: str) -> ThreadPoolExecutor:
    if user_id not in _executors:
        _executors[user_id] = ThreadPoolExecutor(
            max_workers=1, thread_name_prefix=f"write-{user_id}"
        )
    return _executors[user_id]


def enqueue_write(user_id: str, coro) -> None:
    """将写入协程提交到用户专属串行线程池，对调用方完全非阻塞。"""
    loop = asyncio.get_event_loop()
    future = loop.run_in_executor(_get_executor(user_id), asyncio.run, coro)
    _pending_futures.add(future)
    future.add_done_callback(_pending_futures.discard)
