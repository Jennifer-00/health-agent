"""
Critic 失败案例的存取层。

写入：每次 Critic 拦截时追加一条记录到 JSONL。
读取：构建 system prompt 时加载最近 N 条作为反例注入。
"""
import json
import logging
from datetime import datetime
from pathlib import Path

logger = logging.getLogger(__name__)

_LOG_PATH = Path("logs/critic_failures.jsonl")
_MAX_INJECT = 3  # 每次注入的反例条数，太多会稀释 prompt 重点


def write_failure(user_message: str, assistant_response: str, note: str) -> None:
    """追加一条失败案例，同步写入（调用频率低，不需要异步）。"""
    try:
        _LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        record = {
            "ts": datetime.now().isoformat(),
            "user_message": user_message,
            "assistant_response": assistant_response,
            "note": note,
        }
        with _LOG_PATH.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    except Exception as exc:
        logger.warning("[critic_store] write failed: %s", exc)


def load_recent_failures(n: int = _MAX_INJECT) -> list[dict]:
    """读取最近 n 条失败案例。"""
    try:
        if not _LOG_PATH.exists():
            return []
        lines = _LOG_PATH.read_text(encoding="utf-8").splitlines()
        records = [json.loads(l) for l in lines if l.strip()]
        return records[-n:]
    except Exception as exc:
        logger.warning("[critic_store] load failed: %s", exc)
        return []


def format_failure_examples(failures: list[dict]) -> str:
    """把失败案例格式化为 prompt 片段。"""
    if not failures:
        return ""
    lines = ["## 历史错误案例（请勿重复）"]
    for i, f in enumerate(failures, 1):
        lines.append(
            f"{i}. 用户说：「{f['user_message']}」\n"
            f"   错误回复：「{f['assistant_response'][:60]}…」\n"
            f"   问题所在：{f['note']}"
        )
    return "\n".join(lines)
