"""
健康报告生成脚本。

用法：
    python -m agent.skills.report_gen.scripts.run <user_id>

从 Mem0 拉取用户全量记忆，调用 Claude 按 output_format.md 规范生成 Markdown 报告，
打印到 stdout（供 Claude Code skill 机制捕获）。
"""
import asyncio
import os
import sys
from pathlib import Path

import anthropic

from memory.mem0_client import Mem0Client
from agent.skills.report_gen.scripts.formatter import format_report

_REFERENCES_DIR = Path(__file__).parent.parent / "references"


def _load_output_format() -> str:
    path = _REFERENCES_DIR / "output_format.md"
    return path.read_text(encoding="utf-8")


def _memories_to_text(memories: list[dict]) -> str:
    lines = []
    for m in memories:
        if not m:
            continue
        content = m.get("memory") or m.get("content", "")
        if not content:
            continue
        category = (m.get("metadata") or {}).get("category", "")
        prefix = f"[{category}] " if category else ""
        lines.append(f"- {prefix}{content}")
    return "\n".join(lines) if lines else ""


async def generate_report(user_id: str) -> str:
    memories = await Mem0Client(user_id=user_id).get_all()
    memory_text = _memories_to_text(memories)

    if not memory_text:
        return "# 健康摘要报告\n\n> 暂无健康记录，请先与助手对话记录您的健康信息。"

    output_format = _load_output_format()

    system_prompt = f"""你是一位专业的健康档案整理助手。
根据用户的全部健康记忆生成一份结构化 Markdown 健康摘要报告。

严格遵循以下格式规范：
{output_format}

规则：
- 只使用已有记忆中的信息，不添加、不推测任何未记录的内容
- 不做医疗诊断
- 使用中文输出
"""

    user_prompt = f"""以下是该用户的全部健康记忆记录（共 {len(memories)} 条）：

{memory_text}

请按格式规范生成健康摘要报告。"""

    client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
    message = client.messages.create(
        model=os.getenv("REPORT_MODEL", "claude-haiku-4-5-20251001"),
        max_tokens=2048,
        system=system_prompt,
        messages=[{"role": "user", "content": user_prompt}],
    )

    if not message.content or not hasattr(message.content[0], "text"):
        return "# 健康摘要报告\n\n> 报告生成失败：模型返回内容为空，请稍后重试。"
    raw = message.content[0].text
    return format_report(raw)


def main() -> None:
    if len(sys.argv) < 2:
        print("Usage: python -m agent.skills.report_gen.scripts.run <user_id>", file=sys.stderr)
        sys.exit(1)

    user_id = sys.argv[1]
    report = asyncio.run(generate_report(user_id))
    print(report)


if __name__ == "__main__":
    main()
