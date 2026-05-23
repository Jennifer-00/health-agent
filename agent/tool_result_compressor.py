"""
工具结果压缩器 —— 将原始工具返回压缩为 LLM 可用的精简摘要，降低上下文 token 占用。

第一阶段重点支持 search_rag（extractive key_points），其他工具走 generic fallback。
"""

import logging
import re

logger = logging.getLogger(__name__)

# ── 可调常量 ────────────────────────────────────────────────────────────────────
_MAX_RAG_CHUNKS = 5       # search_rag 最多保留 chunk 数
_MAX_KEY_POINTS  = 5       # 每个 chunk 最多提取 key_points 数
_MAX_GENERIC_CHARS = 2000  # generic fallback 截断字符数

# 医疗风险关键词 —— 命中 +0.40 分
_MEDICAL_RISK_WORDS = [
    "过敏", "禁忌", "副作用", "不良反应", "不建议",
    "严重", "立即就医", "剂量", "肝功能", "肾功能", "警告",
    "慎用", "禁用", "毒性", "致死", "休克", "呼吸困难",
]

# 数字/时间/剂量模式 —— 命中 +0.20 分
_NUMBER_PATTERN = re.compile(
    r"(\d+\.?\d*\s*(mg|g|ml|μg|单位|片|粒|次/日|次/天|mg/kg)?)"
    r"|(\d+%\s*)"
    r"|(\d+[-~]\d+)"
)
_DATE_PATTERN = re.compile(r"\d+[年月周日天小时分钟]|每日|每周|空腹|餐后|睡前")

# 中文句子切分
_SENTENCE_SPLIT = re.compile(r"[。！？；\n]+")

# RAG 结果块解析
_RAG_BLOCK = re.compile(
    r"【结果 (?P<rank>\d+)】来源：(?P<source>[^\n]+)\n"
    r"章节：(?P<section>[^\n]+)\n"
    r"内容：(?P<content>.*?)(?=\n【结果 \d+】|$)",
    re.DOTALL,
)


def _split_sentences(text: str) -> list[str]:
    """将文本切为句子列表，过滤空串和纯空白。"""
    parts = _SENTENCE_SPLIT.split(text)
    return [s.strip() for s in parts if s.strip() and len(s.strip()) > 1]


def _score_sentence(sentence: str, query: str) -> float:
    """基于规则对句子打分（0.0–1.0 量级，非归一化）。"""
    score = 0.0

    # 1. query 关键词命中
    if query:
        query_tokens = set(query)
        hit_count = sum(1 for t in query_tokens if t in sentence)
        if hit_count > 0:
            score += min(hit_count / max(len(query_tokens), 1), 1.0) * 0.30

    # 2. 医疗风险词命中
    risk_hits = sum(1 for w in _MEDICAL_RISK_WORDS if w in sentence)
    if risk_hits > 0:
        score += min(risk_hits * 0.40, 0.50)

    # 3. 数字/剂量命中
    if _NUMBER_PATTERN.search(sentence) or _DATE_PATTERN.search(sentence):
        score += 0.20

    return score


def _extract_key_points(content: str, query: str, max_points: int = _MAX_KEY_POINTS) -> list[str]:
    """从 chunk content 中提取 top-N 关键句，按原文顺序输出。"""
    sentences = _split_sentences(content)
    if not sentences:
        return []

    if len(sentences) <= max_points:
        return sentences

    # (原始位置, 得分, 句子)
    indexed: list[tuple[int, float, str]] = []
    for i, sent in enumerate(sentences):
        s = _score_sentence(sent, query)
        if i < 2:
            s += 0.10   # 前两句轻微加权
        indexed.append((i, s, sent))

    indexed.sort(key=lambda x: x[1], reverse=True)  # 按得分降序
    top_indices = sorted([idx for idx, _, _ in indexed[:max_points]])  # 按原文位置排序
    return [sentences[i] for i in top_indices]


def _compress_rag(raw_result: str, query: str) -> str:
    """压缩 search_rag 返回结果：最多保留 top-N chunk，每个 chunk 提取 key_points。"""
    matches = list(_RAG_BLOCK.finditer(raw_result))

    if not matches:
        # 解析不到标准格式，可能是错误信息或降级文本，直接返回
        return raw_result

    raw_chunks = len(matches)
    kept = matches[:_MAX_RAG_CHUNKS]

    parts: list[str] = []
    parts.append("[search_rag compressed result]")
    parts.append(f"raw_chunks: {raw_chunks}")
    parts.append(f"kept_chunks: {len(kept)}")
    parts.append("compression: extractive")
    parts.append("")

    for i, m in enumerate(kept, 1):
        source  = m.group("source").strip()
        section = m.group("section").strip()
        content = m.group("content").strip()

        key_points = _extract_key_points(content, query)

        parts.append(f"{i}. source: {source}")
        parts.append(f"   section: {section}")
        parts.append(f"   key_points:")
        if key_points:
            for kp in key_points:
                parts.append(f"   - {kp}")
        else:
            # 极端情况：句子切分失败，直接截断原文
            parts.append(f"   - {content[:200]}…")

        parts.append("")

    return "\n".join(parts)


def _generic_compress(raw_result: str) -> str:
    """兜底截断：按段落优先，超过阈值则硬截断并标记 [TRUNCATED]."""
    if len(raw_result) <= _MAX_GENERIC_CHARS:
        return raw_result

    # 按段落截断，优先保留完整段落
    paragraphs = raw_result.split("\n\n")
    accumulated = ""
    for para in paragraphs:
        if len(accumulated) + len(para) + 2 > _MAX_GENERIC_CHARS:
            break
        accumulated += para + "\n\n"

    if accumulated:
        return accumulated.rstrip() + "\n\n…[TRUNCATED]"
    # 连一个段落都放不下，硬截断
    return raw_result[:_MAX_GENERIC_CHARS] + "\n\n…[TRUNCATED]"


def compress_tool_result(tool_name: str, raw_result: str, query: str = "") -> str:
    """
    将工具原始返回压缩为 LLM 可用的精简结果。

    返回 llm_result（压缩后文本）。
    raw_chars / llm_chars / compression_ratio 通过日志输出。
    """
    if not isinstance(raw_result, str):
        return str(raw_result)

    raw_chars = len(raw_result)

    if tool_name == "search_rag" and query:
        llm_result = _compress_rag(raw_result, query)
    else:
        llm_result = _generic_compress(raw_result)

    llm_chars = len(llm_result)
    ratio = f"{llm_chars / max(raw_chars, 1) * 100:.1f}%"

    logger.info(
        "[compressor] tool=%s raw_chars=%d llm_chars=%d ratio=%s compressed=%s",
        tool_name, raw_chars, llm_chars, ratio,
        "yes" if llm_chars < raw_chars else "no (passthrough)",
    )

    return llm_result
