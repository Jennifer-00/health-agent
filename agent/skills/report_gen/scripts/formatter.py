"""Markdown 后处理：清理 LLM 输出，确保格式合规。"""
import re
from datetime import datetime


def inject_timestamp(report: str) -> str:
    """若报告顶部缺少生成时间，自动补全。"""
    ts = datetime.now().strftime("%Y-%m-%d %H:%M")
    if "**生成时间**" not in report:
        report = re.sub(
            r"(#\s*健康摘要报告)",
            rf"\1\n**生成时间**：{ts}",
            report,
            count=1,
        )
    return report


def ensure_disclaimer(report: str) -> str:
    """确保报告包含免责声明。"""
    disclaimer = "> **注意**：以下建议仅供参考，不构成医疗诊断，如有疑虑请咨询专业医生。"
    if "不构成医疗诊断" not in report:
        report += f"\n\n{disclaimer}"
    return report


def clean(report: str) -> str:
    """去除多余空行，统一换行符。"""
    report = report.replace("\r\n", "\n")
    report = re.sub(r"\n{3,}", "\n\n", report)
    return report.strip()


def format_report(raw: str) -> str:
    report = clean(raw)
    report = inject_timestamp(report)
    report = ensure_disclaimer(report)
    return report
