"""Markdown 健康报告 → PDF bytes（Playwright headless Chromium 渲染）。

Windows 上 uvicorn 的 asyncio loop 不支持直接 spawn 子进程，
所以用 sync_playwright + asyncio.to_thread 跑在线程池里规避该限制。
"""
import asyncio

import markdown
from playwright.sync_api import sync_playwright

_CSS = """
body {
    font-family: 'Microsoft YaHei', 'PingFang SC', 'Noto Sans CJK SC', sans-serif;
    font-size: 13px;
    line-height: 1.8;
    color: #1a1a1a;
    padding: 48px 56px;
    max-width: 720px;
    margin: 0 auto;
}
h1 { font-size: 22px; color: #065f46; border-bottom: 2px solid #d1fae5; padding-bottom: 8px; }
h2 { font-size: 16px; color: #047857; margin-top: 24px; }
h3 { font-size: 14px; color: #374151; margin-top: 16px; }
hr { border: none; border-top: 1px solid #e5e7eb; margin: 20px 0; }
table { border-collapse: collapse; width: 100%; font-size: 12px; }
th { background: #ecfdf5; color: #065f46; padding: 6px 10px; text-align: left; border: 1px solid #d1fae5; }
td { padding: 5px 10px; border: 1px solid #e5e7eb; }
blockquote { border-left: 3px solid #6ee7b7; margin: 12px 0; padding: 4px 12px; color: #4b5563; background: #f0fdf4; }
ul, ol { padding-left: 20px; }
li { margin: 4px 0; }
strong { color: #111827; }
p { margin: 6px 0; }
"""


def _render_pdf(full_html: str) -> bytes:
    """在同步线程里启动 Chromium，渲染 HTML 并输出 PDF bytes。"""
    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page()
        page.set_content(full_html, wait_until="networkidle")
        pdf_bytes = page.pdf(
            format="A4",
            margin={"top": "20px", "right": "20px", "bottom": "20px", "left": "20px"},
            print_background=True,
        )
        browser.close()
    return pdf_bytes


async def markdown_to_pdf(md_text: str) -> bytes:
    html_body = markdown.markdown(
        md_text,
        extensions=["tables", "nl2br", "sane_lists"],
    )
    full_html = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<style>{_CSS}</style>
</head>
<body>{html_body}</body>
</html>"""
    return await asyncio.to_thread(_render_pdf, full_html)
