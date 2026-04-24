import logging
from datetime import date

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse, Response

from agent.skills.report_gen.scripts.run import generate_report
from agent.skills.report_gen.scripts.pdf_export import markdown_to_pdf

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/report", tags=["report"])


@router.get("/pdf")
async def download_report_pdf(request: Request) -> Response:
    """生成用户健康摘要报告并以 PDF 文件返回。"""
    user_id: str = request.state.user_id
    try:
        md_text = await generate_report(user_id)
        pdf_bytes = await markdown_to_pdf(md_text)
    except Exception as exc:
        logger.exception("[report] PDF generation failed for user=%r: %s", user_id, exc)
        return JSONResponse(status_code=500, content={"detail": str(exc)})

    filename = f"health_report_{date.today().isoformat()}.pdf"
    return Response(
        content=pdf_bytes,
        media_type="application/pdf",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )
