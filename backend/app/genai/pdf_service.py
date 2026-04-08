"""PDF generation helpers for Phase 15/16.

ReportLab is imported lazily so the backend can still start even if the
dependency has not been installed yet in a local environment.
"""

from __future__ import annotations

import base64
from io import BytesIO

from app.genai.config import genai_settings
from app.genai.schemas import AIReportResponse


def build_pdf_bytes(response: AIReportResponse) -> bytes:
    """Generate a compact PDF for the current structured report."""

    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import getSampleStyleSheet
    from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer

    buffer = BytesIO()
    doc = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        title=f"{genai_settings.pdf_title_prefix} - {response.prediction.report_type}",
    )
    styles = getSampleStyleSheet()
    story = [
        Paragraph(
            f"{genai_settings.pdf_title_prefix} ({response.prediction.report_type})",
            styles["Title"],
        ),
        Spacer(1, 12),
        Paragraph(response.prediction.report.summary, styles["BodyText"]),
        Spacer(1, 12),
        Paragraph(
            f"Confidence: {response.prediction.report.confidence.band} "
            f"({response.prediction.report.confidence.score:.2f})",
            styles["BodyText"],
        ),
        Spacer(1, 12),
    ]

    for heading, items in (
        ("Strengths", response.prediction.report.strengths),
        ("Weaknesses", response.prediction.report.weaknesses),
        ("Suggestions", response.prediction.report.suggestions),
    ):
        story.append(Paragraph(heading, styles["Heading2"]))
        for item in items:
            story.append(Paragraph(f"- {item}", styles["BodyText"]))
        story.append(Spacer(1, 8))

    if response.sources.evidence_references:
        story.append(Paragraph("Evidence", styles["Heading2"]))
        for ref in response.sources.evidence_references:
            story.append(
                Paragraph(
                    f"- {ref.source} ({ref.relevance_score:.2f}) {ref.excerpt}",
                    styles["BodyText"],
                )
            )

    doc.build(story)
    return buffer.getvalue()


def build_pdf_base64(response: AIReportResponse) -> str:
    """Return the PDF artifact encoded as base64 for API transport."""

    return base64.b64encode(build_pdf_bytes(response)).decode("ascii")
