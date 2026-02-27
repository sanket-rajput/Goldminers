"""
PDF Report Generator -- Professional medical consultation report.

Generates a clean, elegant PDF with:
  - Branded cover page with session info
  - Symptoms summary
  - Health assessment / diagnosis
  - Medications (if selected)
  - Home remedies (if selected)
  - Diagnostic tests (if selected)
  - Conversation transcript
  - Medical disclaimer

Empty sections are omitted entirely.
"""

from __future__ import annotations

import io
import re
from datetime import datetime

from reportlab.lib.pagesizes import A4
from reportlab.lib.units import mm
from reportlab.lib.colors import HexColor
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_JUSTIFY
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
    HRFlowable,
    PageBreak,
    ListFlowable,
    ListItem,
    KeepTogether,
)

from app.utils import setup_logger

logger = setup_logger(__name__)

# ------------------------------------------------------------------ #
#  Color Palette                                                      #
# ------------------------------------------------------------------ #

class Colors:
    PRIMARY      = HexColor("#0f4c81")
    PRIMARY_MID  = HexColor("#2b6cb0")
    SECONDARY    = HexColor("#276749")
    ACCENT       = HexColor("#5b21b6")
    DANGER       = HexColor("#b91c1c")
    WARNING      = HexColor("#92400e")
    DARK         = HexColor("#111827")
    TEXT         = HexColor("#1f2937")
    TEXT_MID     = HexColor("#4b5563")
    TEXT_LIGHT   = HexColor("#6b7280")
    TEXT_MUTED   = HexColor("#9ca3af")
    BG_LIGHT     = HexColor("#f8fafc")
    BG_BLUE      = HexColor("#f0f5ff")
    BG_GREEN     = HexColor("#f0fdf4")
    BG_AMBER     = HexColor("#fffbeb")
    BG_RED_LIGHT = HexColor("#fef2f2")
    BORDER       = HexColor("#d1d5db")
    BORDER_LIGHT = HexColor("#e5e7eb")
    WHITE        = HexColor("#ffffff")

# ------------------------------------------------------------------ #
#  Paragraph Styles                                                   #
# ------------------------------------------------------------------ #

def _build_styles() -> dict[str, ParagraphStyle]:
    base = getSampleStyleSheet()
    s = {}

    s["cover_title"] = ParagraphStyle(
        "CoverTitle", parent=base["Title"],
        fontSize=38, leading=46, textColor=Colors.PRIMARY,
        alignment=TA_CENTER, spaceAfter=6, fontName="Helvetica-Bold",
    )
    s["cover_subtitle"] = ParagraphStyle(
        "CoverSubtitle", parent=base["Normal"],
        fontSize=13, leading=18, textColor=Colors.TEXT_MID,
        alignment=TA_CENTER, spaceAfter=24, fontName="Helvetica",
    )
    s["cover_date"] = ParagraphStyle(
        "CoverDate", parent=base["Normal"],
        fontSize=10, leading=14, textColor=Colors.TEXT_LIGHT,
        alignment=TA_CENTER, spaceAfter=4,
    )
    s["section_heading"] = ParagraphStyle(
        "SectionHeading", parent=base["Heading1"],
        fontSize=15, leading=20, textColor=Colors.PRIMARY,
        spaceBefore=18, spaceAfter=8, fontName="Helvetica-Bold",
        leftIndent=0, borderWidth=0,
    )
    s["sub_heading"] = ParagraphStyle(
        "SubHeading", parent=base["Heading2"],
        fontSize=11, leading=15, textColor=Colors.DARK,
        spaceBefore=10, spaceAfter=5, fontName="Helvetica-Bold",
    )
    s["body"] = ParagraphStyle(
        "Body", parent=base["Normal"],
        fontSize=10, leading=15, textColor=Colors.TEXT,
        alignment=TA_JUSTIFY, spaceAfter=6,
    )
    s["body_small"] = ParagraphStyle(
        "BodySmall", parent=base["Normal"],
        fontSize=9, leading=13, textColor=Colors.TEXT_LIGHT, spaceAfter=4,
    )
    s["user_msg"] = ParagraphStyle(
        "UserMsg", parent=base["Normal"],
        fontSize=9.5, leading=14, textColor=HexColor("#1e3a5f"),
        leftIndent=10, rightIndent=6, spaceBefore=4, spaceAfter=4,
        backColor=Colors.BG_BLUE, borderPadding=(6, 8, 6, 8),
    )
    s["bot_msg"] = ParagraphStyle(
        "BotMsg", parent=base["Normal"],
        fontSize=9.5, leading=14, textColor=Colors.TEXT,
        leftIndent=10, rightIndent=6, spaceBefore=4, spaceAfter=4,
        backColor=Colors.BG_GREEN, borderPadding=(6, 8, 6, 8),
    )
    s["bullet"] = ParagraphStyle(
        "Bullet", parent=base["Normal"],
        fontSize=10, leading=15, textColor=Colors.TEXT,
        leftIndent=14, spaceAfter=3,
    )
    s["med_name"] = ParagraphStyle(
        "MedName", parent=base["Normal"],
        fontSize=10, leading=14, textColor=Colors.PRIMARY,
        fontName="Helvetica-Bold", spaceAfter=2,
    )
    s["med_detail"] = ParagraphStyle(
        "MedDetail", parent=base["Normal"],
        fontSize=9, leading=13, textColor=Colors.TEXT,
        leftIndent=10, spaceAfter=2,
    )
    s["warning_text"] = ParagraphStyle(
        "WarningText", parent=base["Normal"],
        fontSize=8.5, leading=12, textColor=Colors.WARNING,
        leftIndent=10, spaceAfter=2,
    )
    s["note"] = ParagraphStyle(
        "Note", parent=base["Normal"],
        fontSize=8.5, leading=12, textColor=Colors.TEXT_LIGHT,
        fontName="Helvetica-Oblique", alignment=TA_LEFT,
    )
    s["footer"] = ParagraphStyle(
        "Footer", parent=base["Normal"],
        fontSize=7, leading=10, textColor=Colors.TEXT_MUTED,
        alignment=TA_CENTER,
    )
    return s

# ------------------------------------------------------------------ #
#  Text helpers                                                       #
# ------------------------------------------------------------------ #

def _safe(text: str) -> str:
    if not text:
        return ""
    text = str(text)
    text = text.replace("&", "&amp;")
    text = text.replace("<", "&lt;")
    text = text.replace(">", "&gt;")
    text = text.replace('"', "&quot;")
    text = text.replace("\n", "<br/>")
    return text


def _strip_markdown(text: str) -> str:
    if not text:
        return ""
    text = re.sub(r"\*\*(.+?)\*\*", r"\1", text)
    text = re.sub(r"\*(.+?)\*", r"\1", text)
    text = re.sub(r"^#{1,3}\s+", "", text, flags=re.MULTILINE)
    text = re.sub(r"^[\-\*\u2022]\s+", "", text, flags=re.MULTILINE)
    text = re.sub(r"^-{3,}$", "", text, flags=re.MULTILINE)
    text = re.sub(r"^\u2500{3,}$", "", text, flags=re.MULTILINE)
    text = re.sub(r"`(.+?)`", r"\1", text)
    text = re.sub(
        r"[\U0001F300-\U0001F9FF\u2600-\u26FF\u2700-\u27BF"
        r"\U0001FA00-\U0001FA6F\U0001FA70-\U0001FAFF]+", "", text,
    )
    return text.strip()

# ------------------------------------------------------------------ #
#  Section builders                                                   #
# ------------------------------------------------------------------ #

def _section_header(title: str, styles: dict) -> list:
    return [
        Spacer(1, 6),
        HRFlowable(
            width="100%", thickness=1.5, color=Colors.PRIMARY_MID,
            spaceAfter=3, spaceBefore=10,
        ),
        Paragraph(
            f"<font color='{Colors.PRIMARY.hexval()}'><b>{_safe(title)}</b></font>",
            styles["section_heading"],
        ),
        Spacer(1, 2),
    ]


def _build_cover(timestamp: str, session_id: str, styles: dict) -> list:
    elements: list = []
    elements.append(Spacer(1, 80))
    elements.append(HRFlowable(
        width="50%", thickness=2, color=Colors.PRIMARY,
        spaceAfter=24, hAlign="CENTER",
    ))
    elements.append(Paragraph("RagBluCare", styles["cover_title"]))
    elements.append(Paragraph(
        "AI-Assisted Medical Consultation Report", styles["cover_subtitle"],
    ))
    elements.append(HRFlowable(
        width="30%", thickness=0.5, color=Colors.BORDER,
        spaceAfter=28, spaceBefore=6, hAlign="CENTER",
    ))

    try:
        dt = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
        date_str = dt.strftime("%B %d, %Y")
        time_str = dt.strftime("%I:%M %p")
    except Exception:
        date_str = timestamp.split("T")[0] if "T" in str(timestamp) else str(timestamp)
        time_str = ""

    date_line = f"<b>Date:</b>  {_safe(date_str)}"
    if time_str:
        date_line += f"&nbsp;&nbsp;&nbsp;|&nbsp;&nbsp;&nbsp;<b>Time:</b>  {_safe(time_str)}"
    elements.append(Paragraph(date_line, styles["cover_date"]))

    if session_id:
        elements.append(Spacer(1, 4))
        elements.append(Paragraph(
            f"<b>Session:</b>  {_safe(session_id[:8])}",
            styles["cover_date"],
        ))

    elements.append(Spacer(1, 50))
    elements.append(Paragraph(
        "<i>This report is confidential. Generated by the RagBluCare "
        "AI Medical Diagnostic Assistant.</i>",
        ParagraphStyle(
            "ConfNote", fontSize=8.5, leading=12,
            textColor=Colors.TEXT_MUTED, alignment=TA_CENTER,
        ),
    ))
    elements.append(Spacer(1, 14))
    elements.append(HRFlowable(
        width="70%", thickness=0.5, color=Colors.DANGER,
        spaceAfter=6, hAlign="CENTER",
    ))
    elements.append(Paragraph(
        "<b>DISCLAIMER:</b> This AI-generated report is NOT a substitute for "
        "professional medical advice, diagnosis, or treatment.",
        ParagraphStyle(
            "CoverDisc", fontSize=7.5, leading=10,
            textColor=Colors.DANGER, alignment=TA_CENTER,
        ),
    ))
    elements.append(PageBreak())
    return elements


def _build_symptoms(symptoms: list[str], styles: dict) -> list:
    """Patient symptoms summary. Omitted when empty."""
    if not symptoms:
        return []
    elements = _section_header("Reported Symptoms", styles)

    items = [
        ListItem(
            Paragraph(_safe(s), styles["bullet"]),
            bulletColor=Colors.PRIMARY_MID, bulletFontSize=7,
        )
        for s in symptoms if s and s.strip()
    ]
    if items:
        elements.append(ListFlowable(
            items, bulletType="bullet", bulletFontSize=7,
            bulletColor=Colors.PRIMARY_MID, leftIndent=14,
            spaceBefore=3, spaceAfter=5,
        ))
    elements.append(Spacer(1, 6))
    return elements


def _build_diagnosis(diagnosis: str, styles: dict) -> list:
    if not diagnosis or not diagnosis.strip():
        return []
    elements = _section_header("Health Assessment Summary", styles)

    clean = _strip_markdown(diagnosis)
    paragraphs = clean.split("\n\n")
    for para in paragraphs:
        para = para.strip()
        if not para:
            continue
        lines = para.split("\n")
        bullet_lines = [l for l in lines if l.strip().startswith(("\u2022", "-"))]
        if bullet_lines and len(bullet_lines) >= len(lines) // 2:
            items = []
            for line in lines:
                line = line.strip().lstrip("\u2022-").strip()
                if line:
                    items.append(ListItem(
                        Paragraph(_safe(line), styles["bullet"]),
                        bulletColor=Colors.PRIMARY_MID, bulletFontSize=7,
                    ))
            if items:
                elements.append(ListFlowable(
                    items, bulletType="bullet", bulletFontSize=7,
                    bulletColor=Colors.PRIMARY_MID, leftIndent=14,
                    spaceBefore=3, spaceAfter=4,
                ))
        else:
            elements.append(Paragraph(_safe(para), styles["body"]))
    elements.append(Spacer(1, 6))
    return elements


def _build_medications(medications: list[dict], styles: dict) -> list:
    """Safe OTC medication suggestions. Omitted when empty."""
    if not medications:
        return []
    elements = _section_header("Safe Medicine Suggestions", styles)

    for i, med in enumerate(medications, 1):
        name = med.get("name", "Medication")
        dosage = med.get("dosage", "")
        warnings = med.get("warnings", [])
        age_note = med.get("age_note", "")

        block = []
        block.append(Paragraph(
            f"{i}.  <b>{_safe(name)}</b>", styles["med_name"],
        ))
        if dosage:
            block.append(Paragraph(
                f"<b>Dosage:</b>  {_safe(dosage)}", styles["med_detail"],
            ))
        if age_note:
            block.append(Paragraph(
                f"<b>Note:</b>  {_safe(age_note)}", styles["warning_text"],
            ))
        for w in warnings:
            block.append(Paragraph(
                f"Warning:  {_safe(w)}", styles["warning_text"],
            ))
        block.append(Spacer(1, 4))
        elements.append(KeepTogether(block))

    elements.append(Paragraph(
        "<i>These are over-the-counter suggestions only. Never take medication "
        "without consulting a qualified healthcare provider.</i>",
        styles["note"],
    ))
    elements.append(Spacer(1, 6))
    return elements


def _build_remedies(remedies: list, styles: dict) -> list:
    """Home remedies. Accepts list[dict] or list[str]. Omitted when empty."""
    if not remedies:
        return []

    elements = _section_header("Recommended Home Remedies", styles)

    for entry in remedies:
        if isinstance(entry, dict):
            condition = entry.get("condition", "")
            remedy_list = entry.get("remedies", [])
            if condition:
                elements.append(Paragraph(
                    f"<b>For {_safe(condition)}:</b>", styles["sub_heading"],
                ))
            items = [
                ListItem(
                    Paragraph(_safe(r), styles["bullet"]),
                    bulletColor=Colors.SECONDARY, bulletFontSize=7,
                )
                for r in remedy_list if r and str(r).strip()
            ]
            if items:
                elements.append(ListFlowable(
                    items, bulletType="bullet", bulletFontSize=7,
                    bulletColor=Colors.SECONDARY, leftIndent=14,
                    spaceBefore=3, spaceAfter=5,
                ))
        elif isinstance(entry, str) and entry.strip():
            elements.append(Paragraph(
                f"  {_safe(entry)}", styles["bullet"],
            ))

    elements.append(Paragraph(
        "<i>Home remedies are supplementary and should not replace "
        "prescribed medication or professional medical advice.</i>",
        styles["note"],
    ))
    elements.append(Spacer(1, 6))
    return elements


def _build_tests(tests: str, styles: dict) -> list:
    """Recommended diagnostic tests. Omitted when empty."""
    if not tests or not tests.strip():
        return []
    elements = _section_header("Recommended Diagnostic Tests", styles)

    clean = _strip_markdown(tests)
    for line in clean.split("\n"):
        line = line.strip()
        if not line:
            continue
        if line.startswith("\u2022") or line.startswith("-"):
            line = line.lstrip("\u2022- ").strip()
            elements.append(Paragraph(
                f"  {_safe(line)}", styles["bullet"],
            ))
        else:
            elements.append(Paragraph(_safe(line), styles["body"]))

    elements.append(Spacer(1, 6))
    return elements


def _build_conversation(messages: list[dict], styles: dict) -> list:
    if not messages:
        return []
    elements = _section_header("Consultation Transcript", styles)
    for msg in messages:
        role = msg.get("role", "")
        content = _strip_markdown(msg.get("content", ""))
        content = _safe(content)
        if len(content) > 2500:
            content = content[:2500] + "..."
        if role == "user":
            elements.append(Paragraph(
                f"<b>Patient:</b>  {content}", styles["user_msg"],
            ))
        elif role == "assistant":
            elements.append(Paragraph(
                f"<b>RagBluCare:</b>  {content}", styles["bot_msg"],
            ))
    elements.append(Spacer(1, 6))
    return elements


def _build_disclaimer(styles: dict) -> list:
    elements: list = []
    elements.append(Spacer(1, 14))
    elements.append(HRFlowable(
        width="100%", thickness=1, color=Colors.DANGER, spaceAfter=8,
    ))
    disclaimer_text = (
        "<b>IMPORTANT MEDICAL DISCLAIMER</b><br/><br/>"
        "This report has been generated by RagBluCare, an AI-powered diagnostic "
        "assistant. The information provided is for <b>educational and informational "
        "purposes only</b> and is <b>NOT</b> intended as a substitute for professional "
        "medical advice, diagnosis, or treatment.<br/><br/>"
        "Always seek the advice of your physician or other qualified health provider "
        "with any questions regarding a medical condition. Never disregard professional "
        "medical advice or delay seeking it because of information in this report.<br/><br/>"
        "If you believe you have a medical emergency, contact your doctor or call "
        "emergency services immediately."
    )
    disclaimer_style = ParagraphStyle(
        "DisclaimerBox", fontSize=8, leading=11.5,
        textColor=Colors.DANGER, alignment=TA_LEFT,
        borderWidth=0.8, borderColor=Colors.DANGER,
        borderPadding=12, backColor=Colors.BG_RED_LIGHT,
    )
    elements.append(Paragraph(disclaimer_text, disclaimer_style))
    elements.append(Spacer(1, 10))
    return elements

# ------------------------------------------------------------------ #
#  Page header / footer                                               #
# ------------------------------------------------------------------ #

def _page_chrome(canvas, doc):
    canvas.saveState()
    w, h = A4

    canvas.setStrokeColor(Colors.BORDER_LIGHT)
    canvas.setLineWidth(0.4)
    canvas.line(25 * mm, 17 * mm, w - 25 * mm, 17 * mm)

    canvas.setFont("Helvetica", 7)
    canvas.setFillColor(Colors.TEXT_MUTED)
    canvas.drawString(25 * mm, 12.5 * mm, "RagBluCare  --  AI Medical Diagnostic Report")
    canvas.drawCentredString(w / 2, 12.5 * mm, "Confidential")
    canvas.drawRightString(w - 25 * mm, 12.5 * mm, f"Page {doc.page}")

    if doc.page > 1:
        canvas.setStrokeColor(Colors.PRIMARY)
        canvas.setLineWidth(1)
        canvas.line(25 * mm, h - 17 * mm, w - 25 * mm, h - 17 * mm)
        canvas.setFont("Helvetica-Bold", 8)
        canvas.setFillColor(Colors.PRIMARY)
        canvas.drawString(25 * mm, h - 15 * mm, "RagBluCare")
        canvas.setFont("Helvetica", 7)
        canvas.setFillColor(Colors.TEXT_MUTED)
        canvas.drawRightString(w - 25 * mm, h - 15 * mm, "Medical Consultation Report")

    canvas.restoreState()

# ------------------------------------------------------------------ #
#  Public API                                                         #
# ------------------------------------------------------------------ #

def generate_report_pdf(
    messages: list[dict],
    diagnosis: str = "",
    symptoms: list[str] | None = None,
    medications: list[dict] | None = None,
    remedies: list | None = None,
    tests: str = "",
    session_id: str = "",
    timestamp: str = "",
    **_kwargs,
) -> io.BytesIO:
    """
    Generate a polished medical consultation PDF.

    Only sections with actual content are rendered.

    Args:
        messages:    List of {role, content} conversation dicts.
        diagnosis:   Diagnosis / assessment summary text.
        symptoms:    List of symptom strings.
        medications: List of {name, dosage, warnings} dicts.
        remedies:    List of {condition, remedies} dicts or strings.
        tests:       Test recommendations text.
        session_id:  Session identifier.
        timestamp:   ISO timestamp for the report date.

    Returns:
        BytesIO buffer containing the finished PDF.
    """
    if not timestamp:
        timestamp = datetime.now().isoformat()

    buffer = io.BytesIO()
    doc = SimpleDocTemplate(
        buffer, pagesize=A4,
        rightMargin=25 * mm, leftMargin=25 * mm,
        topMargin=26 * mm, bottomMargin=24 * mm,
        title="RagBluCare Medical Report",
        author="RagBluCare AI",
        subject="Medical Consultation Report",
    )

    styles = _build_styles()
    elements: list = []

    # 1. Cover page
    elements.extend(_build_cover(timestamp, session_id, styles))

    # 2. Symptoms
    elements.extend(_build_symptoms(symptoms or [], styles))

    # 3. Diagnosis / assessment
    elements.extend(_build_diagnosis(diagnosis, styles))

    # 4. Medications
    elements.extend(_build_medications(medications or [], styles))

    # 5. Home remedies
    elements.extend(_build_remedies(remedies or [], styles))

    # 6. Diagnostic tests
    elements.extend(_build_tests(tests, styles))

    # 7. Conversation transcript
    if messages:
        elements.append(PageBreak())
    elements.extend(_build_conversation(messages, styles))

    # 8. Disclaimer
    elements.extend(_build_disclaimer(styles))

    # Footer
    elements.append(Spacer(1, 10))
    date_part = timestamp.split("T")[0] if "T" in timestamp else timestamp
    elements.append(Paragraph(
        f"Report generated on {date_part} by RagBluCare AI Medical Diagnostic Assistant",
        styles["footer"],
    ))

    try:
        doc.build(elements, onFirstPage=_page_chrome, onLaterPages=_page_chrome)
    except Exception as exc:
        logger.error("PDF build error: %s", exc, exc_info=True)
        raise

    buffer.seek(0)
    return buffer
