import os
import math

from reportlab.platypus import BaseDocTemplate, PageTemplate, Frame, Paragraph, Spacer, PageBreak, Image, NextPageTemplate, KeepTogether
from reportlab.platypus.tableofcontents import TableOfContents
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import mm
from reportlab.lib.utils import ImageReader
from reportlab.pdfbase import pdfmetrics
from datetime import datetime
def generate_pdf_report(filepath, data: dict):
    # Paths to logos (ensure these paths are correct in production)
    watermark_logo = data.get('watermark_logo_path', 'logo1.png')
    cover_logo = data.get('cover_logo_path', 'logo2.png')

    def add_footer_and_watermark(canvas, doc):
        canvas.saveState()

        # Footer (skip cover page)
        if doc.page > 1:
            now = datetime.now()
            canvas.setFont("Times-Roman", 8)
            canvas.setFillColor("#0e0e0e")
            canvas.drawCentredString(doc.pagesize[0] / 2.0, 15 * mm,
                                     f"{data['country']} - {now.strftime('%Y-%m-%d')} - {now.strftime('%H:%M:%S')}")
            canvas.drawRightString(doc.pagesize[0] - 20 * mm, 15 * mm, f"Page {doc.page - 1}")
            canvas.setFillColor("#ffd051")
            canvas.drawString(20 * mm, 15 * mm, "SLAI:")
            canvas.setFillColor("#0e0e0e")
            canvas.drawString(28 * mm, 15 * mm, "Scalable Learning Autonomous Intelligence")

        # Watermark (applied to all except cover)
        if os.path.exists(watermark_logo):
            logo = ImageReader(watermark_logo)
            canvas.setFillAlpha(0.15)
            img_width = 100 * mm
            x_pos = (doc.pagesize[0] - img_width) / 2.0
            y_pos = (doc.pagesize[1] - img_width) / 2.0
            canvas.drawImage(logo, x_pos, y_pos, width=img_width, preserveAspectRatio=True, mask='auto')
            canvas.setFillAlpha(1)

        canvas.restoreState()

    doc = BaseDocTemplate(
        filepath,
        pagesize=A4,
        leftMargin=15 * mm,
        rightMargin=15 * mm,
        topMargin=15 * mm,
        bottomMargin=20 * mm
    )

    frame = Frame(doc.leftMargin, doc.bottomMargin, doc.width, doc.height, id='normal')
    template = PageTemplate(
        id='with_footer', 
        frames=frame, 
        onPage=add_footer_and_watermark)
    cover_frame = Frame(doc.leftMargin, doc.bottomMargin, doc.width, doc.height, id='cover')
    cover_template = PageTemplate(id='cover', frames=cover_frame, onPage=add_footer_and_watermark)
    doc.addPageTemplates([cover_template, template])

    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(name='ChapterTitle', parent=styles['Heading1'], spaceAfter=12))
    story = []

    # Cover Page
    if os.path.exists(cover_logo):
        story.append(Image(cover_logo, width=130 * mm, height=130 * mm))
    story.append(Spacer(1, 5 * mm))
    story.append(Paragraph("<b>SLAI</b>", styles['Title']))
    story.append(Spacer(1, 95 * mm))
    story.append(Paragraph(f"<b>Report Title:</b> SLAI Session Report", styles['Normal']))
    story.append(Paragraph(f"<b>Date Range:</b> {data['date_range']}", styles['Normal']))
    story.append(Paragraph(f"<b>Generated On:</b> {data['generated_on']}", styles['Normal']))
    story.append(Paragraph(f"<b>Generated By:</b> {data['generated_by']}", styles['Normal']))
    story.append(PageBreak())

    # Table of Contents
    toc = TableOfContents()
    toc.levelStyles = [
        ParagraphStyle(fontSize=12, name='TOCHeading1', leftIndent=20, firstLineIndent=-20, spaceAfter=5),
    ]
    story.append(Paragraph("Table of Contents", styles['Heading1']))
    story.append(Spacer(1, 6))
    story.append(toc)
    story.append(PageBreak())

    def add_chapter(title, content, chapter_number):
        story.append(PageBreak())
        story.append(NextPageTemplate('with_footer'))
        chapter_title = f"Chapter {chapter_number}: {title}"
        heading = Paragraph(f'<font size=14><b>{chapter_title}</b></font>', styles['ChapterTitle'])
        heading._bookmarkName = f"chapter_{chapter_number}"
        toc.addEntry(0, chapter_title, f"chapter_{chapter_number}")
        story.append(KeepTogether([heading]))
        story.append(Spacer(1, 5 * mm))
        if isinstance(content, str):
            story.append(Paragraph(content, styles['BodyText']))
        elif isinstance(content, list):
            for item in content:
                story.append(Paragraph(item, styles['BodyText']))
        story.append(Spacer(1, 5 * mm))

    chapter_counter = 1
    add_chapter("Interaction History", data['chat_history'], chapter_counter)
    chapter_counter += 1

    chapter_data = [
        ("Executive Summary", data['executive_summary']),
        ("Session Metadata", [
            f"<b>User ID:</b> {data['user_id']}",
            f"<b>Total Interactions:</b> {data['interaction_count']}",
            f"<b>Total Duration:</b> {data['total_duration']}",
            f"<b>Modules Used:</b> {data['modules_used']}",
            f"<b>Export Format:</b> {data['export_format']}",
        ]),
        ("Performance Metrics", [
            f"<b>Average Response Time:</b> {data['avg_response_time']}",
            f"<b>Fastest Response Time:</b> {data['min_response_time']}",
            f"<b>Longest Response Time:</b> {data['max_response_time']}",
            f"<b>Session Uptime:</b> {data['session_uptime']}",
            f"<b>Memory Usage Peak:</b> {data['memory_peak']}",
            f"<b>Risk Warnings Triggered:</b> {data['risk_triggered']}",
            f"<b>SafeAI Interventions:</b> {data['safe_ai']}",
            f"<b>Errors Encountered:</b> {data['errors']}",
        ]),
        ("Risk & Safety Logs", data['risk_logs']),
        ("Notable Actions or Recommendations", data['recommendations']),
        ("Attachments & Export Notes", [
            f"<b>Raw Log File:</b> {data['log_file_path']}",
            f"<b>Chat Export:</b> {data['chat_export_path']}",
            f"<b>Notes:</b> {data['notes']}",
        ]),
    ]

    for title, content in chapter_data:
        chapter_counter += 1
        add_chapter(title, content, chapter_counter)

    doc.build(story)
