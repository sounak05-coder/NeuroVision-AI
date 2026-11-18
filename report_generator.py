
import io
import uuid
from datetime import datetime
from pathlib import Path

import numpy as np
from PIL import Image

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import mm
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
    Image as PDFImage,
    KeepTogether
)


def generate_segmentation_report(
    filename,
    display_mri,
    predicted_mask,
    overlay,
    tumour_pixels,
    tumour_percentage,
    threshold
):
    """
    Generate a PDF report from an existing U-Net segmentation result.

    This function does not perform model inference.
    """

    report_id = str(uuid.uuid4())[:8].upper()
    generated_at = datetime.now().astimezone().strftime(
        "%d %B %Y, %I:%M:%S %p %Z"
    )

    pdf_buffer = io.BytesIO()

    document = SimpleDocTemplate(
        pdf_buffer,
        pagesize=A4,
        rightMargin=18 * mm,
        leftMargin=18 * mm,
        topMargin=16 * mm,
        bottomMargin=16 * mm
    )

    styles = getSampleStyleSheet()

    title_style = ParagraphStyle(
        "ReportTitle",
        parent=styles["Title"],
        fontName="Helvetica-Bold",
        fontSize=21,
        leading=26,
        textColor=colors.HexColor("#087FA5"),
        alignment=TA_CENTER,
        spaceAfter=5 * mm
    )

    subtitle_style = ParagraphStyle(
        "ReportSubtitle",
        parent=styles["Normal"],
        fontSize=9,
        textColor=colors.HexColor("#526779"),
        alignment=TA_CENTER,
        spaceAfter=5 * mm
    )

    heading_style = ParagraphStyle(
        "SectionHeading",
        parent=styles["Heading2"],
        fontSize=13,
        leading=17,
        textColor=colors.HexColor("#075985"),
        spaceBefore=5 * mm,
        spaceAfter=3 * mm
    )

    normal_style = ParagraphStyle(
        "ReportBody",
        parent=styles["BodyText"],
        fontSize=9,
        leading=14,
        textColor=colors.HexColor("#263746")
    )

    small_style = ParagraphStyle(
        "SmallText",
        parent=styles["BodyText"],
        fontSize=8,
        leading=11,
        textColor=colors.HexColor("#526779")
    )

    story = []

    # -------------------------------------------------
    # Report heading
    # -------------------------------------------------

    story.append(Paragraph("NEUROVISION AI", title_style))

    story.append(
        Paragraph(
            "BRAIN MRI TUMOUR SEGMENTATION REPORT",
            subtitle_style
        )
    )

    story.append(
        Paragraph(
            "<b>Report ID:</b> "
            + report_id
            + " &nbsp;&nbsp; <b>Generated:</b> "
            + generated_at,
            small_style
        )
    )

    story.append(Spacer(1, 5 * mm))

    # -------------------------------------------------
    # Source image information
    # -------------------------------------------------

    story.append(Paragraph("1. Image Information", heading_style))

    info_data = [
        ["Uploaded MRI filename", str(filename)],
        ["Image resolution", "256 × 256 pixels (model input)"],
        ["Segmentation threshold", f"{float(threshold):.2f}"]
    ]

    info_table = Table(
        info_data,
        colWidths=[55 * mm, 105 * mm]
    )

    info_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (0, -1), colors.HexColor("#E8F4F8")),
        ("TEXTCOLOR", (0, 0), (-1, -1), colors.HexColor("#263746")),
        ("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold"),
        ("FONTNAME", (1, 0), (1, -1), "Helvetica"),
        ("FONTSIZE", (0, 0), (-1, -1), 8),
        ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#D5E1E8")),
        ("PADDING", (0, 0), (-1, -1), 7)
    ]))

    story.append(info_table)

    # -------------------------------------------------
    # Convert arrays into in-memory PNG images
    # -------------------------------------------------

    def make_image_buffer(array, mode=None):
        array = np.asarray(array)

        if mode == "mask":
            array = (array > 0).astype(np.uint8) * 255
            pil_image = Image.fromarray(array, mode="L")
        else:
            if array.dtype != np.uint8:
                array = np.clip(array, 0, 255).astype(np.uint8)

            pil_image = Image.fromarray(array)

        buffer = io.BytesIO()
        pil_image.save(buffer, format="PNG")
        buffer.seek(0)
        return buffer

    mri_buffer = make_image_buffer(display_mri)
    mask_buffer = make_image_buffer(predicted_mask, mode="mask")
    overlay_buffer = make_image_buffer(overlay)

    # -------------------------------------------------
    # Visual segmentation results
    # -------------------------------------------------

    story.append(Paragraph("2. Segmentation Visualisation", heading_style))

    image_width = 50 * mm
    image_height = 50 * mm

    image_cells = [
        PDFImage(mri_buffer, width=image_width, height=image_height),
        PDFImage(mask_buffer, width=image_width, height=image_height),
        PDFImage(overlay_buffer, width=image_width, height=image_height)
    ]

    image_table = Table(
        [image_cells],
        colWidths=[54 * mm, 54 * mm, 54 * mm]
    )

    image_table.setStyle(TableStyle([
        ("ALIGN", (0, 0), (-1, -1), "CENTER"),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("BOX", (0, 0), (-1, -1), 0.5, colors.HexColor("#C7D8E3")),
        ("INNERGRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#FFFFFF")),
        ("PADDING", (0, 0), (-1, -1), 3)
    ]))

    story.append(image_table)

    captions = Table(
        [[
            Paragraph("Original MRI", small_style),
            Paragraph("Predicted Tumour Mask", small_style),
            Paragraph("MRI with Tumour Overlay", small_style)
        ]],
        colWidths=[54 * mm, 54 * mm, 54 * mm]
    )

    captions.setStyle(TableStyle([
        ("ALIGN", (0, 0), (-1, -1), "CENTER"),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("PADDING", (0, 0), (-1, -1), 4)
    ]))

    story.append(captions)

    # -------------------------------------------------
    # Segmentation statistics
    # -------------------------------------------------

    story.append(Paragraph("3. Segmentation Results", heading_style))

    total_pixels = 256 * 256

    result_data = [
        ["Parameter", "Result"],
        ["Predicted tumour pixels", f"{int(tumour_pixels):,}"],
        ["Predicted area", f"{float(tumour_percentage):.2f}%"],
        ["Total image pixels", f"{total_pixels:,}"],
        ["Binary threshold", f"{float(threshold):.2f}"]
    ]

    result_table = Table(
        result_data,
        colWidths=[80 * mm, 80 * mm]
    )

    result_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#075985")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("BACKGROUND", (0, 1), (-1, -1), colors.HexColor("#F2F7FA")),
        ("TEXTCOLOR", (0, 1), (-1, -1), colors.HexColor("#263746")),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTNAME", (0, 1), (0, -1), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 9),
        ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#D5E1E8")),
        ("PADDING", (0, 0), (-1, -1), 8)
    ]))

    story.append(result_table)

    # -------------------------------------------------
    # Interpretation and disclaimer
    # -------------------------------------------------

    story.append(Paragraph("4. Interpretation", heading_style))

    if int(tumour_pixels) > 0:
        interpretation = (
            "The U-Net model predicted a segmented region at the "
            "selected threshold. This represents an algorithmic "
            "segmentation output."
        )
    else:
        interpretation = (
            "The U-Net model did not predict a tumour region at "
            "the selected threshold."
        )

    story.append(Paragraph(interpretation, normal_style))

    story.append(Spacer(1, 4 * mm))

    story.append(Paragraph("Important Disclaimer", heading_style))

    story.append(
        Paragraph(
            "This report is generated by an academic research prototype. "
            "The segmentation output is not a medical diagnosis and must "
            "not be used for clinical decisions or treatment planning. "
            "Please consult qualified medical professionals for "
            "medical interpretation.",
            small_style
        )
    )

    # -------------------------------------------------
    # Footer
    # -------------------------------------------------

    story.append(Spacer(1, 8 * mm))

    story.append(
        Paragraph(
            "Generated by NeuroVision AI | U-Net Brain Tumour "
            "Segmentation Research Project",
            small_style
        )
    )

    document.build(story)

    pdf_buffer.seek(0)

    return pdf_buffer.getvalue(), report_id
