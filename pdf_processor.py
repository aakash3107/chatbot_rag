"""
pdf_processor.py
Handles: native text extraction (pypdf + pdfplumber) + OCR (Tesseract) for scanned pages.
PyMuPDF removed — not compatible with Python 3.13 yet.
"""

import pypdf
import pdfplumber
import pytesseract
from PIL import Image
import io
import re
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def extract_text_pypdf(pdf_path, page_num):
    """Fast native extraction using pypdf."""
    try:
        reader = pypdf.PdfReader(pdf_path)
        page = reader.pages[page_num]
        return page.extract_text() or ""
    except Exception as e:
        logger.warning(f"pypdf failed on page {page_num}: {e}")
        return ""


def extract_text_pdfplumber(pdf_path, page_num):
    """Better layout-aware extraction using pdfplumber (slower but more accurate)."""
    try:
        with pdfplumber.open(pdf_path) as pdf:
            page = pdf.pages[page_num]
            return page.extract_text() or ""
    except Exception as e:
        logger.warning(f"pdfplumber failed on page {page_num}: {e}")
        return ""


def ocr_page(pdf_path, page_num):
    """Render PDF page as image and run Tesseract OCR."""
    try:
        with pdfplumber.open(pdf_path) as pdf:
            page = pdf.pages[page_num]
            # Render at 300 DPI for good OCR quality
            img = page.to_image(resolution=300).original
            text = pytesseract.image_to_string(img, lang='eng')
            logger.info(f"Page {page_num+1}: OCR extracted {len(text.strip())} chars")
            return text
    except Exception as e:
        logger.warning(f"OCR failed on page {page_num+1}: {e}")
        return ""


def clean_text(text):
    """Remove headers/footers, normalize whitespace, fix common OCR artifacts."""
    if not text:
        return ""

    lines = text.split('\n')
    cleaned_lines = []

    for line in lines:
        line = line.strip()
        if len(line) < 3:
            continue
        if re.match(r'^\d{1,4}$', line):
            continue
        if re.match(r'^(page\s+\d+|chapter\s+\d+|\d+\s*of\s*\d+)$', line, re.IGNORECASE):
            continue
        cleaned_lines.append(line)

    text = '\n'.join(cleaned_lines)
    text = re.sub(r'\n{3,}', '\n\n', text)
    text = re.sub(r'[ \t]{2,}', ' ', text)
    return text.strip()


def extract_pdf(pdf_path, pdf_id, filename):
    """
    Main function: extract all text from a PDF file.
    Strategy: try pypdf first → fallback pdfplumber → fallback OCR.
    Returns list of dicts: {text, page_num, pdf_id, filename}
    """
    logger.info(f"Processing PDF: {filename}")
    pages_data = []

    try:
        # Get total page count
        reader = pypdf.PdfReader(pdf_path)
        total_pages = len(reader.pages)
        logger.info(f"  Total pages: {total_pages}")

        for page_num in range(total_pages):
            # Strategy 1: pypdf (fastest)
            text = extract_text_pypdf(pdf_path, page_num)

            # Strategy 2: pdfplumber if pypdf got too little
            if len(text.strip()) < 50:
                text = extract_text_pdfplumber(pdf_path, page_num)

            # Strategy 3: OCR if still too little (scanned page)
            if len(text.strip()) < 50:
                logger.info(f"Page {page_num+1}: switching to OCR (sparse text)")
                text = ocr_page(pdf_path, page_num)

            clean = clean_text(text)
            if clean:
                pages_data.append({
                    "text": clean,
                    "page_num": page_num + 1,
                    "pdf_id": pdf_id,
                    "filename": filename,
                    "total_pages": total_pages
                })

        logger.info(f"  Extracted {len(pages_data)}/{total_pages} pages with content")

    except Exception as e:
        logger.error(f"Failed to process {filename}: {e}")

    return pages_data
