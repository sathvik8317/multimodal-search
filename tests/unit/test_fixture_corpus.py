import ast
import csv

import fitz  # PyMuPDF
from PIL import Image

from tests.fixtures import CORPUS_DIR


def test_pdf_fixture_has_two_pages_with_real_text_layer():
    pdf_path = CORPUS_DIR / "specs" / "rfc.pdf"
    assert pdf_path.exists()
    doc = fitz.open(pdf_path)
    try:
        assert doc.page_count == 2
        for page in doc:
            assert page.get_text().strip()  # real, non-OCR text layer
    finally:
        doc.close()


def test_pdf_fixture_pages_have_distinct_content():
    pdf_path = CORPUS_DIR / "specs" / "rfc.pdf"
    doc = fitz.open(pdf_path)
    try:
        texts = [page.get_text() for page in doc]
    finally:
        doc.close()
    assert texts[0] != texts[1]


def test_diagram_fixture_is_a_valid_image():
    png_path = CORPUS_DIR / "docs" / "auth-flow.png"
    assert png_path.exists()
    with Image.open(png_path) as img:
        img.verify()
    with Image.open(png_path) as img:
        assert img.format == "PNG"
        assert img.width > 0 and img.height > 0


def test_table_fixture_is_parseable_csv_with_header():
    csv_path = CORPUS_DIR / "data" / "latency.csv"
    assert csv_path.exists()
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    assert reader.fieldnames
    assert len(rows) >= 1
    for row in rows:
        assert set(row.keys()) == set(reader.fieldnames)


def test_code_fixture_is_valid_python_with_known_symbols():
    py_path = CORPUS_DIR / "src" / "ingest" / "base.py"
    assert py_path.exists()
    tree = ast.parse(py_path.read_text())
    class_names = {n.name for n in ast.walk(tree) if isinstance(n, ast.ClassDef)}
    func_names = {n.name for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)}
    assert "PdfIngester" in class_names
    assert "rasterize" in func_names
    assert "_embed_and_write" in func_names
