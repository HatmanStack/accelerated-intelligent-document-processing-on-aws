# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0

"""
TextExtractor function for the TEXT_PATH feature.
This Lambda extracts raw text from various document formats and
structures it to be compatible with the downstream classification and extraction workflow.
"""

import json
import logging
import os
import boto3
import docx2txt
import fitz  # PyMuPDF
from urllib.parse import urlparse
import uuid
import zipfile
import xml.etree.ElementTree as ET

# Rich Text and Word Processing
from striprtf.striprtf import rtf_to_text
from odf import text, teletype
from odf.opendocument import load
import textract

# Markup and Structured Text
import markdown
from bs4 import BeautifulSoup
import yaml
import csv
from docutils.core import publish_parts
import asciidoctor
from pylatexenc.latex2text import LatexNodes2Text

# Ebook formats
import ebooklib
from ebooklib import epub
from mobi import Mobi

logger = logging.getLogger()
logger.setLevel(os.environ.get("LOG_LEVEL", "INFO"))

s3 = boto3.client('s3')

# --- Mock idp_common.models ---
# This is a workaround for the development environment where the idp_common layer is not available.
# In a real deployment, these classes would be imported from the shared idp_common layer.

class Block:
    def __init__(self, text, block_type="LINE", geometry=None, block_id=None):
        self.id = block_id or str(uuid.uuid4())
        self.text = text
        self.block_type = block_type
        self.geometry = geometry or {"BoundingBox": {"Width": 0, "Height": 0, "Left": 0, "Top": 0}}

    def to_dict(self):
        return {
            "Id": self.id,
            "BlockType": self.block_type,
            "Text": self.text,
            "Geometry": self.geometry
        }

class Page:
    def __init__(self, page_number=1, geometry=None):
        self.page_number = page_number
        self.geometry = geometry or {"BoundingBox": {"Width": 1, "Height": 1, "Left": 0, "Top": 0}}
        self.blocks = []

    def add_block(self, block):
        self.blocks.append(block)

    def to_dict(self):
        return {
            "PageNumber": self.page_number,
            "Geometry": self.geometry,
            "Blocks": [block.to_dict() for block in self.blocks]
        }

class Document:
    def __init__(self, s3_uri, document_id):
        self.s3_uri = s3_uri
        self.id = document_id
        self.pages = []
        # Add other fields that might be expected by downstream functions
        self.status = "TEXT_EXTRACTED"
        self.workflow_execution_arn = None
        self.raw_text_s3_uri = None


    def add_page(self, page):
        self.pages.append(page)

    def to_dict(self):
        return {
            "id": self.id,
            "s3_uri": self.s3_uri,
            "pages": [page.to_dict() for page in self.pages],
            "status": self.status,
            "workflow_execution_arn": self.workflow_execution_arn,
            "raw_text_s3_uri": self.raw_text_s3_uri,
        }

# --- End Mock idp_common.models ---

def extract_text_from_docx(file_path):
    """Extracts text from a .docx file."""
    try:
        return docx2txt.process(file_path)
    except Exception as e:
        logger.error(f"Error extracting text from .docx file: {e}")
        return ""

def extract_text_from_pdf(file_path):
    """Extracts text from a .pdf file."""
    try:
        doc = fitz.open(file_path)
        text = "".join(page.get_text() for page in doc)
        doc.close()
        return text
    except Exception as e:
        logger.error(f"Error extracting text from .pdf file: {e}")
        return ""

def extract_text_from_txt(file_path):
    """Extracts text from a plain text file."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()
    except Exception as e:
        logger.error(f"Error extracting text from .txt file: {e}")
        return ""

def extract_text_from_rtf(file_path):
    """Extracts text from an .rtf file."""
    try:
        with open(file_path, 'r') as f:
            return rtf_to_text(f.read())
    except Exception as e:
        logger.error(f"Error extracting text from .rtf file: {e}")
        return ""

def extract_text_from_odt(file_path):
    """Extracts text from an .odt file."""
    try:
        textdoc = load(file_path)
        all_paras = textdoc.getElementsByType(text.P)
        return "\n".join(teletype.extractText(p) for p in all_paras)
    except Exception as e:
        logger.error(f"Error extracting text from .odt file: {e}")
        return ""

def extract_text_from_doc(file_path):
    """
    Extracts text from a .doc file.
    Note: .wpd and .wps are not officially supported due to lack of reliable python libraries.
    `textract` may handle them but it is not guaranteed.
    """
    try:
        return textract.process(file_path).decode('utf-8')
    except Exception as e:
        logger.error(f"Error extracting text from .doc file: {e}")
        return ""

def extract_text_from_pages(file_path):
    """
    Extracts text from an Apple Pages file.
    Note: This is a best-effort implementation and may not work for all .pages files
    due to the complexity of the format.
    """
    text = ""
    try:
        with zipfile.ZipFile(file_path, 'r') as zip_ref:
            for file in zip_ref.namelist():
                if file.endswith('.xml'):
                    with zip_ref.open(file) as xml_file:
                        tree = ET.parse(xml_file)
                        root = tree.getroot()
                        for elem in root.iter():
                            if elem.text:
                                text += elem.text.strip() + "\n"
    except Exception as e:
        logger.error(f"Error extracting text from .pages file: {e}")
    return text

def extract_text_from_html(file_path):
    """Extracts text from an .html file."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            soup = BeautifulSoup(f, 'html.parser')
            return soup.get_text()
    except Exception as e:
        logger.error(f"Error extracting text from .html file: {e}")
        return ""

def extract_text_from_xml(file_path):
    """Extracts text from an .xml file."""
    try:
        tree = ET.parse(file_path)
        root = tree.getroot()
        text = ""
        for elem in root.iter():
            if elem.text:
                text += elem.text.strip() + " "
        return text
    except Exception as e:
        logger.error(f"Error extracting text from .xml file: {e}")
        return ""

def extract_text_from_markdown(file_path):
    """Extracts text from a .md file."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            html = markdown.markdown(f.read())
            soup = BeautifulSoup(html, 'html.parser')
            return soup.get_text()
    except Exception as e:
        logger.error(f"Error extracting text from .md file: {e}")
        return ""

def extract_text_from_json(file_path):
    """Extracts text from a .json file."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            return json.dumps(data, indent=2)
    except Exception as e:
        logger.error(f"Error extracting text from .json file: {e}")
        return ""

def extract_text_from_yaml(file_path):
    """Extracts text from a .yaml file."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = yaml.safe_load(f)
            return yaml.dump(data)
    except Exception as e:
        logger.error(f"Error extracting text from .yaml file: {e}")
        return ""

def extract_text_from_csv(file_path):
    """Extracts text from a .csv file."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            return "\n".join([",".join(row) for row in reader])
    except Exception as e:
        logger.error(f"Error extracting text from .csv file: {e}")
        return ""

def extract_text_from_tsv(file_path):
    """Extracts text from a .tsv file."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            reader = csv.reader(f, delimiter='\t')
            return "\n".join(["\t".join(row) for row in reader])
    except Exception as e:
        logger.error(f"Error extracting text from .tsv file: {e}")
        return ""

def extract_text_from_rst(file_path):
    """Extracts text from a .rst file."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            parts = publish_parts(source=f.read(), writer_name='html')
            html = parts['html_body']
            soup = BeautifulSoup(html, 'html.parser')
            return soup.get_text()
    except Exception as e:
        logger.error(f"Error extracting text from .rst file: {e}")
        return ""

def extract_text_from_asciidoc(file_path):
    """Extracts text from an .asciidoc file."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            html = asciidoctor.convert(f.read(), backend='html5')
            soup = BeautifulSoup(html, 'html.parser')
            return soup.get_text()
    except Exception as e:
        logger.error(f"Error extracting text from .asciidoc file: {e}")
        return ""

def extract_text_from_epub(file_path):
    """Extracts text from an .epub file."""
    try:
        book = epub.read_epub(file_path)
        text = ""
        for item in book.get_items_of_type(ebooklib.ITEM_DOCUMENT):
            soup = BeautifulSoup(item.get_content(), 'html.parser')
            text += soup.get_text() + "\n"
        return text
    except Exception as e:
        logger.error(f"Error extracting text from .epub file: {e}")
        return ""

def extract_text_from_mobi(file_path):
    """Extracts text from a .mobi file."""
    try:
        book = Mobi(file_path)
        book.parse()
        return book.get_text()
    except Exception as e:
        logger.error(f"Error extracting text from .mobi file: {e}")
        return ""

def extract_text_from_tex(file_path):
    """Extracts text from a .tex file."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return LatexNodes2Text().latex_to_text(f.read())
    except Exception as e:
        logger.error(f"Error extracting text from .tex file: {e}")
        return ""

def build_document_structure(raw_text, s3_uri, doc_id, execution_arn):
    """Builds a structured Document object from raw text."""
    doc = Document(s3_uri, doc_id)
    doc.workflow_execution_arn = execution_arn
    page = Page(page_number=1)
    
    lines = raw_text.splitlines()
    for line in lines:
        if line.strip():
            block = Block(text=line.strip(), block_type="LINE")
            page.add_block(block)
    
    doc.add_page(page)
    return doc

def handler(event, context):
    """
    Lambda handler for text extraction and structuring.
    """
    logger.info(f"Event: {json.dumps(event)}")

    WORKING_BUCKET = os.environ.get('WORKING_BUCKET')
    if not WORKING_BUCKET:
        raise ValueError("WORKING_BUCKET environment variable is not set.")

    document_event = event["document"]
    s3_uri = document_event["s3_uri"]
    doc_id = document_event.get("id", str(uuid.uuid4()))
    execution_arn = event.get("execution_arn")
    
    parsed_uri = urlparse(s3_uri)
    bucket = parsed_uri.netloc
    key = parsed_uri.path.lstrip('/')
    
    file_name = os.path.basename(key)
    file_extension = os.path.splitext(file_name)[1].lower()
    
    tmp_path = f"/tmp/{file_name}"

    # Dispatcher for extraction functions
    extraction_dispatcher = {
        '.docx': extract_text_from_docx,
        '.pdf': extract_text_from_pdf,
        '.txt': extract_text_from_txt,
        '.rtf': extract_text_from_rtf,
        '.odt': extract_text_from_odt,
        '.doc': extract_text_from_doc,
        '.pages': extract_text_from_pages,
        '.html': extract_text_from_html,
        '.htm': extract_text_from_html,
        '.xml': extract_text_from_xml,
        '.md': extract_text_from_markdown,
        '.markdown': extract_text_from_markdown,
        '.json': extract_text_from_json,
        '.yaml': extract_text_from_yaml,
        '.yml': extract_text_from_yaml,
        '.csv': extract_text_from_csv,
        '.tsv': extract_text_from_tsv,
        '.rst': extract_text_from_rst,
        '.asciidoc': extract_text_from_asciidoc,
        '.adoc': extract_text_from_asciidoc,
        '.epub': extract_text_from_epub,
        '.mobi': extract_text_from_mobi,
        '.tex': extract_text_from_tex,
        # Plain text formats
        '.py': extract_text_from_txt,
        '.js': extract_text_from_txt,
        '.java': extract_text_from_txt,
        '.sql': extract_text_from_txt,
        '.ini': extract_text_from_txt,
        '.conf': extract_text_from_txt,
        '.config': extract_text_from_txt,
        '.log': extract_text_from_txt,
        '.env': extract_text_from_txt,
    }
    
    try:
        s3.download_file(bucket, key, tmp_path)
        
        extraction_function = extraction_dispatcher.get(file_extension)

        if extraction_function:
            raw_text = extraction_function(tmp_path)
        else:
            logger.warning(f"Unsupported file type: {file_extension}. Treating as plain text.")
            raw_text = extract_text_from_txt(tmp_path)

        os.remove(tmp_path)

        # Save the raw text to a file in the working bucket
        raw_text_output_key = f"text/{doc_id}/{os.path.splitext(file_name)[0]}.txt"
        s3.put_object(
            Bucket=WORKING_BUCKET,
            Key=raw_text_output_key,
            Body=raw_text.encode('utf-8')
        )
        raw_text_s3_uri = f"s3://{WORKING_BUCKET}/{raw_text_output_key}"
        logger.info(f"Raw text saved to {raw_text_s3_uri}")

        structured_document = build_document_structure(raw_text, s3_uri, doc_id, execution_arn)
        structured_document.raw_text_s3_uri = raw_text_s3_uri
        
        return {
            "document": structured_document.to_dict()
        }

    except Exception as e:
        logger.error(f"Error during text extraction and structuring: {e}")
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        raise
