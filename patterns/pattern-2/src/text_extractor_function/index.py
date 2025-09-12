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
    return docx2txt.process(file_path)

def extract_text_from_pdf(file_path):
    """Extracts text from a .pdf file."""
    doc = fitz.open(file_path)
    text = "".join(page.get_text() for page in doc)
    doc.close()
    return text

def extract_text_from_txt(file_path):
    """Extracts text from a .txt or .md file."""
    with open(file_path, 'r', encoding='utf-8') as f:
        return f.read()

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
    
    try:
        s3.download_file(bucket, key, tmp_path)
        
        raw_text = ""
        if file_extension == '.docx':
            raw_text = extract_text_from_docx(tmp_path)
        elif file_extension == '.pdf':
            raw_text = extract_text_from_pdf(tmp_path)
        elif file_extension in ['.txt', '.md']:
            raw_text = extract_text_from_txt(tmp_path)
        else:
            raise ValueError(f"Unsupported file type for text extraction: {file_extension}")

        os.remove(tmp_path)

        # Save the raw text to a file in the working bucket for the S3 backend path
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
        
        # The output of this function should be compatible with what OCRStep produces.
        # It should be a dictionary containing the serialized document.
        return {
            "document": structured_document.to_dict()
        }

    except Exception as e:
        logger.error(f"Error during text extraction and structuring: {e}")
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        raise
