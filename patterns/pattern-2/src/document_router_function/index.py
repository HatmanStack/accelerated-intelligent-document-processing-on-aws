# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0

"""
DocumentRouter function for the TEXT_PATH feature.
This Lambda inspects the uploaded file and routes it to the appropriate workflow.
"""

import json
import logging
import os
import boto3
import fitz  # PyMuPDF
from urllib.parse import urlparse

logger = logging.getLogger()
logger.setLevel(os.environ.get("LOG_LEVEL", "INFO"))

s3 = boto3.client('s3')

# Get the value from an environment variable, defaulting to "false"
IS_TEXT_PATH_ENABLED = os.environ.get("IS_TEXT_PATH_ENABLED", "false").lower() == "true"

def is_pdf_digital(bucket, key):
    """
    Checks if a PDF file is digital (contains text) or image-based.
    """
    tmp_path = f"/tmp/{os.path.basename(key)}"
    try:
        s3.download_file(bucket, key, tmp_path)
        doc = fitz.open(tmp_path)
        
        has_text = any(page.get_text("text").strip() for page in doc)
        
        doc.close()
        os.remove(tmp_path)
        
        if has_text:
            logger.info(f"PDF {key} is digital.")
            return True
        else:
            logger.info(f"PDF {key} is image-based (no text found).")
            return False
    except Exception as e:
        logger.error(f"Error checking PDF {key}: {e}. Defaulting to OCR_PATH.")
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        return False

def handler(event, context):
    """
    Lambda handler for document routing.
    """
    logger.info(f"Event: {json.dumps(event)}")
    
    document = event["document"]
    s3_uri = document["s3_uri"]
    parsed_uri = urlparse(s3_uri)
    bucket = parsed_uri.netloc
    key = parsed_uri.path.lstrip('/')
    
    file_extension = os.path.splitext(key)[1].lower()
    
    path = "OCR_PATH"  # Default path

    if IS_TEXT_PATH_ENABLED:
        logger.info("TEXT_PATH is enabled, checking for eligibility.")
        if file_extension in ['.txt', '.md', '.docx']:
            path = "TEXT_PATH"
        elif file_extension == '.pdf':
            if is_pdf_digital(bucket, key):
                path = "TEXT_PATH"
    else:
        logger.info("TEXT_PATH is disabled, all documents will be routed to OCR_PATH.")

    logger.info(f"Routing document {key} to {path}")
    
    event['path'] = path
    return event
