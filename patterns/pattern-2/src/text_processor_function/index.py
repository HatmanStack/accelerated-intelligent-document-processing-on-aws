# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0

"""
TextProcessorAndChunker function for the S3 Vector Backend.
This Lambda performs semantic chunking of text files.
"""

import json
import logging
import os
import boto3
import re
from urllib.parse import urlparse

logger = logging.getLogger()
logger.setLevel(os.environ.get("LOG_LEVEL", "INFO"))

s3 = boto3.client('s3')

def semantic_chunker(text: str) -> list[dict]:
    """
    Splits text into semantic chunks based on Markdown headers, paragraphs, and code blocks.
    """
    chunks = []
    parts = re.split(r'(```[\s\S]*?```)', text)
    current_header = ""
    
    for part in filter(None, parts):
        part_stripped = part.strip()
        if not part_stripped:
            continue

        if part_stripped.startswith('```') and part_stripped.endswith('```'):
            chunks.append({
                "content": part_stripped,
                "metadata": {"type": "code", "header": current_header}
            })
        else:
            header_split = re.split(r'(?m)(^#+ .*)', part)
            if header_split[0].strip():
                paragraphs = re.split(r'\n\s*\n', header_split[0].strip())
                for para in filter(None, paragraphs):
                    chunks.append({
                        "content": para.strip(),
                        "metadata": {"type": "paragraph", "header": ""}
                    })
            for i in range(1, len(header_split), 2):
                header = header_split[i].strip()
                content = header_split[i+1].strip() if (i+1) < len(header_split) else ""
                if content:
                    paragraphs = re.split(r'\n\s*\n', content)
                    for para in filter(None, paragraphs):
                        chunks.append({
                            "content": para.strip(),
                            "metadata": {"type": "paragraph", "header": header}
                        })
    return chunks

def handler(event, context):
    """
    Lambda handler for text processing and chunking.
    """
    logger.info(f"Event: {json.dumps(event, default=str)}")

    document = event.get("document", {})
    # This function expects the raw text file URI from the TextExtractor output
    s3_uri = document.get("raw_text_s3_uri")
    
    if not s3_uri:
        raise ValueError("Missing 'raw_text_s3_uri' in the event document.")

    parsed_uri = urlparse(s3_uri)
    bucket = parsed_uri.netloc
    key = parsed_uri.path.lstrip('/')
    
    file_name = os.path.basename(key)
    tmp_path = f"/tmp/{file_name}"
    
    try:
        logger.info(f"Downloading s3://{bucket}/{key} for chunking.")
        s3.download_file(bucket, key, tmp_path)
        
        with open(tmp_path, 'r', encoding='utf-8') as f:
            text_content = f.read()
        
        os.remove(tmp_path)
        
        chunks = semantic_chunker(text_content)
        
        # Add chunks to the document object to be passed to the next step
        document['chunks'] = chunks
        
        logger.info(f"Successfully chunked document into {len(chunks)} chunks.")
    
    except Exception as e:
        logger.error(f"Error during chunking: {e}")
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        raise

    return event
