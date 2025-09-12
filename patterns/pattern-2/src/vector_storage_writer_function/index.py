# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0

"""
VectorStorageWriter function for the S3 Vector Backend.
This Lambda saves text chunks and their embeddings to S3.
"""

import json
import logging
import os
import boto3

logger = logging.getLogger()
logger.setLevel(os.environ.get("LOG_LEVEL", "INFO"))

s3 = boto3.client('s3')
VECTORS_BUCKET = os.environ.get('VECTORS_BUCKET')

def handler(event, context):
    """
    Lambda handler for writing vectors to S3.
    """
    logger.info(f"Event: {json.dumps(event, default=str)}")

    if not VECTORS_BUCKET:
        raise ValueError("VECTORS_BUCKET environment variable is not set.")

    document = event.get("document", {})
    chunks = document.get("chunks", [])
    document_id = document.get("id", "unknown-id")
    
    if not chunks:
        logger.warning(f"No chunks found for document {document_id}. Nothing to write to vector storage.")
        return event

    jsonl_content = "\n".join([json.dumps(chunk) for chunk in chunks])

    original_s3_key = document.get("s3_uri", f"s3://unknown-bucket/unknown-prefix/{document_id}.file").split('/', 3)[-1]
    output_filename = f"{os.path.splitext(original_s3_key)[0]}.jsonl"
    output_key = f"vectors/{document_id}/{output_filename}"
    
    logger.info(f"Writing {len(chunks)} chunks to s3://{VECTORS_BUCKET}/{output_key}")

    try:
        s3.put_object(
            Bucket=VECTORS_BUCKET,
            Key=output_key,
            Body=jsonl_content.encode('utf-8'),
            ContentType='application/jsonl'
        )
        
        vector_s3_uri = f"s3://{VECTORS_BUCKET}/{output_key}"
        document['vector_s3_uri'] = vector_s3_uri
        
        logger.info(f"Successfully wrote vectors to: {vector_s3_uri}")

    except Exception as e:
        logger.error(f"Error writing vectors to S3 for document {document_id}: {e}")
        raise

    return event
