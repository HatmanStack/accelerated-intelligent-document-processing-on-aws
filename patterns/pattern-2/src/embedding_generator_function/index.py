# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0

"""
EmbeddingGenerator function for the S3 Vector Backend.
This Lambda generates embeddings for text chunks using Amazon Bedrock.
"""

import json
import logging
import os
import boto3

logger = logging.getLogger()
logger.setLevel(os.environ.get("LOG_LEVEL", "INFO"))

bedrock_runtime = boto3.client('bedrock-runtime')

BEDROCK_MODEL_ID = os.environ.get("BEDROCK_MODEL_ID", "amazon.titan-embed-text-v1")
BATCH_SIZE = 16

def handler(event, context):
    """
    Lambda handler for generating embeddings.
    """
    logger.info(f"Event: {json.dumps(event, default=str)}")

    document = event.get("document", {})
    chunks = document.get("chunks", [])
    
    if not chunks:
        logger.warning("No chunks found in the document. Nothing to embed.")
        return event

    logger.info(f"Generating embeddings for {len(chunks)} chunks using model {BEDROCK_MODEL_ID}")

    for i in range(0, len(chunks), BATCH_SIZE):
        batch_of_chunks = chunks[i:i + BATCH_SIZE]
        input_texts = [chunk['content'] for chunk in batch_of_chunks]

        body = json.dumps({"inputTexts": input_texts})

        try:
            response = bedrock_runtime.invoke_model(
                body=body,
                modelId=BEDROCK_MODEL_ID,
                accept='application/json',
                contentType='application/json'
            )
            
            response_body = json.loads(response.get('body').read())
            batch_embeddings = response_body.get('embedding')
            
            if not batch_embeddings or len(batch_embeddings) != len(batch_of_chunks):
                 raise ValueError("Mismatch between number of chunks and embeddings in Bedrock response.")

            for j, chunk in enumerate(batch_of_chunks):
                chunk['embedding'] = batch_embeddings[j]

        except Exception as e:
            logger.error(f"Bedrock API call failed for batch starting at index {i}: {e}")
            raise

    logger.info("Successfully generated and added embeddings to all chunks.")

    return event
