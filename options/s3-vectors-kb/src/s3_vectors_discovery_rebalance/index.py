import boto3
import json
import os
import logging
import numpy as np
import hdbscan
from typing import List, Dict, Any, Optional, Tuple
from idp_common.bedrock.client import BedrockClient
from idp_common.dynamodb.client import DynamoDBClient
from idp_common.s3vectors.client import S3VectorsClient

# --- Environment Variables ---
LOG_LEVEL = os.environ.get("LOG_LEVEL", "INFO").upper()
STACK_NAME = os.environ["STACK_NAME"]
S3_VECTORS_BUCKET = os.environ["S3_VECTORS_BUCKET"]
S3_VECTORS_CATALOG_TABLE = os.environ["S3_VECTORS_CATALOG_TABLE"]
DYNAMIC_INDEXING_THRESHOLD = int(os.environ["DYNAMIC_INDEXING_THRESHOLD"])
EMBEDDING_MODEL_ID = os.environ["EMBEDDING_MODEL_ID"]
LIGHTWEIGHT_LLM_MODEL_ID = os.environ["LIGHTWEIGHT_LLM_MODEL_ID"]
ALTERNATE_LIGHTWEIGHT_LLM_MODEL_ID = os.environ["ALTERNATE_LIGHTWEIGHT_LLM_MODEL_ID"]
DEFAULT_INDEX_NAME = "default-index" # Assuming the name of the initial index

# --- AWS Service Clients & Logger ---
logger = logging.getLogger(__name__)
logger.setLevel(LOG_LEVEL)

s3vectors_client = S3VectorsClient()
bedrock_client = BedrockClient()
catalog_table = DynamoDBClient(S3_VECTORS_CATALOG_TABLE)
dynamodb_resource = boto3.resource('dynamodb')
table = dynamodb_resource.Table(S3_VECTORS_CATALOG_TABLE)


def handler(event: Dict[str, Any], context: Any) -> Dict[str, Any]:
    """
    Lambda handler for the discovery and rebalancing process.
    """
    logger.info("Starting discovery and rebalance process.")

    try:
        # 1. Entry-Point Guard Clause
        total_vectors = get_total_vector_count()
        logger.info(f"Current total vector count is {total_vectors}.")

        if total_vectors < DYNAMIC_INDEXING_THRESHOLD:
            message = f"Vector count ({total_vectors}) is below threshold ({DYNAMIC_INDEXING_THRESHOLD}). Skipping dynamic indexing."
            logger.info(message)
            return {'statusCode': 200, 'body': json.dumps({'message': message})}

        # If threshold is met, proceed with discovery and rebalancing
        logger.info("Vector count exceeds threshold. Starting dynamic indexing process.")

        # 2. Scan and sample vectors
        logger.info("Scanning and sampling vectors.")
        sampled_vectors = scan_and_sample_vectors(DEFAULT_INDEX_NAME, sample_size=10000)
        if not sampled_vectors:
            raise Exception("No vectors found to process.")

        embeddings = np.array([vec['vector'] for vec in sampled_vectors])

        # 3. Perform clustering with HDBSCAN
        logger.info(f"Performing HDBSCAN clustering on {len(embeddings)} vectors.")
        clusterer = hdbscan.HDBSCAN(min_cluster_size=15, metric='cosine')
        cluster_labels = clusterer.fit_predict(embeddings)

        # 4. Process each cluster
        logger.info(f"Found {len(set(cluster_labels)) - 1} clusters. Processing each one.")
        unique_labels = set(cluster_labels)
        routing_index_items = []

        for label in unique_labels:
            if label == -1:
                # -1 represents outliers, which we will ignore for naming
                continue

            cluster_indices = np.where(cluster_labels == label)[0]
            cluster_embeddings = embeddings[cluster_indices]
            cluster_vectors = [sampled_vectors[i] for i in cluster_indices]

            # 5. Calculate centroid
            centroid = np.mean(cluster_embeddings, axis=0)

            # 6. Name cluster with Bedrock
            cluster_name, cluster_description = name_cluster_with_bedrock(cluster_vectors)

            if not cluster_name:
                logger.warning(f"Could not generate a name for cluster {label}. Skipping.")
                continue

            logger.info(f"Generated name for cluster {label}: {cluster_name}")

            # 7. Prepare item for DynamoDB
            routing_index_items.append({
                'PK': f"INDEX#{cluster_name.replace(' ', '_').lower()}",
                'SK': "METADATA",
                'description': cluster_description,
                'centroid': json.dumps(centroid.tolist()), # Store centroid as JSON string
                'member_count': len(cluster_vectors),
                'created_at': datetime.now(timezone.utc).isoformat()
            })

        # 8. Populate routing index in DynamoDB
        if routing_index_items:
            logger.info(f"Writing {len(routing_index_items)} items to the routing index.")
            batch_write_to_dynamo(routing_index_items)
        else:
            logger.warning("No clusters were successfully named. Routing index not updated.")


        return {
            'statusCode': 200,
            'body': json.dumps({'message': f'Discovery and rebalance process completed successfully. Created {len(routing_index_items)} new indexes.'})
        }

    except Exception as e:
        logger.error("Fatal error during discovery and rebalance process.", exc_info=True)
        return {
            'statusCode': 500,
            'body': json.dumps({'error': str(e)})
        }


def get_total_vector_count() -> int:
    """
    Retrieves the total number of vectors currently in the system.
    This is a placeholder and needs a more robust implementation.
    For now, it counts the number of documents in the catalog.
    A better implementation would be to store a counter in DynamoDB.
    """
    # This is an approximation. A dedicated counter would be better.
    try:
        response = table.scan(
            Select='COUNT',
            FilterExpression="begins_with(PK, :pk_prefix)",
            ExpressionAttributeValues={":pk_prefix": "DOC#"}
        )
        count = response.get('Count', 0)
        # Handle pagination if necessary for very large tables
        while 'LastEvaluatedKey' in response:
            response = table.scan(
                Select='COUNT',
                FilterExpression="begins_with(PK, :pk_prefix)",
                ExpressionAttributeValues={":pk_prefix": "DOC#"},
                ExclusiveStartKey=response['LastEvaluatedKey']
            )
            count += response.get('Count', 0)

        # This counts documents, not vectors. We need to sum the 'number_of_vectors' attribute.
        # This is a placeholder for now.
        logger.warning("Current vector count is an approximation based on document count.")
        return count
    except Exception:
        logger.error("Failed to get total vector count from DynamoDB.", exc_info=True)
        return 0 # Fail safe

def scan_and_sample_vectors(index_name: str, sample_size: int) -> List[Dict]:
    """Scans vectors from an index and returns a random sample."""
    all_vectors = []
    next_token = None
    while True:
        try:
            response = s3vectors_client.list_vectors(
                vectorBucketName=S3_VECTORS_BUCKET,
                indexName=index_name,
                nextToken=next_token
            )
            all_vectors.extend(response.get('vectors', []))
            next_token = response.get('nextToken')
            if not next_token or len(all_vectors) >= sample_size * 2: # Stop if we have enough to sample from
                break
        except Exception as e:
            logger.error(f"Error listing vectors from index {index_name}: {e}")
            break

    if not all_vectors:
        return []

    # Randomly sample the vectors
    sample_indices = np.random.choice(len(all_vectors), size=min(sample_size, len(all_vectors)), replace=False)
    return [all_vectors[i] for i in sample_indices]

def name_cluster_with_bedrock(cluster_vectors: List[Dict], samples_for_naming: int = 5) -> Tuple[Optional[str], Optional[str]]:
    """Generates a name and description for a cluster using Bedrock."""
    if not cluster_vectors:
        return None, None

    # Get sample text content from the cluster's vectors
    sample_texts = []
    for i in np.random.choice(len(cluster_vectors), size=min(samples_for_naming, len(cluster_vectors)), replace=False):
        text = cluster_vectors[i].get('metadata', {}).get('text_content', '')
        if text:
            sample_texts.append(text[:500]) # Truncate for prompt

    if not sample_texts:
        return None, None

    system_prompt = "You are an expert in summarizing and naming document clusters. Based on the following text samples from a cluster, provide a short, descriptive name (3-5 words) and a one-sentence description. Return ONLY a JSON object with 'name' and 'description' keys."
    user_prompt = f"Text samples:\n\n---\n\n" + "\n\n---\n\n".join(sample_texts) + "\n\n---\n\nJSON response:"

    try:
        params = {
            "model_id": LIGHTWEIGHT_LLM_MODEL_ID,
            "system_prompt": system_prompt,
            "content": [{"text": user_prompt}],
            "temperature": 0.2,
            "max_tokens": 200,
        }
        response = bedrock_client.invoke_model(**params)

        # The idp_common BedrockClient returns the parsed text directly
        if isinstance(response, str):
            response_text = response
        else:
            # Fallback for unexpected structure
            response_text = response.get("response", {}).get("output", {}).get("message", {}).get("content", [{}])[0].get("text", "")

        if not response_text:
            raise ValueError("Received empty response from Bedrock.")

        result = json.loads(response_text)
        return result.get('name'), result.get('description')

    except Exception:
        logger.error("Bedrock call for cluster naming failed.", exc_info=True)
        return None, None

def batch_write_to_dynamo(items: List[Dict]):
    """Writes items to DynamoDB in batches of 25."""
    with table.batch_writer() as batch:
        for item in items:
            batch.put_item(Item=item)
