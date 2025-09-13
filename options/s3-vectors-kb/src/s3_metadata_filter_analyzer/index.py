import boto3
import json
import os
import logging
from datetime import datetime
from typing import Dict, List, Any, Optional
from collections import defaultdict

# Import the correct custom client
from idp_common.s3vectors.client import S3VectorsClient
from idp_common.bedrock.client import BedrockClient
from idp_common.s3 import get_s3_client
from idp_common.dynamodb.client import DynamoDBClient

# --- Constants ---
METADATA_ANALYSIS_SAMPLE_SIZE = 1000  # Max vectors to sample for analysis
MIN_FIELD_OCCURRENCE_RATE = 0.1       # Field must be in at least 10% of samples
MAX_FILTERABLE_FIELDS_TO_REPORT = 8   # Max number of fields to report and generate examples for
NON_FILTERABLE_FIELDS = {'text_content', 's3_uri'}


# Initialize clients in the global scope for connection reuse.
logger = logging.getLogger(__name__)
logger.setLevel(os.environ.get("LOG_LEVEL", "INFO").upper())

s3_vectors_client = S3VectorsClient()
s3_client = get_s3_client()
lambda_client = boto3.client('lambda')
bedrock_client = BedrockClient()
dynamodb_resource = boto3.resource('dynamodb')
db_table = dynamodb_resource.Table(os.environ.get("S3_VECTORS_CATALOG_TABLE", "s3-vectors-catalog")) # Added default for local testing

def get_all_index_names() -> List[str]:
    """
    Gets a list of all S3 vector indexes to be scanned.
    In simple mode, returns just the default index.
    In dynamic mode, returns all discovered cluster indexes plus the default.
    """
    default_index = os.environ["S3_VECTORS_INDEX_NAME"]
    try:
        response = db_table.scan(
            FilterExpression="begins_with(PK, :pk_prefix)",
            ExpressionAttributeValues={":pk_prefix": "INDEX#"}
        )
        items = response.get('Items', [])
        if not items:
            logger.info("No routing index found. Operating in simple mode.")
            return [default_index]

        cluster_indexes = [item['PK'].split('#', 1)[1] for item in items]
        all_indexes = list(set(cluster_indexes + [default_index]))
        logger.info(f"Operating in dynamic mode. Found {len(all_indexes)} indexes to scan: {all_indexes}")
        return all_indexes
    except Exception:
        logger.error("Failed to query for routing index. Falling back to simple mode.", exc_info=True)
        return [default_index] # Fail safe to simple mode

def handler(event: Dict[str, Any], context: Any) -> Dict[str, Any]:
    """
    Analyzes S3 Vectors metadata, generates filter examples using an LLM,
    stores them in S3, and updates the query Lambda's environment.
    """
    trigger_source = event.get('trigger_source', 'manual')
    logger.info(f"Starting metadata filter analysis (triggered by: {trigger_source})")

    try:
        metadata_analysis = analyze_metadata()
        if not metadata_analysis.get('filterable_fields'):
            logger.warning("No filterable fields found. Aborting.")
            return {'statusCode': 200, 'body': json.dumps({'message': 'No filterable fields found.'})}

        filter_examples = generate_filter_examples(metadata_analysis)
        s3_location = store_filter_examples(filter_examples)
        
        filterable_fields_keys = list(metadata_analysis['filterable_fields'].keys())
        update_lambda_environment(filterable_fields_keys, filter_examples)

        response_body = {
            'message': 'Filter analysis completed successfully.',
            'trigger_source': trigger_source,
            'filterable_fields': filterable_fields_keys,
            'examples_stored_at': s3_location,
            'example_count': len(filter_examples)
        }
        return {'statusCode': 200, 'body': json.dumps(response_body)}

    except Exception as e:
        logger.error("Failed to complete metadata filter analysis.", exc_info=True)
        return {
            'statusCode': 500,
            'body': json.dumps({'error': 'An internal error occurred.'})
        }


def analyze_metadata() -> Dict[str, Any]:
    """
    Scans a sample of vectors from all relevant indexes to find common metadata fields.
    """
    field_analysis = defaultdict(lambda: {'count': 0, 'sample_values': set(), 'data_type': None})
    all_sampled_vectors = []
    
    indexes_to_scan = get_all_index_names()
    sample_per_index = max(1, METADATA_ANALYSIS_SAMPLE_SIZE // len(indexes_to_scan))

    for index_name in indexes_to_scan:
        try:
            logger.info(f"Analyzing up to {sample_per_index} vectors from index '{index_name}'...")
            response = s3_vectors_client.list_vectors(
                vectorBucketName=os.environ["S3_VECTORS_BUCKET_NAME"],
                indexName=index_name,
                maxResults=sample_per_index,
                returnMetadata=True
            )
            all_sampled_vectors.extend(response.get('vectors', []))
        except Exception:
            logger.warning(f"Could not sample vectors from index '{index_name}'. It might not exist yet.", exc_info=True)

    total_vectors_sampled = len(all_sampled_vectors)
    if total_vectors_sampled == 0:
        logger.warning("No vectors found in any index.")
        return {'total_vectors_sampled': 0, 'filterable_fields': {}}

    for vector in all_sampled_vectors:
        for field, value in vector.get('metadata', {}).items():
            if field not in NON_FILTERABLE_FIELDS:
                info = field_analysis[field]
                info['count'] += 1
                if len(info['sample_values']) < 10:
                    info['sample_values'].add(str(value))
                if info['data_type'] is None:
                    info['data_type'] = type(value).__name__

    min_occurrence = max(1, total_vectors_sampled * MIN_FIELD_OCCURRENCE_RATE)
    filterable_fields = {
        field: {
            'occurrence_rate': round(info['count'] / total_vectors_sampled, 2),
            'sample_values': list(info['sample_values']),
            'data_type': info['data_type']
        }
        for field, info in field_analysis.items() if info['count'] >= min_occurrence
    }

    sorted_fields = sorted(filterable_fields.items(), key=lambda item: item[1]['occurrence_rate'], reverse=True)
    top_fields = dict(sorted_fields[:MAX_FILTERABLE_FIELDS_TO_REPORT])

    logger.info(f"Found {len(top_fields)} filterable fields meeting the threshold from {total_vectors_sampled} sampled vectors.")
    return {'total_vectors_sampled': total_vectors_sampled, 'filterable_fields': top_fields}


def _parse_filter_response(response) -> List[Dict[str, Any]]:
    """Parse filter examples from Bedrock response."""
    # Extract content from the BedrockClient response format
    bedrock_response = response.get('response', {})
    content_blocks = bedrock_response.get('output', {}).get('message', {}).get('content', [])
    content = content_blocks[0].get('text', '[]') if content_blocks else '[]'
    
    content = content.strip()
    if content.startswith('```json'):
        content = content.replace('```json', '').replace('```', '').strip()

    filter_examples = json.loads(content)
    logger.info(f"Successfully generated {len(filter_examples)}")
    return filter_examples


def generate_filter_examples(metadata_analysis: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Generates useful filter examples using a Bedrock LLM."""
    
    filterable_fields = metadata_analysis['filterable_fields']
    prompt = f"""
        You are an expert in S3 Vectors metadata filtering. Based on the following metadata analysis, create useful filter examples that a query Lambda can use as templates.

        Metadata Analysis:
        {json.dumps(filterable_fields, indent=2)}

        S3 VECTORS FILTERING RULES:

        Metadata Constraints:
        - Max 2 KB per vector
        - Types: string, number, boolean, list
        - All fields filterable except: text_content, document_id, s3_uri
        - Use sparingly: confidence, page_number (deterministic)
        - document_type is example only, may not exist

        Filter Operators:
        - $eq, $ne: exact/not equal (string, number, boolean)
        - $gt, $gte, $lt, $lte: numeric comparisons 
        - $in, $nin: array membership
        - $exists: field presence (true/false)
        - $and, $or: logical combinations

        Pattern Examples:
        - Exact match: {{"status": {{"$eq": "processed"}}}}
        - Numeric range: {{"score": {{"$gte": 0.8}}}}
        - Array check: {{"tags": {{"$in": ["urgent", "review"]}}}}
        - Field exists: {{"approved": {{"$exists": true}}}}
        - Multiple conditions: {{"$and": [{{"type": {{"$eq": "report"}}}}, {{"year": {{"$eq": 2024}}}}]}}
        - Alternatives: {{"$or": [{{"priority": {{"$eq": "high"}}}}, {{"due_date": {{"$lt": "2024-01-31"}}}}]}}

        Create 8-10 practical filter examples for real-world document filtering scenarios. Each should be adaptable as a query template.

        CRITICAL: Return ONLY the JSON array, no other text:
        [
        {{
            "name": "descriptive name",
            "description": "what this filter does and when to use it", 
            "use_case": "specific scenario where this filter is useful",
            "filter": {{ "S3 Vectors filter JSON" }}
        }}
        ]
    """

    try:
        params = {
            "model_id": os.environ["LLM_MODEL_ID"],
            "system_prompt": "You are an expert in S3 Vectors metadata filtering.",
            "content": [{"text": prompt}],
            "temperature": 0.1,
            "max_tokens": 4096
        }
        
        response = bedrock_client.invoke_model(**params)
        return _parse_filter_response(response)
        
    except Exception:
        try:
            params["model_id"] = os.environ["ALTERNATIVE_LLM_MODEL_ID"]
            
            response = bedrock_client.invoke_model(**params)
            return _parse_filter_response(response)
        except Exception:
            
            field_names = list(filterable_fields.keys())
            return [
                {
                    'name': 'Default: Field Existence Check',
                    'description': 'Checks if a specific field exists in the metadata.',
                    'use_case': 'A fallback example because the AI model failed to generate examples.',
                    'filter': {field_names[0]: {'$exists': True}} if field_names else {}
                }
            ]

def store_filter_examples(filter_examples: List[Dict[str, Any]]) -> str:
    """Stores the generated filter examples in a versioned and 'latest' S3 object."""
    timestamp = datetime.now().strftime('%Y%m%d-%H%M%S')
    s3_prefix = "metadata-filters/global" # Use a generic prefix for system-wide analysis
    versioned_key = f"{s3_prefix}/filter-examples-{timestamp}.json"
    latest_key = f"{s3_prefix}/filter-examples-latest.json"
    
    filter_data = json.dumps({
        'generated_timestamp_utc': datetime.utcnow().isoformat(),
        'vector_index': os.environ["S3_VECTORS_INDEX_NAME"],
        'examples': filter_examples
    }, indent=2)
    
    s3_client.put_object(Bucket=os.environ["WORKING_BUCKET"], Key=versioned_key, Body=filter_data, ContentType='application/json')
    s3_client.put_object(Bucket=os.environ["WORKING_BUCKET"], Key=latest_key, Body=filter_data, ContentType='application/json')
    
    latest_location = f"s3://{os.environ["WORKING_BUCKET"]}/{versioned_key}"
    logger.info(f"Stored {len(filter_examples)} filter examples at {latest_location}")
    return latest_location

def update_lambda_environment(filterable_fields: List[str], filter_examples: str):
    """Updates the Query Lambda's environment with the latest filter info."""
    try:
        config = lambda_client.get_function_configuration(FunctionName=os.environ["QUERY_LAMBDA_FUNCTION_NAME"])
        env_vars = config.get('Environment', {}).get('Variables', {})

        env_vars['FILTERABLE_METADATA_KEYS'] = ','.join(filterable_fields)
        env_vars['FILTER_EXAMPLES'] = str(filter_examples)

        lambda_client.update_function_configuration(
            FunctionName=os.environ["QUERY_LAMBDA_FUNCTION_NAME"],
            Environment={'Variables': env_vars}
        )
        logger.info("Successfully updated Lambda environment.")
    except Exception:
        logger.error(f"Failed to update environment for Lambda '{os.environ["QUERY_LAMBDA_FUNCTION_NAME"]}'.", exc_info=True)
        raise # Re-raise the exception to fail the handler.