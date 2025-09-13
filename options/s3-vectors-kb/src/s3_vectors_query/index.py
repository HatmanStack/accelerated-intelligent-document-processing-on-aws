import boto3
import json
import os
import logging
import uuid
import numpy as np
from typing import List, Dict, Optional, Any
from datetime import datetime, timezone
from idp_common.bedrock.client import BedrockClient
from idp_common.dynamodb.client import DynamoDBClient
from idp_common.s3vectors.client import S3VectorsClient
from boto3.dynamodb.conditions import Key

# --- Constants ---
SIMILARITY_WEIGHT = 0.5
RERANK_WEIGHT = 0.5

# --- Environment Variables ---
MAX_RESULTS = int(os.environ.get("MAX_RESULTS", "20"))
ACTIVE_FILTERABLE_KEYS = [k.strip() for k in os.environ.get("FILTERABLE_METADATA_KEYS", "").split(',') if k.strip()]
GUARDRAIL_ENV = os.environ["GUARDRAIL_ID_AND_VERSION"]

# --- AWS Service Clients & Logger ---
logger = logging.getLogger(__name__)
logger.setLevel(os.environ.get("LOG_LEVEL", "INFO").upper())
s3vectors_client = S3VectorsClient()
bedrock_client = BedrockClient()
cloudwatch_client = boto3.client('cloudwatch')
catalog_table = DynamoDBClient(os.environ["S3_VECTORS_CATALOG_TABLE"])
dynamodb_resource = boto3.resource('dynamodb')
db_table = dynamodb_resource.Table(os.environ["S3_VECTORS_CATALOG_TABLE"])

# --- Global Cache ---
_routing_index_cache = None

def get_routing_index() -> Optional[List[Dict]]:
    """
    Loads the routing index from DynamoDB and caches it.
    Returns None if no index is found, indicating "simple mode".
    """
    global _routing_index_cache
    if _routing_index_cache is not None:
        return _routing_index_cache

    try:
        response = db_table.scan(
            FilterExpression="begins_with(PK, :pk_prefix)",
            ExpressionAttributeValues={":pk_prefix": "INDEX#"}
        )
        items = response.get('Items', [])
        if not items:
            _routing_index_cache = [] # Cache empty list to signify dynamic mode is off
            return None

        # Process items to be usable
        for item in items:
            item['centroid'] = np.array(json.loads(item['centroid']))

        _routing_index_cache = items
        logger.info(f"Successfully loaded and cached routing index with {len(items)} clusters.")
        return _routing_index_cache
    except Exception:
        logger.error("Failed to load routing index from DynamoDB.", exc_info=True)
        return None # Fail safe to simple mode

def find_top_n_clusters(query_embedding: np.ndarray, routing_index: List[Dict], n: int) -> List[str]:
    """Finds the top N most similar clusters for a given query embedding."""
    if not routing_index:
        return []

    cluster_similarities = []
    for cluster in routing_index:
        centroid = cluster['centroid']
        similarity = np.dot(query_embedding, centroid) / (np.linalg.norm(query_embedding) * np.linalg.norm(centroid))
        cluster_similarities.append({
            'name': cluster['PK'].split('#', 1)[1],
            'similarity': similarity
        })

    # Sort clusters by similarity score in descending order
    cluster_similarities.sort(key=lambda x: x['similarity'], reverse=True)

    # Return the names of the top N clusters
    return [cluster['name'] for cluster in cluster_similarities[:n]]


# --- Main Handler ---
def handler(event: Dict[str, Any], context: Any) -> Dict[str, Any]:
    """Main Lambda handler for processing queries against the S3 Vectors knowledge base."""
    start_time = datetime.now(timezone.utc)
    success = False
    citations_count = 0
    
    try:
        args = event["arguments"]
        query = args["input"]
        session_id = args.get("sessionId")
        
        kb_response = process_query_pipeline(query, session_id)
        
        # Count citations
        citations = kb_response.get('citations', [])
        if citations and citations[0].get('retrievedReferences'):
            citations_count = len(citations[0]['retrievedReferences'])

        success = True
        logger.info("Successfully processed S3 vector query")
        return kb_response
        
    except Exception as e:
        logger.error("Error processing S3 vector query", exc_info=True)
        return {
            'error': 'Failed to process S3 vector query',
            'message': str(e),
            'timestamp': datetime.now(timezone.utc).isoformat()
        }
    finally:
        latency_ms = (datetime.now(timezone.utc) - start_time).total_seconds() * 1000
        emit_query_metrics(latency_ms, success, citations_count)

# --- RAG Pipeline Orchestration ---
def process_query_pipeline(query: str, session_id: Optional[str]) -> Dict[str, Any]:
    """Orchestrates the full RAG pipeline from query to response."""
    session_context = get_session_context(session_id) if session_id else None
    
    query_embedding = bedrock_client.generate_embedding(text=query, model_id=os.environ["EMBEDDING_MODEL_ID"])
    if not query_embedding:
        raise Exception("Failed to generate a query embedding.")

    base_filter = enhance_query_with_filter(query)

    # Retrieve candidate documents from S3 Vectors using a multi-slice approach
    candidates = fetch_candidates(query_embedding, base_filter)

    # Rerank candidates to improve relevance
    reranked_candidates = rerank_candidates(query, candidates)

    # Generate a response using the top candidates
    response = generate_response(query, reranked_candidates, session_context)

    if session_id:
        update_session_context(session_id, query, response)

    return response

def fetch_candidates(query_embedding: List[float], base_filter: Optional[Dict]) -> List[Dict]:
    """
    Fetches candidate vectors from S3 Vectors.
    Adapts between simple (single-index) and dynamic (multi-index) modes.
    """
    all_candidates = []
    per_slice_k = min(MAX_RESULTS, 30)

    routing_index = get_routing_index()

    # --- Dynamic Mode: Two-Hop Search ---
    if routing_index:
        logger.info("Dynamic mode: Performing two-hop search.")
        top_n = 2 # Query top 2 clusters + default
        target_indexes = find_top_n_clusters(np.array(query_embedding), routing_index, n=top_n)
        target_indexes.append(os.environ["S3_VECTORS_INDEX_NAME"]) # Always query default index for outliers

        logger.info(f"Querying indexes: {target_indexes}")
        for index_name in set(target_indexes): # Use set to avoid duplicate queries
            all_candidates.extend(query_s3_vectors(query_embedding, per_slice_k, None, index_name))

    # --- Simple Mode: Original Multi-Slice Approach ---
    else:
        logger.info("Simple mode: Performing single-index search.")
        slice_filters = [
            None,  # Unfiltered slice
            base_filter,
            compose_filter(base_filter, {'confidence': {'$gte': 0.8}}),
        ]
        default_index_name = os.environ["S3_VECTORS_INDEX_NAME"]
        for filter_config in slice_filters:
            if filter_config is None and base_filter is None:
                all_candidates.extend(query_s3_vectors(query_embedding, per_slice_k, None, default_index_name))
            elif filter_config is not None:
                all_candidates.extend(query_s3_vectors(query_embedding, per_slice_k, filter_config, default_index_name))

    # Deduplicate by vector_id, keeping highest score
    seen = {}
    for candidate in all_candidates:
        vid = candidate.get('vector_id')
        if vid and (vid not in seen or candidate.get('score', 0) > seen[vid].get('score', 0)):
            seen[vid] = candidate

    return list(seen.values())

def rerank_candidates(query: str, candidates: List[Dict]) -> List[Dict]:
    """Reranks candidates using a dedicated reranker model for better relevance."""
    if not candidates:
        return []
    
    # Filter candidates with text content
    valid_candidates = []
    for i, candidate in enumerate(candidates):
        text_content = candidate.get("metadata", {}).get("text_content", "")
        if text_content:
            valid_candidates.append({"id": i, "text": text_content, "original": candidate})

    if not valid_candidates:
        return candidates

    # Get rerank scores
    texts = [doc["text"] for doc in valid_candidates]
    rerank_scores = rerank_with_bedrock(query, texts)
    
    # Calculate final scores
    for i, doc in enumerate(valid_candidates):
        similarity_score = doc["original"].get('score', 0.0)
        rerank_score = next((r["score"] for r in rerank_scores if r["id"] == i), 0.0)
        doc["final_score"] = (SIMILARITY_WEIGHT * similarity_score) + (RERANK_WEIGHT * rerank_score)

    # Sort by final score and return original candidates
    valid_candidates.sort(key=lambda x: x["final_score"], reverse=True)
    return [doc["original"] for doc in valid_candidates]

def generate_response(query: str, ranked_candidates: List[Dict], session_context: Optional[Dict]) -> Dict:
    """Generates a final response in Bedrock KB format using an LLM."""
    top_docs = ranked_candidates[:10]  # Use top 10 for final answer generation

    retrieved_references = []
    for doc in top_docs:
        metadata = doc.get('metadata', {})
        text_content = metadata.get('text_content')
        if text_content:
            retrieved_references.append({
                'content': {'text': text_content},
                'location': {'s3Location': {'uri': metadata.get('s3_uri', '')}},
                'metadata': {
                    'document_id': metadata.get('document_id', ''),
                    'page_number': metadata.get('page_number', 0)
                }
            })

    response_text = generate_response_from_chunks(query, retrieved_references, session_context)
    
    result = {
        'output': {'text': response_text},
        'citations': [{'retrievedReferences': retrieved_references}] if retrieved_references else []
    }
    
    if session_context and session_context.get('session_id'):
        result['sessionId'] = session_context['session_id']
    
    return result

# --- S3 Vectors and Bedrock Interactions ---
def query_s3_vectors(query_vector: List[float], top_k: int, metadata_filter: Optional[Dict], index_name: str) -> List[Dict]:
    """Queries the S3 Vectors service and normalizes the response."""
    try:
        params = {
            'vectorBucketName': os.environ["S3_VECTORS_BUCKET"],
            'indexName': index_name,
            'queryVector': query_vector,  # Client will handle the float32 formatting
            'topK': top_k,
            'returnMetadata': True,
            'returnDistance': True
        }
        if metadata_filter:
            params['filter'] = metadata_filter.get('filter')
        print(f'METADATA FILTER {metadata_filter}')
        response = s3vectors_client.query_vectors(**params)
        
        results = []
        for vector in response.get('vectors', []):
            distance = float(vector.get('distance', 1.0))
            # Convert distance to similarity score [0, 1] using exponential decay
            # Works well for both cosine and euclidean distances
            score = 1.0 / (1.0 + distance)
            
            results.append({
                'vector_id': vector.get('key', ''),
                'score': score,
                'metadata': vector.get('metadata', {})
            })
        return results
    except Exception:
        logger.error("Failed to query S3 Vectors", exc_info=True)
        return []

def generate_response_from_chunks(query: str, refs: List[Dict], context: Optional[Dict]) -> str:
    """Generates a final answer from chunks using an LLM."""
    if not refs:
        return "I couldn't find any relevant information to answer your question."
    
    # Format context chunks
    chunks = []
    for ref in refs:
        doc_id = ref['metadata'].get('document_id', 'Unknown')
        text = ref['content']['text']
        chunks.append(f"[Source: {doc_id}]\n{text}")
    
    # Add conversation history if available
    history_text = ""
    if context and context.get('conversation_history'):
        recent_history = context['conversation_history'][-3:]  # Last 3 exchanges
        history_parts = []
        for h in recent_history:
            history_parts.append(f"Human: {h['query']}\nAssistant: {h['response']}")
        history_text = f"\n\nConversation History:\n{chr(10).join(history_parts)}"
    
    system_prompt = "You are a helpful AI assistant. Answer the user's question based only on the provided context. Cite sources as [Source: id]. If the context is insufficient, state that clearly."
    user_prompt = f"Context:\n{chr(10).join(chunks)}{history_text}\n\nQuestion: {query}\n\nAnswer:"
    model_response = invoke_bedrock_llm(system_prompt, user_prompt, os.environ["LLM_MODEL_ID"])
    print(f'MODEL_RESPONSE: {model_response}')
    return model_response

def rerank_with_bedrock(query: str, docs: List[str]) -> List[Dict[str, Any]]:
    """Reranks documents using a lightweight LLM."""
    if not docs:
        return []
    
    # Prepare documents for reranking (truncate to avoid token limits)
    numbered_docs = []
    for i, doc in enumerate(docs):
        truncated_doc = doc[:2000] if len(doc) > 2000 else doc
        numbered_docs.append(f'Doc {i}:\n{truncated_doc}')
    
    system_prompt = "You are a document reranker. You MUST return a JSON array with exactly one object for EVERY document provided. Each object must have 'id' (int) and 'score' (float between 0 and 1) representing relevance to the query. The closer to 1 the more relevant. Return ALL documents, even if some have low relevance scores."
    user_prompt = f"Query: {query}\n\nDocuments:\n{chr(10).join(numbered_docs)}\n\nReturn JSON array with relevance scores:"
    
    try:
        response_text = invoke_bedrock_llm(system_prompt, user_prompt, os.environ["LIGHTWEIGHT_LLM_MODEL_ID"])
        print(f'RESPONSE_RERANK: {response_text}')
        scores = json.loads(response_text)
        
        # Validate response format
        valid_scores = []
        for score in scores:
            if isinstance(score, dict) and 'id' in score and 'score' in score:
                if isinstance(score['id'], int) and isinstance(score['score'], (int, float)):
                    valid_scores.append(score)
        
        return valid_scores
    except (json.JSONDecodeError, TypeError, KeyError):
        logger.warning("Reranking failed - using default scores", exc_info=True)
        return [{"id": i, "score": 0.5} for i in range(len(docs))]

def invoke_bedrock_llm(system_prompt: str, user_prompt: str, model_id: str) -> str:
    """Invokes Bedrock API models."""
    try:
        params = {
            "model_id": model_id,
            "system_prompt": system_prompt,
            "content": [{"text": user_prompt}],
            "temperature": 0.1,
            "top_k": 10,
            "max_tokens": 2048  # Increased to handle larger JSON responses
        }
        
        # Add guardrail if configured
        if GUARDRAIL_ENV and ':' in GUARDRAIL_ENV:
            guardrail_id, guardrail_version = GUARDRAIL_ENV.split(":", 1)
            params["guardrail_config"] = {
                "guardrailId": guardrail_id,
                "guardrailVersion": guardrail_version
            }

        response = bedrock_client.invoke_model(**params) 
        return _parse_bedrock_response(response)
        
    except Exception:
        if model_id in [os.environ["LIGHTWEIGHT_LLM_MODEL_ID"], os.environ["LLM_MODEL_ID"]]:
            try:
                alternative_model = os.environ["ALTERNATIVE_LIGHTWEIGHT_LLM_MODEL_ID"] if model_id == os.environ["LIGHTWEIGHT_LLM_MODEL_ID"] else os.environ["ALTERNATIVE_LLM_MODEL_ID"]
                params["model_id"] = alternative_model
                response = bedrock_client.invoke_model(**params)
                return _parse_bedrock_response(response)
            except Exception:
                logger.error(f"Error invoking Bedrock model {model_id}", exc_info=True)
                raise

def _parse_bedrock_response(response) -> str:
    """Parse text from Bedrock API response structure."""
    if "response" in response and "output" in response["response"]:
        message = response["response"]["output"].get("message", {})
        content = message.get("content", [])
        if content and isinstance(content, list) and len(content) > 0:
            return content[0].get("text", "").strip()
    return ""

# --- Session Management ---
def get_session_context(session_id: str) -> Optional[Dict[str, Any]]:
    """Retrieves session history from DynamoDB."""
    try:
        response = catalog_table.query(
            KeyConditionExpression=Key('PK').eq(f'SESSION#{session_id}'),
            ScanIndexForward=False,
            Limit=10
        )
        items = response.get('Items', [])
        
        # Find session header
        session_header = None
        conversation_items = []
        
        for item in items:
            if item['SK'] == 'SESSION':
                session_header = item
            elif item['SK'].startswith('Q#'):
                conversation_items.append({
                    'query': item.get('query', ''),
                    'response': item.get('response', '')
                })
        
        if not session_header:
            return None
        
        # Add conversation history (reverse to get chronological order)
        session_header['conversation_history'] = list(reversed(conversation_items))
        return session_header
        
    except Exception:
        logger.warning(f"Could not retrieve session context for {session_id}", exc_info=True)
        return None

def update_session_context(session_id: str, query: str, response: Dict):
    """Updates session history in DynamoDB with a TTL."""
    try:
        ttl = int(datetime.now(timezone.utc).timestamp()) + (24 * 3600)  # 24 hour TTL
        now_iso = datetime.now(timezone.utc).isoformat()
        
        # Update session header
        catalog_table.put_item(Item={
            'PK': f'SESSION#{session_id}',
            'SK': 'SESSION',
            'last_updated': now_iso,
            'ttl': ttl
        })
        
        # Add conversation entry
        catalog_table.put_item(Item={
            'PK': f'SESSION#{session_id}',
            'SK': f"Q#{now_iso}#{uuid.uuid4()}",
            'query': query,
            'response': response.get('output', {}).get('text', ''),
            'ttl': ttl
        })
    except Exception:
        logger.warning(f"Could not update session context for {session_id}", exc_info=True)

# --- Utilities ---
def compose_filter(base: Optional[Dict], extra: Dict) -> Optional[Dict]:
    """Safely combines two filter dictionaries."""
    if not base:
        return extra
    if not extra:
        return base
    return {'$and': [base, extra]}

def enhance_query_with_filter(query: str) -> Optional[Dict]:
    """Uses a lightweight LLM to convert a natural language query to a metadata filter."""
    if not ACTIVE_FILTERABLE_KEYS:
        return None
    
    system_prompt = "Return ONLY a valid JSON object for an S3 Vectors metadata filter. No explanations or prose."
    user_prompt = f"Convert this query to a metadata filter. Here are some Examples:{os.environ.get("FILTER_EXAMPLES")} Available fields: {', '.join(ACTIVE_FILTERABLE_KEYS)}\nQuery: {query}\nJSON filter:"

    try:
        response_text = invoke_bedrock_llm(system_prompt, user_prompt, os.environ["LIGHTWEIGHT_LLM_MODEL_ID"])
        if response_text:
            filter_obj = json.loads(response_text)
            # Basic validation - ensure it's a dict
            if isinstance(filter_obj, dict):
                return filter_obj
        return None
    except (json.JSONDecodeError, TypeError):
        logger.warning("Failed to parse LLM-generated filter", exc_info=True)
        return None
    except Exception:
        logger.warning("Failed to enhance query with filter", exc_info=True)
        return None

def emit_query_metrics(latency_ms: float, success: bool, count: int):
    """Not implemented Yet
    Emits CloudWatch metrics for query performance.
    try:
        cloudwatch_client.put_metric_data(
            Namespace=os.environ['STACK_NAME'],
            MetricData=[
                {'MetricName': 'QueryLatency', 'Value': latency_ms, 'Unit': 'Milliseconds'},
                {'MetricName': 'QuerySuccessRate', 'Value': 100.0 if success else 0.0, 'Unit': 'Percent'},
                {'MetricName': 'QueryResultCount', 'Value': count, 'Unit': 'Count'}
            ]
        )
    except Exception:
        logger.warning("Failed to emit CloudWatch metrics.", exc_info=True)
    """
    pass