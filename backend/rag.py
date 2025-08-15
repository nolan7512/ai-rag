from InstructorEmbedding import INSTRUCTOR
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct, Filter, FieldCondition, MatchValue, PointIdsList
import uuid, os
import requests
from qdrant_client.http.exceptions import UnexpectedResponse
from model import judge_question_relevance
from datetime import datetime, timezone  
from docx import Document
from io import BytesIO
from fastapi import UploadFile
from typing import List, Dict, Tuple
import re
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import nltk
from collections import Counter
import unicodedata

# Download required NLTK data
try:
    nltk.download("punkt", quiet=True)
    nltk.download("punkt_tab", quiet=True)
    from nltk.tokenize import sent_tokenize
except:
    print("NLTK download failed, using basic sentence splitting")

# Load model base
model = INSTRUCTOR("hkunlp/instructor-base")
QDRANT_HOST = os.getenv("QDRANT_HOST", "localhost")
qdrant = QdrantClient(host=QDRANT_HOST, port=6333, timeout=120.0)

COLLECTION_NAME = os.getenv("COLLECTION_NAME", "kb")
EMBED_DIM = int(os.getenv("EMBED_DIM", "768"))

# Configuration flags from environment
ENABLE_MULTILINGUAL_FEATURES = os.getenv("ENABLE_MULTILINGUAL_FEATURES", "true").lower() == "true"
ENABLE_STRICT_SEMANTIC_CHECK = os.getenv("ENABLE_STRICT_SEMANTIC_CHECK", "false").lower() == "true"
BUSINESS_DOMAIN_MODE = os.getenv("BUSINESS_DOMAIN_MODE", "true").lower() == "true"  # Cho phép mixed language terms

# Multilingual configurations
LANGUAGE_CONFIGS = {
    'en': {
        'name': 'English',
        'question_prefix': "Represent the question for retrieval:",
        'document_prefix': "Represent the document for retrieval:",
        'chunk_prefix': "Represent the internal document for retrieval:",
        'tfidf_params': {
            'stop_words': 'english',
            'ngram_range': (1, 2),
            'max_features': 1000,
            'min_df': 1,
            'max_df': 0.95
        }
    },
    'vi': {
        'name': 'Vietnamese',
        'question_prefix': "Represent the Vietnamese question for retrieval:",
        'document_prefix': "Represent the Vietnamese document for retrieval:",
        'chunk_prefix': "Represent the Vietnamese internal document for retrieval:",
        'tfidf_params': {
            'stop_words': None,  # Vietnamese stop words not in sklearn
            'ngram_range': (1, 3),  # Vietnamese benefits from longer n-grams
            'max_features': 1500,
            'min_df': 1,
            'max_df': 0.9
        }
    },
    'zh': {
        'name': 'Chinese',
        'question_prefix': "Represent the Chinese question for retrieval:",
        'document_prefix': "Represent the Chinese document for retrieval:",
        'chunk_prefix': "Represent the Chinese internal document for retrieval:",
        'tfidf_params': {
            'stop_words': None,  # Chinese stop words not in sklearn
            'ngram_range': (1, 2),
            'max_features': 2000,  # Chinese needs more features due to characters
            'min_df': 1,
            'max_df': 0.85,
            'analyzer': 'char_wb',  # Character-based for Chinese
            'token_pattern': r'[\u4e00-\u9fff]+|[a-zA-Z0-9]+' # Chinese chars + alphanumeric
        }
    }
}

# Vietnamese stop words
VIETNAMESE_STOP_WORDS = {
    'và', 'của', 'có', 'được', 'một', 'này', 'đó', 'không', 'với', 'trong', 'là', 'để', 
    'những', 'các', 'cho', 'về', 'từ', 'khi', 'đã', 'sẽ', 'bị', 'sau', 'trên', 'dưới',
    'tại', 'theo', 'như', 'nhưng', 'còn', 'chỉ', 'đây', 'đó', 'nào', 'ai', 'gì', 'sao',
    'bao', 'lúc', 'lúc', 'bây', 'giờ', 'rồi', 'thì', 'nên', 'phải', 'cần', 'nó', 'họ'
}

# Business domain terms that should not be translated/processed specially
BUSINESS_ENGLISH_TERMS = {
    'contract', 'insurance', 'policy', 'premium', 'claim', 'coverage', 'deductible',
    'liability', 'benefit', 'term', 'condition', 'clause', 'agreement', 'rider',
    'underwriting', 'actuary', 'broker', 'agent', 'reinsurance', 'copayment',
    'coinsurance', 'exclusion', 'endorsement', 'annuity', 'dividend', 'surrender'
}

# Chinese stop words  
CHINESE_STOP_WORDS = {
    '的', '了', '在', '是', '我', '有', '和', '就', '不', '人', '都', '一', '一个', '上', '也', '很', 
    '到', '说', '要', '去', '你', '会', '着', '没有', '看', '好', '自己', '这', '那', '里', '就是',
    '他', '时候', '过', '出', '什么', '对', '能', '她', '所以', '这样', '但是', '因为', '这个',
    '中', '可以', '为', '从', '与', '及', '或', '等', '以', '所', '由', '其', '而', '之', '以及'
}

# Chỉ tạo collection nếu chưa tồn tại
if not qdrant.collection_exists(collection_name=COLLECTION_NAME):
    qdrant.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(size=EMBED_DIM, distance=Distance.COSINE)
    )

def detect_language(text: str) -> str:
    """
    Detect language with business domain awareness
    Mixed language content (Vietnamese + English terms) is treated as Vietnamese in business context
    """
    if not ENABLE_MULTILINGUAL_FEATURES:
        return 'en'  # Default to English if multilingual disabled
    
    text = text.strip()[:500]  # Check first 500 chars for efficiency
    
    # Count character types
    latin_chars = len(re.findall(r'[a-zA-Z]', text))
    chinese_chars = len(re.findall(r'[\u4e00-\u9fff]', text))
    vietnamese_chars = len(re.findall(r'[àáạảãâầấậẩẫăằắặẳẵèéẹẻẽêềếệểễìíịỉĩòóọỏõôồốộổỗơờớợởỡùúụủũưừứựửữỳýỵỷỹđ]', text, re.IGNORECASE))
    
    total_chars = len(re.sub(r'\s+', '', text))
    
    if total_chars == 0:
        return 'en'  # default
    
    # Calculate ratios
    chinese_ratio = chinese_chars / total_chars
    vietnamese_ratio = vietnamese_chars / total_chars
    latin_ratio = latin_chars / total_chars
    
    # Business domain mode: check for mixed Vietnamese + English business terms
    if BUSINESS_DOMAIN_MODE:
        text_lower = text.lower()
        has_business_terms = any(term in text_lower for term in BUSINESS_ENGLISH_TERMS)
        has_vietnamese_words = any(word in text_lower for word in ['không', 'được', 'những', 'các', 'cho', 'muốn', 'tôi'])
        
        if has_vietnamese_words and has_business_terms:
            return 'vi'  # Treat as Vietnamese with English business terms
    
    # Decision logic
    if chinese_ratio > 0.3:
        return 'zh'
    elif vietnamese_ratio > 0.05 or any(word in text.lower() for word in ['không', 'được', 'những', 'các', 'cho']):
        return 'vi'
    else:
        return 'en'

def preprocess_text_multilingual(text: str, language: str) -> str:
    """
    Preprocess text based on language
    """
    # Basic normalization
    text = unicodedata.normalize('NFKC', text)
    
    if language == 'zh':
        # For Chinese: keep only Chinese characters, numbers, and basic punctuation
        text = re.sub(r'[^\u4e00-\u9fff0-9a-zA-Z\s.,!?;:()""''—]', ' ', text)
    elif language == 'vi':
        # For Vietnamese: preserve Vietnamese diacritics
        text = re.sub(r'[^\w\s.,!?;:()""''—àáạảãâầấậẩẫăằắặẳẵèéẹẻẽêềếệểễìíịỉĩòóọỏõôồốộổỗơờớợởỡùúụủũưừứựửữỳýỵỷỹđÀÁẠẢÃÂẦẤẬẨẪĂẰẮẶẲẴÈÉẸẺẼÊỀẾỆỂỄÌÍỊỈĨÒÓỌỎÕÔỒỐỘỔỖƠỜỚỢỞỠÙÚỤỦŨƯỪỨỰỬỮỲÝỴỶỸĐ]', ' ', text)
    else:  # English
        text = re.sub(r'[^\w\s.,!?;:()""'']', ' ', text)
    
    # Clean up whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def create_multilingual_tfidf_vectorizer(language: str):
    """
    Create TF-IDF vectorizer optimized for specific language
    """
    config = LANGUAGE_CONFIGS.get(language, LANGUAGE_CONFIGS['en'])
    tfidf_params = config['tfidf_params'].copy()
    
    # Add language-specific stop words
    if language == 'vi':
        tfidf_params['stop_words'] = list(VIETNAMESE_STOP_WORDS)
    elif language == 'zh':
        tfidf_params['stop_words'] = list(CHINESE_STOP_WORDS)
    
    return TfidfVectorizer(**tfidf_params)

def calculate_semantic_relevance_multilingual(query: str, context: str) -> float:
    """
    Multi-language semantic relevance calculation with configurable strictness
    """
    if not ENABLE_MULTILINGUAL_FEATURES:
        # Simple fallback for single language mode
        return calculate_simple_semantic_relevance(query, context)
    
    try:
        # Detect languages
        query_lang = detect_language(query)
        context_lang = detect_language(context)
        
        # Use the query language as primary
        primary_lang = query_lang
        config = LANGUAGE_CONFIGS.get(primary_lang, LANGUAGE_CONFIGS['en'])
        
        # In business domain mode, treat mixed language more leniently
        if BUSINESS_DOMAIN_MODE and primary_lang == 'vi':
            # Don't heavily preprocess mixed Vietnamese + English content
            query_clean = preprocess_text_business_mode(query, primary_lang)
            context_clean = preprocess_text_business_mode(context, primary_lang)
        else:
            # Standard preprocessing
            query_clean = preprocess_text_multilingual(query, primary_lang)
            context_clean = preprocess_text_multilingual(context, primary_lang)
        
        # 1. TF-IDF similarity with language-specific settings
        vectorizer = create_multilingual_tfidf_vectorizer(primary_lang)
        
        try:
            tfidf_matrix = vectorizer.fit_transform([query_clean, context_clean])
            if tfidf_matrix.shape[0] >= 2:
                tfidf_sim = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
            else:
                tfidf_sim = 0.0
        except:
            tfidf_sim = 0.0
        
        # 2. Semantic embedding similarity with language-aware prompts
        query_embed = model.encode([[config['question_prefix'], query]])[0]
        context_embed = model.encode([[config['document_prefix'], context]])[0]
        
        # Cosine similarity for embeddings
        embed_sim = np.dot(query_embed, context_embed) / (
            np.linalg.norm(query_embed) * np.linalg.norm(context_embed)
        )
        
        # 3. Language consistency bonus (relaxed for business domain)
        if BUSINESS_DOMAIN_MODE:
            lang_bonus = 1.0  # Don't penalize mixed language in business context
        else:
            lang_bonus = 1.0 if query_lang == context_lang else 0.9
        
        # 4. Combined score with configurable strictness
        if not ENABLE_STRICT_SEMANTIC_CHECK:
            # Relaxed mode: favor recall over precision
            if primary_lang == 'zh':
                combined_score = (0.15 * tfidf_sim + 0.85 * embed_sim) * lang_bonus
            elif primary_lang == 'vi':
                combined_score = (0.25 * tfidf_sim + 0.75 * embed_sim) * lang_bonus
            else:  # English
                combined_score = (0.3 * tfidf_sim + 0.7 * embed_sim) * lang_bonus
        else:
            # Strict mode: original weights
            if primary_lang == 'zh':
                combined_score = (0.25 * tfidf_sim + 0.75 * embed_sim) * lang_bonus
            elif primary_lang == 'vi':
                combined_score = (0.35 * tfidf_sim + 0.65 * embed_sim) * lang_bonus
            else:  # English
                combined_score = (0.4 * tfidf_sim + 0.6 * embed_sim) * lang_bonus
        
        print(f"Lang: {primary_lang}, TF-IDF: {tfidf_sim:.3f}, Embedding: {embed_sim:.3f}, "
              f"Combined: {combined_score:.3f}, Strict: {ENABLE_STRICT_SEMANTIC_CHECK}")
        
        return combined_score
        
    except Exception as e:
        print(f"Error in multilingual semantic relevance: {e}")
        return 0.0

def calculate_simple_semantic_relevance(query: str, context: str) -> float:
    """
    Simple semantic relevance for single language mode
    """
    try:
        # Basic TF-IDF similarity
        vectorizer = TfidfVectorizer(
            stop_words='english',
            ngram_range=(1, 2),
            max_features=1000,
            lowercase=True
        )
        
        tfidf_matrix = vectorizer.fit_transform([query, context])
        tfidf_sim = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
        
        # Basic embedding similarity
        query_embed = model.encode([["Represent the question for retrieval:", query]])[0]
        context_embed = model.encode([["Represent the document for retrieval:", context]])[0]
        
        embed_sim = np.dot(query_embed, context_embed) / (
            np.linalg.norm(query_embed) * np.linalg.norm(context_embed)
        )
        
        # Simple combination
        combined_score = 0.3 * tfidf_sim + 0.7 * embed_sim
        
        print(f"Simple mode - TF-IDF: {tfidf_sim:.3f}, Embedding: {embed_sim:.3f}, Combined: {combined_score:.3f}")
        return combined_score
        
    except Exception as e:
        print(f"Error in simple semantic relevance: {e}")
        return 0.0

def preprocess_text_business_mode(text: str, language: str) -> str:
    """
    Lighter preprocessing for business domain mixed language content
    """
    # Keep business terms intact, minimal processing
    text = unicodedata.normalize('NFKC', text)
    
    if language == 'vi':
        # For Vietnamese business content: preserve English business terms
        # Only clean up excessive punctuation and whitespace
        text = re.sub(r'[^\w\s.,!?;:()""''—àáạảãâầấậẩẫăằắặẳẵèéẹẻẽêềếệểễìíịỉĩòóọỏõôồốộổỗơờớợởỡùúụủũưừứựửữỳýỵỷỹđÀÁẠẢÃÂẦẤẬẨẪĂẰẮẶẲẴÈÉẸẺẼÊỀẾỆỂỄÌÍỊỈĨÒÓỌỎÕÔỒỐỘỔỖƠỜỚỢỞỠÙÚỤỦŨƯỪỨỰỬỮỲÝỴỶỸĐ]', ' ', text)
    else:
        # Standard processing for other languages
        text = preprocess_text_multilingual(text, language)
    
    # Clean up whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def multilingual_sentence_split(text: str, language: str) -> List[str]:
    """
    Language-aware sentence splitting
    """
    if language == 'zh':
        # Chinese sentence endings
        sentences = re.split(r'[。！？；]', text)
        sentences = [s.strip() for s in sentences if s.strip()]
    elif language == 'vi':
        # Vietnamese sentence splitting
        try:
            sentences = sent_tokenize(text)
        except:
            sentences = re.split(r'[.!?;]', text)
            sentences = [s.strip() for s in sentences if s.strip()]
    else:  # English
        try:
            sentences = sent_tokenize(text)
        except:
            sentences = re.split(r'[.!?;]', text)
            sentences = [s.strip() for s in sentences if s.strip()]
    
    return [s for s in sentences if len(s.strip()) > 10]  # Filter very short sentences

def chunk_by_sentences_multilingual(text: str, max_words=200) -> List[str]:
    """
    Multilingual sentence-based chunking
    """
    language = detect_language(text)
    sentences = multilingual_sentence_split(text, language)
    
    chunks = []
    current_chunk = []
    
    # Adjust word counting based on language
    if language == 'zh':
        # For Chinese, count characters instead of words
        max_chars = max_words * 2  # Rough approximation
        for sentence in sentences:
            current_chunk.append(sentence)
            current_text = ''.join(current_chunk)
            if len(current_text) >= max_chars:
                chunks.append('。'.join(current_chunk) + '。')
                current_chunk = []
    else:
        # For English/Vietnamese, count words
        for sentence in sentences:
            current_chunk.append(sentence)
            current_text = ' '.join(current_chunk)
            if len(current_text.split()) >= max_words:
                chunks.append(current_text)
                current_chunk = []
    
    # Add remaining chunk
    if current_chunk:
        if language == 'zh':
            chunks.append('。'.join(current_chunk) + '。')
        else:
            chunks.append(' '.join(current_chunk))
    
    return [chunk for chunk in chunks if chunk.strip()]

def embed_chunks_multilingual(chunks: List[str]) -> List[np.ndarray]:
    """
    Create embeddings with language detection for each chunk
    """
    embeddings = []
    for chunk in chunks:
        language = detect_language(chunk)
        config = LANGUAGE_CONFIGS.get(language, LANGUAGE_CONFIGS['en'])
        
        # Use language-specific instruction
        instruction = config['chunk_prefix']
        embedding = model.encode([[instruction, chunk]])[0]
        embeddings.append(embedding)
    
    return embeddings

def enhanced_should_use_rag_multilingual(
    question: str, 
    semantic_threshold=None,
    min_embedding_score=None,
    fallback_threshold=None
) -> bool:
    """
    Multilingual enhanced RAG decision with environment-based configuration
    """
    # Get thresholds from environment or use defaults
    if semantic_threshold is None:
        semantic_threshold = float(os.getenv("SEMANTIC_THRESHOLD", "0.35" if not ENABLE_STRICT_SEMANTIC_CHECK else "0.45"))
    if min_embedding_score is None:
        min_embedding_score = float(os.getenv("MIN_EMBEDDING_SCORE", "0.25" if not ENABLE_STRICT_SEMANTIC_CHECK else "0.35"))
    if fallback_threshold is None:
        fallback_threshold = float(os.getenv("FALLBACK_THRESHOLD", "0.55" if not ENABLE_STRICT_SEMANTIC_CHECK else "0.60"))
    min_fallback_score = float(os.getenv("MIN_FALLBACK_SCORE", "0.4"))
    max_fallback_score = float(os.getenv("MAX_FALLBACK_SCORE", "0.7"))
    # Detect question language
    question_lang = detect_language(question) if ENABLE_MULTILINGUAL_FEATURES else 'en'
    config = LANGUAGE_CONFIGS.get(question_lang, LANGUAGE_CONFIGS['en'])
    
    print(f"🌍 Language: {config['name']} ({question_lang}), Business Mode: {BUSINESS_DOMAIN_MODE}, Strict: {ENABLE_STRICT_SEMANTIC_CHECK}")
    
    # Adjust thresholds based on language and business mode
    if ENABLE_MULTILINGUAL_FEATURES and not ENABLE_STRICT_SEMANTIC_CHECK:
        if question_lang == 'zh':
            semantic_threshold *= 0.85  # More lenient for Chinese
        elif question_lang == 'vi':
            semantic_threshold *= 0.90  # Slightly more lenient for Vietnamese
    
    # Stage 1: Initial embedding search
    if ENABLE_MULTILINGUAL_FEATURES:
        question_vec = model.encode([[config['question_prefix'], question]])[0]
    else:
        question_vec = model.encode([["Represent the question for retrieval:", question]])[0]
    
    hits = qdrant.search(
        collection_name=COLLECTION_NAME,
        query_vector=question_vec,
        limit=10
    )
    
    if not hits:
        print("❌ No results from vector search")
        return False
    
    top_score = hits[0].score
    print(f"🔍 Top embedding score: {top_score:.3f} (threshold: {min_embedding_score:.3f})")
    
    # Stage 2: Minimum threshold check
    if top_score < min_embedding_score:
        print("❌ Embedding score too low")
        return False
    
    # Stage 3: Semantic relevance check (if enabled)
    if ENABLE_STRICT_SEMANTIC_CHECK or not BUSINESS_DOMAIN_MODE:
        top_contexts = [hit.payload["text"] for hit in hits[:5]]
        combined_context = "\n".join(top_contexts)
        
        if ENABLE_MULTILINGUAL_FEATURES:
            semantic_score = calculate_semantic_relevance_multilingual(question, combined_context)
        else:
            semantic_score = calculate_simple_semantic_relevance(question, combined_context)
        
        print(f"🎯 Semantic score: {semantic_score:.3f} (threshold: {semantic_threshold:.3f})")
        
        # Stage 4: Decision logic
        if semantic_score >= semantic_threshold:
            print("✅ Semantic score meets threshold, using RAG")
            return True
        elif top_score >= fallback_threshold:
            print("🤔 Gray zone, using LLM judge...")
            return judge_question_relevance(question, combined_context)
        else:
            print("❌ Both embedding and semantic scores too low")
            return False
    else:
        # Business domain mode: more permissive, rely mainly on embedding score
        print(f"💼 Business mode: relying on embedding score ({top_score:.3f})")
        if top_score >= min_embedding_score:  # Slightly higher threshold when skipping semantic check
            print("✅ Business mode: using RAG based on embedding score")
            return True
        elif top_score >= min_fallback_score and top_score < max_fallback_score:
            print("🤔 Gray zone, using LLM judge...")
            return judge_question_relevance(question, combined_context)
        else:
            print("❌ Business mode: embedding score insufficient")
            return False


def improved_search_context_multilingual(
    query: str, 
    filter_tag: str = None, 
    top_k: int = 15,
    rerank_top_k: int = 8,
    diversity_penalty: float = 0.1
) -> str:
    """
    Multilingual improved search context
    """
    # Detect query language and get config
    query_lang = detect_language(query)
    config = LANGUAGE_CONFIGS.get(query_lang, LANGUAGE_CONFIGS['en'])
    
    print(f"🔍 Searching in {config['name']} ({query_lang})")
    
    # Stage 1: Initial vector search with language-aware instruction
    query_vec = model.encode([[config['question_prefix'], query]])[0]
    
    filter_query = None
    if filter_tag:
        filter_query = Filter(
            must=[FieldCondition(key="tags", match=MatchValue(value=filter_tag))]
        )
    
    hits = qdrant.search(
        collection_name=COLLECTION_NAME,
        query_vector=query_vec,
        limit=top_k * 2,
        query_filter=filter_query
    )
    
    if not hits:
        return ""
    
    # Stage 2: Re-ranking with multilingual semantic relevance
    reranked_results = []
    for hit in hits:
        text = hit.payload["text"]
        semantic_score = calculate_semantic_relevance_multilingual(query, text)
        
        # Language consistency bonus
        text_lang = detect_language(text)
        lang_consistency = 1.0 if query_lang == text_lang else 0.9
        
        # Combined final score
        final_score = (0.6 * hit.score + 0.4 * semantic_score) * lang_consistency
        
        reranked_results.append({
            "text": text,
            "embedding_score": hit.score,
            "semantic_score": semantic_score,
            "final_score": final_score,
            "language": text_lang,
            "metadata": hit.payload
        })
    
    # Sort by final score
    reranked_results.sort(key=lambda x: x["final_score"], reverse=True)
    
    # Stage 3: Diversity filtering with multilingual awareness
    selected_contexts = []
    selected_texts = []
    
    for result in reranked_results:
        if len(selected_contexts) >= rerank_top_k:
            break
            
        text = result["text"]
        text_lang = result["language"]
        
        # Check similarity with already selected texts
        is_diverse = True
        for existing_text in selected_texts:
            similarity = calculate_text_similarity_multilingual(text, existing_text, text_lang)
            if similarity > (1 - diversity_penalty):
                is_diverse = False
                break
        
        if is_diverse:
            selected_contexts.append(result)
            selected_texts.append(text)
            print(f"✅ Selected ({text_lang}): embed={result['embedding_score']:.3f}, "
                  f"semantic={result['semantic_score']:.3f}, final={result['final_score']:.3f}")
    
    # Stage 4: Join contexts
    final_contexts = [ctx["text"] for ctx in selected_contexts]
    return "\n\n".join(final_contexts)

def calculate_text_similarity_multilingual(text1: str, text2: str, language: str = None) -> float:
    """
    Calculate text similarity with language awareness
    """
    if language is None:
        language = detect_language(text1)
    
    try:
        vectorizer = create_multilingual_tfidf_vectorizer(language)
        
        # Preprocess texts
        text1_clean = preprocess_text_multilingual(text1, language)
        text2_clean = preprocess_text_multilingual(text2, language)
        
        tfidf_matrix = vectorizer.fit_transform([text1_clean, text2_clean])
        if tfidf_matrix.shape[0] >= 2:
            similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
        else:
            similarity = 0.0
        return similarity
    except:
        return 0.0

# Wrapper functions for compatibility
def should_use_rag(question: str) -> bool:
    """Wrapper function for multilingual RAG decision"""
    return enhanced_should_use_rag_multilingual(question)

def search_context(query: str, filter_tag: str = None, top_k: int = 15, score_threshold: float = None) -> str:
    """Wrapper function for multilingual context search"""
    return improved_search_context_multilingual(query, filter_tag, top_k)

def chunk_by_sentences(text: str, max_words: int = 200) -> List[str]:
    """Wrapper function for multilingual chunking"""
    return chunk_by_sentences_multilingual(text, max_words)

def embed_chunks(chunks: List[str]) -> List[np.ndarray]:
    """Wrapper function for multilingual embedding"""
    return embed_chunks_multilingual(chunks)

# Debug and testing functions
def debug_multilingual_search(question: str, limit: int = 5):
    """
    Debug function for multilingual search testing with current configuration
    """
    print(f"\n🔍 MULTILINGUAL DEBUGGING FOR: '{question}'")
    print("=" * 80)
    
    # Show current configuration
    print(f"⚙️  Configuration:")
    print(f"   - Multilingual Features: {ENABLE_MULTILINGUAL_FEATURES}")
    print(f"   - Strict Semantic Check: {ENABLE_STRICT_SEMANTIC_CHECK}")
    print(f"   - Business Domain Mode: {BUSINESS_DOMAIN_MODE}")
    print(f"   - Model: {os.getenv('LLM_MODEL_NAME', 'Not set')}")
    print()
    
    # Language detection
    if ENABLE_MULTILINGUAL_FEATURES:
        lang = detect_language(question)
        config = LANGUAGE_CONFIGS[lang]
        print(f"🌍 Detected Language: {config['name']} ({lang})")
        
        # Check for business terms if Vietnamese
        if lang == 'vi' and BUSINESS_DOMAIN_MODE:
            business_terms_found = [term for term in BUSINESS_ENGLISH_TERMS if term in question.lower()]
            if business_terms_found:
                print(f"💼 Business terms found: {business_terms_found}")
    else:
        print(f"🌍 Language Detection: Disabled (using English)")
    
    print()
    
    # Test RAG decision
    should_use = enhanced_should_use_rag_multilingual(question)
    print(f"\n📊 Final Decision: {'✅ USE RAG' if should_use else '❌ NO RAG'}")
    
    if should_use:
        context = improved_search_context_multilingual(question, top_k=limit) if ENABLE_MULTILINGUAL_FEATURES else improved_search_context_simple(question, top_k=limit)
        print(f"\n📄 RETRIEVED CONTEXT ({len(context)} chars):")
        print("-" * 60)
        
        for i, chunk in enumerate(context.split('\n\n')[:3], 1):
            chunk_lang = detect_language(chunk) if ENABLE_MULTILINGUAL_FEATURES else 'en'
            print(f"Chunk {i} ({chunk_lang}): {chunk[:200]}...")
            print()
    
    return should_use

def improved_search_context_simple(query: str, filter_tag: str = None, top_k: int = 15, rerank_top_k: int = 8) -> str:
    """
    Simple search context for when multilingual features are disabled
    """
    query_vec = model.encode([["Represent the question for retrieval:", query]])[0]
    
    filter_query = None
    if filter_tag:
        filter_query = Filter(
            must=[FieldCondition(key="tags", match=MatchValue(value=filter_tag))]
        )
    
    hits = qdrant.search(
        collection_name=COLLECTION_NAME,
        query_vector=query_vec,
        limit=top_k,
        query_filter=filter_query
    )
    
    if not hits:
        return ""
    
    # Simple selection of top results
    final_contexts = [hit.payload["text"] for hit in hits[:rerank_top_k]]
    return "\n\n".join(final_contexts)

def test_business_scenarios():
    """
    Test function for common business scenarios
    """
    test_cases = [
        # Vietnamese + English business terms (should use RAG in business mode)
        "Tôi muốn biết về contract insurance",
        "Điều khoản policy là gì?",
        "Premium phải trả bao nhiều?",
        "Claim như thế nào?",
        
        # Pure Vietnamese (should use RAG)
        "Điều khoản hợp đồng bảo hiểm là gì?",
        "Tôi cần thông tin về phí bảo hiểm",
        
        # Irrelevant questions (should not use RAG)
        "Hôm nay thời tiết thế nao?",
        "Tôi đói bụng quá",
        "What's the weather today?",
        
        # Edge cases (tricky ones)
        "Insurance contract weather conditions",  # Mixed irrelevant
        "Tôi muốn contract với công ty khác",     # Mixed ambiguous
    ]
    
    print("\n🧪 TESTING BUSINESS SCENARIOS")
    print("=" * 80)
    
    for i, question in enumerate(test_cases, 1):
        print(f"\n{i:2d}. Testing: '{question}'")
        print("-" * 50)
        
        should_use = enhanced_should_use_rag_multilingual(question)
        
        # Quick analysis
        lang = detect_language(question) if ENABLE_MULTILINGUAL_FEATURES else 'en'
        has_business_terms = any(term in question.lower() for term in BUSINESS_ENGLISH_TERMS)
        has_vietnamese = any(word in question.lower() for word in ['tôi', 'là', 'gì', 'như', 'thế', 'nào', 'về'])
        
        print(f"    Language: {lang}, Business terms: {has_business_terms}, Vietnamese: {has_vietnamese}")
        print(f"    Result: {'✅ USE RAG' if should_use else '❌ NO RAG'}")

def get_current_config():
    """
    Return current configuration as dictionary for API endpoint
    """
    return {
        "multilingual_features": ENABLE_MULTILINGUAL_FEATURES,
        "strict_semantic_check": ENABLE_STRICT_SEMANTIC_CHECK, 
        "business_domain_mode": BUSINESS_DOMAIN_MODE,
        "model_name": os.getenv('LLM_MODEL_NAME', 'Not set'),
        "qdrant_host": QDRANT_HOST,
        "collection_name": COLLECTION_NAME,
        "embed_dim": EMBED_DIM,
        "thresholds": {
            "semantic": float(os.getenv("SEMANTIC_THRESHOLD", "0.35" if not ENABLE_STRICT_SEMANTIC_CHECK else "0.45")),
            "min_embedding": float(os.getenv("MIN_EMBEDDING_SCORE", "0.25" if not ENABLE_STRICT_SEMANTIC_CHECK else "0.35")),
            "fallback": float(os.getenv("FALLBACK_THRESHOLD", "0.55" if not ENABLE_STRICT_SEMANTIC_CHECK else "0.60"))
        }
    }

# Store chunks with language detection
def store_chunks(chunks, metadata=None, original_text: str = None):
    """
    Store chunks with language metadata
    """
    embeddings = embed_chunks_multilingual(chunks)
    file_id = metadata.get("file_id", uuid.uuid4().hex)
    file_name = metadata.get("filename", "unknown.docx")
    tags = metadata.get("tags", [])
    uploaded_at = datetime.now(timezone.utc).isoformat()
    source = metadata.get("source", "file")

    points = []

    # Add original text with language detection
    if original_text:
        orig_lang = detect_language(original_text)
        points.append(PointStruct(
            id=uuid.uuid4().hex,
            vector=[0.0] * EMBED_DIM,
            payload={
                "file_id": file_id,
                "filename": file_name,
                "tags": tags,
                "uploaded_at": uploaded_at,
                "source": source,
                "text": original_text,
                "is_original": True,
                "language": orig_lang,
            }
        ))

    # Add chunks with their detected languages
    for vec, chunk in zip(embeddings, chunks):
        chunk_lang = detect_language(chunk)
        points.append(PointStruct(
            id=uuid.uuid4().hex,
            vector=vec,
            payload={
                **(metadata or {}),
                "file_id": file_id,
                "filename": file_name,
                "tags": tags,
                "uploaded_at": uploaded_at,
                "source": source,
                "text": chunk,
                "language": chunk_lang,
            }
        ))

    qdrant.upload_points(collection_name=COLLECTION_NAME, points=points)

# Existing functions remain the same
def clear_collection():
    try:
        qdrant.delete_collection(collection_name=COLLECTION_NAME)
        qdrant.recreate_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(size=EMBED_DIM, distance=Distance.COSINE),
        )
        return {"message": "✅ Collection đã được reset thành công."}
    except UnexpectedResponse as e:
        return {"error": str(e)}

def export_collection():
    try:
        points, _ = qdrant.scroll(
            collection_name=COLLECTION_NAME,
            scroll_filter=None,
            with_payload=True,
            with_vectors=True,
            limit=10000
        )
        return [p.dict() for p in points]
    except Exception as e:
        return {"error": str(e)}

def import_collection_from_raw(raw):
    try:
        points = []
        for p in raw:
            point_id = str(p.get("id") or uuid.uuid4().hex)
            vector = p.get("vector")
            payload = p.get("payload", {})

            if vector is None:
                continue

            points.append(PointStruct(id=point_id, vector=vector, payload=payload))

        qdrant.upload_points(collection_name=COLLECTION_NAME, points=points)
        return {"message": f"✅ Đã import thành công {len(points)} điểm vào collection."}
    except Exception as e:
        return {"error": str(e)}

def list_uploaded_files():
    points, _ = qdrant.scroll(
        collection_name=COLLECTION_NAME,
        with_payload=True,
        with_vectors=False,
        limit=10000,
    )

    seen = {}
    for p in points:
        payload = p.payload
        file_id = payload.get("file_id")
        if file_id and file_id not in seen:
            seen[file_id] = {
                "file_id": file_id,
                "filename": payload.get("filename", ""),
                "tags": payload.get("tags", []),
                "uploaded_at": payload.get("uploaded_at", ""),
                "source": payload.get("source", "file"),
                "language": payload.get("language", "unknown"),
            }

    return list(seen.values())

def delete_file_by_id(file_id: str):
    filter = Filter(
        must=[FieldCondition(key="file_id", match=MatchValue(value=file_id))]
    )
    points, _ = qdrant.scroll(
        collection_name=COLLECTION_NAME,
        with_payload=False,
        with_vectors=False,
        limit=10000,
        scroll_filter=filter,
    )

    ids_to_delete = [pt.id for pt in points]

    if ids_to_delete:
        qdrant.delete(
            collection_name=COLLECTION_NAME,
            points_selector=PointIdsList(points=ids_to_delete)
        )

def update_file(file: UploadFile, file_id: str, tags: List[str] = []):
    try:
        delete_file_by_id(file_id)
        contents = file.file.read()
        doc = Document(BytesIO(contents))
        text = "\n".join(p.text for p in doc.paragraphs)
        chunks = chunk_by_sentences_multilingual(text)

        store_chunks(chunks, metadata={
            "file_id": file_id,
            "filename": file.filename,
            "tags": tags,
            "source": "file",
            "uploaded_at": datetime.now(timezone.utc).isoformat()
        }, original_text=text)

        return {"message": f"File '{file.filename}' updated successfully"}
    except Exception as e:
        raise RuntimeError(f"Lỗi update file: {str(e)}")

def get_text_by_file_id(file_id: str) -> str:
    filter = Filter(
        must=[FieldCondition(key="file_id", match=MatchValue(value=file_id))]
    )
    points, _ = qdrant.scroll(
        collection_name=COLLECTION_NAME,
        scroll_filter=filter,
        with_payload=True,
        with_vectors=False,
        limit=10000
    )

    # Prioritize original text if available
    for p in points:
        if p.payload.get("is_original"):
            return p.payload.get("text", "")

    # Fallback to joined chunks
    sorted_chunks = sorted(points, key=lambda p: p.id)
    return "\n".join(p.payload.get("text", "") for p in sorted_chunks)