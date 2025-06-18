# components.py
import os
import json
import httpx
from langchain_community.embeddings import OllamaEmbeddings
from langchain.vectorstores import FAISS
from sentence_transformers.cross_encoder import CrossEncoder
from rank_bm25 import BM25Okapi

# 導入其他自定義模組
from config import APP_CONFIG
from text_utils import advanced_tokenize

# 導入 LLM 提供商
from langchain_groq import ChatGroq
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.llms import Ollama

def initialize_all():
    """根據配置初始化所有模型和索引，並返回一個包含所有組件的字典。"""
    print("正在初始化所有組件...")
    
    paths_config = APP_CONFIG['Paths']
    models_config = APP_CONFIG['Models']
    # 處理
    if 'Network' in APP_CONFIG:
        network_config = APP_CONFIG['Network']
    else:
    # 如果不存在，給一個空的 SectionProxy 以便後續 .getboolean() 不會出錯
        network_config = {}

    # 載入語料庫
    with open(paths_config['retrieval_corpus_path'], 'r', encoding='utf-8') as f:
        retrieval_data = json.load(f)
    doc_map = {item['definition']: item for item in retrieval_data}

    # 初始化向量檢索 (FAISS)
    embeddings_model = OllamaEmbeddings(model=models_config['embedding_model'])
    vector_db = FAISS.load_local(
        paths_config['embedding_index_path'],
        embeddings_model,
        allow_dangerous_deserialization=True
    )

    # 初始化關鍵詞檢索 (BM25)
    corpus = [item['definition'] for item in retrieval_data]
    tokenized_corpus = [advanced_tokenize(doc) for doc in corpus]
    bm25 = BM25Okapi(tokenized_corpus)

    # 初始化 Re-ranker
    reranker = CrossEncoder(models_config['reranker_model'])

    # 根據配置選擇並初始化 LLM
    provider = models_config['llm_provider']
    temp = float(models_config['llm_temperature'])
    llm = None

    # 【新增】根據配置決定是否禁用 SSL 驗證
    disable_ssl = network_config.getboolean('disable_ssl_verification', fallback=False)
    http_client = None
    if disable_ssl:
        print("警告：已禁用 SSL 憑證驗證。這僅應在受信任的網路環境中使用。")
        http_client = httpx.Client(verify=False)

    if provider == 'Gemini':
        llm = ChatGoogleGenerativeAI(model=models_config['gemini_model'],google_api_key=os.getenv("GOOGLE_API_KEY"), temperature=temp)
        print(f"使用 Gemini 模型: {models_config['gemini_model']}")
    elif provider == 'Groq':
        llm = ChatGroq(model_name=models_config['groq_model'], groq_api_key=os.getenv("GROQ_API_KEY"), temperature=temp, http_client=http_client)
        print(f"使用 Groq 模型: {models_config['groq_model']}")
    elif provider == 'Ollama':
        llm = Ollama(model=models_config['ollama_model'], temperature=temp)
        print(f"使用 Ollama 模型: {models_config['ollama_model']}")
    else:
        raise ValueError(f"不支持的 LLM 提供商: {provider}")

    print("所有組件初始化完成！")
    
    # 將所有組件打包成一個字典返回，更具可讀性
    return {
        "retrieval_data": retrieval_data,
        "doc_map": doc_map,
        "vector_db": vector_db,
        "bm25": bm25,
        "reranker": reranker,
        "llm": llm
    }
