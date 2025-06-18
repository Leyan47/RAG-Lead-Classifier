# retriever.py
from text_utils import advanced_tokenize

def hybrid_retrieve_and_rerank(
    query: str,
    components: dict,  # 接收包含所有組件的字典
    rerank_score_threshold: float,  # 接受閾值參數
    vector_k: int, 
    bm25_k: int, 
    final_k: int,
    verbose: bool
):
    """執行混合搜索、精排，並返回排序後的文檔列表。"""
    if verbose:
        print("\n" + "="*20 + " Hybrid Search & Rerank Details " + "="*20)
        print(f"原始查詢: {query[:100]}...")

    # 從 components 字典中解包所需的組件
    vector_db = components['vector_db']
    bm25 = components['bm25']
    reranker_model = components['reranker']
    retrieval_data = components['retrieval_data']
    doc_content_to_item_map = components['doc_map']

    # 1. 關鍵詞搜索 (BM25)
    tokenized_query = advanced_tokenize(query)
    bm25_scores = bm25.get_scores(tokenized_query)
    top_bm25_indices = sorted(range(len(bm25_scores)), key=lambda i: bm25_scores[i], reverse=True)[:bm25_k]
    bm25_results = [retrieval_data[i] for i in top_bm25_indices]
    
    if verbose:
        print(f"\n--- Step 1: BM25 召回 {len(bm25_results)} 筆 (Top 5 預覽) ---")
        for i in top_bm25_indices[:5]:
            print(f"  - Score: {bm25_scores[i]:.4f} | Label: {retrieval_data[i]['label']}")

    # 2. 向量搜索 (FAISS)
    vector_docs_and_scores = vector_db.similarity_search_with_score(query, k=vector_k)
    vector_results = [doc_content_to_item_map[doc.page_content] for doc, score in vector_docs_and_scores if doc.page_content in doc_content_to_item_map]
    
    if verbose:
        print(f"\n--- Step 2: FAISS 向量搜索召回 {len(vector_results)} 筆 (Top 5 預覽) ---")
        for doc, score in vector_docs_and_scores[:5]:
            print(f"  - Score (L2): {score:.4f} | Label: {doc.metadata.get('label', 'N/A')}")

    # 3. 合併與去重
    combined_results = {item['definition']: item for item in bm25_results + vector_results}
    final_candidates = list(combined_results.values())
    
    if verbose: print(f"\n--- Step 3: 合併去重後，共 {len(final_candidates)} 筆候選進入精排 ---")

    if not final_candidates: return []

    # 4. 精排 (Re-rank)
    pairs_for_rerank = [(query, item['definition']) for item in final_candidates]
    rerank_scores = reranker_model.predict(pairs_for_rerank)
    for i in range(len(final_candidates)):
        final_candidates[i]['rerank_score'] = rerank_scores[i]

    reranked_results = sorted(final_candidates, key=lambda x: x.get('rerank_score', -1), reverse=True)

    if verbose:
        print(f"\n--- Step 4: Re-ranker 精排結果 (Top {final_k}) ---")
        for item in reranked_results[:final_k]:
            print(f"  - Re-rank Score: {item['rerank_score']:.4f} | Label: {item['label']}")

    # 5: 應用相關性閾值 
    # 過濾掉所有分數低於閾值的結果

    # if verbose:
    # # 【調試日誌-接受閾值】
    #     print(f"\n[DEBUG] 準備應用閾值。Threshold value = {rerank_score_threshold}, Type = {type(rerank_score_threshold)}")
    #     print(f"[DEBUG] 檢查分數最低的候選者: Score = {reranked_results[-1]['rerank_score']}, Label = {reranked_results[-1]['label']}")
    #     is_greater = reranked_results[-1]['rerank_score'] >= rerank_score_threshold
    #     print(f"[DEBUG] 比較結果: {reranked_results[-1]['rerank_score']} >= {rerank_score_threshold} is {is_greater}")
    
    thresholded_results = [
        doc for doc in reranked_results if doc['rerank_score'] >= rerank_score_threshold
    ]

    if verbose:
        print(f"\n--- Step 5: 應用閾值 (Threshold = {rerank_score_threshold}) 後 ---")
        print(f"  保留了 {len(thresholded_results)} / {len(reranked_results)} 筆結果。")
        if not thresholded_results and reranked_results:
            print(f"  所有結果的分數都低於閾值。最高分: {reranked_results[0]['rerank_score']:.4f}")

    return thresholded_results[:final_k]
