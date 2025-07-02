# retriever.py
from text_utils import advanced_tokenize

def hybrid_retrieve_and_rerank(
    query: str,
    components: dict,
    rerank_score_threshold: float,
    vector_k: int, 
    bm25_k: int, 
    final_k: int,
    verbose: bool
):
    if verbose:
        print("\n" + "="*20 + " Hybrid Search & Rerank Details (with RRF) " + "="*20)
        print(f"原始查詢: {query[:100]}...")

    # 從 components 字典中解包所需的組件
    vector_db = components['vector_db']
    bm25 = components['bm25']
    reranker_model = components['reranker']
    retrieval_data = components['retrieval_data']
    doc_content_to_item_map = components['doc_map']

    # --- 1. 關鍵詞搜索 (BM25) ---
    tokenized_query = advanced_tokenize(query)
    bm25_scores = bm25.get_scores(tokenized_query)
    top_bm25_indices = sorted(range(len(bm25_scores)), key=lambda i: bm25_scores[i], reverse=True)[:bm25_k]
    # bm25_results 現在是完整的 item 列表
    bm25_results = [retrieval_data[i] for i in top_bm25_indices]
    
    if verbose:
        print(f"\n--- Step 1: BM25 召回 {len(bm25_results)} 筆 (Top 10 預覽) ---")
        for i in top_bm25_indices[:10]:
            print(f"  - Score: {bm25_scores[i]:.4f} | Label: {retrieval_data[i]['label']}")

    # --- 2. 向量搜索 (FAISS) ---
    vector_docs_and_scores = vector_db.similarity_search_with_score(query, k=vector_k)
    # vector_results 現在是完整的 item 列表
    vector_results = [doc_content_to_item_map[doc.page_content] for doc, score in vector_docs_and_scores if doc.page_content in doc_content_to_item_map]
    
    if verbose:
        print(f"\n--- Step 2: FAISS 向量搜索召回 {len(vector_results)} 筆 (Top 10 預覽) ---")
        for doc, score in vector_docs_and_scores[:10]:
            print(f"  - Score (L2): {score:.4f} | Label: {doc.metadata.get('label', 'N/A')}")

    # --- 3. 使用 RRF 合併排序結果 ---
    # 將兩個檢索結果列表傳入 RRF 函式
    fused_scores = reciprocal_rank_fusion([bm25_results, vector_results], k=60)
    
    # 根據 RRF 分數對合併後的結果進行排序
    sorted_fused_definitions = sorted(fused_scores.keys(), key=lambda x: fused_scores[x], reverse=True)
    
    # 將排序後的 definition 轉換回完整的 item 對象
    # 注意：這裡的候選者數量可能多於 final_k，因為我們需要足夠的候選送入 re-ranker
    # 我們可以取一個合理的數量，例如 vector_k + bm25_k
    num_candidates_for_rerank = vector_k + bm25_k
    final_candidates = [doc_content_to_item_map[definition] for definition in sorted_fused_definitions[:num_candidates_for_rerank]]
    
    if verbose:
        print(f"\n--- Step 3: RRF 合併排序後，選取 Top {len(final_candidates)} 筆候選進入精排 ---")
        for definition in sorted_fused_definitions[:5]:
             print(f"  - RRF Score: {fused_scores[definition]:.6f} | Label: {doc_content_to_item_map[definition]['label']}")


    # --- 3.5: 選擇代表樣本並去重 ---
    # 從 RRF 排序結果中，為每個標籤選擇 RRF 分數最高的那個樣本
    # 這樣可以避免同一個標籤的多個樣本都進入精排，減少計算量並聚焦目標
    unique_label_candidates = {}
    for definition in sorted_fused_definitions:
        item = doc_content_to_item_map[definition]
        label = item['label']
        if label not in unique_label_candidates:
            # 因為 sorted_fused_definitions 已經按 RRF 分數從高到低排好序，
            # 所以我們遇到的第一個樣本就是該標籤的最佳代表。
            unique_label_candidates[label] = item
            
    final_candidates = list(unique_label_candidates.values())
    
    # 這裡取一個合理的數量送入精排，例如前 20 個去重後的標籤
    # 或者全部送入，取決於對速度和精度的權衡
    num_candidates_for_rerank = 20
    final_candidates = final_candidates[:num_candidates_for_rerank]

    if verbose:
        print(f"\n--- Step 3.5: 標籤去重後，選取 Top {len(final_candidates)} 筆候選進入精排 ---")
        for item in final_candidates[:5]:
             print(f"  - Label: {item['label']} (來自 RRF 排序)")


    if not final_candidates: return []

    # --- 4. 精排 (Re-rank) ---
    pairs_for_rerank = [(query, item['definition']) for item in final_candidates]
    rerank_scores = reranker_model.predict(pairs_for_rerank)
    for i in range(len(final_candidates)):
        final_candidates[i]['rerank_score'] = rerank_scores[i]
    
    reranked_results = sorted(final_candidates, key=lambda x: x.get('rerank_score', -1), reverse=True)
    
    if verbose:
        print(f"\n--- Step 4: Re-ranker 精排結果 (過濾前 Top {final_k}) ---")
        for item in reranked_results[:final_k]:
            print(f"  - Re-rank Score: {item['rerank_score']:.4f} | Label: {item['label']}")

    # --- 5. 應用相關性閾值篩選 ---
    thresholded_results = [doc for doc in reranked_results if doc['rerank_score'] >= rerank_score_threshold]
    
    # if verbose:
    #     print(f"\n--- Step 5: 應用閾值 (Threshold = {rerank_score_threshold}) 後 ---")
    #     print(f"  保留了 {len(thresholded_results)} / {len(reranked_results)} 筆結果。")
    #     if not thresholded_results and reranked_results:
    #         print(f"  所有結果的分數都低於閾值。最高分: {reranked_results[0]['rerank_score']:.4f}")

    return thresholded_results[:final_k]
