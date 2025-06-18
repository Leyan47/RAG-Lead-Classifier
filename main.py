#%% main.py
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

# 導入其他自定義模組
from config import APP_CONFIG
from components import initialize_all
from retriever import hybrid_retrieve_and_rerank

def get_final_labels(user_input: str, config: dict, components: dict) -> str:
    """
    完整的 RAG 流程，從檢索到最終生成。
    """
    retrieval_config = config['Retrieval']
    llm = components['llm']
    
    # 1. 檢索
    retrieved_docs = hybrid_retrieve_and_rerank(
        query=user_input,
        components=components,
        rerank_score_threshold=float(retrieval_config['rerank_score_threshold']),
        vector_k=int(retrieval_config['vector_k']),
        bm25_k=int(retrieval_config['bm25_k']),
        final_k=int(retrieval_config['final_k']),
        verbose=retrieval_config.getboolean('verbose_logging')
    )
    if not retrieved_docs:
        print("\n所有檢索結果均低於相關性閾值，無需 LLM 判斷。")
        return "無"

    # 2. 檢索後處理：標籤去重
    unique_label_docs = {}
    for doc in retrieved_docs:
        label = doc['label']
        # 因為 retrieved_docs 已經是按分數降序排列的，所以我們遇到的第一個某標籤的文檔，一定是分數最高的。
        if label not in unique_label_docs:
            unique_label_docs[label] = doc
    # unique_label_docs = {doc['label']: doc for doc in reversed(retrieved_docs_with_duplicates)}
    # retrieved_docs = sorted(list(unique_label_docs.values()), key=lambda x: x['rerank_score'], reverse=True)
    
    # 現在 unique_label_docs 裡就是去重後的結果*
    final_candidates = list(unique_label_docs.values())

    if config['Retrieval'].getboolean('verbose_logging'):
        print("\n--- 送入 LLM 前的最終候選列表 (標籤去重後) ---")
        for doc in final_candidates:
            print(f"  - Re-rank Score: {doc['rerank_score']:.4f} | Label: {doc['label']}")


    # 3. 增強 (Augment) - 格式化上下文
    context = "請根據以下候選標籤和其說明，分析使用者需求：\n\n"
    for i, doc in enumerate(final_candidates):
        context += f"候選 {i+1}:\n"
        context += f"  - 標籤名稱: {doc['label']}\n"
        context += f"  - 說明/樣本: {doc['definition']}\n\n"

    # 4. 生成 (Generate)
    template = """
    你是一個頂級的房地產需求分析師。你的任務是基於「顧客帶看後反應」，從「房屋屬性標籤列表」中選出真正符合顧客需求的標籤。

    【重要規則】
    - 請仔細分析使用者需求中的正面表述、負面表述（例如 "不要"、"不需要"、"排除"）和條件語句。
    - 「候選標籤列表」可能包含與使用者需求相矛盾的選項，你必須將它們過濾掉。
    - 你的最終選擇必須嚴格基於「帶看後反應」。

    ---
    【顧客帶看反映】
    {user_input}

    ---
    【候選標籤列表與說明】
    {context}

    ---
    【你的任務】
    分析上述所有資訊，從候選標籤中選擇所有完全符合使用者需求的標籤。
    如果使用者明確表示「不需要」某樣東西，就絕對不要選擇代表「需要」的標籤。

    請只輸出最終選擇的標籤名稱，並用逗號分隔。如果沒有任何標籤完全符合，請回答「無」。

    最終標籤：
    """
    
    prompt = PromptTemplate(template=template, input_variables=["user_input", "context"])
    chain = prompt | llm | StrOutputParser()
    result = chain.invoke({"user_input": user_input, "context": context})
    
    return result.strip()

#%%
if __name__ == "__main__":
    # 初始化
    components = initialize_all()
    
    # 測試
    test_text = [
        # service_no 211100884304
        "在邊間採光、室內格局規劃都還不錯，有一件浴廁沒有對外窗但有建議可加裝6合1冷暖機 廚房做開放式擔心油煙味問題，最後有說對於機械式車位還是會有擔憂，進/出車就要花將 近5分鐘左右的時間，擔心巔峰時間上班會被耽誤",
        # B11101059683
        "採光跟門前階梯無法 覺得無障礙坡度太陡。邊間、簡單整理即可入住，會有考慮。之前有看過一次 覺得比較像辦公室格局 還有多一個地下室 覺得有點多餘。漏水壁癌都需要大整理 就不考慮",
        "借KEY  早安再興三房車位 7樓    壁癌跟需裝潢大整理  不考慮。擔心頂樓未來漏水  防護問題   沒車位不方便",
        "空間夠用，景觀還不錯，但房間都偏小，社區的機械車位也太小，不太合適。樓層較低，屋況也要在整理，社區機械車位不夠大，不合適。屋況較好，空間也夠用，但沒有管理和車位，可能有點不方便，跟太太討論，看是否要再來看"
    ]
    
    print("\n" + "="*20 + " 開始執行 RAG 流程 " + "="*20)

    for text in test_text:
  
        print(f"輸入文字: {text[:50]}...")
        final_labels = get_final_labels(text, APP_CONFIG, components)
        print(f"\n【LLM 最終判斷的標籤】: {final_labels}")
        print("="*60 + "\n")



