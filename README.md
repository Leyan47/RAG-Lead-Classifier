## RAG-RealEstateNeedClassifierr (顧客文本-需求萃取 by RAG LLM)

An intelligent multi-label classifier powered by Retrieval-Augmented Generation (RAG) to automatically analyze customer feedback from property viewings and tag their specific needs and preferences.

一個由檢索增強生成（RAG）驅動的智能多標籤分類器，用於自動分析客戶看房後的回饋，並標記其具體需求與偏好。

## 專案目標 (Project Goal)

在房地產業務中，業務人員需要花費大量時間整理客戶看房後的回饋，從中提煉出客戶的真實需求（如偏好邊間、排除機械車位等）。本專案旨在自動化此過程，通過分析非結構化的對話文本，為每位客戶生成一組精準的需求標籤，從而提高客戶輪廓分析的效率與準確性。

## 技術架構 (Architecture)

本專案採用 **檢索增強生成 (Retrieval-Augmented Generation, RAG)** 架構，流程如下：

1.  **混合檢索 (Hybrid Retrieval)**：
    *   **語義檢索 (Semantic Search)**：使用 `FAISS` 向量資料庫和 `bge-large-zh-v1.5` Embedding 模型，理解用戶輸入的深層語義。
    *   **關鍵詞檢索 (Keyword Search)**：使用 `BM25` 演算法，精準匹配房地產領域的專業術語（如「邊間」、「機械車位」）。
    *   此階段的目標是從包含數百個真實歷史樣本的語料庫中，高效召回一個高相關性的候選標籤列表。

2.  **精準排序 (Re-ranking)**：
    *   使用強大的交叉編碼器模型 `bge-reranker-large` 對混合檢索的結果進行二次排序。
    *   此階段的目標是精準判斷查詢與每個候選樣本之間的真實相關性，並過濾掉分數低於預設閾值的無關選項。

3.  **邏輯裁決 (LLM Adjudication)**：
    *   將經過排序和過濾的候選標籤及其樣本，連同用戶的原始輸入，一同提交給大型語言模型（LLM，如 Gemini, Llama 3 等）。
    *   通過精心設計的 Prompt，引導 LLM 進行最終的邏輯判斷（如處理否定語義、過濾矛盾選項），生成最終的標籤組合。

## 專案結構 (Project Structure)
```
RAG-Lead-Classifier/
│
├── .env # 存放 API 金鑰 (需親自創建)
├── config.ini # 專案配置 (模型、路徑、參數等)
├── retrieval_corpus.json # 檢索語料庫 (基於歷史樣本)
├── evaluation_set.json # 黃金評估測試集
├── embedding_index/ # FAISS 向量索引
│
├── config.py # 讀取配置模組
├── text_utils.py # 文本處理工具 (Jieba 分詞等)
├── components.py # 模型與索引初始化
├── retriever.py # 核心檢索邏輯 (混合搜索 + Re-ranker)
├── main.py # 主程式入口，整合 RAG 流程
└── evaluate.py # 性能評估腳本
```


## 調整配置
所有關鍵參數，如使用的模型、檢索的 K 值、Re-ranker 閾值等，都可以在 config.ini 文件中進行修改，無需變動程式碼。

## 未來方向 (Future Work)
- 擴充語料庫：基於評估結果，針對性地為表現不佳的標籤補充更多、更多樣化的樣本。
- 優化 Prompt Engineering：持續迭代最終裁決階段的 Prompt，提升 LLM 在處理複雜、模糊語義時的表現。
- 自動化模型再訓練/評估流程：建立一個 CI/CD 流程，在語料庫更新後自動重新評估模型性能。
