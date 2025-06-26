## RAG-RealEstateNeedClassifierr (顧客文本-需求萃取 by RAG LLM)

An intelligent multi-label classifier powered by Retrieval-Augmented Generation (RAG) to automatically analyze customer feedback from property viewings and tag their specific needs and preferences.

一個由檢索增強生成（RAG）驅動的智能多標籤分類器，用於自動分析客戶看房後的回饋，並標記其具體需求與偏好。

## 專案目標 (Project Goal)

在房地產業務中，業務人員需要花費大量時間整理客戶看房後的回饋、電話接洽文字、通訊軟體群組歷史對話，從中提煉出客戶的真實需求（如偏好邊間、排除機械車位等）。本專案旨在自動化此過程，通過分析非結構化的對話文本，為每位客戶生成一組精準的需求標籤，從而提高客戶輪廓分析的效率與準確性。

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


## API 服務 (API Service)

本專案通過 **FastAPI** 提供了一個高效、穩定的 RESTful API 服務，將 RAG 分類器的核心功能封裝起來，方便與前端介面或其他後端服務進行整合。

- **高效能**：利用 FastAPI 的異步特性，提供高吞吐量的服務。
- **自動化文檔**：內建 Swagger UI 和 ReDoc，提供交互式的 API 文檔，方便調試與整合。
- **數據驗證**：使用 Pydantic 進行嚴格的請求/回應數據驗證，確保 API 的健壯性。
- **啟動時加載**：所有耗時的模型和索引都在服務啟動時預先加載，確保請求能夠得到快速響應。


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
│
├── api/ # API 相關模組
│ ├── init.py
│ ├── schemas.py # Pydantic 數據模型
│ └── endpoints.py # API 路由和端點
│
└── server.py # 【新增】FastAPI 應用啟動入口
```


## 安裝指南 (Installation)

1.  **克隆儲存庫**
    ```bash
    git clone https://github.com/your-username/RAG-Lead-Classifier.git
    cd RAG-Lead-Classifier
    ```

2.  **創建並激活 Conda 環境** (推薦)
    ```bash
    conda create --name rag-classifier python=3.10
    conda activate rag-classifier
    ```

3.  **安裝依賴套件**
    *   若有 `requirements.txt` 文件，可使用 `pip` 安裝：
        ```bash
        pip install -r requirements.txt
        ```
    *   或使用 `conda` 手動安裝核心依賴：
        ```bash
        conda install pytorch sentence-transformers faiss-cpu -c pytorch
        conda install fastapi uvicorn pydantic email-validator python-dotenv rank_bm25 scikit-learn jieba -c conda-forge
        conda install langchain langchain-community langchain-core langchain-groq langchain-google-genai langchain-openai -c conda-forge
        ```

4.  **配置 API 金鑰**
    *   創建一個名為 `.env` 的文件。
    *   在 `.env` 文件中填入你的 `GOOGLE_API_KEY` 或 `GROQ_API_KEY` 等。
        ```env
        GOOGLE_API_KEY="YOUR_GOOGLE_API_KEY"
        GROQ_API_KEY="YOUR_GROQ_API_KEY"
        ```

5.  **準備模型與索引**
    *   確保 `config.ini` 中的路徑指向正確的語料庫和索引位置。
    *   如果使用本地模型（如 Re-ranker），請確保模型已下載到 `config.ini` 中指定的路徑。
    *   如果尚未建立向量索引，需先執行相關的索引建立腳本。

## 使用方法 (Usage)

### 啟動 API 服務

在專案根目錄下，運行以下命令啟動 FastAPI 伺服器：
```bash
python server.py

伺服器將在 http://127.0.0.1:8000 上運行。所有模型和索引將在啟動時預先加載。

### 訪問 API 文檔與測試
服務啟動後，打開瀏覽器訪問 http://127.0.0.1:8000/docs。
你將看到由 Swagger UI 生成的交互式 API 文檔。你可以直接在此頁面上測試 /api/v1/classify 端點。


## 調整配置
所有關鍵參數，如使用的模型、檢索的 K 值、Re-ranker 閾值等，都可以在 config.ini 文件中進行修改，無需變動程式碼。

## 未來方向 (Future Work)
- 擴充語料庫：基於評估結果，針對性地為表現不佳的標籤補充更多、更多樣化的樣本。
- 優化 Prompt Engineering：持續迭代最終裁決階段的 Prompt，提升 LLM 在處理複雜、模糊語義時的表現。
- 自動化模型再訓練/評估流程：建立一個 CI/CD 流程，在語料庫更新後自動重新評估模型性能。
