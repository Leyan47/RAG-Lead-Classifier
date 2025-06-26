# FastAPI/endpoints.py
from fastapi import APIRouter, Depends, HTTPException
from typing import Dict

# 導入 API 數據模型
from .schemas import ClassifyRequest, ClassifyResponse, HealthCheckResponse

# 導入 RAG 核心函式和組件
from config import APP_CONFIG
from main import get_final_labels # 我們假設 main.py 裡有這個函式

# 這個全域變數用來緩存昂貴的組件
# 避免每次請求都重新初始化
# 我們將在 server.py 中啟動時填充它
APP_COMPONENTS: Dict = {}

router = APIRouter()

# --- 依賴注入 (Dependency Injection) ---
# 這個函式確保在處理請求時，所有 RAG 組件都已準備就緒
def get_components():
    if not APP_COMPONENTS:
        raise HTTPException(
            status_code=503, 
            detail="服務尚未完全初始化，請稍後再試。"
        )
    return APP_COMPONENTS

# --- API 端點定義 ---

@router.get("/health", response_model=HealthCheckResponse, tags=["Monitoring"])
async def health_check():
    """
    執行健康檢查，確認服務是否正在運行。
    """
    return HealthCheckResponse(status="ok")

@router.post("/classify", response_model=ClassifyResponse, tags=["Classification"])
async def classify_text(
    request: ClassifyRequest,
    components: dict = Depends(get_components)
):
    """
    接收文本，並使用 RAG 分類器返回預測的多個標籤。
    """
    try:
        # 調用你的核心 RAG 邏輯
        predicted_str = get_final_labels(
            user_input=request.text,
            config=APP_CONFIG,
            components=components
        )
        
        # 處理 RAG 函式的輸出
        if predicted_str and predicted_str.lower() != '無':
            labels = [label.strip() for label in predicted_str.split(',')]
        else:
            labels = []

        return ClassifyResponse(
            input_text=request.text,
            predicted_labels=labels
        )
    except Exception as e:
        # 捕獲潛在的錯誤，並返回一個有意義的伺服器錯誤
        print(f"處理請求時發生錯誤: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"內部伺服器錯誤: {str(e)}"
        )
