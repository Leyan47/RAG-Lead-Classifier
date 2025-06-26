# server.py
import uvicorn
from fastapi import FastAPI
from contextlib import asynccontextmanager

# 導入你的組件初始化函式和 API 端點
from components import initialize_all
from api.endpoints import router as api_router
from api import endpoints as api_module

@asynccontextmanager
async def lifespan(app: FastAPI):
    # --- 應用程式啟動時執行的程式碼 ---
    print("應用程式啟動中...")
    # 初始化所有 RAG 組件並將其存儲在全域變數中
    api_module.APP_COMPONENTS = initialize_all()
    print("所有 RAG 組件已成功加載。服務準備就緒。")
    
    yield
    
    # --- 應用程式關閉時執行的程式碼 ---
    print("應用程式正在關閉...")
    api_module.APP_COMPONENTS.clear()
    print("資源已清理。")

# 創建 FastAPI 應用實例，並傳入 lifespan 管理器
app = FastAPI(
    title="RealEstate Lead Tagger API",
    description="一個用於分析房地產客戶需求的 RAG 分類器 API。",
    version="1.0.0",
    lifespan=lifespan
)

# 掛載 API 路由
app.include_router(api_router, prefix="/api/v1")

# 主程式入口，用於直接運行伺服器
if __name__ == "__main__":
    # 使用 uvicorn 啟動伺服器
    # reload=True 可以在開發時，當程式碼變更後自動重啟伺-服器
    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=True)
