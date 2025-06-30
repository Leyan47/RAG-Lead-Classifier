# schemas.py
from pydantic import BaseModel, Field
from typing import List, Optional

class ClassifyRequest(BaseModel):
    """
    分類請求的數據模型
    """
    text: str = Field(
        ...,  # ... 表示這個字段是必需的
        min_length=10, 
        max_length=2000,
        title="用戶輸入文本",
        description="需要進行多標籤分類的文本，例如客戶的帶看後反應。"
    )
    # 可選參數，未來可以用來選擇不同的模型或場景
    # scene: Optional[str] = Field("post_viewing", description="應用場景，例如 'post_viewing' 或 'phone_call'")

class ClassifyResponse(BaseModel):
    """
    分類回應的數據模型
    """
    input_text: str = Field(..., description="原始輸入的文本")
    predicted_labels: List[str] = Field(..., description="模型預測出的標籤列表")
    message: str = Field("success", description="操作結果訊息")

class HealthCheckResponse(BaseModel):
    """
    健康檢查的回應模型
    """
    status: str = "ok"
