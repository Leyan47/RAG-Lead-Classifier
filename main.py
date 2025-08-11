#%% main.py
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate

# 導入其他自定義模組
from config import APP_CONFIG
from components import initialize_all
from retriever import hybrid_retrieve_and_rerank
from text_utils import get_tokenizer, split_text_into_chunks

# 定義結構化輸出
from pydantic import BaseModel, Field
from typing import Literal, Optional, Union, List, Dict, Any

import re
import json
import os
import pandas as pd
from tqdm import tqdm
from datetime import datetime



# 0. Use Pydantic 為每個抽取任務定義一個嚴格的資料結構。
# Field 的 description 會被用來生成更詳細的提示，幫助 LLM 理解每個欄位的意義。
class FloorExtraction(BaseModel):
    """描述使用者對樓層需求的資料結構。"""
    type: Literal["specific", "range", "preference", "exclude", "unknown"] = Field(description="需求的類型，例如指定範圍、偏好或排除。")
    value: Optional[Union[int, str]] = Field(description="具體的樓層數字或偏好/排除的類型（例如 '高', '低'）。")
    min: Optional[int] = Field(description="範圍需求的最低樓層。")
    max: Optional[int] = Field(description="範圍需求的最高樓層。")

class SizeExtraction(BaseModel):
    """描述使用者對坪數需求的資料結構。"""
    type: Literal["specific", "range", "min", "max", "unknown"] = Field(description="需求的類型，例如指定坪數、範圍或最小/最大值。")
    value: Optional[int] = Field(description="具體的坪數數字，或最小/最大坪數。")
    min: Optional[int] = Field(description="範圍需求的最小坪數。")
    max: Optional[int] = Field(description="範圍需求的最大坪數。")

class RoomsExtraction(BaseModel):
    """描述使用者對房數需求的資料結構。"""
    type: Literal["specific", "range", "min", "max", "unknown"] = Field(description="需求的類型，例如指定房數、範圍或最小/最大值。")
    value: Optional[int] = Field(description="具體的房數，或最小/最大房數。")
    min: Optional[int] = Field(description="範圍需求的最小房數。")
    max: Optional[int] = Field(description="範圍需求的最大房數。")



# --- 1. 提示詞載入與管理 ---
def load_prompts_from_directory(directory: str) -> dict:
    """
    從指定目錄中讀取所有 .txt 檔案，並將它們載入為提示詞。
    檔案名稱將作為字典的鍵。
    """
    prompts = {}
    try:
        for filename in os.listdir(directory):
            if filename.endswith(".txt"):
                file_path = os.path.join(directory, filename)
                with open(file_path, 'r', encoding='utf-8') as f:
                    # 檔案名稱 (不含 .txt) 作為鍵
                    prompt_key = filename[:-4] 
                    prompts[prompt_key] = f.read()
        print(f"成功從 '{directory}' 目錄載入 {len(prompts)} 個提示詞。")
        return prompts
    except FileNotFoundError:
        print(f"錯誤：提示詞目錄 '{directory}' 不存在。請檢查路徑。")
        return {}
    except Exception as e:
        print(f"載入提示詞時發生未知錯誤: {e}")
        return {}

# 應用程式啟動時載入所有提示詞
PROMPT_TEMPLATES = load_prompts_from_directory('prompts')

# 定義需要進入第二階段萃取的標籤
EXTRACTABLE_TAG_TO_PROMPT_AND_MODEL = { # 這裡的鍵需要對應到 PROMPT_TEMPLATES 中的鍵
    "樓層偏好": ("extraction_floor_preference", FloorExtraction),
    "坪數": ("extraction_size", SizeExtraction),
    "房數需求": ("extraction_rooms", RoomsExtraction)
}



def parse_pydantic_to_final_tag(conceptual_tag: str, data: BaseModel) -> Optional[str]:
    """將 Pydantic 物件直接轉換為人類可讀標籤。"""
    try:
        tag_type = data.type
        value = data.value
        
        if conceptual_tag == "樓層偏好" and isinstance(data, FloorExtraction):
            if tag_type == "range": return f"樓層偏好: {data.min}-{data.max}樓"
            if tag_type == "specific": return f"樓層偏好: {value}樓"
            if tag_type == "preference" and value == "high": return "樓層偏好: 高樓層"
            if tag_type == "preference" and value == "low": return "樓層偏好: 低樓層"
            if tag_type == "exclude" and value in ["high", "low"]: return f"排除：{value}樓層"
            if tag_type == "exclude": return f"排除：{value}樓"

        elif conceptual_tag == "坪數" and isinstance(data, SizeExtraction):
            if tag_type == "range": return f"坪數: {data.min}-{data.max}坪"
            if tag_type == "specific": return f"坪數: {value}坪"
            if tag_type == "min": return f"坪數: {value}坪(含)以上"
            if tag_type == "max": return f"坪數: {value}坪(含)以下"

        elif conceptual_tag == "房數需求" and isinstance(data, RoomsExtraction):
            if tag_type == "range": return f"房數需求: {data.min}-{data.max}房"
            if tag_type == "min": return f"房數需求: {value}房(含)以上"
            if tag_type == "specific": return f"房數需求: {value}房"
            if tag_type == "max": return f"房數需求: {value}房(含)以下"

    except (AttributeError, TypeError) as e:
        print(f"警告：處理來自 '{conceptual_tag}' 的 Pydantic 物件時出錯。錯誤: {e}")
        return None
    return None

def parse_string_to_final_tag(conceptual_tag: str, json_str: str) -> Optional[str]:
    """【後備方案】從字串中解析 JSON，用於不支援 JSON 模式的模型。"""
    try:
        match = re.search(r'\{[\s\S]*\}', json_str)
        if match:
            clean_json_str = match.group(0).replace("'", '"')
            # 簡單地將解析後的 dict 轉換為一個通用的 BaseModel 來復用 Pydantic 解析邏輯
            # 這裡我們使用對應的 Pydantic 模型來驗證和轉換
            _, pydantic_model = EXTRACTABLE_TAG_TO_PROMPT_AND_MODEL[conceptual_tag]
            data_object = pydantic_model.parse_raw(clean_json_str)
            return parse_pydantic_to_final_tag(conceptual_tag, data_object)
        else:
            print(f"警告（後備模式）：在來自 '{conceptual_tag}' 的輸出中未找到 JSON。輸出: '{json_str}'")
            return None
    except Exception as e:
        print(f"警告（後備模式）：解析來自 '{conceptual_tag}' 的字串時失敗。錯誤: {e}")
        return None



def get_final_labels(user_input: str, config: dict, components: dict) -> str:
    """
    完整的 RAG 流程，從檢索到最終生成。
    """

    retrieval_config = config['Retrieval']
    llm = components['llm']
    
    # 1. 檢索
    retrieved_docs = hybrid_retrieve_and_rerank(
        query                  = user_input,# user_input,  expanded_query
        components             = components,
        rerank_score_threshold = float(retrieval_config['rerank_score_threshold']),
        vector_k               = int(retrieval_config['vector_k']),
        bm25_k                 = int(retrieval_config['bm25_k']),
        final_k                = int(retrieval_config['final_k']),
        verbose                = True,
        bm25_zero_score_ratio_threshold  = float(retrieval_config.get('bm25_zero_score_ratio_threshold', 0.8)),
        faiss_high_score_threshold       = float(retrieval_config.get('faiss_high_score_threshold', 300.0)),
        faiss_high_score_ratio_threshold = float(retrieval_config.get('faiss_high_score_ratio_threshold', 0.8))
    )

    if not retrieved_docs:
        print("\n所有檢索結果均低於相關性閾值，無需 LLM 判斷。")
        return [] # "無"

    # 2. 檢索後處理：標籤去重
    unique_label_docs = {}
    for doc in retrieved_docs:
        label = doc['label']
        # 因為 retrieved_docs 已經是按分數降序排列的，所以我們遇到的第一個某標籤的文檔，一定是分數最高的。
        if label not in unique_label_docs:
            unique_label_docs[label] = doc
    
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

    # 4. 生成 (Generate) - 階段一：概念分類
    classification_template = PROMPT_TEMPLATES.get('classification_prompt')
    if not classification_template:
        print("錯誤：找不到 'classification_prompt.txt'。無法進行分類。")
        return []

    prompt = PromptTemplate(template = classification_template, input_variables=["user_input", "context"])
    chain = prompt | llm | StrOutputParser()
    classification_result  = chain.invoke({"user_input": user_input, "context": context})
    
    initial_tags = [tag.strip() for tag in classification_result.split(',') if tag.strip() and tag.strip().lower() != '無']
    if not initial_tags:
        return []
    
    print(f"\n--- 階段一分類結果：识别到 {len(initial_tags)} 個概念/標籤 ---")
    print(initial_tags)
    
    # --- STAGE 2: 細粒度抽取 ---

    final_results = []
    print("\n--- 階段二抽取開始 ---")
    for tag in initial_tags:
        if tag in EXTRACTABLE_TAG_TO_PROMPT_AND_MODEL:
            prompt_key, pydantic_model = EXTRACTABLE_TAG_TO_PROMPT_AND_MODEL[tag]

            system_prompt_template = PROMPT_TEMPLATES.get(prompt_key)
            
            if not system_prompt_template:
                print(f"警告：找不到標籤 '{tag}' 的提示詞檔案。")
                continue

            print(f"  正在為 '{tag}' 執行信息抽取 (優先使用 JSON 模式)...")
            final_tag = None
            
            # 使用 ChatPromptTemplate
            try:

                chat_prompt = ChatPromptTemplate.from_messages([
                    ("system", system_prompt_template),
                    ("human", "{user_input}") # 將用戶輸入作為 Human Message
                ])

                
                structured_llm = llm.with_structured_output(pydantic_model)  # 綁定結構化輸出到 LLM
                chain = chat_prompt | structured_llm
                extracted_data_object = chain.invoke({"user_input": user_input})  # invoke 的變數與 chat_prompt 中定義的變數完全匹配
                
                print(f"    JSON 模式成功！LLM 輸出物件: {extracted_data_object.dict()}")
                final_tag = parse_pydantic_to_final_tag(tag, extracted_data_object)

            except Exception as e:
                # 後備方案
                print(f"    警告：執行 JSON 模式時發生錯誤 ({e})。退回到字串解析模式。")
                # 為了安全，我們也讓後備方案使用 ChatPromptTemplate
                fallback_prompt = ChatPromptTemplate.from_messages([
                    ("system", system_prompt_template + "\n\n請務必只輸出 JSON 物件。"), # 額外強調格式
                    ("human", "{user_input}")
                ])
                chain_fallback = fallback_prompt | llm | StrOutputParser()
                extracted_json_str = chain_fallback.invoke({"user_input": user_input})
                print(f"    後備模式 LLM 輸出字串: {extracted_json_str}")
                final_tag = parse_string_to_final_tag(tag, extracted_json_str)

            if final_tag:
                final_results.append(final_tag)
        else:
            final_results.append(tag) # 加入不用二階段萃取標籤


    final_results_removeNegative = [tag for tag in final_results if not tag.startswith('_')]        
    return final_results_removeNegative



def load_processed_customers(file_path: str) -> set:
    """
    從 JSONL 檔案中讀取所有已處理的客戶 cus_no。
    
    Args:
        file_path: JSONL 檔案的路徑。
        
    Returns:
        一個包含所有已處理 cus_no 的集合 (set)。
    """
    processed_ids = set()
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                # 忽略可能的空行
                if line.strip():
                    try:
                        data = json.loads(line)
                        # 從 JSON 中提取 cus_no
                        if 'cus_no' in data:
                            processed_ids.add(data['cus_no'])
                    except json.JSONDecodeError:
                        print(f"警告：檔案中發現無效的 JSON 行，已略過: {line.strip()}")
                        continue
    except FileNotFoundError:
        # 如果檔案不存在（代表是第一次運行），返回一個空集合是正常行為
        print("輸出檔案尚未建立，將從頭開始處理。")
        pass
    
    return processed_ids


#%% 多筆運行
if __name__ == "__main__":

    # --- 0. 初始化 RAG 組件與路徑設定 ---
    components = initialize_all()
    
    input_path = 'batch1_without_noise.csv'
    output_path = "batch1_rag_results.jsonl" 

    tokenizer = components['tokenizer']
    token_threshold = components['token_threshold']

    # --- 1. 載入並預處理資料 ---
    print(f"正在從 '{input_path}' 載入資料...")
    try:
        # 假設你的 CSV 總是有 header
        df_raw = pd.read_csv(input_path)
        print("--- 數據載入初步檢查 ---")
        print(f"df_raw 的維度: {df_raw.shape}")
        print("df_raw 的前 5 行:")
        print(df_raw.head())
        print("\ndf_raw 的欄位資訊:")
        df_raw.info()
    except Exception as e:
        print(f"讀取 CSV 時發生錯誤: {e}")
        exit() # 如果文件讀取失敗，直接退出

    # 清理和篩選數據
    print("\n--- 'msg' 欄位內容抽樣檢查 ---")

    df_clean = df_raw[df_raw['msg'].notna() & (df_raw['msg'] != '')].copy()

    # 從 msg (JSON 字串) 中提取 "帶看回饋"
    def extract_feedback(msg_str):
        try:
            # 首先确保输入是字符串类型
            if not isinstance(msg_str, str):
                return ""

            msg_json = json.loads(msg_str)
            
            # 【核心修正点】检查解析后的对象是否为字典
            if isinstance(msg_json, dict):
                # 只有当它是字典时，才使用 .get() 方法
                return msg_json.get("帶看回饋", "")
            else:
                # 如果解析出来是数字、列表或其他类型，则不是我们想要的格式
                return ""
                
        except (json.JSONDecodeError, TypeError):
            # 如果字符串不是合法的 JSON，或者输入类型不是字符串，则返回空字符串
            return ""

    df_clean['feedback_text'] = df_clean['msg'].apply(extract_feedback)
    df_processed = df_clean[df_clean['feedback_text'] != ''].copy()
    
    print(f"資料載入完成。共找到 {len(df_processed)} 條有效的帶看回饋。")

    # --- 2. 計算需要處理的客戶列表 ---
    all_customers = set(df_processed['cus_no'].unique())
    processed_customers = load_processed_customers(output_path)
    customers_to_process = sorted(list(all_customers - processed_customers))

    print(f"共找到 {len(all_customers)} 位獨立客戶。")
    print(f"已在 '{output_path}' 中找到 {len(processed_customers)} 位已處理客戶。")
    
    if not customers_to_process:
        print("所有客戶皆已處理完成！程式結束。")
        exit()

    print(f"將開始處理剩餘的 {len(customers_to_process)} 位客戶。")

    # --- 3. 迭代處理未判斷之客戶 ---
    
    # 按客戶分組，方便合併文本
    grouped = df_processed.groupby('cus_no')
    
    successful_count = 0
    failed_customers = []

    # 使用 'a' (append) 模式打開檔案
    with open(output_path, 'a', encoding='utf-8') as f:
        # 使用 tqdm 顯示進度條
        for cus_no in tqdm(customers_to_process, desc="正在處理客戶"):
            try:
                # 獲取該客戶的所有回饋
                customer_feedbacks = grouped.get_group(cus_no)['feedback_text'].tolist()
                
                # 將所有回饋合併成一個長文本，用換行符分隔
                combined_text = "\n".join(customer_feedbacks)

                text_chunks = split_text_into_chunks(
                    text=combined_text,
                    tokenizer=tokenizer,
                    max_tokens_per_chunk=token_threshold
                )

                if len(text_chunks) > 1:
                    print(f"\n客戶 {cus_no} 的文本過長，已分割成 {len(text_chunks)} 塊進行處理。")

                all_attributes_for_customer = set()

                # 遍歷每一個文本塊，獨立執行 RAG 流程
                for i, chunk in enumerate(text_chunks):
                    if len(text_chunks) > 1:
                        print(f"  正在處理塊 {i+1}/{len(text_chunks)}...")
                    
                    # 調用你的核心 RAG 函式
                    # 注意：現在 get_final_attributes 只需要傳入 config 和 components
                    extracted_attributes_for_chunk = get_final_labels(
                        user_input=chunk, 
                        config=APP_CONFIG, 
                        components=components # components 字典包含了所有需要的東西
                    )
                    
                    if extracted_attributes_for_chunk:
                        all_attributes_for_customer.update(extracted_attributes_for_chunk)
                
                # 構建要儲存的 JSON 物件
                result_json = {
                    "cus_no": cus_no,
                    "combined_feedback": combined_text,
                    "extracted_attributes": sorted(list(all_attributes_for_customer)),
                    "processed_at": datetime.now().isoformat() # 增加時間戳
                }
                
                # 寫入文件
                f.write(json.dumps(result_json, ensure_ascii=False) + '\n')
                f.flush() # 立即將緩存寫入磁碟，更安全
                successful_count += 1

            except KeyError:
                # 如果 groupby 中找不到 cus_no (理論上不應該發生)
                print(f"警告：在分組數據中找不到客戶 {cus_no}，已略過。")
                continue
            except Exception as e:
                print(f"\n處理客戶 {cus_no} 時發生嚴重錯誤: {e}")
                import traceback
                traceback.print_exc() # 打印詳細的錯誤堆疊
                failed_customers.append(cus_no)
                continue

    # --- 4. 打印最終報告 ---
    print("\n--- 批次處理完成 ---")
    print(f"成功處理並儲存: {successful_count} 位客戶")
    if failed_customers:
        print(f"失敗: {len(failed_customers)} 位客戶")
        print("失敗的客戶編號:", failed_customers)
    print(f"結果已儲存至: {output_path}")



#%% 單筆測試
# if __name__ == "__main__":
#    # 初始化
#    components = initialize_all()
    
#    # 測試
#    test_text = [
#        "在邊間採光、室內格局規劃都還不錯，有一件浴廁沒有對外窗但有建議可加裝6合1冷暖機 廚房做開放式擔心油煙味問題，最後有說對於機械式車位還是會有擔憂，進/出車就要花將 近5分鐘左右的時間，擔心巔峰時間上班會被耽誤",
#        "採光跟門前階梯無法 覺得無障礙坡度太陡。邊間、簡單整理即可入住，會有考慮。之前有看過一次 覺得比較像辦公室格局 還有多一個地下室 覺得有點多餘。漏水壁癌都需要大整理 就不考慮",
#        "借KEY  早安再興三房車位 7樓    壁癌跟需裝潢大整理  不考慮。擔心頂樓未來漏水  防護問題   沒車位不方便",
#        "空間夠用，景觀還不錯，但房間都偏小，社區的機械車位也太小，不太合適。樓層較低，屋況也要在整理，社區機械車位不夠大，不合適。屋況較好，空間也夠用，但沒有管理和車位，可能有點不方便，跟太太討論，看是否要再來看"
#    ]
    
#    print("\n" + "="*20 + " 開始執行 RAG 流程 " + "="*20)

#    for text in test_text:
  
#        print(f"輸入文字: {text[:50]}...")
#        final_labels = get_final_labels(text, APP_CONFIG, components)
#        print(f"\n【LLM 最終判斷的標籤】: {final_labels}")
#        print("="*60 + "\n")



