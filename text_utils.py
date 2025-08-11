# text_utils.py
import re
import string
import jieba
import jieba.analyse

# 設置 jieba 分詞模式
jieba.setLogLevel(jieba.logging.INFO)

# 停用詞列表
STOP_WORDS = set([
    '的', '了', '在', '是', '我', '有', '和', '就', '不', '人', '都', '一', '一個', '上', '也', '很', '到', '說', '要', '去',\
    '你', '會', '著', '沒有', '看', '好', '自己', '這', '那', '他', '她', '它', '們', '個', '中', '來', '為', '與', '及', '或',\
    '但', '而', '等', '如', '所', '其', '此', '該', '些', '這些', '那些', '可以', '能夠', '應該', '必須', '可能', '也許', '大概',\
    '或許', '因為', '所以', '但是', '然而', '不過', '而且', '另外', '此外', '除了', '包括', '根據', '關於', '對於', '由於', '透過',\
    '通過', '經過', "想要", "想想", "覺得", "也", "是", "又", "都", "而且", "因為", "因此", "還", "中", "討論", "評估", "真", "的",\
    "了", "要", "一下", "遺下", "再", "在", "太太", "先生", "夫妻", "女兒", "兒子", "爸爸", "媽媽", "但是", "目前", "和", "跟", "回去",\
    "回家", "看看", "這間", "阿", "啊", "ㄚ", "家人"
])

def preprocess_text(text):
    if not isinstance(text, str): return ""
    text = re.sub(r'\s+', ' ', text).strip()
    text = re.sub(r'[^\u4e00-\u9fff\w\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text

def advanced_tokenize(text, use_hmm=True, remove_stopwords=True, min_word_len=1):
    if not text: return []
    text = preprocess_text(text)
    words = jieba.lcut(text, HMM=use_hmm)
    filtered_words = []
    for word in words:
        word = word.strip()
        if (len(word) >= min_word_len and 
            word not in string.punctuation and 
            word != ' ' and
            (not remove_stopwords or word not in STOP_WORDS)):
            filtered_words.append(word)
    return filtered_words

# Tokenizer 和 Chunking 相關函式

def get_tokenizer(tokenizer_path: str):
    """
    從本地路徑加載一個 transformers tokenizer。
    """
    try:
        # from_pretrained 可以直接接收一個本地文件夾路徑
        return AutoTokenizer.from_pretrained(tokenizer_path)
    except OSError:
        print(f"錯誤：在指定路徑 '{tokenizer_path}' 找不到 Tokenizer 文件。")
        # 你可以選擇在這裡拋出異常或返回 None
        raise

def count_tokens(text: str, tokenizer) -> int:
    """使用 transformers tokenizer 計算文本中的 token 數量。"""
    # 我們不關心 attention_mask 等，只需要 input_ids 的長度
    return len(tokenizer.encode(text))

def split_text_into_chunks(
    text: str, 
    tokenizer,
    max_tokens_per_chunk: int
) -> list[str]:
    """
    將長文本按最大 token 數量分割成多個塊。
    這是一個簡單的實現，可以根據需要變得更複雜（例如，考慮句子邊界）。
    """
    if count_tokens(text, tokenizer) <= max_tokens_per_chunk:
        return [text] # 如果文本不超過長度，直接返回

    chunks = []
    # 簡單實現：按行分割，然後合併成塊
    lines = text.splitlines()
    current_chunk_lines = []
    current_chunk_tokens = 0

    for line in lines:
        line_tokens = count_tokens(line, tokenizer)
        
        if current_chunk_tokens + line_tokens > max_tokens_per_chunk and current_chunk_lines:
            # 當前塊已滿，將其加入列表並開始新塊
            chunks.append("\n".join(current_chunk_lines))
            current_chunk_lines = []
            current_chunk_tokens = 0
            
        current_chunk_lines.append(line)
        current_chunk_tokens += line_tokens

    # 加入最後剩餘的塊
    if current_chunk_lines:
        chunks.append("\n".join(current_chunk_lines))

    return chunks
