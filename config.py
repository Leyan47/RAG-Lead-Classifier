# config.py
import configparser
import os
from dotenv import load_dotenv

def load_app_config():
    """
    加載 .env 和 config.ini 文件，並返回一個配置物件。
    """
    # 從 .env 文件加載環境變數
    load_dotenv()
    
    # 從 config.ini 讀取配置
    config = configparser.ConfigParser()
    config.read('config.ini', encoding='utf-8')
    
    print("配置加載完成。")
    return config

# 創建一個全域配置實例，以便其他模組可以直接導入使用
# 這樣可以確保配置文件只被讀取一次
APP_CONFIG = load_app_config()
