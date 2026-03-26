"""
Story Weaver 系统配置文件
"""
import os
from pathlib import Path

# 项目根目录
PROJECT_ROOT = Path(__file__).parent.absolute()

# 模型配置
class ModelConfig:
    # NLU模型
    INTENT_MODEL = "distilbert-base-uncased-finetuned-sst-2-english"
    ENTITY_MODEL = "dslim/bert-base-multilingual-cased-ner-hrl"
    
    # NLG模型配置 - 动态故事推进
    TEXT_GENERATION_MODEL = "distilgpt2"  # 轻量级模型，用于约束故事生成
    
    # NLG配置
    USE_LLM_GENERATION = True  # ✓ 启用动态LLM生成（带约束）
    # 这个系统会用约束提示词确保故事始终在HP宇宙内

    
    # 嵌入模型
    EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
    
    # 语言
    LANGUAGE = "zh"  # 中文

# 数据库配置
class DataConfig:
    # 知识库路径
    KNOWLEDGE_BASE_PATH = PROJECT_ROOT / "data" / "knowledge_base"
    CHARACTER_GRAPH_PATH = KNOWLEDGE_BASE_PATH / "character_graph.json"
    PLOT_SEGMENTS_PATH = KNOWLEDGE_BASE_PATH / "plot_segments.json"
    CONSISTENCY_RULES_PATH = KNOWLEDGE_BASE_PATH / "consistency_rules.json"
    WORLD_STATE_PATH = KNOWLEDGE_BASE_PATH / "world_state.json"
    
    # RAG索引
    RAG_INDEX_PATH = PROJECT_ROOT / "data" / "rag_index"

# 系统配置
class SystemConfig:
    # 性能
    MAX_RESPONSE_TIME = 2.0  # 秒
    BATCH_SIZE = 32
    
    # 置信度阈值
    INTENT_CONFIDENCE_THRESHOLD = 0.7
    ENTITY_CONFIDENCE_THRESHOLD = 0.6
    
    # RAG配置
    RAG_TOP_K = 5  # 检索最相关的K个片段
    
    # 多轮交互
    MAX_HISTORY_LENGTH = 10  # 保留最近10轮对话
    
    # 日志
    LOG_DIR = PROJECT_ROOT / "logs"
    INTERACTION_LOG_FILE = LOG_DIR / "interactions.jsonl"
    ERROR_LOG_FILE = LOG_DIR / "errors.log"
    
    # Web接口
    WEB_HOST = "0.0.0.0"
    WEB_PORT = 5001
    DEBUG_MODE = False

# 故事背景配置
class StoryConfig:
    # 故事设定：哈利波特
    UNIVERSE = "Harry Potter"
    SETTING = "Wizarding World"
    
    # 故事参数
    MAX_GENERATED_OPTIONS = 3  # 生成的选项数
    RESPONSE_LENGTH = "medium"  # short, medium, long
    
    # 初始故事点
    INITIAL_SCENE = "Hogwarts Castle, Great Hall"
    INITIAL_TIME_PERIOD = "During Harry's Fifth Year"

# 创建必要的目录
def setup_directories():
    """创建必要的导出目录"""
    system_config = SystemConfig()
    system_config.LOG_DIR.mkdir(parents=True, exist_ok=True)
    
    data_config = DataConfig()
    data_config.KNOWLEDGE_BASE_PATH.mkdir(parents=True, exist_ok=True)
    data_config.RAG_INDEX_PATH.mkdir(parents=True, exist_ok=True)

# 初始化时执行
if __name__ != "__main__":
    setup_directories()
