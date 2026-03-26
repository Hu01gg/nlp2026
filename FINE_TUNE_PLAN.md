# 在 Kaggle 上微调 distilgpt2 方案

## 概述
使用 Kaggle Notebooks（免费 16GB GPU）微调 distilgpt2 适配中文哈利波特故事生成。

---

## 第一步：准备训练数据

### 1.1 生成训练数据集
在本地运行以下脚本生成哈利波特故事数据：

```python
# generate_training_data.py
import json
from pathlib import Path

# 从现有知识库和模板生成训练数据
training_data = []

story_templates = {
    "Hogwarts Castle": [
        "哈利在霍格沃茨城堡的长廊里行走，古老的肖像从墙上注视着他。魔法楼梯在脚下响起。",
        "赫敏在图书馆里翻阅古老的魔法书，寻找关键的咒语。突然，一本书自己飞了起来。",
        "罗恩在格兰芬多休息室里，炉火照亮了他思考的脸。他决定加入这个危险的任务。"
    ],
    "Forbidden Forest": [
        "禁林深处传来古老的魔法能量。哈利踏入森林，感受到神秘生物的存在。危险与机遇并存。",
        "森林中闪烁的蓝光指引着他们前进。树木在微风中沙沙作响，仿佛在诉说古老的秘密。",
        "他们发现了一个被遗忘的魔法遗迹，石头上刻着奇异的符号。这可能是关键线索。"
    ],
    "Diagon Alley": [
        "对角巷人头攒动，魔法商店的橱窗闪闪发光。哈利和朋友们穿过人群，寻找需要的魔法物品。",
        "一位年长的巫师在店里递给哈利一个古老的魔法物品。'这对你很重要，'他低声说。",
        "对角巷的隐秘区域隐藏着更多的秘密。只有真正的巫师才能找到这些地方。"
    ]
}

# 转换为训练格式
for location, stories in story_templates.items():
    for story in stories:
        training_data.append({
            "text": story
        })

# 保存为 JSONL 格式（Hugging Face 标准）
with open("training_data.jsonl", "w", encoding="utf-8") as f:
    for item in training_data:
        f.write(json.dumps(item, ensure_ascii=False) + "\n")

print(f"✓ 生成了 {len(training_data)} 条训练样本")
```

### 1.2 上传数据到 Kaggle
```bash
# 将训练数据上传到 Kaggle
kaggle datasets create -p training_data --public
# 或在 Kaggle 网站上手动上传 training_data.jsonl
```

---

## 第二步：在 Kaggle 上创建微调 Notebook

### 2.1 创建 Kaggle Notebook
1. 登录 [kaggle.com](https://www.kaggle.com)
2. 创建新 Notebook（Python）
3. 选择 GPU 加速器（免费 16GB）

### 2.2 Kaggle Notebook 代码

```python
# Kaggle Notebook：微调 distilgpt2

# 安装依赖
!pip install -q transformers datasets torch accelerate

# 导入
import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel, TextDataset, DataCollatorForLanguageModeling
from transformers import Trainer, TrainingArguments
from datasets import load_dataset

print(f"GPU 可用: {torch.cuda.is_available()}")
print(f"GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None'}")

# 1. 加载数据
dataset = load_dataset(
    "json",
    data_files="/kaggle/input/your-dataset/training_data.jsonl",  # 替换为你的数据集名称
    split="train"
)

print(f"✓ 加载 {len(dataset)} 条训练样本")

# 2. 初始化模型和分词器
model_name = "distilgpt2"
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
model = GPT2LMHeadModel.from_pretrained(model_name)

# 添加中文 token（可选但推荐）
tokenizer.add_special_tokens({'pad_token': '[PAD]'})
model.resize_token_embeddings(len(tokenizer))

print(f"✓ 加载模型: {model_name}")

# 3. 数据预处理
def preprocess_function(examples):
    return tokenizer(
        examples["text"],
        truncation=True,
        max_length=256,
        padding="max_length"
    )

tokenized_dataset = dataset.map(preprocess_function, batched=True, remove_columns=["text"])

print(f"✓ 数据预处理完成")

# 4. 数据整理器
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False  # GPT-2 使用因果语言建模，不是 MLM
)

# 5. 训练配置
training_args = TrainingArguments(
    output_dir="/kaggle/working/fine_tuned_model",
    overwrite_output_dir=True,
    num_train_epochs=3,
    per_device_train_batch_size=8,
    save_steps=100,
    save_total_limit=2,
    logging_steps=50,
    learning_rate=5e-5,
    weight_decay=0.01,
    warmup_steps=100,
)

# 6. 创建 Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    data_collator=data_collator,
)

# 7. 开始微调
print("\n🚀 开始微调...")
trainer.train()

# 8. 保存模型
model.save_pretrained("/kaggle/working/fine_tuned_model")
tokenizer.save_pretrained("/kaggle/working/fine_tuned_model")

print("\n✓ 微调完成！模型已保存到 /kaggle/working/fine_tuned_model")
```

### 2.3 运行 Notebook
1. 复制上述代码到 Kaggle Notebook
2. 运行所有单元格
3. 等待训练完成（约 20-30 分钟）

---

## 第三步：下载微调后的模型

### 3.1 从 Kaggle 下载
```bash
# 使用 Kaggle API 下载
kaggle kernels output your-username/notebook-slug -p ./downloaded_model

# 或在网页上手动下载 "fine_tuned_model" 文件夹
```

### 3.2 本地准备模型文件
```bash
# 将模型放到项目目录
mkdir -p fine_tuned_models/distilgpt2_zh
cp -r downloaded_model/* fine_tuned_models/distilgpt2_zh/
```

---

## 第四步：集成到项目

### 4.1 更新配置
编辑 `config.py`：

```python
class ModelConfig:
    # 使用微调后的中文模型
    TEXT_GENERATION_MODEL = "./fine_tuned_models/distilgpt2_zh"
    USE_LLM_GENERATION = True  # 重新启用 LLM
```

### 4.2 恢复 LLM 生成
编辑 `story_weaver/nlg/generator.py`：

```python
def _generate_dynamic_story(self, user_action: str, game_state: Dict, 
                           retrieved_context: List[Dict],
                           intent: str) -> str:
    """生成动态推进的故事"""
    
    character = game_state.get('player_character', '巫师')
    location = game_state.get('current_location', '霍格沃茨')
    
    # 使用微调后的 LLM 生成
    if self.use_llm and self.model and self.tokenizer:
        story = self._llm_generate_with_rag_context(
            user_action, character, location, intent, retrieved_context
        )
        if story:
            return story
    
    # 回退到模板
    print(f"[NLG] ℹ️ LLM 生成失败，使用模板回退")
    return self._template_generate_story(
        user_action, character, location, intent
    )
```

### 4.3 重启程序
```bash
python app.py
```

---

## 第五步：验证效果

测试生成输出是否改善：
```python
from story_weaver.core import StoryWeaver

weaver = StoryWeaver()
weaver.start_new_game()

result = weaver.process_user_input("去禁林探险")
print(result['narrative'])  # 应该看到连贯的中文故事
```

---

## 核心优势

✅ **完全免费** - Kaggle 免费提供 GPU  
✅ **中文优化** - 数据都是中文故事  
✅ **快速迭代** - 可多次微调改进  
✅ **离线使用** - 微调后的模型可离线运行  

---

## 故障排查

| 问题 | 解决方案 |
|------|--------|
| GPU 内存不足 | 减小 batch_size（8→4）或 max_length（256→128） |
| 模型仍然输出垃圾 | 增加训练数据或训练轮数（3→5 epochs） |
| 导入模型错误 | 检查路径是否正确，重新下载 |

---

## 下一步改进

1. **收集真实用户数据** - 用实际交互结果优化训练集
2. **迁移到更大模型** - ChatGLM-6B、Qwen-7B
3. **多语言微调** - 支持英文 + 中文混合生成
4. **RLHF 对齐** - 用人工反馈优化生成质量

