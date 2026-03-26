# Kaggle 微调 DistilGPT2 中文故事生成模型指南

## 📋 完整方案

### 第一步：准备 Kaggle 环境

#### 1. 创建 Kaggle 账号
- 访问 https://www.kaggle.com/settings/account
- 点击 "Create New API Token"
- 下载 `kaggle.json`

#### 2. 安装 Kaggle CLI
```bash
pip install kaggle

# 配置认证
mkdir -p ~/.kaggle
cp kaggle.json ~/.kaggle/
chmod 600 ~/.kaggle/kaggle.json
```

#### 3. 创建 Kaggle Dataset
```bash
# 创建本地目录
mkdir -p ~/kaggle-dataset/hp-story-training
cd ~/kaggle-dataset

# 复制训练数据（见第二步）
# 创建 dataset-metadata.json
```

---

### 第二步：准备训练数据

创建文件 `/workspaces/nlp2026/prepare_training_data.py`：

```python
import json
from pathlib import Path

def generate_training_data():
    """从知识库和模板生成中文故事训练数据"""
    
    training_examples = [
        # 格式：（提示词，期望输出）
        ("哈利在霍格沃茨。他走进了教室。故事继续：", "教授还没有到达，几个学生已经坐在了座位上。哈利找到了靠窗的位置，坐了下来。"),
        ("赫敏在对角巷。她进入了一家书店。故事继续：", "书店的架子上摆满了各种魔法书籍。赫敏的眼睛发亮，立即走向了神咒部分。"),
        ("罗恩在禁林。他听到了奇怪的声音。故事继续：", "他握紧了魔杖，小心翼翼地向声音传来的方向走去。月光照亮了林间的小路。"),
        ("邓布利多在魔法部。他召开了紧急会议。故事继续：", "各位部长都已经入座。邓布利多的表情异常严肃，说明情况很糟。"),
        ("斯内普在地下教室。他在调制魔药。故事继续：", "他用玻璃棒搅拌着坩埚中的液体，浓烈的蒸汽升起，房间里弥漫着奇异的气味。"),
        
        # 添加更多示例...（至少30条以上为佳）
    ]
    
    # 保存为 JSON
    output_path = Path("/workspaces/nlp2026/data/training_data.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump({
            "examples": training_examples,
            "count": len(training_examples)
        }, f, ensure_ascii=False, indent=2)
    
    print(f"✓ 生成了 {len(training_examples)} 条训练样本")

if __name__ == "__main__":
    generate_training_data()
```

运行生成：
```bash
cd /workspaces/nlp2026
python prepare_training_data.py
```

---

### 第三步：Kaggle Notebook 微调脚本

在 Kaggle 上创建新 Notebook，复制下面代码：

```python
# ============ Kaggle 微调脚本 ============

import json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, TextDataset, DataCollatorForLanguageModeling, Trainer, TrainingArguments
from pathlib import Path

# 1. 上传训练数据到 /kaggle/input/hp-story-training/
TRAIN_DATA_PATH = "/kaggle/input/hp-story-training/training_data.json"
OUTPUT_DIR = "/kaggle/working/distilgpt2-finetuned-chinese"

# 2. 加载模型和分词器
print("📥 加载基础模型...")
model_name = "distilgpt2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# 添加特殊 token（中文特化）
special_tokens = {
    "pad_token": "[PAD]",
    "eos_token": "[EOS]",
}
num_added_tokens = tokenizer.add_special_tokens(special_tokens)
model.resize_token_embeddings(len(tokenizer))

# 3. 准备数据
print("📊 准备训练数据...")
with open(TRAIN_DATA_PATH, 'r', encoding='utf-8') as f:
    data = json.load(f)

# 格式化为文本
train_text = ""
for prompt, completion in data['examples']:
    train_text += f"{prompt} {completion}\n"

# 保存为临时文件
train_file = "/kaggle/working/train.txt"
with open(train_file, 'w', encoding='utf-8') as f:
    f.write(train_text)

# 4. 微调参数
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    overwrite_output_dir=True,
    num_train_epochs=3,  # 简短版本用3个 epoch
    per_device_train_batch_size=2,  # Kaggle GPU 内存限制
    save_steps=50,
    save_total_limit=2,
    logging_steps=10,
    learning_rate=5e-5,
    weight_decay=0.01,
    warmup_steps=100,
    fp16=True,  # 使用混合精度加速
)

# 5. 创建数据加载器
train_dataset = TextDataset(
    tokenizer=tokenizer,
    file_path=train_file,
    block_size=128
)

data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False
)

# 6. 微调
print("🚀 开始微调...")
trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=train_dataset,
)

trainer.train()

# 7. 保存微调模型
print("💾 保存模型...")
model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)

print(f"✓ 模型已保存到 {OUTPUT_DIR}")
print("\n📦 接下来的步骤：")
print("1. 下载微调模型文件（目录）")
print("2. 上传到项目 /models/distilgpt2-finetuned-chinese/")
print("3. 更新 config.py 指向新模型")
```

---

### 第四步：Kaggle 执行步骤

1. **创建 Dataset**
   ```bash
   kaggle datasets create -p ~/kaggle-dataset/hp-story-training -u
   # 记下 dataset ID：username/dataset-name
   ```

2. **在 Kaggle 新建 Notebook**
   - 选择 GPU 加速
   - 添加数据源：你刚上传的 dataset
   - 粘贴上面的微调脚本
   - 运行 Notebook

3. **下载微调模型**
   - Notebook 完成后，下载 `/kaggle/working/distilgpt2-finetuned-chinese/`

---

### 第五步：集成到项目

创建目录和配置：

```bash
# 1. 创建模型目录
mkdir -p /workspaces/nlp2026/models/distilgpt2-finetuned-chinese

# 2. 上传微调后的模型文件到该目录
# pytorch_model.bin
# config.json
# tokenizer.json
# etc.

# 3. 更新 config.py
```

修改 `/workspaces/nlp2026/config.py`：

```python
class ModelConfig:
    # 改为微调后的模型
    TEXT_GENERATION_MODEL = "./models/distilgpt2-finetuned-chinese"
    # 或者如果上传到 HuggingFace Hub：
    # TEXT_GENERATION_MODEL = "your-username/distilgpt2-finetuned-chinese"
```

修改 `/workspaces/nlp2026/story_weaver/nlg/generator.py`：

```python
def _generate_dynamic_story(self, user_action: str, game_state: Dict, 
                           retrieved_context: List[Dict],
                           intent: str) -> str:
    """生成动态推进的故事 - 使用微调后的 distilgpt2"""
    
    character = game_state.get('player_character', '巫师')
    location = game_state.get('current_location', '霍格沃茨')
    
    # 启用微调模型生成
    if self.use_llm and self.model and self.tokenizer:
        story = self._llm_generate_with_rag_context(
            user_action, character, location, intent, retrieved_context
        )
        if story:
            return story
    
    # 回退到模板
    print(f"[NLG] ℹ️ 微调模型生成失败，使用模板回退")
    return self._template_generate_story(
        user_action, character, location, intent
    )
```

---

### 第六步：验证

启动项目并测试：

```bash
cd /workspaces/nlp2026
python app.py
# 访问 http://localhost:5001
```

---

## 📊 预期结果

| 指标 | 微调前 | 微调后 |
|------|--------|--------|
| 中文输出质量 | ❌ 垃圾字符 | ✅ 流畅叙事 |
| 重复问题 | ❌ 严重 | ✅ 消除 |
| 故事连贯性 | ❌ 差 | ✅ 优秀 |
| 推理时间 | ✅ 快（<1s） | ✅ 快（<1s） |

---

## 🚀 快速开始（精简版）

```bash
# 1. 生成训练数据
python prepare_training_data.py

# 2. 上传到 Kaggle
kaggle datasets create -p ~/kaggle-dataset -u

# 3. 在 Kaggle Notebook 运行微调脚本

# 4. 下载模型，放入 /models/

# 5. 更新 config.py

# 6. 重启应用
python app.py
```

---

## 💡 优化建议

- **增加训练数据**：收集更多中文故事（100+条更佳）
- **调整超参数**：根据 loss 曲线调整 learning rate 和 epochs
- **评估指标**：用测试集验证生成质量
- **上传到 HuggingFace**：方便分享和版本管理

---

## ❓ 常见问题

**Q：Kaggle GPU 不够？**  
A：用 CPU 模式（慢 10 倍）或分批微调较小数据集

**Q：模型文件太大？**  
A：DistilGPT2 只有 ~350MB，Kaggle 便宜存储足够

**Q：如何持续改进？**  
A：每次添加新故事数据后重新微调一轮

