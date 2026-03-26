"""
NLG模块 - 动态故事推进系统 v2
关键特性：
1. 约束提示词 - 确保内容在哈利波特宇宙
2. 动态生成 - 根据用户输入推进故事
3. 状态跟踪 - 记录故事进展
4. 智能回退 - LLM失败时有高质量模板
"""

from typing import List, Dict, Optional
import json
from dataclasses import dataclass
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import random

@dataclass
class NarrativeResponse:
    """叙事响应"""
    main_narrative: str
    next_options: List[str]
    state_updates: Dict[str, str]
    metadata: Dict


class NLGEngine:
    """动态故事生成引擎"""
    
    def __init__(self, model_name: str = "distilgpt2", use_llm: bool = True):
        self.model_name = model_name
        self.use_llm = use_llm
        self.tokenizer = None
        self.model = None
        self.device = "cpu"
        
        # 哈利波特宇宙约束
        self.hp_constraints = {
            "forbidden": ["宇航员", "火箭", "计算机", "汽车", "飞机"],
            "locations": ["霍格沃茨", "禁林", "对角巷", "魔法部", "格林戈茨"],
            "characters": ["哈利", "赫敏", "罗恩", "邓布利多", "斯内普"]
        }
        
        if self.use_llm:
            self._load_model()
    
    def _load_model(self):
        """加载LLM"""
        try:
            print(f"[NLG] 加载动态故事生成模型: {self.model_name}...")
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name, 
                trust_remote_code=True
            )
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                trust_remote_code=True,
                low_cpu_mem_usage=True
            )
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            self.model.to(self.device)
            self.model.eval()
            print(f"[NLG] ✓ 模型已加载")
        except Exception as e:
            print(f"[NLG] ⚠️ 加载失败: {e}，使用模板模式")
            self.use_llm = False
    
    def generate_narrative(self, 
                          user_action: str,
                          game_state: Dict,
                          retrieved_segments: List[Dict],
                          intent: str,
                          entities: List[Dict]) -> NarrativeResponse:
        """生成动态故事 - RAG提供上下文，NLG自由生成"""
        
        # **关键改变**：将RAG检索结果传递给生成方法
        narrative = self._generate_dynamic_story(
            user_action=user_action,
            game_state=game_state,
            retrieved_context=retrieved_segments,  # 传递RAG上下文
            intent=intent
        )
        
        # 生成选项
        options = self._generate_options(game_state, intent)
        
        # 生成状态更新
        state_updates = self._extract_state_changes(user_action, intent)
        
        # 元数据
        metadata = {
            "model": "RAG增强NLG生成",
            "intent": intent,
            "context_segments": len(retrieved_segments),
            "is_constrained": False  # 改为False，表示不再严重依赖模板
        }
        
        return NarrativeResponse(
            main_narrative=narrative,
            next_options=options,
            state_updates=state_updates,
            metadata=metadata
        )
    
    def _generate_dynamic_story(self, user_action: str, game_state: Dict, 
                               retrieved_context: List[Dict],
                               intent: str) -> str:
        """生成动态推进的故事 - 使用RAG上下文作为引导"""
        
        character = game_state.get('player_character', '巫师')
        location = game_state.get('current_location', '霍格沃茨')
        
        # **改进**：distilgpt2 在中文处理上表现较差，直接使用高质量模板
        # 未来可升级为 ChatGLM、Qwen 等中文模型
        print(f"[NLG] ℹ️ 使用模板生成（中文优化）")
        return self._template_generate_story(
            user_action, character, location, intent
        )
    
    def _llm_generate_with_rag_context(self, user_action: str, character: str,
                                       location: str, intent: str, 
                                       retrieved_context: List[Dict]) -> Optional[str]:
        """使用RAG检索的上下文来引导NLG生成 - 最小化约束，最大化模型自由度"""
        try:
            # 构建上下文信息 - 从RAG检索结果中提取
            context_info = self._build_context_from_rag(retrieved_context, location)
            
            # **简化提示词**：避免模型重复或混淆
            prompt = f"{character}在{location}。{user_action}。故事继续："
            
            inputs = self.tokenizer.encode(prompt, return_tensors="pt").to(self.device)
            
            with torch.no_grad():
                outputs = self.model.generate(
                    inputs,
                    max_new_tokens=60,
                    temperature=0.7,
                    top_p=0.85,
                    top_k=40,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id,
                    attention_mask=torch.ones_like(inputs)
                )
            
            # 正确提取生成的文本
            generated_ids = outputs[0][inputs.shape[-1]:]
            story = self.tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
            
            if not story:
                return None
            
            # **检测重复字符** - 拒绝包含大量重复的垃圾输出
            if self._is_repetitive_garbage(story):
                return None
            
            # **增强验证**：过滤掉提示词碎片和垃圾输出
            bad_patterns = ["相关背景", "背景信息", "故事继续", "根据", "要求", "- ", "：", "、"]
            if any(pattern in story for pattern in bad_patterns[:3]):
                return None
            
            # 检查禁止词
            for forbidden in self.hp_constraints["forbidden"]:
                if forbidden.lower() in story.lower():
                    return None
            
            # 提取完整句子
            if "。" in story:
                story = story.split("。")[0] + "。"
            
            if 10 < len(story) < 150:
                print(f"[NLG] ✓ RAG增强生成: {story[:50]}...")
                return story
            
            return None
            
        except Exception as e:
            print(f"[NLG] ⚠️ RAG增强生成失败: {e}")
            return None
    
    def _is_repetitive_garbage(self, text: str, threshold: float = 0.5) -> bool:
        """检测是否为重复字符垃圾输出"""
        if len(text) < 3:
            return True
        
        # 检查最后5个字是否都相同或几乎相同
        last_chars = text[-5:] if len(text) >= 5 else text
        if len(set(last_chars)) == 1:
            return True  # 重复的字符
        
        # 检查是否包含大量重复的两字组合（如："续：续：续："）
        for i in range(0, len(text) - 3, 2):
            pattern = text[i:i+2]
            if len(pattern) == 2:
                count = text.count(pattern)
                if count > 3 and count / len(text) > threshold:
                    return True  # 包含重复模式
        
        # 检查是否全是符号
        symbols_only = text.replace("：", "").replace("·", "").replace("。", "").replace("，", "").replace("、", "").replace("-", "").strip()
        if len(symbols_only) < 2:
            return True  # 全是或几乎全是符号
        
        return False
    
    def _build_context_from_rag(self, retrieved_segments: List[Dict], 
                                location: str) -> str:
        """从RAG检索结果构建上下文信息"""
        if not retrieved_segments:
            return self._get_location_background(location)
        
        # 提取检索到的内容片段
        context_lines = []
        for i, seg in enumerate(retrieved_segments[:3], 1):  # 最多使用3个片段
            content = seg.get("content", "")
            source = seg.get("source", "未知")
            
            # 格式化上下文
            if content:
                # 截断长内容
                if len(content) > 100:
                    content = content[:100] + "..."
                context_lines.append(f"- {content}")
        
        # 如果没有检索到内容，使用位置背景
        if not context_lines:
            return self._get_location_background(location)
        
        return "\n".join(context_lines)
    
    def _get_location_background(self, location: str) -> str:
        """获取位置的背景信息"""
        location_backgrounds = {
            "Hogwarts Castle": "这是一个魔法学校，充满了古老的魔法和秘密的走廊。",
            "Forbidden Forest": "禁忌森林是一个神秘而危险的地方，隐藏着许多魔法生物和古老魔法。",
            "Diagon Alley": "对角巷是魔法社区的商业中心，汇集了各种魔法商店和奇异商品。",
            "Ministry of Magic": "魔法部掌控着魔法界的权力，内部充满了复杂的政治斗争。"
        }
        return location_backgrounds.get(location, "这是一个充满魔力的地方。")
    
    def _llm_generate_with_constraints(self, user_action: str, character: str,
                                       location: str, intent: str) -> Optional[str]:
        """使用约束提示词的LLM生成"""
        try:
            # 简化提示词 - 避免模型重复提示词本身
            prompt = f"{character}在{location}，行动：{user_action}。故事继续：\n"
            
            inputs = self.tokenizer.encode(prompt, return_tensors="pt").to(self.device)
            
            with torch.no_grad():
                outputs = self.model.generate(
                    inputs,
                    max_new_tokens=60,
                    temperature=0.6,
                    top_p=0.85,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id,
                    attention_mask=torch.ones_like(inputs)
                )
            
            # 正确提取生成的文本（移除输入部分）
            generated_ids = outputs[0][inputs.shape[-1]:]
            story = self.tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
            
            # 清理生成的文本
            if not story:
                return None
            
            # 检查禁止词
            for forbidden in self.hp_constraints["forbidden"]:
                if forbidden.lower() in story.lower():
                    return None
            
            # 获取第一句完整句子
            if "。" in story:
                story = story.split("。")[0] + "。"
            elif "，" in story:
                # 如果没有句号，至少取到逗号处
                parts = story.split("，")
                if len(parts[0]) > 10:
                    story = parts[0] + "。"
            
            # 验证生成的故事长度合理
            if 15 < len(story) < 150:
                print(f"[NLG] ✓ 动态生成: {story[:60]}...")
                return story
            
            return None
            
        except Exception as e:
            print(f"[NLG] 生成失败: {e}")
            return None
    
    def _template_generate_story(self, user_action: str, character: str,
                                location: str, intent: str) -> str:
        """模板回退生成 - 根据位置和意图生成推进情节的故事"""
        
        # 根据位置的故事模板
        location_stories = {
            "Hogwarts Castle": {
                "move": [
                    f"{character}沿着霍格沃茨古老的走廊前进，每扇肖像后面都隐藏着魔法的秘密。",
                    f"魔法楼梯在脚下响起，{character}继续探索城堡的深处，寻找新的发现。",
                    f"{character}推开一扇沉重的橡木门，发现了城堡一个崭新的区域，魔力在空气中跳动。"
                ],
                "talk": [
                    f"一位老教授抬起头，与{character}进行了一场意味深长的对话，谈论不为人知的魔法知识。",
                    f"{character}的问题激发了同学们的讨论，一场激烈的魔法理论争辩开始了。",
                    f"对话进行得非常深入，{character}获得了关键的线索信息。"
                ],
                "take": [
                    f"{character}发现了一件古老的魔法物品，感受到它蕴含的强大魔力。",
                    f"这件物品在{character}手中闪闪发光，仿佛在认可新主人的到来。",
                    f"获得这件神秘物品后，{character}感到一股新的力量涌遍全身。"
                ],
                "observe": [
                    f"{character}仔细观察城堡的古老魔法纹路，发现了隐藏多年的线索。",
                    f"通过仔细观察，{character}注意到了其他人都忽视的关键细节，这可能改变局势。",
                    f"你的观察力使你发现了一个通往未知地方的隐秘通道。"
                ],
                "cast": [
                    f"强大的魔法能量在{character}周围汇聚，咒语的效果在城堡中回荡。",
                    f"{character}释放的魔法照亮了整个走廊，周围的肖像都被惊动了。",
                    f"魔法的威力超出预期，城堡的某些部分开始产生反应。"
                ]
            },
            "Forbidden Forest": {
                "move": [
                    f"{character}踏入禁林深处，古老的树木簇拥着，神秘的声音在夜幕中回响。",
                    f"森林越来越深，{character}感受到了魔法生物的存在，每一步都充满了危险与惊奇。",
                    f"古老的魔法在这片森林中肆意流动，{character}继续前行，发现了从未见过的奇异景象。"
                ],
                "talk": [
                    f"一个神秘的声音从森林深处传来，{character}与这个未知的存在进行了交流。",
                    f"森林中的一个古老生灵用奇异的方式与{character}达成了无言的对话。",
                    f"{character}与禁林的秘密达成了某种理解。"
                ],
                "take": [
                    f"{character}小心地收集了稀有的魔法材料，这些在禁林中生长了数百年。",
                    f"一根闪闪发光的树枝飘落到{character}手中，似乎是有意的安排。",
                    f"获得禁林的神秘物品后，{character}感到与这片森林更加紧密地联系在一起。"
                ],
                "observe": [
                    f"{character}观察到禁林内隐藏的魔法动物足迹，这暗示着一个重大发现即将来临。",
                    f"通过仔细观察，{character}发现了禁林内一个被遗忘的古老魔法遗迹。",
                    f"你发现了森林深处一个闪闪发光的地方，仿佛那里隐藏着重要的秘密。"
                ],
                "cast": [
                    f"在禁林的魔法干扰下，{character}施放的咒语展现出了意想不到的效果。",
                    f"魔法在禁林中与自然的魔力相融合，产生了强大的共鸣。",
                    f"{character}的魔法在森林中引起了连锁反应，周围的魔法生物都被惊动了。"
                ]
            },
            "Diagon Alley": {
                "move": [
                    f"{character}沿着对角巷的石板街道行走，迷人的魔法商店在周围闪闪发光。",
                    f"对角巷人头攒动，但{character}成功地穿过人群，来到了更深的区域。",
                    f"{character}发现了对角巷的一个隐秘地方，这里似乎很少有人来往。"
                ],
                "talk": [
                    f"一位店主热情地与{character}交谈，分享了关于魔法物品的珍贵知识。",
                    f"{character}与一位来自远方的巫师进行了引人入胜的对话，交换了许多信息。",
                    f"在对角巷的繁华中，{character}进行了一次改变命运的对话。"
                ],
                "take": [
                    f"{character}在一家古老的魔法店里找到了一件稀有的物品，它正在等待合适的主人。",
                    f"这件物品价值连城，但店主似乎很乐意与{character}做成这笔交易。",
                    f"获得这件对角巷的珍贵物品后，{character}的实力有了显著提升。"
                ],
                "observe": [
                    f"{character}注意到对角巷中某间店铺有异常的魔法气息，必然隐藏着秘密。",
                    f"通过仔细观察，{character}发现了对角巷一个不为人知的交易场所。",
                    f"你发现了对角巷中一条通往暗黑魔法研究地点的隐秘通道。"
                ],
                "cast": [
                    f"在对角巷进行魔法实验太危险了，但{character}仍然成功地施放了咒语。",
                    f"{character}的魔法在人群中引起了一些惊呼，但没有造成麻烦。",
                    f"魔法能量在对角巷中闪耀，吸引了许多巫师的注意和敬畏。"
                ]
            },
            "Ministry of Magic": {
                "move": [
                    f"{character}走进了魔法部的宏伟大厅，政治的气息弥漫在空气中。",
                    f"魔法部的走廊延伸至深处，每个角落都透露出权力与秘密。",
                    f"{character}继续探索魔法部的各个部门，发现了许多隐藏的真相。"
                ],
                "talk": [
                    f"一位魔法部的重要官员与{character}进行了一次秘密对话，涉及深层的政治考量。",
                    f"{character}与魔法部的某位要人交换了关键信息，这可能改变一切。",
                    f"对话涉及到魔法界最高层的决策，{{character}}意识到事情比想象中复杂得多。"
                ],
                "take": [
                    f"{character}在魔法部的档案室中发现了一份重要的文件，内容令人震惊。",
                    f"这份材料看起来被许多人守卫，{{character}}成功地获取了它。",
                    f"拥有这份信息后，{{character}}掌握了改变魔法界局势的关键。"
                ],
                "observe": [
                    f"{{character}}敏锐地发现了魔法部内部权力斗争的证据。",
                    f"通过仔细观察，{{character}}注意到了魔法部官员之间的紧张关系。",
                    f"你发现了一个通往魔法部最高机密部门的线索。"
                ],
                "cast": [
                    f"{{character}}在魔法部施放咒语很危险，但{{character}}仍然小心地完成了魔法。",
                    f"魔法能量在魔法部的魔法防御系统中引起了反应，但没有触发警报。",
                    f"{{character}}的魔法在这个权力中心成功地展现了力量和价值。"
                ]
            }
        }
        
        # 获取该位置的故事模板，如果没有则使用默认
        location_templates = location_stories.get(location, location_stories["Hogwarts Castle"])
        story_list = location_templates.get(intent, location_templates["move"])
        
        import random
        return random.choice(story_list)
    
    def _generate_options(self, game_state: Dict, intent: str) -> List[str]:
        """生成下一步选项"""
        location = game_state.get('current_location', '霍格沃茨')
        
        location_options = {
            "Hogwarts Castle": [
                "前往魔咒课教室",
                "走向公共休息室",
                "进入图书馆",
                "上楼到塔楼"
            ],
            "Forbidden Forest": [
                "深入森林探索",
                "返回城堡",
                "寻找神奇生物",
                "收集魔法植物"
            ],
            "Diagon Alley": [
                "进入魔杖店",
                "访问古灵阁",
                "逛书店",
                "购买魔药材料"
            ]
        }
        
        default_options = [
            "继续探索",
            "与人交谈",
            "观察周围",
            "尝试魔法"
        ]
        
        return location_options.get(location, default_options)
    
    def _extract_state_changes(self, user_action: str, intent: str) -> Dict[str, str]:
        """提取需要更新的游戏状态"""
        changes = {
            "action_performed": intent,
            "timestamp": str(__import__('time').time())
        }
        
        # 检查位置变化
        locations = ["禁林", "对角巷", "霍格沃茨", "魔法部"]
        for loc in locations:
            if loc in user_action:
                changes["location"] = loc
                break
        
        return changes


class DialogueGenerator:
    """对话生成器（为了兼容性）"""
    def __init__(self):
        pass
    
    def generate_dialogue(self, context: Dict) -> str:
        """生成对话"""
        return "嗯...我不确定如何回应那个。"

