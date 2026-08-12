"""
AI 小说写作工作流 - DeepSeek API
用法:
  python novel_writer.py new [章节号] [--note "额外要求"]   # 生成新章节
  python novel_writer.py revise <章节号> [--note "精修要求"] # 精修章节
  python novel_writer.py list                               # 查看章节列表
  python novel_writer.py init                               # 初始化模板
"""
import os
import json
import argparse
from datetime import datetime

# ================= 1. 配置区 =================
LLM_API_KEY = os.getenv("LLM_API_KEY")
LLM_BASE_URL = os.getenv("LLM_BASE_URL", "https://api.deepseek.com/v1")
LLM_MODEL = os.getenv("LLM_MODEL", "deepseek-chat")

# 写作参数
CHAPTER_WORD_COUNT = int(os.getenv("CHAPTER_WORD_COUNT", "2500"))
TEMPERATURE = float(os.getenv("TEMPERATURE", "0.85"))
MAX_TOKENS = int(os.getenv("MAX_TOKENS", "4096"))
PREV_SUMMARY_COUNT = int(os.getenv("PREV_SUMMARY_COUNT", "3"))

# 路径
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
CHAPTERS_DIR = os.path.join(DATA_DIR, "chapters")
OUTLINE_FILE = os.path.join(DATA_DIR, "outline.md")
CHARACTERS_FILE = os.path.join(DATA_DIR, "characters.md")
WORLD_FILE = os.path.join(DATA_DIR, "world_setting.md")
PROGRESS_FILE = os.path.join(DATA_DIR, "progress.json")
PROMPTS_DIR = os.path.join(BASE_DIR, "prompts")


# ================= 2. 工具函数 =================
def read_file(path):
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return f.read().strip()
    return ""


def call_llm(prompt, system_prompt=None, temperature=None, max_tokens=None):
    """调用大模型 API，返回文本。"""
    if not LLM_API_KEY:
        raise ValueError("未设置 LLM_API_KEY 环境变量")

    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})

    headers = {
        "Authorization": f"Bearer {LLM_API_KEY}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": LLM_MODEL,
        "messages": messages,
        "temperature": temperature or TEMPERATURE,
        "max_tokens": max_tokens or MAX_TOKENS,
        "top_p": 0.9,
        "presence_penalty": 0.3,
        "frequency_penalty": 0.3,
        "stream": False,
    }

    import requests
    response = requests.post(
        f"{LLM_BASE_URL}/chat/completions",
        headers=headers,
        json=payload,
        timeout=120,
    )
    response.raise_for_status()
    result = response.json()

    if "choices" not in result or not result["choices"]:
        raise ValueError(f"API 响应异常: {result}")

    return result["choices"][0]["message"]["content"].strip()


def load_progress():
    if os.path.exists(PROGRESS_FILE):
        with open(PROGRESS_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return {"last_chapter": 0, "summaries": {}, "created_at": datetime.now().isoformat()}


def save_progress(progress):
    os.makedirs(DATA_DIR, exist_ok=True)
    with open(PROGRESS_FILE, "w", encoding="utf-8") as f:
        json.dump(progress, f, ensure_ascii=False, indent=2)


# ================= 3. 上下文组装 =================
def build_context(progress, chapter_num):
    """组装大纲 + 人设 + 世界观 + 前文摘要。"""
    parts = []

    outline = read_file(OUTLINE_FILE)
    if outline:
        parts.append(f"【全书大纲】\n{outline}")

    characters = read_file(CHARACTERS_FILE)
    if characters:
        parts.append(f"【人物设定】\n{characters}")

    world = read_file(WORLD_FILE)
    if world:
        parts.append(f"【世界观设定】\n{world}")

    # 前 N 章摘要
    summaries = progress.get("summaries", {})
    prev_nums = sorted([
        chapter_num - i for i in range(1, PREV_SUMMARY_COUNT + 1)
        if chapter_num - i > 0 and str(chapter_num - i) in summaries
    ])
    if prev_nums:
        prev_texts = [f"第{n}章摘要：{summaries[str(n)]}" for n in prev_nums]
        parts.append("【前文回顾】\n" + "\n".join(prev_texts))

    return "\n\n".join(parts)


# ================= 4. 章节生成 =================
def generate_chapter(chapter_num=None, note=""):
    """生成一章并保存。"""
    progress = load_progress()

    if chapter_num is None:
        chapter_num = progress["last_chapter"] + 1

    chapter_path = os.path.join(CHAPTERS_DIR, f"ch{chapter_num:04d}.md")
    if os.path.exists(chapter_path):
        print(f"⚠️  第{chapter_num}章已存在：{chapter_path}")
        return read_file(chapter_path)

    # 组装上下文
    context = build_context(progress, chapter_num)
    system_prompt = read_file(os.path.join(PROMPTS_DIR, "system.md"))

    prompt = f"""{context}

【本次任务】
请写第{chapter_num}章，约{CHAPTER_WORD_COUNT}字。

要求：
- 严格遵循全书大纲走向
- 保持人物性格一致
- 每500字左右一个爽点或反转
- 章末留钩子让读者想继续看
- 直接输出正文，以"第{chapter_num}章"开头"""

    if note:
        prompt += f"\n\n【额外要求】\n{note}"

    print(f"📝 正在生成第{chapter_num}章（模型: {LLM_MODEL}）...")
    content = call_llm(prompt, system_prompt=system_prompt)

    # 保存章节
    os.makedirs(CHAPTERS_DIR, exist_ok=True)
    with open(chapter_path, "w", encoding="utf-8") as f:
        f.write(content)

    # 生成摘要
    print(f"📋 正在生成第{chapter_num}章摘要...")
    summary_prompt = f"请用200字以内概括以下章节的核心剧情、人物状态变化和关键伏笔：\n\n{content}"
    summary = call_llm(summary_prompt, temperature=0.3, max_tokens=300)

    progress.setdefault("summaries", {})[str(chapter_num)] = summary
    progress["last_chapter"] = max(progress["last_chapter"], chapter_num)
    progress["last_updated"] = datetime.now().isoformat()
    save_progress(progress)

    word_count = len(content)
    print(f"✅ 第{chapter_num}章已保存（{word_count}字）：{chapter_path}")
    print(f"\n{'='*50}")
    print(content[:600] + ("..." if len(content) > 600 else ""))
    print(f"{'='*50}")
    return content


# ================= 5. 章节精修 =================
def revise_chapter(chapter_num, note=""):
    """精修已有章节，原版自动备份。"""
    chapter_path = os.path.join(CHAPTERS_DIR, f"ch{chapter_num:04d}.md")
    original = read_file(chapter_path)
    if not original:
        print(f"❌ 第{chapter_num}章不存在：{chapter_path}")
        return

    system_prompt = read_file(os.path.join(PROMPTS_DIR, "system.md"))
    revise_guide = read_file(os.path.join(PROMPTS_DIR, "revise.md"))
    if not revise_guide:
        revise_guide = "请对以下章节进行精修，提升节奏感、对话质量和爽点密度，保持剧情和人物不变。"

    prompt = f"{revise_guide}\n\n"
    if note:
        prompt += f"【精修要求】\n{note}\n\n"
    prompt += f"【原文】\n{original}"

    print(f"🔧 正在精修第{chapter_num}章...")
    revised = call_llm(prompt, system_prompt=system_prompt)

    # 备份原版
    backup_path = os.path.join(CHAPTERS_DIR, f"ch{chapter_num:04d}.bak.md")
    with open(backup_path, "w", encoding="utf-8") as f:
        f.write(original)

    with open(chapter_path, "w", encoding="utf-8") as f:
        f.write(revised)

    print(f"✅ 第{chapter_num}章精修完成，原版备份：{backup_path}")
    print(f"   原文字数：{len(original)} → 精修后：{len(revised)}")
    return revised


# ================= 6. 章节列表 =================
def list_chapters():
    """列出所有已有章节。"""
    progress = load_progress()
    print(f"📚 当前进度：第{progress['last_chapter']}章")
    print(f"📂 章节目录：{CHAPTERS_DIR}\n")

    if not os.path.exists(CHAPTERS_DIR):
        print("  （暂无章节）")
        return

    import re
    files = sorted([
        f for f in os.listdir(CHAPTERS_DIR)
        if f.startswith("ch") and f.endswith(".md") and ".bak." not in f
    ])

    for fname in files:
        path = os.path.join(CHAPTERS_DIR, fname)
        text = read_file(path)
        match = re.search(r"ch(\d+)", fname)
        num = int(match.group(1)) if match else 0
        summary = progress.get("summaries", {}).get(str(num), "")
        print(f"  第{num}章  {len(text)}字  {fname}")
        if summary:
            print(f"    → {summary[:80]}...")


# ================= 7. 初始化 =================
def init_project():
    """初始化数据目录和模板文件。"""
    os.makedirs(CHAPTERS_DIR, exist_ok=True)

    templates = {
        OUTLINE_FILE: (
            "# 全书大纲\n\n"
            "## 核心设定\n"
            "（一句话概括故事）\n\n"
            "## 主线\n"
            "（主角的目标和成长路径）\n\n"
            "## 分卷规划\n"
            "### 第一卷：卷名（第1-50章）\n"
            "- 第1-10章：\n"
            "- 第11-20章：\n"
            "- 第21-30章：\n"
            "- ...\n\n"
            "### 第二卷：...\n"
        ),
        CHARACTERS_FILE: (
            "# 人物设定\n\n"
            "## 主角\n"
            "- 姓名：\n"
            "- 年龄：\n"
            "- 身份：\n"
            "- 性格：\n"
            "- 金手指：\n"
            "- 核心动机：\n"
            "- 口头禅/行为习惯：\n\n"
            "## 主要配角\n"
            "### 角色2\n"
            "- 与主角关系：\n"
            "- 性格：\n"
            "- 在剧情中的作用：\n\n"
            "## 反派\n"
            "### 反派1\n"
            "- ...\n"
        ),
        WORLD_FILE: (
            "# 世界观设定\n\n"
            "## 时代背景\n"
            "（故事发生的时间、地点、社会环境）\n\n"
            "## 核心规则\n"
            "（金手指规则、力量体系等）\n\n"
            "## 势力分布\n"
            "（主要势力及其关系）\n"
        ),
    }

    for path, content in templates.items():
        if not os.path.exists(path):
            with open(path, "w", encoding="utf-8") as f:
                f.write(content)
            print(f"📄 创建模板：{path}")
        else:
            print(f"⏭️  已存在：{path}")

    # prompts 目录
    os.makedirs(PROMPTS_DIR, exist_ok=True)
    system_prompt_path = os.path.join(PROMPTS_DIR, "system.md")
    revise_prompt_path = os.path.join(PROMPTS_DIR, "revise.md")
    if not os.path.exists(system_prompt_path):
        with open(system_prompt_path, "w", encoding="utf-8") as f:
            f.write(DEFAULT_SYSTEM_PROMPT)
        print(f"📄 创建写作Prompt：{system_prompt_path}")
    if not os.path.exists(revise_prompt_path):
        with open(revise_prompt_path, "w", encoding="utf-8") as f:
            f.write(DEFAULT_REVISE_PROMPT)
        print(f"📄 创建精修Prompt：{revise_prompt_path}")

    print(f"\n✅ 初始化完成！")
    print(f"下一步：编辑以下文件填入设定，然后运行 python {os.path.basename(__file__)} new")
    print(f"  - {OUTLINE_FILE}")
    print(f"  - {CHARACTERS_FILE}")
    print(f"  - {WORLD_FILE}")


# ================= 8. 默认 Prompt =================
DEFAULT_SYSTEM_PROMPT = """# 你的身份
你是一位经验丰富的网络小说写手，擅长番茄小说平台的男频都市题材。你深谙"黄金三章"法则、爽点节奏和读者心理。

# 写作铁律
1. 首段必须有冲突或悬念，300字内抓住读者
2. 每500-800字一个爽点/反转/钩子
3. 每章2000-2500字，章末必须留钩子
4. 多用对话和动作，少用大段心理描写和环境描写
5. 语言直白有画面感，拒绝文艺腔和长难句
6. 节奏要快，不写废话，不堆砌辞藻
7. 人物性格通过行为展现，不要直接告诉读者"他是个XX的人"

# 番茄平台要点
- 读者画像：18-30岁，下沉市场为主，喜欢代入感强的故事
- 核心体验：爽、快、燃、反转
- 禁忌：开头慢热、大段设定说明、主角窝囊太久

# 输出格式
直接输出章节正文，不要加"第X章"以外的标题、注释、说明文字。
"""

DEFAULT_REVISE_PROMPT = """# 章节精修要求
你是一位资深网文编辑，请对以下章节进行精修。

## 精修原则
1. 节奏：删减冗余描写，确保每500字有爽点/反转/钩子
2. 对话：让对话更有张力，每句话都有信息量或情绪
3. 开头：首段必须有冲突或悬念
4. 结尾：章末必须留钩子
5. 代入感：增加主角的感官细节和即时反应
6. 语言：短句为主，口语化，有画面感
7. 保持：剧情走向、人物关系、核心事件不变

## 输出
直接输出精修后的完整章节正文。
"""


# ================= 9. 主入口 =================
def main():
    parser = argparse.ArgumentParser(description="AI 小说写作工作流 (DeepSeek)")
    sub = parser.add_subparsers(dest="command")

    # new
    p_new = sub.add_parser("new", help="生成新章节")
    p_new.add_argument("chapter", type=int, nargs="?", default=None, help="章节号（默认自动递增）")
    p_new.add_argument("--note", "-n", default="", help="额外写作要求")

    # revise
    p_rev = sub.add_parser("revise", help="精修章节")
    p_rev.add_argument("chapter", type=int, help="章节号")
    p_rev.add_argument("--note", "-n", default="", help="精修要求")

    # list
    sub.add_parser("list", help="查看章节列表")

    # init
    sub.add_parser("init", help="初始化模板文件")

    args = parser.parse_args()

    if args.command == "new":
        generate_chapter(chapter_num=args.chapter, note=args.note)
    elif args.command == "revise":
        revise_chapter(chapter_num=args.chapter, note=args.note)
    elif args.command == "list":
        list_chapters()
    elif args.command == "init":
        init_project()
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
