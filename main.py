import os
import re
import json
import time
import requests
import markdown
from datetime import datetime
from jinja2 import Environment, FileSystemLoader

# ================= 1. 配置区 =================
LLM_API_KEY = os.getenv("LLM_API_KEY")
LLM_BASE_URL = os.getenv("LLM_BASE_URL", "https://api.deepseek.com/v1")
LLM_MODEL = os.getenv("LLM_MODEL", "deepseek-chat")
WEBHOOK_URL = os.getenv("WEBHOOK_URL", "")

HTTP_TIMEOUT = 30          # 单次请求超时（秒）
HTTP_RETRIES = 3           # 重试次数
RECENT_DAYS = 7            # published_at 最近 N 天内
TOP_PAPERS_COUNT = "8-12"  # 统一 Top 论文数量描述

# 长关键词（子串匹配）
SUBSTRING_KEYWORDS = [
    "reinforcement learning",
    "agentic",
    "agent",
    "llm training",
    "alignment",
    "reward model",
]

# 精确短词匹配（避免 "rl" 误伤 world/early 等）
RL_EXACT_PATTERN = re.compile(r"\b(rl|rlhf|grpo|ppo|dpo)\b", re.IGNORECASE)

# RSI / 自进化关键词
RSI_KEYWORDS = [
    "recursive self-improvement",
    "recursive self improvement",
    "self-improvement",
    "self improvement",
    "self-evolving",
    "self evolving",
    "self-evolution",
    "self evolution",
    "self-rewarding",
    "self rewarding",
    "self-play",
    "self play",
    "self-training",
    "self training",
    "self-instruct",
    "self instruct",
    "self-refine",
    "self refine",
    "reflexion",
    "self-debug",
    "self debug",
    "open-ended",
    "open ended",
    "self-modifying",
    "self modifying",
    "godel machine",
    "self-referential",
    "self referential",
]


# ================= 2. 通用工具 =================
def http_get_with_retry(url, **kwargs):
    last_err = None
    for attempt in range(HTTP_RETRIES):
        try:
            resp = requests.get(url, timeout=HTTP_TIMEOUT, **kwargs)
            return resp
        except Exception as e:
            last_err = e
            if attempt < HTTP_RETRIES - 1:
                time.sleep(2 ** attempt)
    print(f"⚠️ 请求失败 {url}: {last_err}")
    return None


def call_llm(prompt, temperature=0.2, max_tokens=8000):
    headers = {
        "Authorization": f"Bearer {LLM_API_KEY}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": LLM_MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    for attempt in range(HTTP_RETRIES):
        try:
            resp = requests.post(
                f"{LLM_BASE_URL}/chat/completions",
                headers=headers,
                json=payload,
                timeout=HTTP_TIMEOUT * 4,
            )
            resp.raise_for_status()
            data = resp.json()
            if "choices" not in data or not data["choices"]:
                print(f"❌ LLM 响应异常: {data}")
                return None
            return data["choices"][0]["message"]["content"].strip()
        except Exception as e:
            print(f"⚠️ LLM 调用失败 (attempt {attempt + 1}): {e}")
            if attempt < HTTP_RETRIES - 1:
                time.sleep(2 ** attempt)
    return None


def parse_json_response(content):
    if not content:
        return None
    content = content.replace("```json", "").replace("```", "").strip()
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        try:
            from json_repair import repair_json
            return json.loads(repair_json(content))
        except Exception as e:
            print(f"❌ JSON 修复后仍然失败: {e}")
            print(f"📄 内容末尾 500 字符:\n{content[-500:]}")
            return None


def _is_recent(published_str, days=RECENT_DAYS):
    if not published_str:
        return True
    try:
        pub = datetime.strptime(str(published_str)[:10], "%Y-%m-%d")
        return (datetime.now() - pub).days <= days
    except Exception:
        return True


# ================= 3. 获取论文与框架数据 =================
def fetch_papers():
    """
    HuggingFace Daily Papers + 关键 GitHub 训练框架 Release。
    - 关键词过滤（RL 精确匹配 + 长关键词子串匹配）
    - published_at 最近 7 天过滤
    - RSI 关键词命中时 topic 标记为 RSI
    """
    url = "https://huggingface.co/api/daily_papers"
    resp = http_get_with_retry(url)
    if resp is None or resp.status_code != 200:
        print("❌ 获取 HuggingFace 论文失败")
        return []

    papers = resp.json()
    filtered = []

    for item in papers[:20]:
        paper = item.get("paper", {})
        title = paper.get("title", "") or ""
        summary = paper.get("summary", "") or ""
        published = paper.get("publishedAt", "") or ""
        combined = f"{title} {summary}".lower()

        # 关键词匹配
        matched = bool(RL_EXACT_PATTERN.search(combined))
        if not matched:
            matched = any(kw in combined for kw in SUBSTRING_KEYWORDS)
        if not matched:
            continue

        # 时效性过滤
        if not _is_recent(published):
            continue

        is_rsi = any(kw in combined for kw in RSI_KEYWORDS)
        topic = "RSI / Self-Evolution" if is_rsi else "RL & Agentic Training"

        # 作者处理
        authors_list = paper.get("authors", []) or []
        if authors_list and isinstance(authors_list[0], dict):
            author_names = [a.get("name", "") for a in authors_list if isinstance(a, dict)]
        else:
            author_names = [a for a in authors_list if isinstance(a, str)]
        authors_str = ", ".join(author_names[:3])
        if len(author_names) > 3:
            authors_str += " et al."

        filtered.append({
            "index": len(filtered) + 1,
            "title": paper.get("title"),
            "url": f"https://huggingface.co/papers/{paper.get('id')}",
            "authors": authors_str,
            "abstract": paper.get("summary"),
            "published": published[:10],
            "categories": "cs.AI, cs.LG",
            "topic": topic,
        })

    # 抓取训练框架最新 Release（最近 7 天内才纳入）
    github_repos = ["volcengine/verl", "modelscope/ms-swift"]
    for repo in github_repos:
        try:
            gh_url = f"https://api.github.com/repos/{repo}/releases/latest"
            gh_resp = http_get_with_retry(gh_url)
            if gh_resp is None or gh_resp.status_code != 200:
                continue
            release = gh_resp.json()
            pub = release.get("published_at", "") or ""
            if not _is_recent(pub):
                continue

            filtered.append({
                "index": len(filtered) + 1,
                "title": f"[Framework Update] {repo} Latest Release: {release.get('name', 'Unknown')}",
                "url": release.get("html_url", f"https://github.com/{repo}"),
                "authors": repo,
                "abstract": f"Release Note: {(release.get('body') or 'No description')[:1000]}",
                "published": pub[:10],
                "categories": "Framework, GitHub",
                "topic": "Framework Updates",
            })
        except Exception as e:
            print(f"⚠️ 获取 {repo} release 失败: {e}")

    return filtered


# ================= 4. 第一次 LLM 调用：评分筛选 =================
def score_papers(papers):
    """第一次调用：对每条论文做 5 维度评分，并分类到具体赛道。"""
    if not papers:
        return {}

    papers_text = ""
    for p in papers:
        abstract_text = (p.get("abstract") or "")[:1500]
        papers_text += (
            f"条目{p['index']}：\n"
            f"- 类型：{p.get('topic', 'Paper')}\n"
            f"- 标题：{p['title']}\n"
            f"- 摘要：{abstract_text}\n"
            f"- 发布日期：{p['published']}\n\n"
        )

    prompt = f"""# 角色
你是AI前沿论文评审专家。请对下列条目进行5维度评分（总分100分），并分配赛道。

# 评分维度（仅学术论文）
1. 机构权威性(30分): 顶校大厂=25-30；知名院校=15-24；普通=5-14
2. 关键词相关性(30分): 直接命中(RL, RLHF, GRPO, PPO, agentic, LLM training, recursive self-improvement, self-evolving, self-rewarding, self-play 等)=25-30；间接=15-24
3. 时效性(20分): 3天内=18-20；1周内=12-17；2周内=6-11
4. 分类匹配度(10分): 核心分类(cs.LG, cs.AI, cs.CL, stat.ML)=8-10；相关=4-7
5. 标题热度信号(10分): 含突破性关键词=8-10；有一定吸引力=4-7

# 赛道分类（只能选一个）
- "rl": 强化学习 (RL)
- "agentic_rl": Agentic RL / Coding Agent
- "llm_training": LLM 训练/对齐
- "rsi": RSI / 自进化（递归自我改进、自举、自生成数据/课程、自我奖励、自修改代码/工具、开放式进化）
- "framework": 训练框架更新（[Framework Update] 开头，不参与评分）
- "reject": 不相关

# RSI 判定标准
- 强 RSI：改进后的系统能提升"自我改进能力"本身，形成递归闭环
- 弱 RSI：一次性 self-refine / reflexion / self-debug，仅作为线索，不强行归为强 RSI

# recommendation 字段
- "strong": 强烈推荐（总分≥80，或赛道内 Top）
- "worth_knowing": 值得知道（总分 60-79）
- "reject": 不推荐（<60）

# 输出 JSON（严格，不要包含 ```json 等 Markdown 标记）
{{
  "scores": [
    {{
      "index": 1,
      "total_score": 85,
      "score_breakdown": {{
        "institution_authority": 25,
        "keyword_relevance": 28,
        "timeliness": 18,
        "category_match": 8,
        "title_heat": 6
      }},
      "track": "rl",
      "recommendation": "strong"
    }}
  ]
}}

# 待评分条目列表
{papers_text}
"""

    content = call_llm(prompt, temperature=0.2, max_tokens=6000)
    data = parse_json_response(content)
    if not data or "scores" not in data:
        print("⚠️ 评分结果解析失败，使用空评分继续。")
        return {}

    score_map = {}
    for s in data["scores"]:
        if isinstance(s, dict) and "index" in s:
            score_map[s["index"]] = s
    return score_map


# ================= 5. 第二次 LLM 调用：生成完整报告 =================
def generate_report_with_llm(papers, score_map):
    if not papers:
        return None

    current_date = datetime.now().strftime("%Y-%m-%d")

    # 将论文 + 预评分合并为输入
    papers_text = ""
    for p in papers:
        s = score_map.get(p["index"], {})
        score_info = ""
        if s and s.get("total_score") is not None:
            score_info = (
                f"- 预评分：总分 {s.get('total_score')}，赛道={s.get('track')}，建议={s.get('recommendation')}\n"
                f"- 评分明细：{json.dumps(s.get('score_breakdown', {}), ensure_ascii=False)}\n"
            )
        papers_text += (
            f"条目{p['index']}：\n"
            f"- 类型：{p.get('topic', 'Paper')}\n"
            f"- 标题：{p['title']}\n"
            f"- 链接：{p['url']}\n"
            f"- 作者/来源：{p['authors']}\n"
            f"- 摘要/Release Note：{p['abstract']}\n"
            f"- 发布日期：{p['published']}\n"
            f"- 分类：{p['categories']}\n"
            f"{score_info}\n"
        )

    # JSON 模板（使用单层花括号，避免 f-string 嵌套报错）
    json_template = """
{
  "report_date": "DATE_PLACEHOLDER",
  "executive_summary": {
    "rl": "强化学习(RL)领域本周最重要发现（1-2句话）",
    "agentic_rl": "Agentic RL / Coding Agent 领域本周最重要发现（1-2句话）",
    "llm_training": "LLM 训练/对齐领域本周最重要发现（1-2句话）",
    "rsi_self_evolution": "RSI / 自进化领域本周最重要发现（1-2句话）",
    "framework_updates": "训练框架（如 verl, ms-swift）本周 GitHub 最新动态总结（1-2句话，若无更新则写'本周无重要框架更新'）"
  },
  "cross_insights": [
    {"title": "洞察标题", "content": "洞察详细描述"}
  ],
  "paper_tracks": [
    {
      "track_name": "🧠 强化学习 (RL)",
      "recommended": [
        {
          "title": "论文标题",
          "url": "链接",
          "authors": "作者信息",
          "published": "发布日期",
          "total_score": 85,
          "score_breakdown": {
            "institution_authority": 25,
            "keyword_relevance": 28,
            "timeliness": 18,
            "category_match": 8,
            "title_heat": 6
          },
          "core_contribution": "核心贡献详述（问题→方法→结果，严格基于原文）",
          "technical_highlights": "技术亮点",
          "differentiation": "与现有工作的差异",
          "code_availability": "代码/复现信息",
          "application_direction": "落地方向",
          "difficulty": "Low"
        }
      ],
      "worth_knowing": [
        {
          "title": "论文标题",
          "url": "链接",
          "one_line_summary": "一句话总结",
          "highlights": "亮点",
          "application_tip": "落地提示"
        }
      ]
    },
    {
      "track_name": "🤖 Agentic RL / Coding Agent",
      "recommended": [
        {
          "title": "论文标题",
          "url": "链接",
          "authors": "作者信息",
          "published": "发布日期",
          "total_score": 85,
          "score_breakdown": {
            "institution_authority": 25,
            "keyword_relevance": 28,
            "timeliness": 18,
            "category_match": 8,
            "title_heat": 6
          },
          "core_contribution": "核心贡献详述（问题→方法→结果，严格基于原文）",
          "technical_highlights": "技术亮点",
          "differentiation": "与现有工作的差异",
          "code_availability": "代码/复现信息",
          "application_direction": "落地方向",
          "difficulty": "Low"
        }
      ],
      "worth_knowing": [
        {
          "title": "论文标题",
          "url": "链接",
          "one_line_summary": "一句话总结",
          "highlights": "亮点",
          "application_tip": "落地提示"
        }
      ]
    },
    {
      "track_name": "🏋️ LLM 训练/对齐",
      "recommended": [
        {
          "title": "论文标题",
          "url": "链接",
          "authors": "作者信息",
          "published": "发布日期",
          "total_score": 85,
          "score_breakdown": {
            "institution_authority": 25,
            "keyword_relevance": 28,
            "timeliness": 18,
            "category_match": 8,
            "title_heat": 6
          },
          "core_contribution": "核心贡献详述（问题→方法→结果，严格基于原文）",
          "technical_highlights": "技术亮点",
          "differentiation": "与现有工作的差异",
          "code_availability": "代码/复现信息",
          "application_direction": "落地方向",
          "difficulty": "Low"
        }
      ],
      "worth_knowing": [
        {
          "title": "论文标题",
          "url": "链接",
          "one_line_summary": "一句话总结",
          "highlights": "亮点",
          "application_tip": "落地提示"
        }
      ]
    },
    {
      "track_name": "🔄 RSI / 自进化",
      "recommended": [
        {
          "title": "论文标题",
          "url": "链接",
          "authors": "作者信息",
          "published": "发布日期",
          "total_score": 85,
          "score_breakdown": {
            "institution_authority": 25,
            "keyword_relevance": 28,
            "timeliness": 18,
            "category_match": 8,
            "title_heat": 6
          },
          "core_contribution": "核心贡献详述（问题→方法→结果，严格基于原文）",
          "technical_highlights": "技术亮点",
          "differentiation": "与现有工作的差异",
          "code_availability": "代码/复现信息",
          "application_direction": "落地方向",
          "difficulty": "Low"
        }
      ],
      "worth_knowing": [
        {
          "title": "论文标题",
          "url": "链接",
          "one_line_summary": "一句话总结",
          "highlights": "亮点",
          "application_tip": "落地提示"
        }
      ]
    },
    {
      "track_name": "🛠️ 训练框架动态",
      "recommended": [
        {
          "title": "框架名称及版本 (如: volcengine/verl v0.8.0)",
          "url": "GitHub Release 链接",
          "authors": "N/A",
          "published": "发布日期",
          "total_score": "N/A",
          "score_breakdown": {
            "institution_authority": "N/A",
            "keyword_relevance": "N/A",
            "timeliness": "N/A",
            "category_match": "N/A",
            "title_heat": "N/A"
          },
          "core_contribution": "核心更新内容总结（1-2句话）",
          "technical_highlights": "重要新特性",
          "differentiation": "N/A",
          "code_availability": "N/A",
          "application_direction": "N/A",
          "difficulty": "N/A"
        }
      ],
      "worth_knowing": []
    }
  ],
  "top_papers": [
    {"title": "论文标题", "url": "链接", "reason": "推荐理由（2-3句话）"}
  ]
}
"""

    json_template = json_template.replace("DATE_PLACEHOLDER", current_date)

    prompt = f"""# 角色定义
你是AI前沿论文评审专家兼周报编辑，专注于以下领域：
1. 强化学习（RL）：PPO, GRPO, RLHF, DPO 等算法及其在 LLM 上的应用
2. Agentic RL：智能体学习、Coding Agent、多智能体系统等（不限制具体场景）
3. LLM 训练与对齐：预训练、监督微调、偏好对齐等
4. RSI / 自进化：递归自我改进、自举、自生成数据/课程、自我奖励、自修改代码/工具、开放式进化等
5. 训练框架动态：verl, ms-swift 等主流训练框架的 GitHub 更新

⚠️ 注意：不关注纯机器人（Robotics）或纯具身智能硬件相关的研究，除非涉及核心 RL 算法创新。

# 任务目标
基于第一次预评分结果，对**学术论文**进行筛选与报告生成：
- 选出总分较高的 Top {TOP_PAPERS_COUNT} 篇学术论文
- 将选中论文合理分配到前 4 个赛道（RL / Agentic RL / LLM 训练对齐 / RSI 自进化）的「强烈推荐」或「值得知道」中
- 将 [Framework Update] 条目放入第 5 个赛道（"🛠️ 训练框架动态"）的「强烈推荐」中
- 生成完整周报结构化 JSON

# RSI / 自进化判定标准
- 强 RSI：改进后的系统能提升"自我改进能力"本身，形成递归闭环；如自修改代码、自生成数据并再训练、自奖励、开放式进化、Gödel Machine 类
- 弱 RSI：一次性 self-refine / reflexion / self-debug，仅作为相关线索，不强行归入强 RSI

# 输出 JSON 结构 (严格遵循，不要包含 ```json 等 Markdown 标记)
{json_template}

# 约束
- ⚠️ 绝对真实性约束：所有「核心贡献」、「技术亮点」必须严格基于提供的 abstract/release note 原文总结，严禁编造任何未提及的实验数据、指标或结论！
- ⚠️ 框架更新约束：[Framework Update] 开头的条目仅放入 "🛠️ 训练框架动态" 赛道，评分字段填 "N/A"，且不得出现在其他学术赛道或 top_papers 中。
- 每篇论文的「核心贡献」控制在 100-150 字以内。
- 「值得知道」部分只保留「一句话总结」和「亮点」。
- 如果论文超过 10 篇，优先保证「强烈推荐」部分的完整性。
- 仅返回纯 JSON，绝对不要包含任何额外的解释文本或 Markdown 标记。

# 待分析内容列表：
{papers_text}
"""

    content = call_llm(prompt, temperature=0.2, max_tokens=18000)
    report_data = parse_json_response(content)

    if report_data:
        report_data = _enforce_framework_track(report_data)
    return report_data


def _enforce_framework_track(report_data):
    """强制清洗：确保 [Framework Update] 条目只出现在框架赛道中。"""
    tracks = report_data.get("paper_tracks", [])
    framework_titles = set()

    # 收集框架条目
    for track in tracks:
        if "训练框架动态" in track.get("track_name", ""):
            for p in track.get("recommended", []):
                if p.get("title"):
                    framework_titles.add(p["title"])
            continue

    # 从其他赛道剔除
    for track in tracks:
        if "训练框架动态" in track.get("track_name", ""):
            continue
        for key in ("recommended", "worth_knowing"):
            items = track.get(key, [])
            track[key] = [
                p for p in items
                if not (
                    str(p.get("title", "")).startswith("[Framework Update]")
                    or p.get("title") in framework_titles
                )
            ]

    # top_papers 剔除
    top = report_data.get("top_papers", [])
    report_data["top_papers"] = [
        p for p in top
        if not str(p.get("title", "")).startswith("[Framework Update]")
        and p.get("title") not in framework_titles
    ]
    return report_data


# ================= 6. Markdown 转 HTML =================
def process_markdown_to_html(report_data):
    md = markdown.Markdown(extensions=["extra"])

    def convert_dict(d):
        if isinstance(d, dict):
            return {k: convert_dict(v) for k, v in d.items()}
        elif isinstance(d, list):
            return [convert_dict(i) for i in d]
        elif isinstance(d, str):
            html = md.convert(d)
            if html.startswith("<p>") and html.endswith("</p>"):
                return html[3:-4]
            return html
        return d

    return convert_dict(report_data)


# ================= 7. 渲染 HTML =================
def render_html(report_data):
    os.makedirs("templates", exist_ok=True)
    template_path = os.path.join("templates", "report.html")
    if not os.path.exists(template_path):
        print(f"❌ 模板文件不存在: {template_path}")
        print("   请先创建 templates/report.html 再运行。")
        return

    env = Environment(loader=FileSystemLoader("templates"))
    template = env.get_template("report.html")
    html_content = template.render(**report_data)

    with open("index.html", "w", encoding="utf-8") as f:
        f.write(html_content)
    print("✅ 精美 HTML 周报已生成: index.html")


# ================= 8. 保存 Markdown =================
def save_as_markdown(report_data):
    os.makedirs("reports", exist_ok=True)
    date_str = report_data.get("report_date", datetime.now().strftime("%Y-%m-%d"))
    filename = f"reports/{date_str}.md"

    md_content = f"# 🤖 AI 前沿周报 - {date_str}\n\n"

    # 1. 执行摘要
    md_content += "## 📌 执行摘要\n\n"
    summary = report_data.get("executive_summary", {})
    md_content += f"- 🧠 **强化学习 (RL)**: {summary.get('rl', '本周无显著更新')}\n"
    md_content += f"- 🤖 **Agentic RL / Coding Agent**: {summary.get('agentic_rl', '本周无显著更新')}\n"
    md_content += f"- 🏋️ **LLM 训练/对齐**: {summary.get('llm_training', '本周无显著更新')}\n"
    md_content += f"- 🔄 **RSI / 自进化**: {summary.get('rsi_self_evolution', '本周无显著更新')}\n"
    md_content += f"- 🛠️ **训练框架动态**: {summary.get('framework_updates', '本周无显著更新')}\n\n"

    # 2. 跨领域洞察
    if report_data.get("cross_insights"):
        md_content += "## 💡 跨领域洞察\n\n"
        for insight in report_data["cross_insights"]:
            md_content += f"### {insight['title']}\n{insight['content']}\n\n"

    # 3. 论文赛道
    for track in report_data.get("paper_tracks", []):
        md_content += f"## 📚 {track['track_name']}\n\n"

        if track.get("recommended"):
            md_content += "### 🌟 强烈推荐\n\n"
            for p in track["recommended"]:
                md_content += f"#### [{p['title']}]({p['url']})\n\n"
                md_content += f"**作者**: {p.get('authors', 'N/A')} | **发布**: {p.get('published', 'N/A')}\n\n"

                sb = p.get("score_breakdown", {}) or {}
                if isinstance(sb.get("institution_authority"), str) and sb.get("institution_authority") == "N/A":
                    md_content += "*<sub>📊 框架更新，不参与学术评分</sub>*\n\n"
                else:
                    md_content += (
                        f"*<sub>📊 评分明细: 权威 {sb.get('institution_authority', '-')}"
                        f" | 相关 {sb.get('keyword_relevance', '-')}"
                        f" | 时效 {sb.get('timeliness', '-')}"
                        f" | 分类 {sb.get('category_match', '-')}"
                        f" | 热度 {sb.get('title_heat', '-')}"
                        f" ➔ **总分: {p.get('total_score', 'N/A')}**</sub>*\n\n"
                    )

                md_content += f"{p.get('core_contribution', '')}\n\n"
                if p.get("technical_highlights"):
                    md_content += f"**技术亮点**: {p['technical_highlights']}\n\n"
                md_content += f"**落地难度**: {p.get('difficulty', 'N/A')}\n\n---\n\n"

        if track.get("worth_knowing"):
            md_content += "### 📖 值得知道\n\n"
            for p in track["worth_knowing"]:
                md_content += f"- **[{p['title']}]({p['url']})**: {p.get('one_line_summary', '')}\n"
                if p.get("application_tip"):
                    md_content += f"  - 💡 {p['application_tip']}\n"
            md_content += "\n"

    # 4. Top Papers
    if report_data.get("top_papers"):
        md_content += "## 🏆 值得深入阅读的 Top Papers\n\n"
        for i, p in enumerate(report_data["top_papers"], 1):
            md_content += f"{i}. **[{p['title']}]({p['url']})**\n   > {p.get('reason', '')}\n\n"

    with open(filename, "w", encoding="utf-8") as f:
        f.write(md_content)
    print(f"✅ Markdown 周报已保存: {filename}")
    return filename


# ================= 9. 推送通知 =================
def push_notification(page_url):
    if not WEBHOOK_URL:
        print("⚠️ 未配置 WEBHOOK_URL，跳过推送。")
        return

    payload = {
        "msgtype": "text",
        "text": {
            "content": f"🚀 本周 AI 前沿周报已生成！\n✨ 点击查看精美网页版：{page_url}"
        },
    }
    try:
        requests.post(WEBHOOK_URL, json=payload, timeout=HTTP_TIMEOUT)
        print("✅ 推送通知已发送")
    except Exception as e:
        print(f"⚠️ 推送失败: {e}")


# ================= 主流程 =================
if __name__ == "__main__":
    print("1. 正在获取最新论文与框架数据...")
    papers = fetch_papers()
    print(f"   找到 {len(papers)} 条相关条目。")

    if not papers:
        print("❌ 未找到条目，退出。")
        raise SystemExit(0)

    print("2. 第一次 LLM 调用：论文评分与赛道分类...")
    score_map = score_papers(papers)
    print(f"   已评分 {len(score_map)} 条。")

    print("3. 第二次 LLM 调用：生成完整周报...")
    report_data = generate_report_with_llm(papers, score_map)

    if not report_data:
        print("❌ 大模型未能返回有效的 JSON 数据，请检查 API 配置或重试。")
        raise SystemExit(1)

    print("4. 正在处理 Markdown 格式并渲染精美 HTML...")
    clean_data = process_markdown_to_html(report_data)
    render_html(clean_data)

    print("5. 正在生成 Markdown 归档文件...")
    save_as_markdown(report_data)

    # GitHub Pages URL 兜底
    owner = os.getenv("GITHUB_REPOSITORY_OWNER", "")
    repo_full = os.getenv("GITHUB_REPOSITORY", "")
    if owner and repo_full:
        repo_name = repo_full.split("/")[-1]
        github_pages_url = f"https://{owner}.github.io/{repo_name}/"
    else:
        github_pages_url = "（本地运行，未配置 GitHub Pages 地址）"

    push_notification(github_pages_url)
