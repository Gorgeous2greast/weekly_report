import os
import re
import json
import time
import requests
import feedparser
import markdown
from datetime import datetime, timedelta
from typing import Optional
from jinja2 import Environment, FileSystemLoader

# ================= 1. 配置区 =================
LLM_API_KEY = os.getenv("LLM_API_KEY")
LLM_BASE_URL = os.getenv("LLM_BASE_URL", "https://api.deepseek.com/v1")
LLM_MODEL = os.getenv("LLM_MODEL", "deepseek-chat")
WEBHOOK_URL = os.getenv("WEBHOOK_URL", "")

HTTP_TIMEOUT = 30
HTTP_RETRIES = 3
RECENT_DAYS = 7
TOP_PAPERS_COUNT = "8-12"

# ===== arXiv 查询：分类锚定 + 布尔表达式 =====
# 不再依赖标题/摘要的宽泛子串匹配，而是用 cat: 限定方向，再用布尔表达式捕捉具体内容。
ARXIV_QUERIES = [
    # Coding Agent：软件工程 + NLP/LLM + AI 三个入口
    'cat:cs.SE AND (abs:agent OR abs:"code generation" OR abs:"program repair" OR abs:"issue resolution")',
    'cat:cs.CL AND (abs:"coding agent" OR abs:"code agent" OR abs:"software engineering" OR abs:"repo-level")',
    'cat:cs.AI AND (abs:agent OR abs:planning) AND (abs:code OR abs:program)',
    # RSI：直接使用 recursive self-improvement 及相关强信号，排除一次性的 self-refine/reflexion
    'ti:"recursive self-improvement" OR abs:"recursive self-improvement"',
    'abs:"self-evolving" OR abs:"self-evolution"',
    'abs:"self-rewarding" OR abs:"self-play"',
    'abs:"godel machine" OR abs:"self-referential"',
    'abs:"self-generated data" OR abs:"self-improvement loop"',
]

# 弱 RSI 信号：不作为 arXiv 查询条件，仅在已入选论文中做“赛道归类”时使用
WEAK_RSI_KEYWORDS = [
    "self-refine", "self refine", "reflexion", "self-debug", "self debug",
    "self-instruct", "self instruct",
]

# 长关键词（用于 HF Daily Papers 的补充过滤，arXiv 源不依赖此列表）
SUBSTRING_KEYWORDS = [
    "reinforcement learning", "agentic", "llm agent", "coding agent",
    "multi-agent", "tool agent", "llm training", "alignment", "reward model",
]

RL_EXACT_PATTERN = re.compile(r"\b(rl|rlhf|grpo|ppo|dpo)\b", re.IGNORECASE)

# ===== 排除关键词：机器人 / 具身 / 世界模型 =====
EXCLUDE_PATTERN = re.compile(
    r"\b("
    r"robots?|robotics?|robotic|"
    r"embodied|embodiment|"
    r"humanoid|quadruped|bipedal|"
    r"locomotion|grasping|dexterous|"
    r"autonomous driving|self-driving|"
    r"world[- ]models?|world simulator|"
    r"vision-language-action|vla"
    r")\b",
    re.IGNORECASE,
)

# ===== GitHub 源：在 verl / ms-swift 之外，补充 RSI 基准与 coding agent 核心仓库 =====
GITHUB_REPOS = [
    # 原有训练框架（保留）
    "volcengine/verl",
    "modelscope/ms-swift",
    # RSI 基准与框架
    "scaleapi/rsi-benchmark",
    "quqxui/RSI-Flow",
    # Coding Agent 代表性仓库
    "SWE-agent/SWE-agent",
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


def _has_exclusion(text: str) -> bool:
    """在标题或摘要全文上独立检查排除模式，任一命中即视为需要丢弃。"""
    return bool(EXCLUDE_PATTERN.search(text or ""))


# ================= 3. 论文获取：多源聚合 =================

def _fetch_arxiv_papers() -> list[dict]:
    """
    按分类锚定的布尔查询从 arXiv 拉取论文。
    arXiv 返回 Atom XML，使用 feedparser 解析。
    """
    base = "https://export.arxiv.org/api/query"
    results = []
    seen_ids = set()

    for query in ARXIV_QUERIES:
        params = {
            "search_query": query,
            "start": 0,
            "max_results": 50,
            "sortBy": "submittedDate",
            "sortOrder": "descending",
        }
        try:
            resp = http_get_with_retry(base, params=params)
            if resp is None or resp.status_code != 200:
                print(f"⚠️ arXiv 查询失败: {query[:60]}...")
                continue
            feed = feedparser.parse(resp.content)
        except Exception as e:
            print(f"⚠️ arXiv 解析失败 [{query[:50]}]: {e}")
            continue

        for entry in feed.entries:
            arxiv_id = entry.get("id", "").split("/abs/")[-1].split("v")[0]
            if arxiv_id in seen_ids:
                continue

            title = (entry.get("title") or "").strip().replace("\n", " ")
            summary = (entry.get("summary") or "").strip().replace("\n", " ")
            published = entry.get("published", "")[:10]

            # 摘要级排除：标题和摘要分别独立检查，任一中命中即丢弃
            if _has_exclusion(title) or _has_exclusion(summary):
                continue
            if not _is_recent(published):
                continue

            # 作者处理
            authors_raw = entry.get("authors", [])
            author_names = [a.get("name", "") for a in authors_raw if a.get("name")]
            authors_str = ", ".join(author_names[:3])
            if len(author_names) > 3:
                authors_str += " et al."

            # 分类
            categories = entry.get("tags", [])
            cat_str = ", ".join(t.get("term", "") for t in categories if t.get("term"))
            if not cat_str:
                cat_str = "cs.AI, cs.LG"

            seen_ids.add(arxiv_id)
            results.append({
                "source": "arxiv_query",
                "arxiv_id": arxiv_id,
                "title": title,
                "url": f"https://arxiv.org/abs/{arxiv_id}",
                "authors": authors_str or "N/A",
                "abstract": summary,
                "published": published,
                "categories": cat_str,
            })

    return results


def _fetch_hf_daily() -> list[dict]:
    """HuggingFace Daily Papers 作为补充源，保留关键词过滤，但排除逻辑更严格。"""
    url = "https://huggingface.co/api/daily_papers"
    resp = http_get_with_retry(url)
    if resp is None or resp.status_code != 200:
        print("⚠️ HuggingFace Daily Papers 获取失败，跳过该源。")
        return []

    papers = resp.json()
    results = []

    for item in papers[:60]:
        paper = item.get("paper", {})
        title = (paper.get("title") or "").strip()
        summary = (paper.get("summary") or "").strip()
        published = paper.get("publishedAt", "") or ""

        if _has_exclusion(title) or _has_exclusion(summary):
            continue
        if not _is_recent(published):
            continue

        combined = f"{title} {summary}".lower()
        matched = bool(RL_EXACT_PATTERN.search(combined)) or any(
            kw in combined for kw in SUBSTRING_KEYWORDS
        )
        if not matched:
            continue

        authors_list = paper.get("authors", []) or []
        if authors_list and isinstance(authors_list[0], dict):
            author_names = [a.get("name", "") for a in authors_list if isinstance(a, dict)]
        else:
            author_names = [a for a in authors_list if isinstance(a, str)]
        authors_str = ", ".join(author_names[:3])
        if len(author_names) > 3:
            authors_str += " et al."

        arxiv_id = paper.get("id", "")
        results.append({
            "source": "hf_daily",
            "arxiv_id": arxiv_id,
            "title": title,
            "url": f"https://huggingface.co/papers/{arxiv_id}",
            "authors": authors_str or "N/A",
            "abstract": summary,
            "published": published[:10],
            "categories": "cs.AI, cs.LG",
        })

    return results


def _fetch_github_releases() -> list[dict]:
    """抓取训练框架 / RSI 基准 / Coding Agent 仓库的 latest release。"""
    results = []
    for repo in GITHUB_REPOS:
        try:
            url = f"https://api.github.com/repos/{repo}/releases/latest"
            resp = http_get_with_retry(url)
            if resp is None or resp.status_code != 200:
                continue
            release = resp.json()
            pub = release.get("published_at", "") or ""
            if not _is_recent(pub):
                continue

            body = (release.get("body") or "")[:1200]
            title = f"[Framework Update] {repo} Latest Release: {release.get('name', 'Unknown')}"

            if _has_exclusion(title) or _has_exclusion(body):
                continue

            results.append({
                "source": "github_release",
                "arxiv_id": None,
                "title": title,
                "url": release.get("html_url", f"https://github.com/{repo}"),
                "authors": repo,
                "abstract": f"Release Note: {body}",
                "published": pub[:10],
                "categories": "Framework, GitHub",
            })
        except Exception as e:
            print(f"⚠️ 获取 {repo} release 失败: {e}")

    return results


def _deduplicate(papers: list[dict]) -> list[dict]:
    """按 arXiv ID 和标题去重，arXiv 源优先保留。"""
    seen_ids = set()
    seen_titles = set()
    deduped = []

    # arXiv 源排前面，确保重复时保留 arXiv 版本
    papers.sort(key=lambda p: 0 if p.get("source") == "arxiv_query" else 1)

    for p in papers:
        aid = p.get("arxiv_id")
        title_key = p["title"].strip().lower()[:120]
        if aid and aid in seen_ids:
            continue
        if title_key in seen_titles:
            continue
        if aid:
            seen_ids.add(aid)
        seen_titles.add(title_key)
        deduped.append(p)

    return deduped


def _assign_topic_and_index(papers: list[dict]) -> list[dict]:
    """为每条论文分配 topic 标签和全局 index，供 LLM 评分使用。"""
    for i, p in enumerate(papers, 1):
        p["index"] = i
        combined = f"{p['title']} {p.get('abstract', '')}".lower()

        if p.get("source") == "github_release":
            p["topic"] = "Framework Updates"
            continue

        # RSI 判定：强信号 → RSI；弱信号（self-refine 等）仅在已入选论文中标记
        has_strong_rsi = any(
            kw in combined for kw in [
                "recursive self-improvement", "recursive self improvement",
                "self-evolving", "self evolving", "self-evolution", "self evolution",
                "self-rewarding", "self rewarding", "self-play", "self play",
                "godel machine", "self-referential", "self-referential",
                "self-generated data", "self-improvement loop",
            ]
        )
        has_weak_rsi = any(kw in combined for kw in WEAK_RSI_KEYWORDS)

        if has_strong_rsi:
            p["topic"] = "RSI / Self-Evolution"
        elif has_weak_rsi:
            p["topic"] = "RSI / Self-Evolution (Weak)"
        elif "coding agent" in combined or "code agent" in combined or "swe-bench" in combined:
            p["topic"] = "Coding Agent"
        elif "agentic" in combined or "multi-agent" in combined or "tool agent" in combined:
            p["topic"] = "Agentic RL"
        else:
            p["topic"] = "RL & LLM Training"

    return papers


def fetch_papers() -> list[dict]:
    """
    多源聚合入口：
    1. arXiv 分类布尔查询（主力，coding agent + RSI 定向召回）
    2. HuggingFace Daily Papers（补充，关键词过滤）
    3. GitHub Releases（框架 / RSI 基准 / coding agent 工程动态）
    → 统一去重、摘要级排除、时效过滤 → 分配 topic 和 index
    """
    print("   → 拉取 arXiv（分类布尔查询）...")
    arxiv_papers = _fetch_arxiv_papers()
    print(f"      arXiv 返回 {len(arxiv_papers)} 条")

    print("   → 拉取 HuggingFace Daily Papers ...")
    hf_papers = _fetch_hf_daily()
    print(f"      HF Daily 返回 {len(hf_papers)} 条")

    print("   → 拉取 GitHub Releases ...")
    gh_releases = _fetch_github_releases()
    print(f"      GitHub Releases 返回 {len(gh_releases)} 条")

    merged = arxiv_papers + hf_papers + gh_releases
    merged = _deduplicate(merged)
    merged = _assign_topic_and_index(merged)

    print(f"   多源合并去重后共 {len(merged)} 条")
    return merged


# ================= 4. 第一次 LLM 调用：评分筛选 =================
def score_papers(papers):
    if not papers:
        return {}

    papers_text = ""
    for p in papers:
        abstract_text = (p.get("abstract") or "")[:1000]
        papers_text += (
            f"条目{p['index']}：\n"
            f"- 来源：{p.get('source', 'unknown')}\n"
            f"- 类型：{p.get('topic', 'Paper')}\n"
            f"- 标题：{p['title']}\n"
            f"- 摘要：{abstract_text}\n"
            f"- 发布日期：{p['published']}\n\n"
        )

    prompt = f"""# 角色
你是AI前沿论文评审专家。请对下列条目进行5维度评分（总分100分），并分配赛道。

# 排除规则（一票否决，total_score=0, track="reject", recommendation="reject"）
- 机器人 / 具身智能硬件 / 机械臂 / 运动控制 / 自动驾驶 / 导航 → 直接 reject
- 世界模型（World Model）、世界模拟器 → 直接 reject
- 即使 RL 关键词命中，只要核心场景是上述方向，也 reject

# 评分维度（仅学术论文）
1. 机构权威性(30分): 顶校大厂=25-30；知名院校=15-24；普通=5-14
2. 关键词相关性(30分): 直接命中(RL, RLHF, GRPO, PPO, agentic, coding agent, LLM training, recursive self-improvement, self-evolving, self-rewarding, self-play 等)=25-30；间接=15-24
3. 时效性(20分): 3天内=18-20；1周内=12-17；2周内=6-11
4. 分类匹配度(10分): 核心分类(cs.LG, cs.AI, cs.CL, cs.SE, stat.ML)=8-10；相关=4-7
5. 标题热度信号(10分): 含突破性关键词=8-10；有一定吸引力=4-7

# 赛道分类（只能选一个）
- "rl": 强化学习 (RL)
- "coding_agent": Coding Agent / SWE-bench / 代码生成与修复
- "agentic_rl": 通用 Agentic RL / 多智能体系统（非纯 coding）
- "llm_training": LLM 训练/对齐
- "rsi": RSI / 自进化（递归自我改进、自举、自生成数据/课程、自我奖励、自修改代码/工具、开放式进化）
- "framework": 训练框架 / 基准仓库更新（[Framework Update] 开头，不参与评分）
- "reject": 不相关

# RSI 判定标准（严格区分强/弱）
- 强 RSI（递归闭环：自修改、自生成数据再训练、自奖励、开放式进化、Gödel Machine 类）→ track="rsi", recommendation="strong"
- 弱 RSI（一次性 self-refine / reflexion / self-debug / self-instruct）→ 仅当论文同时涉及 coding agent 或训练闭环时，才归入 track="rsi"，recommendation 最多为 "worth_knowing"；否则按其主要内容归入其他赛道，不要强行放入 RSI。

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
      "track": "coding_agent",
      "recommendation": "strong"
    }}
  ]
}}

# 待评分条目列表
{papers_text}
"""

    content = call_llm(prompt, temperature=0.2, max_tokens=16000)
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
            f"- 来源：{p.get('source', 'unknown')}\n"
            f"- 类型：{p.get('topic', 'Paper')}\n"
            f"- 标题：{p['title']}\n"
            f"- 链接：{p['url']}\n"
            f"- 作者/来源：{p['authors']}\n"
            f"- 摘要/Release Note：{(p.get('abstract') or '')[:1200]}\n"
            f"- 发布日期：{p['published']}\n"
            f"- 分类：{p['categories']}\n"
            f"{score_info}\n"
        )

    json_template = """
{
  "report_date": "DATE_PLACEHOLDER",
  "executive_summary": {
    "rl": "强化学习(RL)领域本周最重要发现（1-2句话）",
    "coding_agent": "Coding Agent / SWE-bench 领域本周最重要发现（1-2句话）",
    "agentic_rl": "通用 Agentic RL / 多智能体领域本周最重要发现（1-2句话）",
    "llm_training": "LLM 训练/对齐领域本周最重要发现（1-2句话）",
    "rsi_self_evolution": "RSI / 自进化领域本周最重要发现（1-2句话）",
    "framework_updates": "训练框架 / RSI 基准 / Coding Agent 仓库本周动态总结（1-2句话，若无更新则写'本周无重要更新'）"
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
      "track_name": "💻 Coding Agent",
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
      "track_name": "🤖 Agentic RL / 多智能体",
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
      "track_name": "🛠️ 框架与基准动态",
      "recommended": [
        {
          "title": "框架/仓库名称及版本",
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
2. Coding Agent：SWE-bench、代码生成与修复、仓库级软件工程 Agent
3. 通用 Agentic RL / 多智能体系统
4. LLM 训练与对齐
5. RSI / 自进化：递归自我改进、自举、自生成数据/课程、自我奖励、自修改代码/工具、开放式进化
6. 框架与基准动态：训练框架、RSI 基准、Coding Agent 核心仓库的 GitHub 更新

⚠️ 不关注纯机器人（Robotics）、具身智能硬件、世界模型（World Model）相关的研究。

# 任务目标
基于第一次预评分结果，对**学术论文**进行筛选与报告生成：
- 选出总分较高的 Top {TOP_PAPERS_COUNT} 篇学术论文
- 将选中论文合理分配到前 5 个赛道（RL / Coding Agent / Agentic RL / LLM 训练对齐 / RSI 自进化）的「强烈推荐」或「值得知道」中
- 将 [Framework Update] 条目放入第 6 个赛道（"🛠️ 框架与基准动态"）的「强烈推荐」中
- 生成完整周报结构化 JSON

# RSI / 自进化判定标准
- 强 RSI：改进后的系统能提升"自我改进能力"本身，形成递归闭环；如自修改代码、自生成数据并再训练、自奖励、开放式进化、Gödel Machine 类
- 弱 RSI（self-refine / reflexion / self-debug / self-instruct）：仅作为相关线索，仅当与 coding agent 或训练闭环结合时才归入 RSI 赛道，recommendation 最多 "worth_knowing"；否则按主要内容归入其他赛道。

# 输出 JSON 结构 (严格遵循，不要包含 ```json 等 Markdown 标记)
{json_template}

# 约束
- ⚠️ 绝对真实性约束：所有「核心贡献」、「技术亮点」必须严格基于提供的 abstract/release note 原文总结，严禁编造任何未提及的实验数据、指标或结论！
- ⚠️ 框架更新约束：[Framework Update] 开头的条目仅放入 "🛠️ 框架与基准动态" 赛道，评分字段填 "N/A"，且不得出现在其他学术赛道或 top_papers 中。
- ⚠️ 排除约束：严禁收录机器人、具身智能硬件、世界模型相关论文；即使预评分给了高分，也必须丢弃。
- ⚠️ RSI 质量约束：不要为了填充 RSI 赛道而将弱 RSI（self-refine 等）强行放入。若本周 RSI 强信号论文不足，RSI 赛道可以只有 1 篇甚至为空，但需在 executive_summary.rsi_self_evolution 中如实说明"本周 RSI 强信号论文较少"。
- 每篇论文的「核心贡献」控制在 100-150 字以内。
- 「值得知道」部分只保留「一句话总结」和「亮点」。
- 仅返回纯 JSON，绝对不要包含任何额外的解释文本或 Markdown 标记。

# 待分析内容列表：
{papers_text}
"""

    content = call_llm(prompt, temperature=0.2, max_tokens=18000)
    report_data = parse_json_response(content)

    if report_data:
        report_data = _enforce_framework_track(report_data)
        report_data = _enforce_exclusions(report_data)
    return report_data


def _enforce_framework_track(report_data):
    """强制清洗：确保 [Framework Update] 条目只出现在框架赛道中。"""
    tracks = report_data.get("paper_tracks", [])
    framework_titles = set()

    for track in tracks:
        if "框架与基准动态" in track.get("track_name", "") or "训练框架动态" in track.get("track_name", ""):
            for p in track.get("recommended", []):
                if p.get("title"):
                    framework_titles.add(p["title"])
            continue

    for track in tracks:
        if "框架与基准动态" in track.get("track_name", "") or "训练框架动态" in track.get("track_name", ""):
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

    top = report_data.get("top_papers", [])
    report_data["top_papers"] = [
        p for p in top
        if not str(p.get("title", "")).startswith("[Framework Update]")
        and p.get("title") not in framework_titles
    ]
    return report_data


def _enforce_exclusions(report_data):
    """兜底：从报告里剔除机器人 / 世界模型条目。"""
    for track in report_data.get("paper_tracks", []):
        for key in ("recommended", "worth_knowing"):
            items = track.get(key, [])
            track[key] = [
                p for p in items
                if not EXCLUDE_PATTERN.search(
                    f"{p.get('title','')} {p.get('core_contribution','')} "
                    f"{p.get('one_line_summary','')} {p.get('technical_highlights','')}"
                )
            ]
    top = report_data.get("top_papers", [])
    report_data["top_papers"] = [
        p for p in top
        if not EXCLUDE_PATTERN.search(f"{p.get('title','')} {p.get('reason','')}")
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

    md_content += "## 📌 执行摘要\n\n"
    summary = report_data.get("executive_summary", {})
    for key, label in [
        ("rl", "🧠 **强化学习 (RL)**"),
        ("coding_agent", "💻 **Coding Agent**"),
        ("agentic_rl", "🤖 **Agentic RL / 多智能体**"),
        ("llm_training", "🏋️ **LLM 训练/对齐**"),
        ("rsi_self_evolution", "🔄 **RSI / 自进化**"),
        ("framework_updates", "🛠️ **框架与基准动态**"),
    ]:
        md_content += f"- {label}: {summary.get(key, '本周无显著更新')}\n"
    md_content += "\n"

    if report_data.get("cross_insights"):
        md_content += "## 💡 跨领域洞察\n\n"
        for insight in report_data["cross_insights"]:
            md_content += f"### {insight['title']}\n{insight['content']}\n\n"

    for track in report_data.get("paper_tracks", []):
        md_content += f"## 📚 {track['track_name']}\n\n"

        if track.get("recommended"):
            md_content += "### 🌟 强烈推荐\n\n"
            for p in track["recommended"]:
                md_content += f"#### [{p['title']}]({p['url']})\n\n"
                md_content += f"**作者**: {p.get('authors', 'N/A')} | **发布**: {p.get('published', 'N/A')}\n\n"

                sb = p.get("score_breakdown", {}) or {}
                if isinstance(sb.get("institution_authority"), str) and sb.get("institution_authority") == "N/A":
                    md_content += "*<sub>📊 框架/基准更新，不参与学术评分</sub>*\n\n"
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
    print("1. 正在从多源获取论文与框架数据...")
    papers = fetch_papers()
    print(f"   最终候选 {len(papers)} 条。")

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

    owner = os.getenv("GITHUB_REPOSITORY_OWNER", "")
    repo_full = os.getenv("GITHUB_REPOSITORY", "")
    if owner and repo_full:
        repo_name = repo_full.split("/")[-1]
        github_pages_url = f"https://{owner}.github.io/{repo_name}/"
    else:
        github_pages_url = "（本地运行，未配置 GitHub Pages 地址）"

    push_notification(github_pages_url)
