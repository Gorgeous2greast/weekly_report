import os
import json
import requests
import markdown
from datetime import datetime
from jinja2 import Environment, FileSystemLoader

# ================= 1. 配置区 =================
LLM_API_KEY = os.getenv("LLM_API_KEY")
LLM_BASE_URL = os.getenv("LLM_BASE_URL", "https://api.deepseek.com/v1")
LLM_MODEL = os.getenv("LLM_MODEL", "deepseek-chat")

# ================= 2. 获取论文数据 (复用 HF API，但过滤更精准) =================
def fetch_specialized_papers():
    """
    使用 HuggingFace Daily Papers API，但精准过滤 Agentic Retrieval & Reward 相关论文
    """
    url = "https://huggingface.co/api/daily_papers"
    response = requests.get(url, timeout=10)
    papers = response.json()
    
    #  精准关键词：必须同时包含 (Agentic/RL) + (Retrieval/RAG) + (Reward/Optimization)
    # 或者标题/摘要中明确提到这些组合
    filtered = []
    
    for item in papers[:50]:  # 多看一些，取前 50 篇
        paper = item.get("paper", {})
        title = paper.get("title", "").lower()
        summary = paper.get("summary", "").lower()
        
        # 组合关键词检查
        has_agent_rl = any(kw in title or kw in summary for kw in ["agentic", "reinforcement learning", "rlhf", "ppo", "grpo"])
        has_retrieval = any(kw in title or kw in summary for kw in ["retrieval", "rag", "search", "dynamic retrieval"])
        has_reward = any(kw in title or kw in summary for kw in ["reward", "optimization", "shaping", "preference"])
        
        # 必须同时满足：有 RL/Agent + 有 Retrieval + 有 Reward/Optimization
        if has_agent_rl and has_retrieval and has_reward:
            # 处理作者
            authors_list = paper.get("authors", [])
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
                "published": paper.get("publishedAt", "")[:10],
                "categories": "cs.AI, cs.LG",
                "topic": "Agentic Retrieval & Reward Optimization"
            })
    
    # 如果 HF 没有找到足够的论文，补充一些经典关键词放宽搜索
    if len(filtered) < 5:
        print("⚠️ HF 找到论文较少，放宽关键词重新搜索...")
        for item in papers[:50]:
            paper = item.get("paper", {})
            title = paper.get("title", "").lower()
            summary = paper.get("summary", "").lower()
            
            # 放宽条件：只要有 (Agent/RL) 和 (Retrieval/RAG) 即可
            if any(kw in title or kw in summary for kw in ["agentic", "rl", "rlhf"]) and \
               any(kw in title or kw in summary for kw in ["retrieval", "rag", "search"]):
                
                # 避免重复
                if not any(p["url"] == f"https://huggingface.co/papers/{paper.get('id')}" for p in filtered):
                    authors_list = paper.get("authors", [])
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
                        "published": paper.get("publishedAt", "")[:10],
                        "categories": "cs.AI, cs.LG",
                        "topic": "Agentic Retrieval & Reward Optimization"
                    })
    
    return filtered[:15]  # 最多返回 15 篇

# ================= 3. 调用大模型进行深度思路提取 =================
def analyze_reward_strategies(papers):
    if not papers:
        return None

    papers_text = ""
    for p in papers:
        papers_text += f"论文{p['index']}：\n- 标题：{p['title']}\n- 链接：{p['url']}\n- 摘要：{p['abstract']}\n\n"

    # 🌟 专项定制 Prompt：聚焦奖励函数与动态检索优化
    prompt = f"""# 角色定义
你是 Agentic RL 与 Reward Shaping 领域的顶级研究员。当前团队的核心痛点是：**如何优化大模型在自主动态检索 (Dynamic Retrieval / Agentic Search) 过程中的奖励函数 (Reward Function)**。

# 任务目标
阅读提供的论文列表，深度挖掘其中关于"检索策略优化"、"奖励函数设计"、"动态反馈机制"的新思路，并输出结构化的调研洞察报告。

# 分析维度 (请在总结中重点关注)
1. **Reward Formulation**：论文是如何设计奖励信号的？(如：稀疏/稠密奖励、过程奖励 Process Reward、基于信息增益的奖励、基于不确定性的惩罚等)。
2. **Retrieval Action Space**：模型在检索时的动作空间是如何定义的？(如：决定何时检索、检索多少文档、如何重写查询)。
3. **Novelty & Inspiration**：该方法与传统 RLHF 或标准 RAG 相比，创新点在哪里？对我们当前任务有何直接启发？

# 输出 JSON 结构 (严格遵循，不要包含 ```json 标记)
{{
  "research_date": "{datetime.now().strftime('%Y-%m-%d')}",
  "core_paradigms": [
    {{
      "paradigm_name": "研究范式名称 (如：基于过程奖励的检索步长控制)",
      "description": "该范式的核心思想（2-3句话）",
      "representative_papers": [
        {{
          "title": "论文标题",
          "url": "链接",
          "reward_design": "该论文具体的奖励函数设计思路或公式文字描述",
          "key_inspiration": "对我们优化动态检索任务的直接启发（1-2句话）"
        }}
      ]
    }}
  ],
  "contradictions_or_gaps": "当前这些研究在奖励函数设计上存在的矛盾、局限性或未解决的空白（1-2段话，这往往是我们创新的突破口）",
  "top_3_actionable_ideas": [
    "我们可以立即尝试的具体思路 1（如：引入某某指标作为检索动作的内在奖励）",
    "我们可以立即尝试的具体思路 2",
    "我们可以立即尝试的具体思路 3"
  ]
}}

# 约束
- 绝对真实性：所有奖励设计思路必须严格基于摘要原文，严禁编造公式或指标。
- 聚焦核心：如果论文只是泛泛而谈 Agent，未涉及具体的检索或奖励优化，请忽略。
- 仅返回纯 JSON，不要包含任何额外的解释文本。

# 待分析论文列表：
{papers_text}
"""

    headers = {"Authorization": f"Bearer {LLM_API_KEY}", "Content-Type": "application/json"}
    payload = {
        "model": LLM_MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.1,
        "max_tokens": 8000
    }
    
    response = requests.post(f"{LLM_BASE_URL}/chat/completions", headers=headers, json=payload)
    response.raise_for_status()
    
    content = response.json()["choices"][0]["message"]["content"].strip()
    content = content.replace("```json", "").replace("```", "").strip()
    
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        try:
            from json_repair import repair_json
            return json.loads(repair_json(content))
        except Exception as e2:
            print(f" JSON 解析失败: {e2}\n末尾内容: {content[-300:]}")
            return None

# ================= 4. 渲染专项调研 Markdown 报告 =================
def save_research_report(report_data):
    os.makedirs("research_reports", exist_ok=True)
    date_str = report_data.get("research_date", datetime.now().strftime("%Y-%m-%d"))
    filename = f"research_reports/Agentic_Retrieval_Reward_{date_str}.md"
    
    md = f"#  Agentic RL 动态检索奖励优化专项调研 - {date_str}\n\n"
    
    # 1. 核心研究范式
    md += "## 🧠 核心研究范式 (Core Paradigms)\n\n"
    for paradigm in report_data.get("core_paradigms", []):
        md += f"### {paradigm['paradigm_name']}\n"
        md += f"{paradigm['description']}\n\n"
        
        for p in paradigm.get("representative_papers", []):
            md += f"#### 📄 [{p['title']}]({p['url']})\n"
            md += f"- **奖励设计**: {p.get('reward_design', 'N/A')}\n"
            md += f"- **💡 启发**: {p.get('key_inspiration', 'N/A')}\n\n"
        md += "---\n\n"
        
    # 2. 现有研究的局限与空白
    md += "## ⚠️ 现有研究的局限与空白 (Gaps & Contradictions)\n\n"
    md += f"{report_data.get('contradictions_or_gaps', '暂无')}\n\n"
    
    # 3. 可落地的行动思路
    md += "## 🚀 Top 3 可立即尝试的行动思路 (Actionable Ideas)\n\n"
    for i, idea in enumerate(report_data.get("top_3_actionable_ideas", []), 1):
        md += f"{i}. **{idea}**\n\n"
        
    with open(filename, "w", encoding="utf-8") as f:
        f.write(md)
    print(f"✅ 专项调研报告已保存: {filename}")
    return filename

# ================= 主流程 =================
if __name__ == "__main__":
    print("1. 正在 HuggingFace 定向检索 Agentic Retrieval & Reward 论文...")
    papers = fetch_specialized_papers()
    print(f"   找到 {len(papers)} 篇高度相关论文。")
    
    if not papers:
        print("❌ 未找到相关论文，请检查网络或 API。")
        exit(1)
        
    print("2. 正在调用大模型深度提取奖励函数设计思路...")
    report_data = analyze_reward_strategies(papers)
    
    if report_data:
        print("3. 正在生成专项调研 Markdown 报告...")
        save_research_report(report_data)
        print("✅ 调研完成！请查看 research_reports/ 目录下的最新文件。")
    else:
        print("❌ 大模型未能返回有效的 JSON 数据。")
