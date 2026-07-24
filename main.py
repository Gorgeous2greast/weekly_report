import os
import json
import requests
import markdown
from datetime import datetime
from jinja2 import Environment, FileSystemLoader

# ================= 1. 配置区 =================
LLM_API_KEY = os.getenv("LLM_API_KEY")
LLM_BASE_URL = os.getenv("LLM_BASE_URL", "https://api.deepseek.com/v1") # 替换为你的实际 Base URL
LLM_MODEL = os.getenv("LLM_MODEL", "deepseek-chat") # 或 qwen-plus, doubao-seed 等
WEBHOOK_URL = os.getenv("WEBHOOK_URL", "")

# ================= 2. 获取论文数据 =================
def fetch_papers():
    """
    这里使用 HuggingFace Daily Papers API 作为示例。
    """
    url = "https://huggingface.co/api/daily_papers"
    response = requests.get(url)
    papers = response.json()
    
    # 简单过滤：只保留标题或摘要包含关键词的论文
    keywords = ["rl", "reinforcement", "agent", "agentic", "training", "ppo", "grpo", "rlhf"]
    filtered = []
    
    for item in papers[:30]: # 取前30篇进行深度评分
        paper = item.get("paper", {})
        title = paper.get("title", "").lower()
        summary = paper.get("summary", "").lower()
        
        if any(kw in title or kw in summary for kw in keywords):
            # 🌟 步骤 1: 先在外面处理好 authors 字符串
            authors_list = paper.get("authors", [])
            if authors_list and isinstance(authors_list[0], dict):
                # 如果是字典列表，提取 name 字段
                author_names = [a.get("name", "") for a in authors_list if isinstance(a, dict)]
            else:
                # 如果是字符串列表，直接使用
                author_names = [a for a in authors_list if isinstance(a, str)]
            
            authors_str = ", ".join(author_names[:3])
            if len(author_names) > 3:
                authors_str += " et al."
            
            # 🌟 步骤 2: 将处理好的字符串放入字典中
            filtered.append({
                "index": len(filtered) + 1,
                "title": paper.get("title"),
                "url": f"https://huggingface.co/papers/{paper.get('id')}",
                "authors": authors_str,  # 直接使用处理好的变量
                "abstract": paper.get("summary"),
                "published": paper.get("publishedAt", "")[:10],
                "categories": "cs.AI, cs.LG", 
                "topic": "RL & Agentic Training"
            })

    # 🌟 新增：抓取训练框架的最新动态 (满足带教要求)
    github_repos = ["volcengine/verl", "modelscope/ms-swift"]
    for repo in github_repos:
        try:
            gh_url = f"https://api.github.com/repos/{repo}/releases/latest"
            gh_resp = requests.get(gh_url, timeout=5)
            if gh_resp.status_code == 200:
                release = gh_resp.json()
                filtered.append({
                    "index": len(filtered) + 1,
                    "title": f"[Framework Update] {repo} Latest Release: {release.get('name', 'Unknown')}",
                    "url": release.get('html_url', f"https://github.com/{repo}"),
                    "authors": repo,
                    "abstract": f"Release Note: {release.get('body', 'No description')[:1000]}", # 截取前1000字
                    "published": release.get('published_at', '')[:10],
                    "categories": "Framework, GitHub",
                    "topic": "Framework Updates" # 特殊标记
                })
        except Exception:
            pass # 如果网络超时或无 release，静默跳过
    
    return filtered

# ================= 3. 调用大模型 (融合打分与报告生成) =================
def generate_report_with_llm(papers):
    if not papers:
        return None

    # 将论文列表格式化为 Prompt 输入
    papers_text = ""
    for p in papers:
        papers_text += f"论文{p['index']}：\n- 标题：{p['title']}\n- 链接：{p['url']}\n- 作者：{p['authors']}\n- 摘要：{p['abstract']}\n- 发布日期：{p['published']}\n- 分类：{p['categories']}\n\n"

    # 🌟 融合版超级 Prompt：一次性完成 5 维度评分 + 深度分析 + 结构化 JSON 输出
    prompt = f"""# 角色定义
你是AI前沿论文评审专家兼周报编辑，专注于强化学习、RLHF、LLM训练、Agentic RL等领域。

# 任务目标
对提供的论文列表进行5维度评分（总分100分），选出Top 6-10篇高质量论文，并直接生成一份完整的周报结构化 JSON 数据。

# 评分维度（总分100分，用于内部评估排序）
1. 机构权威性(30分): 顶校大厂=25-30；知名院校=15-24；普通=5-14
2. 关键词相关性(30分): 直接命中(RL, RLHF, GRPO, PPO, agentic, LLM training等)=25-30；间接=15-24
3. 时效性(20分): 3天内=18-20；1周内=12-17；2周内=6-11
4. 分类匹配度(10分): 核心分类(cs.LG, cs.AI, cs.CL, stat.ML)=8-10；相关=4-7
5. 标题热度信号(10分): 含突破性关键词=8-10；有一定吸引力=4-7

# 筛选与分析规则
- 按总分降序排列，选出Top 6-10篇。
- 将选中论文分为「强烈推荐」(Top 3-5) 和「值得知道」(其余) 两个等级。
- 提炼跨领域洞察（识别不同研究方向之间的关联）。
- 撰写 3-5 句话的执行摘要，概括本周最重要发现。

# 输出 JSON 结构 (严格遵循，不要包含 ```json 等 Markdown 标记)
{{
  "report_date": "YYYY-MM-DD",
  "executive_summary": {
    "rl": "强化学习(RL)领域本周最重要发现（1-2句话）",
    "agentic_rl": "Agentic RL / Coding Agent 领域本周最重要发现（1-2句话）",
    "llm_training": "LLM 训练/对齐领域本周最重要发现（1-2句话）",
    "framework_updates": "训练框架（如 verl, ms-swift）本周 GitHub 最新动态总结（1-2句话）"
  },
  "cross_insights": [
    {"title": "洞察标题", "content": "洞察详细描述"}
  ],
  "paper_tracks": [
    {
      "track_name": "强化学习（RL）与 Agentic 训练",
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
          "core_contribution": "核心贡献详述（问题→方法→结果，严格基于原文，严禁编造数据）",
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
    }
  ],
  "top_papers": [
    {"title": "论文标题", "url": "链接", "reason": "推荐理由（2-3句话）"}
  ]
}}

# 约束
- ⚠️ 绝对真实性约束：所有「核心贡献」、「技术亮点」必须严格基于提供的 abstract/release note 原文总结，严禁编造任何未提及的实验数据、指标或结论！
- 每篇论文的「核心贡献」控制在 100-150 字以内。
- 「值得知道」部分只保留「一句话总结」和「亮点」。
- 如果论文超过 10 篇，优先保证「强烈推荐」部分的完整性。
- 仅返回纯 JSON，绝对不要包含 ```json 等 Markdown 标记。

# 待分析论文列表：
{papers_text}
"""

    headers = {"Authorization": f"Bearer {LLM_API_KEY}", "Content-Type": "application/json"}
    payload = {
        "model": LLM_MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.2,
        "max_tokens": 16000
    }
    
    response = requests.post(f"{LLM_BASE_URL}/chat/completions", headers=headers, json=payload)
    response.raise_for_status()
    
    # 🔍 调试：打印完整响应
    response_json = response.json()
    print(f"✅ API 响应成功！完整响应结构：{response_json.keys()}")
    
    # 检查是否有 choices
    if "choices" not in response_json or len(response_json["choices"]) == 0:
        print(f"❌ 响应中没有 choices！完整响应: {response_json}")
        return None
    
    # 获取内容
    try:
        content = response_json["choices"][0]["message"]["content"].strip()
    except (KeyError, IndexError) as e:
        print(f"❌ 无法获取内容: {e}")
        print(f"响应结构: {response_json}")
        return None
    
    print(f"📄 获取到内容，长度: {len(content)} 字符")
    
    # 清理 Markdown 标记
    content = content.replace("```json", "").replace("```", "").strip()
    
    print(f"📄 清理后内容前 200 字符:\n{content[:200]}...")
    
    try:
        return json.loads(content)
    except json.JSONDecodeError as e:
        print(f"❌ JSON 解析失败: {e}")
        print(f"📄 完整内容前 1000 字符:\n{content[:1000]}")
        return None

# ================= 4. Markdown 转 HTML 处理 =================
def process_markdown_to_html(report_data):
    """
    大模型输出的文本可能包含 Markdown (如 **加粗**)。
    此函数将其转换为 HTML，以便在 Jinja2 模板中用 | safe 完美渲染。
    """
    md = markdown.Markdown(extensions=['extra'])
    
    # 递归处理字典中的字符串字段
    def convert_dict(d):
        if isinstance(d, dict):
            return {k: convert_dict(v) for k, v in d.items()}
        elif isinstance(d, list):
            return [convert_dict(i) for i in d]
        elif isinstance(d, str):
            # 将 Markdown 转换为 HTML，并移除包裹的 <p> 标签以保持紧凑
            html = md.convert(d)
            if html.startswith("<p>") and html.endswith("</p>"):
                return html[3:-4]
            return html
        return d

    return convert_dict(report_data)

# ================= 5. 渲染 HTML 页面 =================
def render_html(report_data):
    # 确保 templates 目录存在
    os.makedirs("templates", exist_ok=True)
    
    env = Environment(loader=FileSystemLoader('templates'))
    template = env.get_template('report.html')
    
    html_content = template.render(**report_data)
    
    with open("index.html", "w", encoding="utf-8") as f:
        f.write(html_content)
    print("✅ 精美 HTML 周报已生成: index.html")

# ================= 6. 推送通知 =================
def push_notification(page_url):
    if not WEBHOOK_URL:
        print("⚠️ 未配置 WEBHOOK_URL，跳过推送。")
        return
    
    payload = {
        "msgtype": "text",
        "text": {
            "content": f"🚀 本周 AI 前沿周报已生成！\n✨ 点击查看精美网页版：{page_url}"
        }
    }
    requests.post(WEBHOOK_URL, json=payload)
    print("✅ 推送通知已发送")


# ================= 新增：生成 Markdown 文件 =================
def save_as_markdown(report_data):
    os.makedirs("reports", exist_ok=True)
    date_str = report_data.get("report_date", datetime.now().strftime("%Y-%m-%d"))
    filename = f"reports/{date_str}.md"
    
    md_content = f"# 🤖 AI 前沿周报 - {date_str}\n\n"
    
    # 1. 执行摘要 (严格按 4 个固定维度)
    md_content += "## 📌 执行摘要\n\n"
    summary = report_data.get("executive_summary", {})
    md_content += f"- 🧠 **强化学习 (RL)**: {summary.get('rl', '本周无显著更新')}\n"
    md_content += f"- 🤖 **Agentic RL / Coding Agent**: {summary.get('agentic_rl', '本周无显著更新')}\n"
    md_content += f"- 🏋️ **LLM 训练/对齐**: {summary.get('llm_training', '本周无显著更新')}\n"
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
                
                # 🌟 新增：5 维评分小字展示
                sb = p.get("score_breakdown", {})
                md_content += f"*<sub>📊 评分明细: 权威 {sb.get('institution_authority', '-')} | 相关 {sb.get('keyword_relevance', '-')} | 时效 {sb.get('timeliness', '-')} | 分类 {sb.get('category_match', '-')} | 热度 {sb.get('title_heat', '-')} ➔ **总分: {p.get('total_score', 'N/A')}</sub>*\n\n"
                
                md_content += f"{p.get('core_contribution', '')}\n\n"
                if p.get('technical_highlights'):
                    md_content += f"**技术亮点**: {p['technical_highlights']}\n\n"
                md_content += f"**落地难度**: {p.get('difficulty', 'N/A')}\n\n---\n\n"
                
        if track.get("worth_knowing"):
            md_content += "### 📖 值得知道\n\n"
            for p in track["worth_knowing"]:
                md_content += f"- **[{p['title']}]({p['url']})**: {p.get('one_line_summary', '')}\n"
                if p.get('application_tip'):
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
    
# ================= 主流程 =================
if __name__ == "__main__":
    print("1. 正在获取最新论文数据...")
    papers = fetch_papers()
    print(f"   找到 {len(papers)} 篇相关论文。")
    
    print("2. 正在调用大模型进行 5 维度评分与深度分析...")
    report_data = generate_report_with_llm(papers)
    
    if report_data:
        print("3. 正在处理 Markdown 格式并渲染精美 HTML...")
        clean_data = process_markdown_to_html(report_data)
        render_html(clean_data)

        #  新增：同时生成 Markdown 文件
        print("4. 正在生成 Markdown 归档文件...")
        save_as_markdown(report_data)
        
        # 注意：这里的 URL 需要替换为你实际的 GitHub Pages 地址
        # 格式通常为: https://<你的GitHub用户名>.github.io/<你的仓库名>/
        github_pages_url = f"https://{os.getenv('GITHUB_REPOSITORY_OWNER')}.github.io/{os.getenv('GITHUB_REPOSITORY').split('/')[-1]}/"
        push_notification(github_pages_url)
    else:
        print("❌ 大模型未能返回有效的 JSON 数据，请检查 API 配置或重试。")
