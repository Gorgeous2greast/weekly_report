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
    
    for item in papers[:5]: # 取前30篇进行深度评分
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
  "report_date": "{datetime.now().strftime('%Y-%m-%d')}",
  "executive_summary": ["本周最重要发现1（2-3句话）", "发现2", "发现3"],
  "cross_insights": [
    {{"title": "洞察标题", "content": "洞察详细描述"}}
  ],
  "paper_tracks": [
    {{
      "track_name": "强化学习（RL）与 Agentic 训练",
      "recommended": [
        {{
          "title": "论文标题",
          "url": "链接",
          "authors": "作者信息",
          "published": "发布日期",
          "total_score": 85,
          "core_contribution": "核心贡献详述（问题→方法→结果，可使用 <br> 换行）",
          "technical_highlights": "技术亮点",
          "differentiation": "与现有工作的差异",
          "code_availability": "代码/复现信息",
          "application_direction": "落地方向",
          "difficulty": "Low"
        }}
      ],
      "worth_knowing": [
        {{
          "title": "论文标题",
          "url": "链接",
          "one_line_summary": "一句话总结",
          "highlights": "亮点",
          "application_tip": "落地提示"
        }}
      ]
    }}
  ],
  "top_papers": [
    {{"title": "论文标题", "url": "链接", "reason": "推荐理由（2-3句话说明为什么值得深入阅读）"}}
  ]
}}

# 约束
- 仅返回纯 JSON，绝对不要包含其他文本或 Markdown 代码块标记。
- 评分要客观严格，体现区分度。
- 论文分析要专业、具体、有技术深度。

# 待分析论文列表：
{papers_text}
"""

    headers = {"Authorization": f"Bearer {LLM_API_KEY}", "Content-Type": "application/json"}
    payload = {
        "model": LLM_MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.2,
        "max_tokens": 4000
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
        
        # 注意：这里的 URL 需要替换为你实际的 GitHub Pages 地址
        # 格式通常为: https://<你的GitHub用户名>.github.io/<你的仓库名>/
        github_pages_url = f"https://{os.getenv('GITHUB_REPOSITORY_OWNER')}.github.io/{os.getenv('GITHUB_REPOSITORY').split('/')[-1]}/"
        push_notification(github_pages_url)
    else:
        print("❌ 大模型未能返回有效的 JSON 数据，请检查 API 配置或重试。")
