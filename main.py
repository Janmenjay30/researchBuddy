from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
import os
from datetime import datetime

load_dotenv()

llm=ChatGoogleGenerativeAI(
    model="gemini-3.1-flash-lite-preview",
    temperature=0.3,
    max_output_tokens=2048,
    top_p=0.95,)


from langchain_community.tools import DuckDuckGoSearchResults

search = DuckDuckGoSearchResults()


def report_to_text(report):
    if isinstance(report, str):
        return report

    if isinstance(report, list):
        parts = []
        for item in report:
            if isinstance(item, dict) and "text" in item:
                parts.append(str(item["text"]))
            else:
                parts.append(str(item))
        return "\n\n".join(parts)

    return str(report)


def save_report_to_file(topic, report):
    os.makedirs("reports", exist_ok=True)

    safe_topic = "".join(c if c.isalnum() or c in ("-", "_") else "_" for c in topic).strip("_")
    safe_topic = safe_topic or "research_topic"

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    file_name = f"{safe_topic}_{timestamp}.txt"
    file_path = os.path.join("reports", file_name)

    with open(file_path, "w", encoding="utf-8") as f:
        f.write(report_to_text(report))

    return file_path

def create_research_plan(topic):
    prompt = f"""
You are a research assistant. 
Your task is to create a research plan for the topic: {topic}.
return only the queries as a list
"""
    return llm.invoke(prompt).content


def search_web(query):
    results=[]

    if isinstance(query, list):
        query_items = query
    else:
        query_items = str(query).split("\n")

    for q in query_items:
        if isinstance(q, dict):
            text = q.get("query") or q.get("q") or q.get("text") or str(q)
        else:
            text = str(q)

        for line in text.split("\n"):
            clean_query = line.strip().lstrip("- ").strip()
            if not clean_query:
                continue
            try:
                results.append(search.run(clean_query))
            except Exception as e:
                results.append(f"Search failed for '{clean_query}': {e}")

    return results


def analyze_results(topic,results):
    prompt = f"""
you are a research assistant.

Topic:{topic}

Search results:
{results}

extract key insights and facts
"""
    return llm.invoke(prompt).content


def generate_report(topic,insights):
    prompt = f"""
Create a structured research report

Topic: {topic}

Insights:
{insights}

1. Overview
2. Key Developments
3. Important Companies
4. Future Outlook
    """
    return llm.invoke(prompt).content


def research_agent(topic):
    print("🧠 Planning research...")
    queries = create_research_plan(topic)

    print("🔎 Searching web...")
    results = search_web(queries)

    print("📊 Analyzing results...")
    insights = analyze_results(topic, results)

    print("📝 Writing report...")
    report = generate_report(topic, insights)

    return report


while True:
    topic=input("Enter a research topic (or 'exit' to quit): ")
    if topic.lower() == 'exit':
        break
    report = research_agent(topic)
    output_file = save_report_to_file(topic, report)
    print("\n📄 Research Report:\n"
          "-------------------\n")
    print(report)
    print(f"\n💾 Saved report to: {output_file}")
