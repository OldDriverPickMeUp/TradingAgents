import os
from langchain_openai import ChatOpenAI
from langchain_tavily import TavilySearch
from langchain_core.messages import HumanMessage, SystemMessage
from pprint import pprint


def get_config():
    return {
        "backend_url": os.getenv("DASHSCOPE_SDK_BASE_URL"),
        "quick_think_llm": "qwen-plus",
        "api_key": os.getenv("DASHSCOPE_API_KEY"),
    }


def get_stock_news_openai(ticker, curr_date):
    config = get_config()
    llm = ChatOpenAI(base_url=config["backend_url"],
                     api_key=config["api_key"], model=config["quick_think_llm"])
    search_tool = TavilySearch(
        api_key=os.getenv("TAVILY_API_KEY"),
        topic="news"
    )

    llm = llm.bind_tools([search_tool])
    messages = [SystemMessage(
        f"Can you search Social Media for {ticker} from 7 days before {curr_date} to {curr_date}? Make sure you only get the data posted during that period.")]
    has_tool_call = False
    for _ in range(5):
        ai_msg = llm.invoke(messages)
        messages.append(ai_msg)

        if not ai_msg.tool_calls:
            break
        for tool_call in ai_msg.tool_calls:
            tool_msg = search_tool.invoke(tool_call)
            has_tool_call = True
            messages.append(tool_msg)
    if not has_tool_call:
        return ""
    return messages[-1].content


if __name__ == "__main__":
    import dotenv
    dotenv.load_dotenv()
    ticker = "AAPL"
    curr_date = "2025-07-10"
    news = get_stock_news_openai(ticker, curr_date)
    print(news)
