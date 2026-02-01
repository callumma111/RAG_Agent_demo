import os

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_community.chat_models import ChatTongyi
from langchain_core.tools import tool

load_dotenv()
model = ChatTongyi(
    model="qwen3-max",
    # model="qwen-max",
    api_key=os.getenv("DASHSCOPE_API_KEY"),
)


@tool(description="查询天气")
def get_weather() -> str:
    return "晴天"


agent = create_agent(
    model=model,  # 智能体的大脑LLM
    tools=[get_weather],  # 向智能体提供工具列表
    system_prompt="你是一个聊天助手，可以回答用户问题。",
)
res = agent.invoke(
    {
        "messages": [
            {"role": "user", "content": "明天深圳的天气如何？"},
        ]
    }
)
for msg in res["messages"]:
    print(type(msg).__name__, msg.content)
