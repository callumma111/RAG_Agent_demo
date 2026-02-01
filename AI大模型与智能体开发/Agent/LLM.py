import os

from dotenv import load_dotenv
from langchain_community.chat_models import ChatTongyi

load_dotenv()

# 聊天模型
llm = ChatTongyi(
    model="qwen3-max",
    # model="qwen-max",
    api_key=os.getenv("DASHSCOPE_API_KEY"),
)
