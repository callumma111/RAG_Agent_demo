from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
import os
from dotenv import load_dotenv

load_dotenv()  # 加载.env文件中的环境变量

model = ChatOllama(model="qwen3:4b")

Message = [
    SystemMessage("你是一个边塞诗人"),
    HumanMessage("给我写一唐诗"),
    AIMessage("新芽破土向青崖，细雨垂珠润物华。忽有山禽穿雾过，衔将落瓣到云涯。"),
    HumanMessage("再写一个")
]

# 调用stream向模型提问
res = model.stream(input=Message)

for chunk in res:
    print(chunk.content, end="", flush=True)
