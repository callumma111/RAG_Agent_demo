from langchain_community.chat_models.tongyi import ChatTongyi
import os
from dotenv import load_dotenv

load_dotenv()  # 加载.env文件中的环境变量

# qwen3-max是聊天模型
model = ChatTongyi(
    model="qwen3-max",
    # model="qwen-max",
    api_key=os.getenv("DASHSCOPE_API_KEY"),
)

Messages = [
    ("system", "你是一个边塞诗人"),
    ("human", "给我写一唐诗"),
    ("ai", "新 芽破土向 青崖，细 雨垂珠润 物华。忽 有山禽穿雾 过，衔将 落瓣到云涯 。"),
    ("human", "再写一个")
]

# 调用stream向模型提问
res = model.stream(input=Messages)

for chunk in res:
    print(chunk.content, end=" ", flush=True)
