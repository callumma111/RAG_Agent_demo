import os

from dotenv import load_dotenv
from langchain_community.llms.tongyi import Tongyi

load_dotenv()  # 加载.env文件中的环境变量

# 不用qwen3-max，qwen3-max是聊天模型，qwen-max 是大语言模型
model = Tongyi(
    model="qwen-plus-2025-12-01",
    # model="qwen-max",
    api_key=os.getenv("DASHSCOPE_API_KEY"),
)

# 调用stream向模型提问
res = model.stream(input="你能做什么")

for chunk in res:
    print(chunk, end=" ", flush=True)
