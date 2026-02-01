from langchain_core.prompts import PromptTemplate
import os

from dotenv import load_dotenv
from langchain_community.llms.tongyi import Tongyi

load_dotenv()  # 加载.env文件中的环境变量
prompt_template = PromptTemplate.from_template(
    "我邻居姓{lastname},刚刚生了一个{gender}，帮我给他取个名字,简单回答"
)

prompt_text = prompt_template.format(lastname="李", gender="女孩")

print(prompt_text)

# 不用qwen3-max，qwen3-max是聊天模型，qwen-max 是大语言模型
model = Tongyi(
    model="qwen-plus-2025-12-01",
    # model="qwen-max",
    api_key=os.getenv("DASHSCOPE_API_KEY"),
)

# 调用invoke向模型提问
res = model.invoke(input=prompt_text)

print(res)

