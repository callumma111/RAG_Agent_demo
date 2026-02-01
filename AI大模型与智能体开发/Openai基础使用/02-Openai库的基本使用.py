import os

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()  # 加载.env文件中的环境变量

#  获取client对象
client = OpenAI(
    # 若没有配置环境变量，请用百炼API Key将下行替换为：api_key="sk-xxx",
    api_key=os.getenv("DASHSCOPE_API_KEY"),
    # api_key='sk-6b89c6e782df478782a04a97b93ecea7',
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
)

#  调用模型
completion = client.chat.completions.create(
    # model="qwen3-max",
    model="deepseek-v3.2",
    messages=[
        {"role": "system", "content": "你是一个python变成专家，并且不说废话"},
        {"role": "assistant", "content": "好的，我是编写专家，你要问什么，并且不说废话"},
        {"role": "user", "content": "你是谁？"},
    ],
)
#  处理结果
print(completion.choices[0].message.content)
