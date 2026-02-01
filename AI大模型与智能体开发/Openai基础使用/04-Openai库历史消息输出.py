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
    model="deepseek-v3.2",
    messages=[
        {"role": "system", "content": "你是一个AI助理，回答非常简介"},
        {"role": "user", "content": "小明有两条狗"},
        {"role": "assistant", "content": "好的"},
        {"role": "user", "content": "小红有三条狗"},
        {"role": "assistant", "content": "好的"},
        {"role": "user", "content": "一共有几条狗"},
    ],
    stream=True  # 开启流式输出
)
#  处理结果
for chunk in completion:
    print(
        chunk.choices[0].delta.content,
        end=" ",
        flush=True  # 立即刷新缓冲区
)
