import os

from dotenv import load_dotenv
from langchain_community.chat_models import ChatTongyi
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

chat_prompt_template = ChatPromptTemplate.from_messages(
    [
        ("system", "你是一个边塞诗人，可以作诗。"),
        MessagesPlaceholder("history"),
        ("human", "请再来一首唐诗"),
    ]
)

history_data = [
    ("human", "你来写一个唐诗"),
    ("ai", "床前明月光，疑是地上霜，举头望明月，低头思故乡"),
    ("human", "好诗再来一个"),
    ("ai", "锄禾日当午，汗滴禾下锄，谁知盘中餐，粒粒皆辛苦"),
]

# StringPromptValue  to_string()
prompt_text = chat_prompt_template.invoke({"history": history_data})
print(prompt_text)

load_dotenv()
model = ChatTongyi(
    model="qwen3-max",
    # model="qwen-max",
    api_key=os.getenv("DASHSCOPE_API_KEY"),
)

# 调用invoke向模型提问
res = model.invoke(prompt_text)

print(res, type(res))
