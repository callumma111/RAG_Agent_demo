import os

from dotenv import load_dotenv
from langchain_community.chat_models.tongyi import ChatTongyi
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.chat_history import InMemoryChatMessageHistory

# 初始化模型、提示模板和解析器
load_dotenv()
model = ChatTongyi(
    # model="qwen3-max",
    model="deepseek-v3.2",
    api_key=os.getenv("DASHSCOPE_API_KEY"),
)
# prompt = PromptTemplate.from_template(
#     "你需要根据会话历史回应用户问题。对话历史：{chat_history}，用户提问：{input}，请回答"
# )

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "你需要根据会话历史回应用户问题。对话历史："),
        MessagesPlaceholder("chat_history"),
        ("human", "请回答如下问题：{input}")
    ]
)

str_parser = StrOutputParser()


def print_prompt(full_prompt):
    print("=" * 20, full_prompt.to_string, "=" * 20, )
    return full_prompt


# 构建基础调用链
base_chain = prompt | print_prompt | model | str_parser

# 定义会话历史存储和获取函数
store = {}  # key为session_id，value为InMemoryChatMessageHistory对象


def get_history(session_id):
    if session_id not in store:
        store[session_id] = InMemoryChatMessageHistory()
    return store[session_id]


# 构建带对话历史的增强链
conversation_chain = RunnableWithMessageHistory(
    base_chain,
    get_history,
    input_messages_key="input",
    history_messages_key="chat_history"
)

if __name__ == '__main__':
    # 配置会话ID，用于隔离不同用户的对话历史
    session_config = {
        "configurable": {
            "session_id": "user_001"
        }
    }

    # 多轮对话测试
    res = conversation_chain.invoke(input={"input": "小明有2个猫"}, config=session_config)
    print("第1次执行：", res)

    res = conversation_chain.invoke(input={"input": "小刚有1只狗"}, config=session_config)
    print("第2次执行：", res)

    res = conversation_chain.invoke(input={"input": "总共有几个宠物"}, config=session_config)
    print("第3次执行：", res)
