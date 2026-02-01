import os

from dotenv import load_dotenv
from langchain_community.chat_models.tongyi import ChatTongyi
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.vectorstores import InMemoryVectorStore
from langchain_community.embeddings import DashScopeEmbeddings
from langchain_core.runnables import RunnablePassthrough

load_dotenv()
# 初始化大语言模型
model = ChatTongyi(
    model="qwen3-max",
    api_key=os.getenv("DASHSCOPE_API_KEY")
)

# 构建RAG提示模板
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "以我提供的已知参考资料为主，简洁和专业的回答用户问题。参考资料: {context}。"),
        ("user", "用户提问: {input}")
    ]
)

# 初始化内存向量存储和嵌入模型
vector_store = InMemoryVectorStore(embedding=DashScopeEmbeddings(
    model="text-embedding-v4",
    dashscope_api_key=os.getenv("DASHSCOPE_API_KEY")
))

# 准备资料，添加到向量库
vector_store.add_texts([
    "减肥就是要少吃多练",
    "在减脂期间吃东西很重要,清淡少油控制卡路里摄入并运动起来",
    "跑步是很好的运动哦"
])

# 用户提问
input_text = "怎么减肥？"

# langchain中向量存储对象，有一个方法: as_retriever，可以返回一个Runnable接口的子类实例对象
retriever = vector_store.as_retriever(search_kwargs={"k": 2})


# 辅助函数：打印提示词内容
def print_prompt(prompt):
    print(prompt.to_string())
    print("=" * 20)
    return prompt


def format_func(docs: list[Document]):
    if not docs:
        return "无相关参考资料"

    formatted_str = "["
    for doc in docs:
        formatted_str += doc.page_content
    formatted_str += "]"

    return formatted_str


# chain
chain = (
        {"input": RunnablePassthrough(),
         "context": retriever | format_func} | prompt | print_prompt | model | StrOutputParser()
)

res = chain.invoke(input_text)
print(res)

"""
retriever:
    - 输入: 用户的提问      str
    - 输出: 向量库的检索结果  list[Document]
prompt:
    - 输入: 用户的提问 + 向量库的检索结果  dict
    - 输出: 完整的提示词         PromptValue
"""
