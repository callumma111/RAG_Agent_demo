import os

from dotenv import load_dotenv
from langchain_community.chat_models.tongyi import ChatTongyi
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.vectorstores import InMemoryVectorStore
from langchain_community.embeddings import DashScopeEmbeddings

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

# 检索向量库，获取相关参考资料
result = vector_store.similarity_search(input_text, k=2)
reference_text = "["
for doc in result:
    reference_text += doc.page_content
reference_text += "]"

# 辅助函数：打印提示词内容
def print_prompt(prompt):
    print(prompt.to_string())
    print("="*20)
    return prompt

# 构建完整调用链
chain = prompt | print_prompt | model | StrOutputParser()

# 执行调用并输出结果
res = chain.invoke({"input": input_text, "context": reference_text})
print(res)