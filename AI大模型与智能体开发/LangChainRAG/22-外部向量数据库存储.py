import os

from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_community.embeddings import DashScopeEmbeddings
from langchain_community.document_loaders import CSVLoader

load_dotenv()
# 初始化内存向量存储

# Chroma 向量数据库（轻量级的）
# 确保 langchain-chroma chromadb 这两个库安装了的，没有的话请pip install
vector_store = Chroma(
    collection_name="test",            # 当前向量存储起个名字，类似数据库的表名称
    embedding_function=DashScopeEmbeddings(
        dashscope_api_key=os.getenv("DASHSCOPE_API_KEY")
    ),  # 嵌入模型
    persist_directory="./chroma_db"   # 指定数据存放的文件夹
)
#
# # 加载CSV文档
# loader = CSVLoader(
#     file_path="./data/info.csv",
#     encoding="utf-8",
#     source_column="source",  # 指定每条数据的来源字段
# )
# documents = loader.load()
#
# # 向量存储的新增、删除、检索
# vector_store.add_documents(
#     documents=documents,
#     ids=["id"+str(i) for i in range(1, len(documents)+1)]  # 为每个文档生成唯一ID
# )
#
# # 删除指定ID的文档
# vector_store.delete(["id1", "id2"])

# 相似性检索
result = vector_store.similarity_search(
    query="Python是不是简单易学呀",
    k=3,  # 指定返回结果的数量
    filter={"source": "黑马程序员"}
)

print(result)
