from langchain_community.embeddings import DashScopeEmbeddings
import os
from dotenv import load_dotenv

load_dotenv()  # 加载.env文件中的环境变量
# 创建模型对象，默认是text-embeddings-v1
model = DashScopeEmbeddings(
    dashscope_api_key=os.getenv("DASHSCOPE_API_KEY")
)

# 不用invoke，stream
print(model.embed_query("我喜欢你"))
print(model.embed_documents(["我喜欢你", "我不喜欢你", "哈哈哈"]))
