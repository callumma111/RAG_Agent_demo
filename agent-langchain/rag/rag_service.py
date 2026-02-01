from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from rag.vector_store import VectorStoreService
from model.factory import chat_model
from utils.prompt_loader import load_rag_prompts


class RagSummarizeService:
    #  初始化 RAG服务
    def __init__(self):
        self.vector_store = VectorStoreService()  # 初始化向量库
        self.retriever = self.vector_store.get_retriever()  # 获取检索器
        self.prompt_text = load_rag_prompts()  # 加载提示词
        self.prompt_template = PromptTemplate.from_template(self.prompt_text)  # 创建提示词模板
        self.model = chat_model  # 创建模型
        self.chain = self._init_chain()  # 创建链

    def _init_chain(self):
        chain = self.prompt_template | self.model | StrOutputParser()
        return chain

    # 检索文档
    def retrieve_docs(self, query: str) -> list[Document]:
        return self.retriever.invoke(query)

    # 总结
    def rag_summarize(self, query: str) -> str:
        # key: input, for user query
        # key: context, 参考资料
        input_dict = {}

        context_docs = self.retrieve_docs(query)
        context = ""
        counter = 0
        for doc in context_docs:
            counter += 1
            context += f"【参考资料{counter}】：参考资料：{doc.page_content} | 参考元数据：{doc.metadata}\n"
        input_dict["input"] = query
        input_dict["context"] = context

        return self.chain.invoke(input_dict)


# for testing
if __name__ == '__main__':
    rag = RagSummarizeService()

    print(rag.rag_summarize("小户型适合哪种扫地机器人？"))
