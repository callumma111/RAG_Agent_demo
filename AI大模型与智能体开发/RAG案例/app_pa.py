import time
import streamlit as st

import config_data
from rag import RagService  # 从rag.py导入RagService

# 标题
st.title("智能客服")
st.divider()  # 分隔符

# 初始化对话历史
if "message" not in st.session_state:
    st.session_state["message"] = [{"role": "assistant", "content": "你好，有什么可以帮助你？"}]

# 初始化RAG服务
if "rag" not in st.session_state:
    st.session_state["rag"] = RagService()

# 渲染所有对话历史
for message in st.session_state["message"]:
    st.chat_message(message["role"]).write(message["content"])

# 在页面最下方提供用户输入栏
prompt = st.chat_input()

if prompt:
    # 在页面输出用户的提问
    st.chat_message("user").write(prompt)
    st.session_state["message"].append({"role": "user", "content": prompt})

    ai_res_list = []
    with st.spinner("AI思考中..."):
        # 调用RAG服务的流式接口
        res_stream = st.session_state["rag"].chain.stream({"input": prompt}, config_data.session_config)

        # 捕获生成器的输出，同时缓存到列表
        def capture(generator, cache_list):
            for chunk in generator:
                cache_list.append(chunk)
                yield chunk

        # 流式输出AI回复
        st.chat_message("assistant").write_stream(capture(res_stream, ai_res_list))
        # 将完整回复保存到对话历史
        st.session_state["message"].append({"role": "assistant", "content": "".join(ai_res_list)})