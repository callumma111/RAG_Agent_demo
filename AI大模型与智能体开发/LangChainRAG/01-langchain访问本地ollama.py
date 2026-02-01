from langchain_ollama import OllamaLLM

model = OllamaLLM(model="qwen3:4b")

# 调用invoke向模型提问
res = model.invoke(input="你能做什么")

print(res)
