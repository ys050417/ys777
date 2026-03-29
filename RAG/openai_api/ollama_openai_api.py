# OpenAI 形式访问 Ollama 大模型 API
from openai import OpenAI

# 加载本地的大模型服务
api_key = 'ollama'  # 固定写 ollama
base_url = 'http://localhost:11434/v1'  # Ollama 默认端口

# 创建客户端
client = OpenAI(
    api_key=api_key,
    base_url=base_url
)

# 发送请求到大模型 —— 流式输出
response = client.chat.completions.create(
    model='qwen2.5:0.5b',  # 你拉取的 Ollama 模型名
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "你好"}
    ],
    max_tokens=150,
    temperature=0.7,
    stream=True  # 流式输出
)

# 逐块打印返回结果
for chunk in response:
    chunk_message = chunk.choices[0].delta.content
    if chunk_message:
        print(chunk_message, end='')