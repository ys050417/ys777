# 导入所需库
from fastapi import FastAPI, Body
from openai import AsyncOpenAI  # 异步OpenAI客户端，适配FastAPI异步特性
from typing import List
from fastapi.responses import StreamingResponse

# 1. 初始化FastAPI应用实例
app = FastAPI(
    title="Qwen2.5 ChatBot FastAPI后端",
    description="基于Ollama+Qwen2.5的流式对话后端API",
    version="1.0.0"
)

# 2. 初始化异步OpenAI客户端，对接本地Ollama服务
# Ollama默认API地址：http://localhost:11434/v1，api_key任意填写（如ollama）
API_KEY = "ollama"
BASE_URL = "http://localhost:11434/v1"
aclient = AsyncOpenAI(
    api_key=API_KEY,
    base_url=BASE_URL
)

# 3. 初始化对话消息列表（用于存储上下文）
messages = []


# 4. 定义核心对话接口：POST请求 /chat
@app.post("/chat", summary="流式对话接口", description="接收用户输入，返回模型流式回复")
async def chat(
        query: str = Body(..., description="用户的输入问题"),
        sys_prompt: str = Body("你是一个有用的助手。", description="系统提示词，定义模型角色"),
        history: List = Body([], description="历史对话记录，格式：[{role: user/assistant, content: 内容}]"),
        history_len: int = Body(1, description="保留历史对话的轮数，1表示保留最近1轮"),
        temperature: float = Body(0.5, description="LLM采样温度，0-2，值越高越随机"),
        top_p: float = Body(0.5, description="LLM采样top_p，0-1，值越高多样性越强"),
        max_tokens: int = Body(None, description="模型最大生成token数，None表示不限制")
):
    global messages  # 引用全局消息列表
    # 控制历史对话长度，只保留最近N轮（每轮包含user+assistant，故×2）
    if history_len > 0:
        history = history[-2 * history_len:]

    # 重置并构建新的消息列表（系统提示词+历史对话+当前用户输入）
    messages.clear()
    messages.append({"role": "system", "content": sys_prompt})  # 系统角色
    messages.extend(history)  # 追加历史对话
    messages.append({"role": "user", "content": query})  # 追加当前用户输入

    # 调用Ollama的Qwen2.5模型，流式生成回复
    response = await aclient.chat.completions.create(
        model="qwen2.5:0.5b",  # 本地Ollama加载的模型名称，需与实际一致
        messages=messages,
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
        stream=True  # 开启流式输出，核心！
    )

    # 定义流式生成响应的函数
    async def generate_response():
        async for chunk in response:
            # 提取流式返回的内容，过滤空值
            chunk_msg = chunk.choices[0].delta.content
            if chunk_msg:
                yield chunk_msg  # 逐块返回内容

    # 返回流式响应，媒体类型为纯文本
    return StreamingResponse(generate_response(), media_type="text/plain")


# 5. 启动服务
if __name__ == "__main__":
    import uvicorn

    # 启动uvicorn服务器，监听0.0.0.0:6066，允许局域网访问
    uvicorn.run(
        app="fastapi_chatbot:app",
        host="0.0.0.0",
        port=6066,
        log_level="info"  # 日志级别，info即可
    )