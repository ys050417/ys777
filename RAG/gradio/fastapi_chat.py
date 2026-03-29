# fastapi_chat.py 后端核心代码
from fastapi import FastAPI, Body
from openai import AsyncOpenAI
from typing import List
from fastapi.responses import StreamingResponse

# 初始化FastAPI应用
app = FastAPI()

# 配置AsyncOpenAI客户端，对接本地Ollama服务
API_KEY = 'ollama'  # Ollama固定值
BASE_URL = 'http://localhost:11434/v1'  # Ollama默认OpenAI兼容接口地址
aclient = AsyncOpenAI(api_key=API_KEY, base_url=BASE_URL)

# 初始化对话消息列表
messages = []

# 定义POST接口：/chat，接收前端请求，返回流式响应
@app.post("/chat")
async def chat(
    query: str = Body(..., description="用户输入的问题"),
    sys_prompt: str = Body("你是一个有用的助手。", description="系统提示词，定义模型角色"),
    history: List = Body([], description="历史对话记录"),
    history_len: int = Body(1, description="保留历史对话的轮数"),
    temperature: float = Body(0.5, description="模型采样温度，越低越严谨"),
    top_p: float = Body(0.5, description="采样概率，越低越集中"),
    max_tokens: int = Body(None, description="模型最大生成token数")
):
    global messages
    # 控制历史对话长度，避免上下文过长
    if history_len > 0:
        history = history[-2 * history_len:]  # 一轮对话包含user+assistant，故×2
    # 重置消息列表，拼接系统提示+历史对话+当前问题
    messages.clear()
    messages.append({"role": "system", "content": sys_prompt})
    messages.extend(history)
    messages.append({"role": "user", "content": query})

    # 调用Ollama的Qwen2.5模型，流式生成响应
    response = await aclient.chat.completions.create(
        model="qwen2.5:0.5b",  # Ollama中运行的Qwen2.5模型名称（需与Ollama一致）
        messages=messages,
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
        stream=True  # 流式输出，前端实时显示
    )

    # 流式生成响应内容，返回给前端
    async def generate_response():
        async for chunk in response:
            chunk_msg = chunk.choices[0].delta.content
            if chunk_msg:  # 过滤空内容
                yield chunk_msg

    # 返回流式响应，媒体类型为纯文本
    return StreamingResponse(generate_response(), media_type="text/plain")

# 启动FastAPI服务
if __name__ == "__main__":
    import uvicorn
    # 监听0.0.0.0，端口6066（前端需对接此端口），开启日志
    uvicorn.run(app, host="0.0.0.0", port=6066, log_level="info")