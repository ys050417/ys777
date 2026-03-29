from fastapi import FastAPI, Body
from openai import AsyncOpenAI
from typing import List
from fastapi.responses import StreamingResponse

# 初始化FastAPI应用
app = FastAPI()

# 初始化异步OpenAI客户端，对接本地Ollama服务
api_key = 'ollama'
base_url = 'http://localhost:11434/v1'
aclient = AsyncOpenAI(api_key=api_key, base_url=base_url)

# 初始化对话列表
messages = []


# 定义/chat接口，添加temperature和top_p参数
@app.post("/chat")
async def chat(
        query: str = Body(..., description="用户输入问题"),
        sys_prompt: str = Body("你是一个有用的助手。", description="系统提示词"),
        history: List = Body([], description="历史对话记录"),
        history_len: int = Body(1, description="保留历史对话的轮数"),
        # 核心：添加temperature和top_p参数，设置合理默认值和范围说明
        temperature: float = Body(0.5, ge=0.01, le=2.0, description="温度参数，控制生成随机性，0.01最确定，2.0最随机"),
        top_p: float = Body(0.5, ge=0.01, le=1.0, description="采样参数，控制生成多样性，0.01最集中，1.0最多样"),
        max_tokens: int = Body(1024, description="最大生成token数")
):
    global messages
    # 控制历史记录长度
    if history_len > 0:
        history = history[-2 * history_len:]
    # 重构对话消息
    messages.clear()
    messages.append({"role": "system", "content": sys_prompt})
    messages.extend(history)
    messages.append({"role": "user", "content": query})

    # 调用Ollama接口，传递temperature和top_p参数
    response = await aclient.chat.completions.create(
        model="qwen2.5:0.5b",
        messages=messages,
        max_tokens=max_tokens,
        temperature=temperature,  # 透传温度参数
        top_p=top_p,  # 透传top_p参数
        stream=True
    )

    # 流式返回结果
    async def generate_response():
        async for chunk in response:
            chunk_msg = chunk.choices[0].delta.content
            if chunk_msg:
                yield chunk_msg

    return StreamingResponse(generate_response(), media_type="text/plain")


# 启动服务（PyCharm中直接运行该文件即可）
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=6066, log_level="info")