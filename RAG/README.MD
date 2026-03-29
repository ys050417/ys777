# 使⽤开源⼤语⾔模型 Qwen2.5 搭建 — RAG 、 agent

​		本案例将在基于 Ollama 完成 Qwen2.5 本地部署的基础上，进一步结合 Ollama，围绕推理优化与加速、对话机器人搭建、LangChain 链式调用与 Agent 应用、RAG 的文本加载与分割、常见向量数据库的使用，以及 RAG 的向量化与效果优化等内容展开实践，完整呈现一个可落地的本地大模型应用流程。

## 一、ollama安装及模型下载

### （一）下载网址以及安装

- 默认安装到C盘

```
https://ollama.com/download
```

### （二）常用命令

#### 1.启动终端

```
win+r
```

#### 2.启动 Ollama 服务（必须先运行）

- 作用：启动本地 Ollama 后台服务，不运行这个命令，无法使用模型
- 运行后保持终端开启，关闭则服务停止

```
ollama serve
```

#### 3.查看已安装的模型

```
ollama list
```

#### 4.下载 / 拉取模型

```
ollama pull 模型名称
```

#### 5.直接运行模型（自动下载 + 启动对话）

```
ollama run 模型名称
```

#### 6.删除不需要的模型

```
ollama rm 模型名称
```

#### 7.查看 Ollama 版本

```
ollama --version
```

#### 8.停止 Ollama 服务

```
ollama serve
```

## 二、OpenAI 形式访问 ollama ⼤模型 API

### （一）安装依赖包

在 PyCharm 底部打开 Terminal，执行命令

```
pip install openai==1.71.0
```

### （二）创建ollama_openai_api.py

```
from openai import OpenAI

# 加载本地的大模型服务
api_key = 'ollama'  # 固定写 ollama
base_url = 'http://localhost:11434/v1'  # Ollama 默认端口

# 创建客户端
client = OpenAI(
    api_key=api_key,
    base_url=base_url
)

# 发送请求到大模型
response = client.chat.completions.create(
    model='qwen2.5:0.5b',  # 拉取的 Ollama 模型名
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "你好"}
    ],
    max_tokens=150,
    temperature=0.7,
    stream=True  
)

# 逐块打印返回结果
for chunk in response:
    chunk_message = chunk.choices[0].delta.content
    if chunk_message:
        print(chunk_message, end='')
```

### （三）运行前必须检查 3 点

- Ollama 必须正在运行
- 模型已经拉取
- 端口 11434 没被占用

### （四）运行

```
右键+Run 'ollama_openai_api'
```

## 三、对话机器⼈ fastapi

### （一）fastapi 介绍

- FastAPI 是一个现代化、高性能的 Python Web 框架，专注于快速开发和生产就绪的 API。
- 它的性能媲美 Node.js 和 Go ，是基于 Python 的框架中最快的之⼀。

- 具有高性能与标准支持、类型提示与数据验证、依赖注入系统、安全性与认证、异步编程与性能优化*、*数据库集成与操作、自动化文档与调试、部署与生产就绪等功能。

### （二）安装相关依赖

- FastAPI+uvicorn（Web框架+服务器）

```
pip install fastapi==0.115.12
```

```
pip install uvicorn==0.34.0
```

- OpenAI客户端（对接Ollama的OpenAI兼容API）

```
pip install openai==1.71.0
```

- 验证Ollama连接

```
pip install requests
```

### （三）fastapi_chatbot.py

```
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
```

### （四）必检配置

|       检查内容       |                           检查方法                           |
| :------------------: | :----------------------------------------------------------: |
|   Ollama 是否启动    | 打开电脑终端，输入`ollama ps`，查看是否有 Qwen2.5 模型在运行，无则执行`ollama run qwen2.5:0.5b` |
|   模型名称是否一致   | 代码中`model="qwen2.5:0.5b"`必须与 Ollama 中实际模型名一致（ollama list） |
|    端口是否被占用    | `6066`（FastAPI）、`11434`（Ollama）若被占用，可修改代码中端口号 |
| 虚拟环境依赖是否完整 |        确保`fastapi`/`uvicorn`/`openai`版本与文档一致        |

### （五）常见问题

|          问题现象           |                           解决方法                           |
| :-------------------------: | :----------------------------------------------------------: |
| 启动时报ModuleNotFoundError |       重新在 PyCharm 终端安装对应包，确保虚拟环境激活        |
| 接口测试返回ConnectionError |          检查 Ollama 是否启动，或 BASE_URL 是否正确          |
|       流式输出无内容        |                     检查模型是否正常响应                     |
|       历史对话不生效        | 确保前端传参`history`的格式为`[{"role":"user","content":"xxx"},{"role":"assistant","content":"xxx"}]` |

## 四、对话机器⼈ streamlit

### （一）streamlit简介

Streamlit 是一款极为便捷的 Python 库，能够帮助开发者快速构建数据驱动的 Web 应用。在本项目中，将利用 Streamlit 搭建聊天界面，并实现与后端的交互，从而提供直观、高效的使用体验。

### （二）安装相关依赖

```
pip install streamlit==1.39.0
pip install requests
pip install fastapi==0.115.12 uvicorn==0.34.0
pip install openai==1.71.0
```

### （三）创建fastapi_backend.py

```
from fastapi import FastAPI, Body
from openai import AsyncOpenAI
from typing import List
from fastapi.responses import StreamingResponse

app = FastAPI()

# 连接本地 Ollama 大模型
api_key = 'ollama'
base_url = 'http://localhost:11434/v1'
aclient = AsyncOpenAI(api_key=api_key, base_url=base_url)

messages = []

@app.post("/chat")
async def chat(
    query: str = Body(..., description="用户输入"),
    sys_prompt: str = Body("你是一个有用的助手。", description="系统提示词"),
    history: List = Body([], description="历史对话"),
    history_len: int = Body(1, description="保留历史对话轮数"),
    temperature: float = Body(0.5, description="温度"),
    top_p: float = Body(0.5, description="top_p"),
    max_tokens: int = Body(None, description="最大token")
):
    global messages
    if history_len > 0:
        history = history[-2 * history_len:]

    messages.clear()
    messages.append({"role": "system", "content": sys_prompt})
    messages.extend(history)
    messages.append({"role": "user", "content": query})

    response = await aclient.chat.completions.create(
        model="qwen2.5:0.5b",
        messages=messages,
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
        stream=True
    )

    async def generate_response():
        async for chunk in response:
            chunk_msg = chunk.choices[0].delta.content
            if chunk_msg:
                yield chunk_msg

    return StreamingResponse(generate_response(), media_type="text/plain")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=6066, log_level="info")
```

### （四）创建聊天界面streamlit_frontend.py

```
import streamlit as st
import requests

# 后端地址（必须和 fastapi 一致）
backend_url = "http://127.0.0.1:6066/chat"

# 页面配置
st.set_page_config(page_title="本地聊天机器人", page_icon="🤖", layout="centered")
st.title("🤖 Qwen2.5 本地对话机器人")

# 清空历史函数
def clear_chat_history():
    st.session_state.history = []

# 侧边栏参数配置
with st.sidebar:
    st.title("参数设置")
    sys_prompt = st.text_input("系统提示词:", value="You are a helpful assistant.")
    history_len = st.slider("保留历史轮数:", 1, 10, 1)
    temperature = st.slider("temperature:", 0.01, 2.0, 0.5, 0.01)
    top_p = st.slider("top_p:", 0.01, 1.0, 0.5, 0.01)
    max_tokens = st.slider("max_tokens:", 256, 4096, 1024, 8)
    stream = st.checkbox("流式输出", value=True)
    st.button("清空聊天历史", on_click=clear_chat_history)

# 初始化历史
if "history" not in st.session_state:
    st.session_state.history = []

# 显示历史消息
for msg in st.session_state.history:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# 用户输入
if prompt := st.chat_input("请输入消息..."):
    with st.chat_message("user"):
        st.markdown(prompt)

    # 构造请求
    data = {
        "query": prompt,
        "sys_prompt": sys_prompt,
        "history_len": history_len,
        "history": st.session_state.history,
        "temperature": temperature,
        "top_p": top_p,
        "max_tokens": max_tokens
    }

    # 请求后端
    response = requests.post(backend_url, json=data, stream=True)
    if response.status_code == 200:
        chunks = ""
        assistant_msg = st.chat_message("assistant")
        msg_placeholder = assistant_msg.markdown("")

        # 流式显示
        for chunk in response.iter_content(chunk_size=None, decode_unicode=True):
            chunks += chunk
            msg_placeholder.markdown(chunks)

        # 保存历史
        st.session_state.history.append({"role": "user", "content": prompt})
        st.session_state.history.append({"role": "assistant", "content": chunks})
```

### （五）运行

**保持后端运行窗口不关闭**，在 PyCharm 中打开一个新的 Terminal，确认新终端已激活和后端相同的虚拟环境，执行启动命令

```
 streamlit run [文件位置]
```

```
 streamlit run E:\python\大模型应用开发\ollama\streamlit\streamlit_frontend.py
```

### （六）易犯错误

不能直接在 PyCharm 里点运行按钮（三角箭头）启动Streamlit ，需要在保持后端运行的基础上，在终端中输入命令启动前端

## 五、对话机器⼈ gradio

### （一）gradio简介

**Gradio** 是由 Hugging Face 推出的一个开源 Python 库，专为快速构建机器学习模型的交互式 Web 应用而设计。它允许开发者通过简单的 Python 接口，将机器学习模型、Python 函数或 API 快速转换为用户友好的 Web 界面，无需任何前端开发经验。

- 核心特点

|     简单易用      |                只需几行代码即可创建交互式界面                |
| :---------------: | :----------------------------------------------------------: |
| 多种输入/输出支持 |           支持文本、图像、音频、视频等多种数据类型           |
|     即时分享      |       通过 *share=True* 参数，快速生成可公开访问的链接       |
|     兼容性强      | 支持主流机器学习框架，如 TensorFlow、PyTorch、Scikit-learn 等 |
|     灵活部署      |       可在本地运行、嵌入 Jupyter Notebook 或部署到云端       |
|     流式传输      |            支持实时数据处理，如视频流、语音识别等            |

### （二）安装依赖包

|                     依赖包                      |                      安装指令                       |
| :---------------------------------------------: | :-------------------------------------------------: |
|             FastAPI+uvicorn（后端）             |    pip install fastapi==0.115.12 uvicorn==0.34.0    |
|                 Gradio（前端）                  |              pip install gradio==5.0.2              |
|      OpenAI（对接Ollama的OpenAI兼容接口）       |             pip install openai==1.71.0              |
| 验证Qwen2.5模型的transformers（若已安装可跳过） | pip install transformers==4.45.0 modelscope==1.18.1 |

### （三）后端fastapi_chat.py

```
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
```

### （四）前端gradio_chat.py

```
# gradio_chat.py 前端核心代码
import gradio as gr
import requests

# 配置FastAPI后端接口地址（与后端的host+port+接口一致）
BACKEND_URL = "http://127.0.0.1:6066/chat"

# 核心函数：对接后端接口，发送请求并接收流式响应
def chat_with_backend(
    prompt,  # 用户当前输入的问题
    history,  # Gradio的聊天历史记录
    sys_prompt,  # 系统提示词
    history_len,  # 保留历史对话轮数
    temperature,  # 采样温度
    top_p,  # 采样概率
    max_tokens,  # 最大生成token数
    stream  # 是否流式输出
):
    # 处理Gradio的历史记录格式，去除多余的metadata字段，适配后端
    history_clean = [{"role": h.get("role"), "content": h.get("content")} for h in history]
    # 构建请求参数，与FastAPI的/chat接口参数一一对应
    request_data = {
        "query": prompt,
        "sys_prompt": sys_prompt,
        "history": history_clean,
        "history_len": history_len,
        "temperature": temperature,
        "top_p": top_p,
        "max_tokens": max_tokens
    }
    # 发送POST请求到后端，开启流式响应
    response = requests.post(BACKEND_URL, json=request_data, stream=True)
    # 接收后端的流式数据并实时返回
    if response.status_code == 200:
        chunks = ""
        if stream:
            for chunk in response.iter_content(chunk_size=None, decode_unicode=True):
                chunks += chunk
                yield chunks  # 流式返回，Gradio实时更新
        else:
            for chunk in response.iter_content(chunk_size=None, decode_unicode=True):
                chunks += chunk
            yield chunks

# 构建Gradio前端界面
with gr.Blocks(fill_width=True, fill_height=True, title="Qwen2.5聊天机器人") as demo:
    # 标签页标题
    with gr.Tab("🤖 Qwen2.5 对话机器人"):
        gr.Markdown("### 📚 基于Qwen2.5+Ollama+FastAPI+Gradio搭建的本地对话机器人")
        # 行布局：左侧参数面板 + 右侧聊天窗口
        with gr.Row():
            # 左侧：参数调节面板（比例1）
            with gr.Column(scale=1, variant="panel"):
                sys_prompt = gr.Textbox(
                    label="📝 系统提示词",
                    value="你是一个专业的人工智能助手，回答简洁、准确、有逻辑。",
                    lines=3
                )
                history_len = gr.Slider(
                    label="📜 保留历史对话轮数",
                    minimum=1, maximum=10, value=1, step=1
                )
                temperature = gr.Slider(
                    label="🌡️ 采样温度 (0.01-2.0)",
                    minimum=0.01, maximum=2.0, value=0.5, step=0.01
                )
                top_p = gr.Slider(
                    label="🎲 采样Top-P (0.01-1.0)",
                    minimum=0.01, maximum=1.0, value=0.5, step=0.01
                )
                max_tokens = gr.Slider(
                    label="🔢 最大生成Token数",
                    minimum=512, maximum=4096, value=1024, step=8
                )
                stream = gr.Checkbox(
                    label="⚡ 流式输出",
                    value=True,
                    info="开启后实时显示回答，关闭后一次性显示"
                )
            # 右侧：聊天窗口（比例10，占主要界面）
            with gr.Column(scale=10):
                # 聊天机器人组件，支持messages格式
                chatbot = gr.Chatbot(
                    type="messages",
                    height=600,
                    bubble_full_width=False,
                    avatar_images=("user.png", "robot.png")  # 可选：自定义头像，无则注释
                )
                # 聊天交互界面，绑定核心函数和参数
                gr.ChatInterface(
                    fn=chat_with_backend,  # 核心交互函数
                    type="messages",  # 消息格式与chatbot一致
                    chatbot=chatbot,  # 绑定聊天窗口
                    # 绑定左侧面板的所有参数
                    additional_inputs=[
                        sys_prompt, history_len, temperature, top_p, max_tokens, stream
                    ]
                )

# 启动Gradio服务
if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,  # 设为True可生成公网链接，无需则False
        inbrowser=True  # 自动打开浏览器
    )
```

## 六、在对话机器人的前后端代码中添加参数配置项

### （一）temperature、top_p简介

- temperature：温度参数，通常控制⽣成内容的随机性。值越低，⽣成内容越确定；值越⾼，⽣成内容越随机。
- top_p：采样参数，控制⽣成时选择词汇的多样性。值越⾼，⽣成结果越具多样性。

### （二）确认本地环境已搭建

```
pip install ollama openai==1.71.0 fastapi==0.115.12 uvicorn==0.34.0 streamlit==1.39.0 gradio==5.0.2
```

### （三）确认 Ollama 服务已启动（PyCharm 终端 / 系统终端均可）

```
ollama run qwen2.5:0.5b
```

### （四）后端fastapi_backend.py

核心：在 FastAPI 的`/chat`接口中添加 temperature 和 top_p 入参，并将参数传递给 Ollama 的 OpenAI 兼容接口，实现参数的后端透传。

```
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
```

### （五）前端streamlit_frontend.py

核心：在 Streamlit 前端添加可交互的参数调节控件（滑块），让用户可以可视化调整`temperature`和`top_p`，并将参数传递给 FastAPI 后端，实现前端调参、后端执行。

```
import streamlit as st
import requests

# 对接FastAPI后端地址
backend_url = "http://127.0.0.1:6066/chat"

# 页面基础配置
st.set_page_config(page_title="Qwen2.5聊天机器人", page_icon="🤖", layout="centered")
st.title("🤖 Qwen2.5 聊天机器人")

# 定义清空聊天历史的函数
def clear_chat_history():
    st.session_state.history = []

# 侧边栏：添加参数调节控件（核心）
with st.sidebar:
    st.header("⚙️ 模型参数配置")
    sys_prompt = st.text_input("系统提示词：", value="你是一个专业的AI助手，回答简洁准确。")
    history_len = st.slider("保留历史对话轮数：", min_value=1, max_value=10, value=1, step=1)
    # 核心1：temperature滑块，范围0.01-2.0（与后端一致），步长0.01
    temperature = st.slider(
        "temperature（生成随机性）：",
        min_value=0.01, max_value=2.0, value=0.5, step=0.01,
        help="值越低，生成越确定/保守；值越高，生成越随机/有创意"
    )
    # 核心2：top_p滑块，范围0.01-1.0（与后端一致），步长0.01
    top_p = st.slider(
        "top_p（生成多样性）：",
        min_value=0.01, max_value=1.0, value=0.5, step=0.01,
        help="值越低，生成词汇越集中；值越高，生成词汇越多样（与temperature配合使用）"
    )
    max_tokens = st.slider("最大生成token数：", min_value=256, max_value=4096, value=1024, step=8)
    stream = st.checkbox("流式输出", value=True)
    st.button("🗑️ 清空聊天历史", on_click=clear_chat_history)

# 初始化聊天历史
if "history" not in st.session_state:
    st.session_state.history = []

# 显示历史聊天记录
for msg in st.session_state.history:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# 接收用户输入
if prompt := st.chat_input("请输入你的问题..."):
    # 显示用户输入
    with st.chat_message("user"):
        st.markdown(prompt)
    # 构建请求参数，包含temperature和top_p
    data = {
        "query": prompt,
        "sys_prompt": sys_prompt,
        "history_len": history_len,
        "history": st.session_state.history,
        "temperature": temperature,  # 传递温度参数
        "top_p": top_p,              # 传递top_p参数
        "max_tokens": max_tokens
    }
    # 向后端发送请求
    response = requests.post(backend_url, json=data, stream=True)
    # 处理响应并显示
    if response.status_code == 200:
        chunks = ""
        assistant_placeholder = st.chat_message("assistant")
        assistant_text = assistant_placeholder.markdown("")
        if stream:
            for chunk in response.iter_content(chunk_size=None, decode_unicode=True):
                chunks += chunk
                assistant_text.markdown(chunks)
        else:
            for chunk in response.iter_content(chunk_size=None, decode_unicode=True):
                chunks += chunk
            assistant_text.markdown(chunks)
        # 更新聊天历史
        st.session_state.history.append({"role": "user", "content": prompt})
        st.session_state.history.append({"role": "assistant", "content": chunks})
    else:
        st.error(f"后端请求失败，状态码：{response.status_code}")
```

### （六）运行

#### 1.确保**后端 fastapi_backend.py**已在 PyCharm 中运行

#### 2.确认 Ollama 服务在运行

```
ollama run qwen2.5:0.5b
```

#### 3.新建 PyCharm 终端 → 执行以下命令启动 Streamlit 前端

```
streamlit run [文件地址]
```

```
 streamlit run E:\python\大模型应用开发\ollama\Parameter\streamlit_frontend.py
```

#### 3.自动打开浏览器页面`http://localhost:8501`，若未打开可手动复制地址访问

### （七）不同搭配效果

端页面左侧模型参数配置栏可直接拖动滑块调整`temperature`和`top_p`，结合文档中的参数应用场景 ，测试不同组合的生成效果，PyCharm 中可通过控制台查看前端日志、后端窗口查看接口请求日志，定位参数传递问题。

|      场景      |            搭配             |                             效果                             |
| :------------: | :-------------------------: | :----------------------------------------------------------: |
|    代码生成    | 低温度 + 低 top_p，追求精准 | 生成的代码语法正确、注释规范，无多余创意性内容，符合固定模式 |
|    创意写作    | 高温度 + 高 top_p，追求多样 |     生成的内容富有想象力，词汇多样，无重复句式，创意性强     |
| 聊天机器人回复 |  中等参数，兼顾一致与多样   |       回复兼顾实用性和多样性，推荐理由具体，无过度发散       |
|  代码注释生成  |   极低参数，追求简洁准确    |            注释简洁明了，符合编程惯例，无多余话语            |

## 七、Langchain-chain 链

LangChain的链（Chain）是将多个组件组合在一起以实现复杂应用的核心机制，允许开发者创建模块化和可复用的工作流。

### （一）什么是链（Chain）

在LangChain中，链是对组件调用的序列，可以包含其他链。它允许将多个模块化组件（如模型、提示模板、记忆模块等）以特定方式组合，从而实现共同的用例。链的基本接口简单而强大，极大地简化了复杂应用的实现，并使其更模块化，便于调试和维护。

### （二）链的工作原理

链的核心运行逻辑是以串行化数据流的方式传递处理结果，前一环节的输出自动作为下一环节的输入，最终完成复杂任务的端到端执行。通过将多个链组合在一起，或将链与其他组件组合，可以构建更复杂的工作流。

### （三）链的类型

|   类型   |                             说明                             |
| :------: | :----------------------------------------------------------: |
| LLMChain | 最基本的链类型，结合提示模板和语言模型。它接受用户输入，格式化提示，并返回模型的响应。 |
|  顺序链  | 如SimpleSequentialChain和SequentialChain，允许按顺序执行多个步骤，处理多个输入和输出。 |
| 自定义链 |    开发者可以通过继承Chain类来实现自定义链，满足特定需求     |

### （四）安装 Langchain 相关核心依赖

|                   依赖库                    |                 安装命令                 |
| :-----------------------------------------: | :--------------------------------------: |
|               核心Langchain库               |      pip install langchain==0.3.23       |
|        OpenAI接口适配（对接ollama）         |   pip install langchain-openai==0.3.12   |
| 检索链所需依赖（Embedding+向量库+社区组件） | pip install sentence-transformers==3.3.0 |
| 检索链所需依赖（Embedding+向量库+社区组件） |       pip install faiss-cpu==1.9.0       |
| 检索链所需依赖（Embedding+向量库+社区组件） | pip install langchain-huggingface==0.1.2 |
| 检索链所需依赖（Embedding+向量库+社区组件） | pip install langchain-community==0.3.21  |

### （五）下载 Embedding 模型（用于检索链）

```
from modelscope.hub.snapshot_download import snapshot_download
# 下载Embedding模型到本地models目录
emb_model_dir = snapshot_download('AI-ModelScope/bge-large-zh-v1.5', cache_dir='models')
print("Embedding模型下载完成，路径：", emb_model_dir)
```

### （六）实现Langchain-chain 三大核心链

```
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# 通用配置：对接ollama的Qwen2.5模型（OpenAI兼容接口）
from langchain_openai import ChatOpenAI

# 初始化大语言模型（ollama本地服务，默认端口11434）
chat_model = ChatOpenAI(
    openai_api_key="ollama",
    base_url="http://localhost:11434/v1",
    model="qwen2.5:0.5b"
)

# 1. 基础LLMChain（新版本LCEL写法）
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

system_message = "你是一个贴心的智能助手，回答简洁易懂。"
human_message = "{user_question}"
chat_prompt = ChatPromptTemplate.from_messages([
    ("system", system_message),
    ("human", human_message),
])
llm_chain = chat_prompt | chat_model | StrOutputParser()

# 2. 检索链RetrievalQA（RAG核心）
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import FakeEmbeddings
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA

# 自动创建测试文件
if not os.path.exists("sanguoyanyi.txt"):
    with open("sanguoyanyi.txt", "w", encoding="utf-8") as f:
        f.write("三国演义中刘备的五虎上将是：关羽、张飞、赵云、马超、黄忠。")

# 加载文档
loader = TextLoader("sanguoyanyi.txt", encoding='utf-8')
docs = loader.load()
text_splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=20)
chunks = text_splitter.split_documents(docs)

# 离线嵌入，不联网
embedding = FakeEmbeddings(size=1024)

# 构建向量库
vs = FAISS.from_documents(chunks, embedding)
retriever = vs.as_retriever()

# 提示模板
qa_prompt = ChatPromptTemplate.from_messages([
    ("system", "根据以下已知信息回答用户问题，仅用已知信息，不要编造：\n{context}"),
    ("human", "{question}"),
])

# 构建RAG链
retrieval_qa_chain = RetrievalQA.from_chain_type(
    llm=chat_model,
    chain_type="stuff",
    retriever=retriever,
    chain_type_kwargs={"prompt": qa_prompt}
)

# 3. 自定义链
custom_prompt = ChatPromptTemplate.from_template("说出一句包含{topic}的古诗，仅输出诗句，不要多余内容。")
custom_chain = custom_prompt | chat_model | StrOutputParser()

# ==================== 运行 ====================
if __name__ == "__main__":
    # 1 基础链
    print("=== LLMChain测试结果 ===")
    response = llm_chain.invoke({"user_question": "你好，介绍一下你自己"})
    print(response)

    # 2 RAG检索链
    print("\n=== RetrievalQA链测试结果 ===")
    qa_response = retrieval_qa_chain.invoke({"query": "五虎上将有哪些？"})
    print(qa_response["result"])

    # 检索文档
    related_docs = retriever.invoke("五虎上将有哪些？")
    print("\n=== 检索到的相关文档 ===")
    for i, doc in enumerate(related_docs):
        print(f"文档{i + 1}：{doc.page_content}")

    # 3 自定义链
    print("\n=== 自定义链测试结果 ===")
    poem_response = custom_chain.invoke({"topic": "花"})
    print(poem_response)
```

### （七）成语接龙实现

```
# 解决OMP报错（必须放第一行）
import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# 模型配置（你原来的配置 + 超时设置）
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
import time

chat_model = ChatOpenAI(
    openai_api_key="ollama",
    base_url="http://localhost:11434/v1",
    model="qwen2.5:0.5b",
    temperature=0.1,  # 降低随机性
    timeout=30  # 30秒超时，防止卡住
)

# ======================= 极简提示词（必回复） =======================
prompt = ChatPromptTemplate.from_template("""
成语接龙：{idiom}
接最后一个字，只输出一个四字成语，不会就说“接不上”。
""")

chain = prompt | chat_model | StrOutputParser()


# ======================= 游戏逻辑（加加载提示） =======================
def play():
    print("=== 纯AI大模型成语接龙（无本地词库）===")
    print("输入 exit 退出\n")

    while True:
        user = input("你：").strip()

        if user == "exit":
            print("=== 游戏结束 ===")
            break

        if len(user) != 4:
            print("我：请输入四字成语！\n")
            continue

        print("我正在思考...")
        try:
            res = chain.invoke({"idiom": user})
            # 清理输出，只保留成语
            res = res.strip().split()[0] if res else "接不上"
            print(f"我：{res}\n")
        except Exception as e:
            print(f"我：接不上（错误：{e}）\n")


if __name__ == "__main__":
    play()
```

## 八、Langchain-agent 代理

### （一）Langchain-agent 代理简介

在 Langchain 框架中，Agent（代理）可以理解为一类具备智能决策能力的执行单元。它能根据用户输入的指令或所处的环境上下文，自主判断并调用合适的工具——例如搜索引擎、数据库、API 等——来完成相应任务。这类代理通过预设的逻辑流程或通过学习获得的策略，协助开发者实现自动化、动态适应，并能感知上下文的计算与决策过程。

### （二）核心依赖

```
pip install langchain==0.3.23 langchain-openai==0.3.12 pydantic==2.8.2 requests==2.32.3
```

### （三）验证Python 解释器和依赖

```
import langchain
import openai
import requests
import pydantic
print("langchain版本：", langchain.__version__)
print("环境验证成功！")
```

### （四）申请工具 API

- 打开心知天气官网：https://www.seniverse.com/，注册账号并登录

- 进入控制台，申请免费版 API，获取专属api_key
- 验证 API 可用性：可在 PyCharm 终端用 curl 测试

```
curl "https://api.seniverse.com/v3/weather/now.json?key=your_api_key&location=成都&language=zh-Hans&unit=c"
```

### （五）创建agent_weather_demo.py

```
# 工具调用依赖
import requests
from pydantic import Field
# Langchain Agent核心依赖
from langchain_openai import ChatOpenAI
from langchain.agents import Tool, create_react_agent, AgentExecutor
from langchain_core.prompts import PromptTemplate

# 定义心知天气API工具类
class WeatherTool:
    city: str = Field(description="City name, include city and county")

    def __init__(self, api_key) -> None:
        self.api_key = api_key

    def run(self, city):
        city = city.split("\n")[0]
        url = f"https://api.seniverse.com/v3/weather/now.json?key={self.api_key}&location={city}&language=zh-Hans&unit=c"
        try:
            response = requests.get(url)
            if response.status_code == 200:
                data = response.json()
                weather = data["results"][0]["now"]["text"]
                tem = data["results"][0]["now"]["temperature"]
                return f"{city}的天气是{weather}，温度是{tem}°C"
            else:
                return f"无法获取{city}的天气信息，API返回状态码：{response.status_code}"
        except Exception as e:
            return f"天气查询失败：{str(e)}"

# ====================== 替换为自己的心知天气API Key ======================
API_KEY = "SBJVysU9a4KvOtgHs"
weather_tool = WeatherTool(API_KEY)

# ====================== 模型配置（ollama 本地 Qwen2.5）======================
chat_model = ChatOpenAI(
    openai_api_key="ollama",
    base_url="http://localhost:11434/v1",
    model="qwen2.5:0.5b",
    temperature=0  # 固定输出，减少格式错误
)

# ====================== 工具封装 ======================
tools = [
    Tool(
        name="weather check",
        func=weather_tool.run,
        description="检查指定城市的天气情况，输入参数为城市名称"
    )
]

# ====================== 强化版提示词，强制模型遵守格式 ======================
template = """
你必须严格按照下面的格式输出，绝对不能改变格式！
必须包含：Thought、Action、Action Input、Observation、Final Answer

可用工具：
{tools}

严格格式如下：
Question: {input}
Thought: 思考过程
Action: 工具名称（必须是 [{tool_names}] 中的一个）
Action Input: 传入工具的参数
Observation: 工具返回的结果
Thought: 我现在知道最终答案了
Final Answer: 最终回答用户的内容

现在开始！
Question: {input}
Thought: {agent_scratchpad}
"""

prompt = PromptTemplate.from_template(template)

# ====================== 创建Agent ======================
agent = create_react_agent(
    llm=chat_model,
    tools=tools,
    prompt=prompt,
    stop_sequence=["\nObservation"]
)

agent_executor = AgentExecutor.from_agent_and_tools(
    agent=agent,
    tools=tools,
    verbose=True,
    handle_parsing_errors=True  # 自动处理格式错误，让Agent重试
)

# ====================== 测试运行 ======================
if __name__ == "__main__":
    query = "成都天气怎么样？"
    response = agent_executor.invoke({"input": query})
    print("="*50)
    print("Agent最终回答：", response["output"])
```

### （六）超级Agent

这是一个基于规则的多功能Agent，具备以下特点：

- 天气查询：调用心知天气API，返回指定城市的实时天气和温度

- 时间查询：返回当前系统时间，精确到秒

- 计算器：支持基本数学运算，包括加减乘除、乘方和括号

```
import requests
import datetime
import re


# ====================== 工具1：天气 ======================
def get_weather(city):
    API_KEY = "SBJVysU9a4KvOtgHs"
    url = f"https://api.seniverse.com/v3/weather/now.json?key={API_KEY}&location={city}&language=zh-Hans&unit=c"
    try:
        res = requests.get(url, timeout=3)
        data = res.json()
        return f"{city}：{data['results'][0]['now']['text']}，{data['results'][0]['now']['temperature']}℃"
    except:
        return "天气查询失败"


# ====================== 工具2：时间 ======================
def get_time():
    return f"当前时间：{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"


# ====================== 工具3：计算器 ======================
def calc(exp):
    try:
        # 替换中文运算符
        exp = exp.replace('^', '**').replace('×', '*').replace('÷', '/')
        # 只保留数字、运算符和小数点
        exp = re.sub(r'[^\d\+\-\*\/\.\(\)]', '', exp)
        if not exp:
            return "请输入有效的计算表达式"
        return f"结果：{eval(exp)}"
    except:
        return "计算错误"


# ====================== ✅ 修改后的 Agent ======================
def my_agent(query):
    q = query.lower()

    # 规则判断
    if "天气" in q:
        # 提取城市名：去掉"天气"、"的"等关键词，保留中文城市名
        city = query.replace("天气", "").replace("的", "").strip()

        # 如果提取后为空或者提取后的内容不是有效城市名，使用默认城市
        if not city or city == "":
            city = "成都"

        return get_weather(city)

    elif "时间" in q or "几点" in q:
        return get_time()

    elif "计算" in q or "+" in q or "-" in q or "*" in q or "/" in q or "^" in q or "×" in q or "÷" in q:
        # 提取表达式：去掉"计算"、"等于"等中文词
        exp = query.replace("计算", "").replace("等于", "").replace("多少", "").replace("?", "").replace("？", "").strip()
        return calc(exp)

    else:
        return "我可以回答：天气、时间、计算"


# ====================== 运行 ======================
if __name__ == "__main__":
    print("=== 超级agent ===")
    print("支持功能：天气查询、时间查询、计算器")
    print("示例：'成都天气'、'现在几点'、'计算 12+34'")
    print("输入 'exit' 或 '退出' 结束程序")

    while True:
        q = input("\n请输入问题：")
        if q in ["exit", "退出"]:
            break
        print("\n✅ 回答：", my_agent(q))
```
