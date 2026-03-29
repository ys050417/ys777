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