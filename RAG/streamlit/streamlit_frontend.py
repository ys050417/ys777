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