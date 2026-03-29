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