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