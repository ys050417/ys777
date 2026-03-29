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