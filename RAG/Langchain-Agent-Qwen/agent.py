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