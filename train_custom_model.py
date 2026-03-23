# 导入所需库
import jieba
import re
from gensim.models import Word2Vec
import warnings

warnings.filterwarnings("ignore")


# 定义文本预处理函数（核心）
def preprocess_chinese_text(file_path):
    """
    处理中文文本：读取文件→清洗→分词→转换为训练格式
    :param file_path: 语料文件路径
    :return: 二维列表，格式为[[词1,词2,词3,...], ...]
    """
    # 1. 读取文件（指定utf-8编码，避免乱码）
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            text = f.read()
        print("文本读取成功！")
    except FileNotFoundError:
        print(f"错误：文件 {file_path} 不存在")
        return None
    except UnicodeDecodeError:
        print("错误：文件编码非utf-8，建议转换编码后重试")
        return None

    # 2. 文本清洗（仅保留中文，剔除所有无关字符）
    # 剔除非中文字符（正则：\u4e00-\u9fff是中文unicode范围）
    text = re.sub(r'[^\u4e00-\u9fff]', '', text)
    # 剔除多余空白符（空格、制表符）
    text = re.sub(r'\s+', '', text)
    # 剔除换行符
    text = re.sub(r'\n+', '', text)
    print(f"文本清洗完成，剩余字符数：{len(text)}")

    # 3. 中文分词（jieba.cut返回生成器，转列表）
    seg_list = jieba.cut(text, cut_all=False)  # cut_all=False为精确分词
    word_list = list(seg_list)
    print(f"分词完成，总词数：{len(word_list)}")

    # 4. 转换为gensim训练格式（二维列表，每个子列表是一个句子的分词结果）
    # 若语料是长文本，可按句号/换行分割成多个句子，这里简化为一个大列表
    sentences = [word_list]
    return sentences

# 定义语料路径（替换为你的实际路径）
corpus_path = r"E:\python\大模型应用开发\Word2Vec_Project\data\corpus\sgyy.txt"

# 调用预处理函数
sentences = preprocess_chinese_text(corpus_path)
if sentences is None:
    exit()  # 预处理失败则退出

# 定义训练参数（新手推荐默认值，后续可调优）
params = {
    "vector_size": 100,    # 词向量维度（常用100/200/300）
    "window": 5,           # 上下文窗口大小（目标词前后各5个词）
    "min_count": 5,        # 最小词频（仅保留出现≥5次的词，剔除低频无意义词）
    "workers": 4,          # 并行训练线程数（等于CPU核心数最佳）
    "sg": 0,               # 训练算法：0=CBOW（快、省内存），1=Skip-gram（效果好）
    "epochs": 10           # 训练轮数（迭代次数）
}

# 初始化并训练模型
print("\n开始训练模型...")
model = Word2Vec(
    sentences=sentences,
    vector_size=params["vector_size"],
    window=params["window"],
    min_count=params["min_count"],
    workers=params["workers"],
    sg=params["sg"],
    epochs=params["epochs"]
)
print("模型训练完成！")

# 定义模型保存路径
model_save_path = "../model/sanguo_w2v.model"

# 保存模型
model.save(model_save_path)
print(f"模型已保存至：{model_save_path}")

# 可选：保存词向量为txt格式（方便查看）
vector_save_path = "../model/sanguo_vectors.txt"
model.wv.save_word2vec_format(vector_save_path, binary=False)
print(f"词向量已保存至：{vector_save_path}")