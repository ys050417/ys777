from gensim.models import Word2Vec
import warnings
warnings.filterwarnings("ignore")

# 1. 加载训练好的模型
model_path = "../model/sanguo_w2v.model"
try:
    model = Word2Vec.load(model_path)
    print("自定义模型加载成功！")
except FileNotFoundError:
    print(f"错误：模型文件 {model_path} 不存在，请先训练模型")
    exit()

# 2. 功能1：查询单个词的相似词
def get_similar_words(word, topn=10):
    """查询相似词"""
    try:
        similar_words = model.wv.most_similar(word, topn=topn)
        print(f"\n=== 与「{word}」最相似的{topn}个词 ===")
        for i, (w, score) in enumerate(similar_words, 1):
            print(f"{i}. {w} (相似度：{score:.4f})")
    except KeyError:
        print(f"错误：词「{word}」不在模型词典中")

# 3. 功能2：中文类比推理
def analogy推理(word1, word2, word3, topn=5):
    """
    类比推理：word1 - word2 + word3 = ?
    示例：刘备 - 关羽 + 张飞 = ?
    """
    try:
        result = model.wv.most_similar(positive=[word1, word3], negative=[word2], topn=topn)
        print(f"\n=== 类比推理：{word1} - {word2} + {word3} ===")
        for i, (w, score) in enumerate(result, 1):
            print(f"{i}. {w} (相似度：{score:.4f})")
    except KeyError as e:
        print(f"错误：词「{e}」不在模型词典中")

# 4. 调用功能（测试）
get_similar_words("玄德")
get_similar_words("诸葛")
analogy推理("玄德", "翼德", "云长")