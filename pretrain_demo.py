from gensim.models import KeyedVectors
import warnings
warnings.filterwarnings("ignore")

model_path = r"E:\python\大模型应用开发\Word2Vec_Project\data\pretrain\GoogleNews-vectors-negative300.bin"

# 加载模型（binary=True表示二进制格式）
print("开始加载预训练模型...（约1-2分钟）")
try:
    model = KeyedVectors.load_word2vec_format(model_path, binary=True)
    print("模型加载成功！")
except Exception as e:
    print(f"模型加载失败：{e}")
    exit()

word_pairs = [("cat", "dog"), ("apple", "banana"), ("car", "bus")]

# 遍历计算相似性
for word1, word2 in word_pairs:
    try:
        similarity = model.similarity(word1, word2)
        print(f"'{word1}' 和 '{word2}' 的相似性：{similarity:.4f}")
    except KeyError as e:
        print(f"词 '{e}' 不在模型词典中，跳过")

# 定义类比关系：king - man + woman = ?
print("\n=== 类比推理 ===")
try:
    result = model.most_similar(positive=["woman", "king"], negative=["man"], topn=5)
    print("king - man + woman 的结果（Top5）：")
    for i, (word, score) in enumerate(result, 1):
        print(f"{i}. {word} (相似度：{score:.4f})")
except KeyError as e:
    print(f"类比失败：词 '{e}' 不在词典中")