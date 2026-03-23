# 导入依赖
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import jieba
import numpy as np
import re
from collections import Counter
import time

# 检查GPU是否可用
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"使用设备: {device}")
if torch.cuda.is_available():
    print(f"GPU型号: {torch.cuda.get_device_name(0)}")


# ===================== 核心修复：整合文本加载+分词逻辑 =====================
def read_and_tokenize_chinese_text(file_path, stop_words=None, min_sentence_len=5):
    """
    读取中文文本文件，完成清洗、分句、分词、过滤全流程
    :param file_path: 文本文件路径（字符串，核心修复点）
    :param stop_words: 自定义停止词集合（默认使用基础虚词集合）
    :param min_sentence_len: 最小句子长度（过滤过短文本）
    :return:
        tokenized_corpus: 分词后的句子列表（每个元素是单词列表）
        raw_sentences: 原始清洗后的文本句子列表（用于展示）
    """
    # 初始化默认停止词（虚词）
    if stop_words is None:
        stop_words = {'之', '乎', '者', '也', '以', '于', '而', '则', '其', '乃', '若', '且', '故'}

    # 第一步：读取并清洗文本
    with open(file_path, 'r', encoding='utf-8') as file:
        text = file.read()
        # 1. 去掉行首数字标签（兼容原load_and_preprocess_data的逻辑）
        text = re.sub(r'^\d+\s*', '', text, flags=re.MULTILINE)
        # 2. 只保留汉字和句末标点（。！？；），过滤数字、字母、乱码、空格
        text = re.sub(r'[^\u4e00-\u9fff。！？；]', '', text)
        # 3. 去掉多余换行，统一句末标点为。（简化句子切分）
        text = re.sub(r'\n+', '', text)
        text = re.sub(r'[！？；]', '。', text)

    # 第二步：分句 + 过滤短句子
    raw_sentences = []
    tokenized_corpus = []
    # 按句号切分原始句子
    split_sentences = text.split('。')
    total_sentences = len(split_sentences)

    for i, sent in enumerate(split_sentences):
        # 过滤空句子/短句子
        sent = sent.strip()
        if len(sent) < min_sentence_len:
            continue
        raw_sentences.append(sent)

        # jieba精确分词
        words = jieba.lcut(sent)

        # 多层过滤：去空格、去停止词、去单字
        words = [
            word.strip()
            for word in words
            if len(word.strip()) > 1 and word not in stop_words
        ]

        # 保留至少2个有效词的句子
        if len(words) > 1:
            tokenized_corpus.append(words)

        # 进度显示（每200条打印一次）
        if i % 200 == 0 and i > 0:
            print(f"进度：已处理 {i}/{total_sentences} 条原始句子，有效分词句子数：{len(tokenized_corpus)}")

    # 最终进度提示
    print(f"\n处理完成！")
    print(f"原始句子总数：{total_sentences}")
    print(f"过滤后有效原始句子数：{len(raw_sentences)}")
    print(f"分词后有效句子数：{len(tokenized_corpus)}")

    return tokenized_corpus, raw_sentences


# ===================== 加载数据 + 分词（修复参数传递） =====================
file_path = r'E:\python\大模型应用开发\Word2Vec_Project\data\corpus\zhizhitongjian01(1).txt'
# 调用整合后的函数：传入文件路径（字符串），而非列表
tokenized_corpus, raw_sentences = read_and_tokenize_chinese_text(
    file_path,
    min_sentence_len=5  # 对应原代码中len(content) > 5的逻辑
)

# 打印数据示例（验证修复效果）
print("\n加载的数据示例：")
print("前5条原始文本：")
for i in range(min(5, len(raw_sentences))):
    print(f"{i + 1}: {raw_sentences[i][:50]}...")

print("\n前3条分词结果：")
for i in range(min(3, len(tokenized_corpus))):
    print(f"原文: {raw_sentences[i][:30]}...")
    print(f"分词: {tokenized_corpus[i][:10]}...")


# ===================== 词汇统计分析 =====================
def analyze_vocabulary(tokenized_corpus):
    """分析词汇统计信息"""
    all_words = [word for sentence in tokenized_corpus for word in sentence]
    word_freq = Counter(all_words)

    print("\n词汇统计信息：")
    print(f"总词汇量: {len(all_words)}")
    print(f"唯一词汇数: {len(word_freq)}")
    print(f"平均句子长度: {np.mean([len(sentence) for sentence in tokenized_corpus]):.2f}")
    print(f"最长句子长度: {max([len(sentence) for sentence in tokenized_corpus])}")
    print(f"最短句子长度: {min([len(sentence) for sentence in tokenized_corpus])}")

    # 显示最高频词汇
    print("\n前20个最高频词汇：")
    for word, freq in word_freq.most_common(20):
        print(f"{word}: {freq}次")

    return word_freq


word_frequency = analyze_vocabulary(tokenized_corpus)


# ===================== 构建词汇表 =====================
def build_vocab(tokenized_corpus, min_count=5):
    """构建词汇表和索引映射"""
    # 统计词频
    word_counts = Counter([word for sentence in tokenized_corpus for word in sentence])

    # 过滤低频词
    vocab = {word: count for word, count in word_counts.items() if count >= min_count}

    # 创建索引映射
    idx_to_word = ['<PAD>', '<UNK>'] + list(vocab.keys())
    word_to_idx = {word: idx for idx, word in enumerate(idx_to_word)}

    print(f"\n词汇表大小: {len(word_to_idx)} (包含 {len(word_counts) - len(vocab)} 个低频词被过滤)")

    return word_to_idx, idx_to_word, vocab


word_to_idx, idx_to_word, vocab = build_vocab(tokenized_corpus, min_count=5)


# ===================== 创建训练数据（优化负采样） =====================
def create_training_data(tokenized_corpus, word_to_idx, window_size=5, num_negatives=5):
    """创建Word2Vec训练数据（Skip-gram with Negative Sampling）"""
    training_data = []
    vocab_size = len(word_to_idx)
    unk_idx = word_to_idx.get('<UNK>', 0)

    # 计算词频分布用于负采样
    word_counts = np.zeros(vocab_size)
    for word, idx in word_to_idx.items():
        if word in vocab:
            word_counts[idx] = vocab[word]

    # 负采样分布（按词频的3/4次方）
    word_distribution = np.power(word_counts, 0.75)
    # 避免除以0
    if word_distribution.sum() == 0:
        word_distribution = np.ones(vocab_size) / vocab_size
    else:
        word_distribution = word_distribution / word_distribution.sum()

    for sentence in tokenized_corpus:
        # 转换为索引
        sentence_indices = [word_to_idx.get(word, unk_idx) for word in sentence]

        # 跳过过短句子
        if len(sentence_indices) < 2:
            continue

        for i, target_word_idx in enumerate(sentence_indices):
            # 获取上下文窗口
            start = max(0, i - window_size)
            end = min(len(sentence_indices), i + window_size + 1)

            for j in range(start, end):
                if j != i:  # 跳过目标词本身
                    context_word_idx = sentence_indices[j]
                    training_data.append((target_word_idx, context_word_idx))

    print(f"\n创建了 {len(training_data)} 个训练样本")
    return training_data, word_distribution


# 创建训练数据
training_data, word_distribution = create_training_data(
    tokenized_corpus, word_to_idx, window_size=5, num_negatives=5
)


# ===================== 自定义Dataset（优化负采样） =====================
class Word2VecDataset(Dataset):
    def __init__(self, training_data, word_distribution, num_negatives=5):
        self.training_data = training_data
        self.word_distribution = word_distribution
        self.num_negatives = num_negatives
        self.vocab_size = len(word_distribution)

    def __len__(self):
        return len(self.training_data)

    def __getitem__(self, idx):
        target, context = self.training_data[idx]

        # 优化负采样：批量采样+去重，避免循环
        negative_samples = np.random.choice(
            self.vocab_size,
            size=self.num_negatives * 2,  # 多采一倍
            p=self.word_distribution,
            replace=False
        )
        # 过滤掉目标词和上下文词
        negative_samples = [n for n in negative_samples if n != target and n != context][:self.num_negatives]
        # 兜底：如果不够5个，补充随机样本
        while len(negative_samples) < self.num_negatives:
            negative = np.random.choice(self.vocab_size, p=self.word_distribution)
            if negative != target and negative != context:
                negative_samples.append(negative)

        return {
            'target': torch.tensor(target, dtype=torch.long),
            'context': torch.tensor(context, dtype=torch.long),
            'negatives': torch.tensor(negative_samples, dtype=torch.long)
        }


# ===================== Word2Vec模型 =====================
class Word2VecModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim=100):  # 维度调至100，适配890万文本
        super(Word2VecModel, self).__init__()
        self.target_embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.context_embeddings = nn.Embedding(vocab_size, embedding_dim)

        # 初始化权重（更稳定的初始化）
        nn.init.xavier_uniform_(self.target_embeddings.weight)
        nn.init.xavier_uniform_(self.context_embeddings.weight)

    def forward(self, target_word, context_word, negative_words):
        # 获取词向量
        target_embed = self.target_embeddings(target_word)  # [batch_size, embedding_dim]
        context_embed = self.context_embeddings(context_word)  # [batch_size, embedding_dim]
        negative_embed = self.context_embeddings(negative_words)  # [batch_size, num_negatives, embedding_dim]

        # 计算正样本得分
        positive_score = torch.sum(target_embed * context_embed, dim=1)  # [batch_size]
        positive_score = torch.clamp(positive_score, max=10, min=-10)

        # 计算负样本得分
        target_embed_expanded = target_embed.unsqueeze(1)  # [batch_size, 1, embedding_dim]
        negative_score = torch.bmm(negative_embed,
                                   target_embed_expanded.transpose(1, 2))  # [batch_size, num_negatives, 1]
        negative_score = torch.clamp(negative_score.squeeze(2), max=10, min=-10)  # [batch_size, num_negatives]

        return positive_score, negative_score


# ===================== 损失函数（增加数值稳定） =====================
def skipgram_loss(positive_score, negative_score):
    """Skip-gram with Negative Sampling损失函数（优化数值稳定性）"""
    # 正样本损失（增加小常数避免log(0)）
    positive_loss = -torch.log(torch.sigmoid(positive_score) + 1e-8)

    # 负样本损失
    negative_loss = -torch.sum(torch.log(torch.sigmoid(-negative_score) + 1e-8), dim=1)

    return (positive_loss + negative_loss).mean()


# ===================== 训练函数（核心优化） =====================
def train_word2vec_gpu(model, dataset, batch_size=1024, epochs=5, learning_rate=0.003):
    """在GPU上训练Word2Vec模型（优化学习率+梯度裁剪+调度器）"""
    model = model.to(device)

    # 创建DataLoader（增加预取，提升速度）
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
    )

    # 优化器（增加权重衰减，防止过拟合）
    optimizer = optim.Adam(
        model.parameters(),
        lr=learning_rate,
        weight_decay=1e-5  # 权重衰减
    )

    # 学习率调度器（更合理的衰减策略）
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.5,
        patience=2,
        verbose=True
    )

    print(f"\n开始训练...")
    print(f"批量大小: {batch_size}")
    print(f"训练轮数: {epochs}")
    print(f"优化器: Adam, 初始学习率: {learning_rate}")

    losses = []
    best_loss = float('inf')

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        start_time = time.time()

        for batch_idx, batch in enumerate(dataloader):
            # 将数据移到GPU
            target_words = batch['target'].to(device)
            context_words = batch['context'].to(device)
            negative_words = batch['negatives'].to(device)

            # 前向传播
            optimizer.zero_grad()
            positive_score, negative_score = model(target_words, context_words, negative_words)

            # 计算损失
            loss = skipgram_loss(positive_score, negative_score)

            # 反向传播（梯度裁剪，防止爆炸）
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # 核心：梯度裁剪
            optimizer.step()

            total_loss += loss.item()

            # 每200个batch打印一次（减少打印频率，提升速度）
            if batch_idx % 200 == 0:
                print(f"Epoch {epoch + 1}/{epochs} | Batch {batch_idx}/{len(dataloader)} | Loss: {loss.item():.4f}")

        # 计算平均损失
        avg_loss = total_loss / len(dataloader)
        losses.append(avg_loss)

        # 更新学习率
        scheduler.step(avg_loss)

        # 保存最优模型
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), "best_word2vec_model.pth")

        epoch_time = time.time() - start_time
        print(
            f"\nEpoch {epoch + 1}/{epochs} 完成 | 平均损失: {avg_loss:.4f} | 最佳损失: {best_loss:.4f} | 时间: {epoch_time:.2f}秒")

    print("\n训练完成！最优模型已保存为 best_word2vec_model.pth")
    return model, losses


# ===================== 初始化训练 =====================
# 创建数据集和模型
dataset = Word2VecDataset(training_data, word_distribution, num_negatives=5)
model = Word2VecModel(vocab_size=len(word_to_idx), embedding_dim=100)  # 维度调至100

# 训练模型（核心修改：训练轮数改为1轮）
trained_model, losses = train_word2vec_gpu(
    model, dataset, batch_size=1024, epochs=1, learning_rate=0.001  # epochs=1 关键修改处
)


# ===================== 词向量提取与保存 =====================
def get_word_vectors(model, word_to_idx):
    """从训练好的模型中提取词向量并保存为 .pt 文件"""
    model.eval()
    with torch.no_grad():
        # 获取目标词向量
        all_indices = torch.arange(len(word_to_idx)).to(device)
        word_vectors = model.target_embeddings(all_indices).detach().cpu()  # 保留为 PyTorch 张量

    # 创建词向量字典（键为词，值为张量）
    word_vectors_dict = {}
    for word, idx in word_to_idx.items():
        word_vectors_dict[word] = word_vectors[idx]

    return word_vectors_dict, word_vectors


word_vectors_dict, all_vectors = get_word_vectors(trained_model, word_to_idx)


# 保存词向量为 .pt 文件
def save_word_vectors_pt(word_vectors_dict, output_path):
    """将词向量字典保存为 .pt 文件"""
    torch.save(word_vectors_dict, output_path)
    print(f"\n词向量已保存到: {output_path}")


# 加载词向量从 .pt 文件
def load_word_vectors_pt(input_path):
    """从 .pt 文件加载词向量字典"""
    word_vectors_dict = torch.load(input_path)
    print(f"词向量已从 {input_path} 加载")
    return word_vectors_dict


# 保存词向量
save_word_vectors_pt(word_vectors_dict, "word_vectors.pt")

# 加载词向量
loaded_word_vectors_dict = load_word_vectors_pt("word_vectors.pt")


# ===================== 兼容gensim的接口 =====================
class PyTorchWord2VecWrapper:
    def __init__(self, word_vectors_dict, word_to_idx, idx_to_word):
        self.wv = self.WordVectors(word_vectors_dict, word_to_idx, idx_to_word)
        self.vector_size = list(word_vectors_dict.values())[0].shape[0]

    class WordVectors:
        def __init__(self, word_vectors_dict, word_to_idx, idx_to_word):
            self.vectors_dict = word_vectors_dict
            self.word_to_idx = word_to_idx
            self.idx_to_word = idx_to_word
            self.key_to_index = word_to_idx
            self.vectors = np.stack(list(word_vectors_dict.values()))

        def __getitem__(self, word):
            return self.vectors_dict.get(word, None)

        def __contains__(self, word):
            return word in self.vectors_dict

        def similarity(self, word1, word2):
            """计算两个词的余弦相似度"""
            if word1 not in self.vectors_dict or word2 not in self.vectors_dict:
                raise KeyError(f"词语不在词汇表中: {word1} 或 {word2}")

            vec1 = self.vectors_dict[word1].numpy() if torch.is_tensor(self.vectors_dict[word1]) else self.vectors_dict[
                word1]
            vec2 = self.vectors_dict[word2].numpy() if torch.is_tensor(self.vectors_dict[word2]) else self.vectors_dict[
                word2]

            # 计算余弦相似度
            norm1 = np.linalg.norm(vec1)
            norm2 = np.linalg.norm(vec2)

            if norm1 == 0 or norm2 == 0:
                return 0.0

            similarity = np.dot(vec1, vec2) / (norm1 * norm2)
            return similarity

        def most_similar(self, word, topn=10):
            """查找与给定词最相似的词"""
            if word not in self.vectors_dict:
                raise KeyError(f"词语不在词汇表中: {word}")

            target_vec = self.vectors_dict[word].numpy() if torch.is_tensor(self.vectors_dict[word]) else \
            self.vectors_dict[word]
            similarities = []

            for w, vec in self.vectors_dict.items():
                if w == word:
                    continue

                vec_np = vec.numpy() if torch.is_tensor(vec) else vec
                norm_target = np.linalg.norm(target_vec)
                norm_vec = np.linalg.norm(vec_np)

                if norm_target == 0 or norm_vec == 0:
                    sim = 0.0
                else:
                    sim = np.dot(target_vec, vec_np) / (norm_target * norm_vec)

                similarities.append((w, sim))

            # 按相似度排序
            similarities.sort(key=lambda x: x[1], reverse=True)
            return similarities[:topn]


# ===================== 测试模型 =====================
# 创建包装器
w2v_model = PyTorchWord2VecWrapper(loaded_word_vectors_dict, word_to_idx, idx_to_word)

# 测试相似词查找功能
print("\n" + "=" * 50)
print("词向量模型测试")
print("=" * 50)

test_words = ['项羽', '刘邦', '关羽', '孙权']
print("\n相似词查找测试：")
for word in test_words:
    if word in w2v_model.wv:
        similar_words = w2v_model.wv.most_similar(word, topn=5)
        print(f"\n与'{word}'最相似的词：")
        for similar, score in similar_words:
            print(f"  {similar}: {score:.3f}")
    else:
        print(f"'{word}'不在词汇表中")

# 词汇相似度计算
print("\n词汇相似度计算：")
word_pairs = [('项羽', '刘邦'), ('刘备', '关羽'), ('咸阳', '洛阳'), ('嬴政', '荆轲')]

for word1, word2 in word_pairs:
    if word1 in w2v_model.wv and word2 in w2v_model.wv:
        similarity = w2v_model.wv.similarity(word1, word2)
        print(f"'{word1}' 和 '{word2}' 的相似度: {similarity:.3f}")
    else:
        print(f"词汇对 ({word1}, {word2}) 中有词不在词汇表中")