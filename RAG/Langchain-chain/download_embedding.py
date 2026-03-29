from modelscope.hub.snapshot_download import snapshot_download
# 下载Embedding模型到本地models目录
emb_model_dir = snapshot_download('AI-ModelScope/bge-large-zh-v1.5', cache_dir='models')
print("Embedding模型下载完成，路径：", emb_model_dir)