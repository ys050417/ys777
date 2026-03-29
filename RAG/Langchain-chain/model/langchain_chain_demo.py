import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# 通用配置：对接ollama的Qwen2.5模型（OpenAI兼容接口）
from langchain_openai import ChatOpenAI

# 初始化大语言模型（ollama本地服务，默认端口11434）
chat_model = ChatOpenAI(
    openai_api_key="ollama",
    base_url="http://localhost:11434/v1",
    model="qwen2.5:0.5b"
)

# 1. 基础LLMChain（新版本LCEL写法）
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

system_message = "你是一个贴心的智能助手，回答简洁易懂。"
human_message = "{user_question}"
chat_prompt = ChatPromptTemplate.from_messages([
    ("system", system_message),
    ("human", human_message),
])
llm_chain = chat_prompt | chat_model | StrOutputParser()

# 2. 检索链RetrievalQA（RAG核心）
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import FakeEmbeddings
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA

# 自动创建测试文件
if not os.path.exists("sanguoyanyi.txt"):
    with open("sanguoyanyi.txt", "w", encoding="utf-8") as f:
        f.write("三国演义中刘备的五虎上将是：关羽、张飞、赵云、马超、黄忠。")

# 加载文档
loader = TextLoader("sanguoyanyi.txt", encoding='utf-8')
docs = loader.load()
text_splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=20)
chunks = text_splitter.split_documents(docs)

# 离线嵌入，不联网
embedding = FakeEmbeddings(size=1024)

# 构建向量库
vs = FAISS.from_documents(chunks, embedding)
retriever = vs.as_retriever()

# 提示模板
qa_prompt = ChatPromptTemplate.from_messages([
    ("system", "根据以下已知信息回答用户问题，仅用已知信息，不要编造：\n{context}"),
    ("human", "{question}"),
])

# 构建RAG链
retrieval_qa_chain = RetrievalQA.from_chain_type(
    llm=chat_model,
    chain_type="stuff",
    retriever=retriever,
    chain_type_kwargs={"prompt": qa_prompt}
)

# 3. 自定义链
custom_prompt = ChatPromptTemplate.from_template("说出一句包含{topic}的古诗，仅输出诗句，不要多余内容。")
custom_chain = custom_prompt | chat_model | StrOutputParser()

# ==================== 运行 ====================
if __name__ == "__main__":
    # 1 基础链
    print("=== LLMChain测试结果 ===")
    response = llm_chain.invoke({"user_question": "你好，介绍一下你自己"})
    print(response)

    # 2 RAG检索链
    print("\n=== RetrievalQA链测试结果 ===")
    qa_response = retrieval_qa_chain.invoke({"query": "五虎上将有哪些？"})
    print(qa_response["result"])

    # 检索文档
    related_docs = retriever.invoke("五虎上将有哪些？")
    print("\n=== 检索到的相关文档 ===")
    for i, doc in enumerate(related_docs):
        print(f"文档{i + 1}：{doc.page_content}")

    # 3 自定义链
    print("\n=== 自定义链测试结果 ===")
    poem_response = custom_chain.invoke({"topic": "花"})
    print(poem_response)