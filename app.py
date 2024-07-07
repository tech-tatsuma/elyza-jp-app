# 各ライブラリのインポート
import streamlit as st
import tempfile
from langchain_community.llms import LlamaCpp
from langchain_core.output_parsers import StrOutputParser
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
import asyncio
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain.agents import initialize_agent
from langchain.agents import AgentType
from langchain.tools import DuckDuckGoSearchRun
from langchain.agents import Tool

# 使用するモデルの名前
MODEL_NAME = "Llama-3-ELYZA-JP-8B-q4_k_m.gguf"
embeddings = HuggingFaceEmbeddings(model_name="intfloat/multilingual-e5-base")

# モデルの準備
llm = LlamaCpp(
    model_path=MODEL_NAME,
)

st.title("Local AI Assistant")

simplechain = llm | StrOutputParser()

st.sidebar.title("設定")
uploaded_file = st.sidebar.file_uploader("PDFファイルをアップロード", type="pdf")
chunk_size = st.sidebar.slider("chunk size", 128, 1024, 256)
chunk_overlap = st.sidebar.slider("chunk overlap", 0, 120, 40)
isagent = st.sidebar.toggle("Use Web Search")

# アップロードされたPDFからretrieverを生成する処理を関数にまとめる
def process_uploaded_file(uploaded_file):
    with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        tmp_file_path = tmp_file.name

    loader = PyPDFLoader(tmp_file_path)
    documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    texts = text_splitter.split_documents(documents)

    # embeddingsを計算
    vectorstore = FAISS.from_documents(
        documents=texts, embedding=embeddings
    )
    return vectorstore.as_retriever()

# アップロードされたPDFからretrieverを生成
if uploaded_file is not None:
    # session_stateに保存されているファイルと異なる場合、新たにファイルを処理
    if 'uploaded_file_name' not in st.session_state or st.session_state.uploaded_file_name != uploaded_file.name:
        st.session_state.uploaded_file_name = uploaded_file.name
        st.session_state.retriever = process_uploaded_file(uploaded_file)

# セッション内のメッセージが指定されていない場合のデフォルト値
if "messages" not in st.session_state: # session_stateの中にmessagesが含まれていない場合
    st.session_state.messages = [] # メッセージを空のリストで初期化

# セッション内でチャット履歴をクリアするかどうかの状態変数
if "Clear" not in st.session_state: # session_stateの中にClearが含まれていない場合
    st.session_state.Clear = False # ClearをFalseで初期化

# 以前のメッセージを表示
for message in st.session_state.messages: # メッセージの数だけ繰り返す
    with st.chat_message(message["role"]): # メッセージのロールを指定
        st.markdown(message["content"]) # メッセージの内容を表示

# ユーザーからの新しい入力を取得
if prompt := st.chat_input("質問を入力してください"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    if uploaded_file is not None:
        system_prompt = (
            "You are an assistant for question-answering tasks. "
            "Use the following pieces of retrieved context to answer "
            "the question. If you don't know the answer, say that you "
            "don't know. Use three sentences maximum and keep the "
            "answer concise."
            "\n\n"
            "{context}"
        )

        ragprompt = ChatPromptTemplate.from_messages(
            [
                ("system", system_prompt),
                ("human", "{input}"),
            ]
        )

        question_answer_chain = create_stuff_documents_chain(llm, ragprompt)

        rag_chain = create_retrieval_chain(st.session_state.retriever, question_answer_chain)

    elif isagent:
        tools = []
        search = DuckDuckGoSearchRun()
        tools.append(
            Tool(
                name="duckduckgo-search",
                func=search.run,
                description="useful for when you need to search for latest information in web"
            )
        )
        agent = initialize_agent(tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True)

    else:
        chain = simplechain

    with st.chat_message("assistant"):
        message_placeholder = st.empty() # 一時的なプレースホルダーを作成
        full_response = "" # レスポンスを格納する変数

        if uploaded_file is not None:
            for response in rag_chain.stream({"input": prompt}):
                if answer_chunk := response.get("answer"):
                    print("answer_chunk: ", answer_chunk)
                    full_response += answer_chunk
                    message_placeholder.markdown(full_response + "▌")

        elif isagent:
            full_response = agent.run(prompt)
        elif uploaded_file is None and not isagent:
            for response in chain.stream(prompt):
                full_response += response
                message_placeholder.markdown(full_response + "▌")

        else:
            full_response = "web検索とRAG機能を同時に利用することはできません．"
        message_placeholder.markdown(full_response) # 最終レスポンスを表示
    st.session_state.messages.append({"role": "assistant", "content": full_response})
    st.session_state.Clear = True # チャット履歴のクリアボタンを有効にする

# チャット履歴をクリアするボタンが押されたら、メッセージをリセット
if st.session_state.Clear:
    if st.button('clear chat history'):
        st.session_state.messages = [] # メッセージのリセット
        full_response = ""
        st.session_state.Clear = False # クリア状態をリセット
        st.experimental_rerun() # 画面を更新