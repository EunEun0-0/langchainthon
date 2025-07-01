import os
import streamlit as st
import pandas as pd
import time
import sys

# --- 1. 페이지 기본 설정 (가장 먼저 호출) ---
st.set_page_config(
    page_title="세봇이 🤖",
    page_icon="🧾",
    layout="centered",
)

# --- 2. 기본 설정 및 라이브러리 임포트 ---
# ChromaDB 사용을 위한 pysqlite3 설정 (Streamlit Cloud 배포 시 필요)
if sys.version_info.major == 3 and sys.version_info.minor >= 10:
    try:
        __import__('pysqlite3')
        sys.modules['sqlite3'] = sys.modules.get('pysqlite3')
    except ImportError:
        pass

from langchain_community.document_loaders import PyPDFLoader, UnstructuredFileLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# --- 3. API 키 설정 (보안 강화) ---
try:
    os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]
except (KeyError, FileNotFoundError):
    st.error("🚨 OpenAI API 키를 찾을 수 없습니다. .streamlit/secrets.toml 파일에 키를 설정해주세요.")
    st.stop()


# --- 4. LangChain 백엔드 함수 ---

@st.cache_resource
def load_and_split_documents(file_paths):
    """여러 파일을 로드하고 분할하는 함수"""
    all_pages = []
    for file_path in file_paths:
        if file_path.endswith('.pdf'):
            loader = PyPDFLoader(file_path)
            all_pages.extend(loader.load_and_split())
        else:
            loader = UnstructuredFileLoader(file_path)
            all_pages.extend(loader.load_and_split())
    return all_pages

@st.cache_resource
def get_vectorstore(_docs):
    """문서들로부터 벡터 저장소를 생성하거나 로드하는 함수"""
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    split_docs = text_splitter.split_documents(_docs)
    
    vectorstore = Chroma.from_documents(
        documents=split_docs,
        embedding=OpenAIEmbeddings(model='text-embedding-3-small')
    )
    return vectorstore

def format_docs(docs):
    """검색된 문서들을 하나의 문자열로 포맷하는 함수"""
    return "\n\n".join(doc.page_content for doc in docs)

@st.cache_resource
def build_rag_chain():
    """RAG 체인을 빌드하는 함수"""
    try:
        base_docs_paths = [
            os.path.join("data", "2025.1기 확정 부가가치세 신고안내 매뉴얼.pdf"),
            os.path.join("data", "2024년 제2기 확정 부가가치세 신고안내.pdf"),
            os.path.join("data", "부가가치세_국세청_유권해석사례.pdf"),
            os.path.join("data", "부가가치세_실무사례.pdf")
        ]
        base_documents = load_and_split_documents(base_docs_paths)
        vectorstore = get_vectorstore(base_documents)
        retriever = vectorstore.as_retriever()
    except Exception as e:
        st.error(f"🚨 기본 PDF 문서를 로드하는 데 실패했습니다: {e}")
        st.info("'data' 폴더에 PDF 파일이 있는지 확인해주세요.")
        st.stop()


    qa_system_prompt = """
    [지시사항]
    당신은 대한민국 부가가치세법을 기반으로 한 전문 Q&A 봇입니다. 사용자의 질문에 대해 정확하고, 상세하며, 법률적 근거를 바탕으로 답변해야 합니다.
    만약 사용자가 [첨부된 파일 내용]을 제공하면, 반드시 그 내용을 최우선으로 참고하여 답변해야 합니다.
    
    [답변 생성 원칙]
    1. **정확성:** 반드시 대한민국 부가가치세법 및 관련 규정에 부합하는 내용만을 답변합니다.
    2. **구체성:** 추상적인 답변 대신 구체적인 상황과 연결하여 설명하고 예시를 들어 이해를 돕습니다.
    3. **근거 제시:** 단순히 "네/아니오"로 끝나는 답변이 아닌, 왜 그런지 이유와 근거를 함께 제시합니다.
    4. **최신성:** 기본 지식은 25년 매뉴얼을 따르되, 과거와 관련된 질문에는 24년 매뉴얼을 참고합니다.
    5. **파일 내용 우선:** 사용자가 파일을 첨부했다면, 검색된 일반 지식보다 첨부 파일의 내용을 우선하여 답변합니다.
    
    [답변 구조]
    - **핵심 요약:** 질문에 대한 간결하고 직접적인 답변.
    - **상세 설명:** 핵심 요약에 대한 구체적인 근거와 추가 정보.
    - **주의사항/참고:** 추가적으로 알아야 할 점, 예외 사항, 전문가 상담 권유 등.
    
    ---
    [검색된 일반 지식]
    {context}
    ---
    """

    qa_prompt = ChatPromptTemplate.from_messages([
        ("system", qa_system_prompt),
        ("human", "{input}"),
    ])

    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    
    rag_chain = (
        {"context": retriever | format_docs, "input": RunnablePassthrough()}
        | qa_prompt
        | llm
        | StrOutputParser()
    )
    return rag_chain

# --- 5. 메인 로직 ---
rag_chain = build_rag_chain()

# --- 세션 상태 초기화 ---
if "messages" not in st.session_state:
    st.session_state.messages = []
if "file_context" not in st.session_state:
    st.session_state.file_context = None
if "file_name" not in st.session_state:
    st.session_state.file_name = None

# --- 사이드바 UI 구성 ---
with st.sidebar:
    st.title("🧾 파일 업로드")
    st.info("세금계산서(XLSX, CSV)나 관련 PDF 파일을 업로드하여 질문할 수 있습니다.")

    uploaded_file = st.file_uploader(
        "파일을 선택하세요.",
        type=['xlsx', 'csv', 'pdf'],
        label_visibility="collapsed"
    )

    if uploaded_file:
        try:
            file_name = uploaded_file.name
            df = None
            
            if file_name.endswith('.xlsx'):
                df = pd.read_excel(uploaded_file)
            elif file_name.endswith('.csv'):
                df = pd.read_csv(uploaded_file)
            elif file_name.endswith('.pdf'):
                st.warning("PDF 파일은 현재 미리보기를 지원하지 않지만, 내용에 대한 질문은 가능합니다.")
                st.session_state.file_name = file_name
                st.session_state.file_context = "PDF 파일이 업로드되었습니다. 파일 내용에 대해 질문해주세요."
            
            if df is not None:
                st.session_state.file_context = df.to_markdown()
                st.session_state.file_name = file_name
                st.subheader("✅ 업로드 파일 미리보기")
                st.dataframe(df, height=300, use_container_width=True)

        except Exception as e:
            st.error(f"파일 처리 중 오류 발생: {e}")
            st.session_state.file_context = None
            st.session_state.file_name = None
    
    if st.session_state.file_name:
        if st.button("🔄 업로드 파일 초기화", use_container_width=True):
            st.session_state.file_context = None
            st.session_state.file_name = None
            st.success("파일이 초기화되었습니다. 일반 질문 모드로 전환합니다.")
            time.sleep(1)
            st.rerun()

# --- 메인 화면 UI 구성 ---
st.title("세봇이 🤖")
st.write("부가가치세 관련 질문에 답변해 드립니다. 왼쪽 사이드바에 파일을 올려 특정 문서에 대해 질문할 수도 있습니다.")
st.write("---")

if st.session_state.file_name:
    st.info(f"🧾 **'{st.session_state.file_name}' 파일에 대해 질문합니다.**")
else:
    st.info("💡 **일반 부가가치세 질문 모드입니다.**")

# --- 채팅 대화 기록 표시 ---
if not st.session_state.messages:
    st.session_state.messages.append({"role": "assistant", "content": "안녕하세요! 부가가치세에 대해 궁금한 점을 물어보세요."})

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

# --- 예시 질문 버튼 ---
if len(st.session_state.messages) <= 1:
    st.write("**예시 질문:**")
    cols = st.columns(3)
    example_questions = ["간이과세자 부가세는?", "매입세액공제란?", "세금계산서 발급의무"]
    
    # 예시 버튼 클릭 시 메시지 추가 및 rerun
    if cols[0].button(example_questions[0], use_container_width=True):
        st.session_state.messages.append({"role": "user", "content": example_questions[0]})
        st.rerun()
    if cols[1].button(example_questions[1], use_container_width=True):
        st.session_state.messages.append({"role": "user", "content": example_questions[1]})
        st.rerun()
    if cols[2].button(example_questions[2], use_container_width=True):
        st.session_state.messages.append({"role": "user", "content": example_questions[2]})
        st.rerun()

# --- 사용자 채팅 입력 ---
if user_input := st.chat_input("질문을 입력해주세요 :)"):
    st.session_state.messages.append({"role": "user", "content": user_input})
    st.rerun()

# --- 챗봇 응답 생성 로직 ---
# 마지막 메시지가 사용자의 것이고, 아직 봇이 응답하지 않았을 경우에만 실행
if st.session_state.messages and st.session_state.messages[-1]["role"] == "user":
    user_prompt = st.session_state.messages[-1]["content"]
    
    final_prompt = user_prompt
    if st.session_state.file_context:
        final_prompt = f"""
[첨부된 파일 내용]
{st.session_state.file_context}
---
위 첨부파일 내용을 바탕으로 아래 질문에 답변해 주세요.

[사용자 질문]
{user_prompt}
"""
    with st.chat_message("assistant"):
        with st.spinner("답변을 생성하는 중..."):
            response = rag_chain.invoke(final_prompt)
            st.write(response)
            # 응답을 state에 추가
            st.session_state.messages.append({"role": "assistant", "content": response})
