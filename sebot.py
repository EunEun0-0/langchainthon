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
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain.tools import tool
from langchain.tools.retriever import create_retriever_tool
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage

# --- 3. API 키 설정 (보안 강화) ---
try:
    # Streamlit Secrets에서 API 키 로드
    os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]
    # Tavily Search API 키 (웹 검색 도구용, 선택 사항)
    os.environ["TAVILY_API_KEY"] = st.secrets["TAVILY_API_KEY"]
except (KeyError, FileNotFoundError):
    st.error("🚨 API 키를 찾을 수 없습니다. .streamlit/secrets.toml 파일에 키를 설정해주세요.")
    st.stop()


# --- 4. LangChain 백엔드 함수 ---
@tool
def calculator(expression: str) -> str:
    """
    사용자로부터 받은 수학적 표현식을 계산합니다.
    이 도구는 덧셈, 뺄셈, 곱셈, 나눗셈 등 기본적인 사칙연산을 처리할 수 있습니다.
    예시: "35000 * 0.1" 또는 "10000 + 2500"
    """
    try:
        # 경고: eval()은 안전하지 않을 수 있으므로 실제 프로덕션에서는
        #      더 안전한 라이브러리(예: numexpr) 사용을 권장합니다.
        result = eval(expression)
        return f"계산 결과: {result}"
    except Exception as e:
        return f"계산 오류: {e}"

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
def get_retriever():
    """문서로부터 retriever를 생성하는 함수 (RAG 도구의 기반)"""
    try:
        base_docs_paths = [
            os.path.join("data", "2025.1기 확정 부가가치세 신고안내 매뉴얼.pdf"),
            os.path.join("data", "2024년 제2기 확정 부가가치세 신고안내.pdf"),
            os.path.join("data", "부가가치세_국세청_유권해석사례.pdf"),
            os.path.join("data", "부가가치세_실무사례.pdf"),
            os.path.join("data", "부가세 신고할 때 자주 묻는 질문들_토스페이먼츠.pdf")
        ]
        
        all_pages = []
        for file_path in base_docs_paths:
            loader = PyPDFLoader(file_path)
            all_pages.extend(loader.load_and_split())

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        split_docs = text_splitter.split_documents(all_pages)
        
        vectorstore = Chroma.from_documents(
            documents=split_docs,
            embedding=OpenAIEmbeddings(model='text-embedding-3-small')
        )
        return vectorstore.as_retriever()
    except Exception as e:
        st.error(f"🚨 기본 PDF 문서를 로드하는 데 실패했습니다: {e}")
        st.info("'data' 폴더에 PDF 파일이 있는지 확인해주세요.")
        st.stop()

@st.cache_resource
def build_agent_executor():
    """필요한 도구들을 포함한 에이전트 실행기를 빌드합니다."""
    
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    
    # --- [도구 2] RAG 검색 도구 생성 ---
    retriever = get_retriever()
    retriever_tool = create_retriever_tool(
        retriever,
        "vat_law_search", # 도구 이름
        "대한민국 부가가치세(VAT) 법률, 규정, 신고 절차에 대한 정보를 검색합니다. 세금 용어, 신고 기한, 공제 항목 등에 대한 질문에 사용하세요." # 도구 설명
    )
    
    # --- [도구 3] 웹 검색 도구 (선택 사항) ---
    # 최신 정보를 위해 웹 검색이 필요할 경우 사용
    web_search_tool = TavilySearchResults(max_results=2)

    # 에이전트가 사용할 도구 리스트
    tools = [calculator, retriever_tool, web_search_tool]

    # 에이전트 프롬프트 설정
    # LLM이 도구를 어떻게 사용할지, 어떤 역할을 할지 지시합니다.
    prompt = ChatPromptTemplate.from_messages([
        ("system", """
        당신은 전문 세무 상담사 AI 에이전트 '세봇이'입니다.
        사용자의 질문에 답변하기 위해 사용 가능한 도구들을 적극적으로 활용해야 합니다.
        이전 대화 내용을 참고하여 맥락에 맞는 답변을 하세요.

        - **vat_law_search**: 부가가치세 법률, 규정, 절차에 대한 질문일 때 사용하세요.
        - **calculator**: 숫자 계산이 필요할 때 사용하세요.
        - **tavily_search_results_json**: 최신 정보나 법률 외 일반 정보가 필요할 때 사용하세요.
        
        답변은 항상 한국어로, 친절하고 명확하게 제공해야 합니다.
        """),
        MessagesPlaceholder(variable_name="chat_history", optional=True),
        ("human", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ])

    agent = create_tool_calling_agent(llm, tools, prompt)
    agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)
    
    return agent_executor


# --- [추가] 대화 기록을 문자열로 변환하는 함수 ---
def format_chat_history(messages):
    """세션의 대화 기록을 LLM에 전달할 형식의 문자열로 변환합니다."""
    if not messages:
        return ""
    # 마지막 6개 메시지 (3쌍의 질문/답변)를 컨텍스트로 사용
    history = []
    for msg in messages[-6:]:
        role = "사용자" if msg["role"] == "user" else "챗봇"
        history.append(f"{role}: {msg['content']}")
    return "\n".join(history)

# --- 5. 메인 로직 ---
agent_executor = build_agent_executor()

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
    
    chat_history_str = format_chat_history(st.session_state.messages[:-1])
    
    file_context_str = ""
    if st.session_state.file_context:
        file_context_str = f"""
[첨부된 파일 내용]
{st.session_state.file_context}
---
"""
    final_prompt = f"""
{file_context_str}
[이전 대화 내용]
{chat_history_str}
---
위 파일 내용과 대화 맥락을 바탕으로 아래 질문에 답변해 주세요.

[사용자 질문]
{user_prompt}
"""
    agent_input = {
        "input": final_input,
        "chat_history": chat_history,
    }
    
    with st.chat_message("assistant"):
        with st.spinner("생각 중..."):
            response = agent_executor.invoke(agent_input)
            final_answer = response.get('output', '오류: 답변을 생성하지 못했습니다.')
            
            st.session_state.messages.append({"role": "assistant", "content": final_answer})
            st.rerun()
