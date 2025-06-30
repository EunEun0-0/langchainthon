import os
import streamlit as st
import tempfile
import sys
# sys.version_info.major가 3이고, sys.version_info.minor가 10 이상인 경우만 실행하는 것이 안전
if sys.version_info.major == 3 and sys.version_info.minor >= 10: # Python 3.10 이상일 때
    try:
        __import__('pysqlite3')
        sys.modules['sqlite3'] = sys.modules['pysqlite3']
    except ImportError:
        pass # pysqlite3가 없으면 원래 sqlite3 사용 (이 경우 오류가 발생할 것임)
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser



os.environ["OPENAI_API_KEY"] = st.secrets['OPENAI_API_KEY']

st.title("tax supporter")

if "messages" not in st.session_state:
    st.session_state["messages"] = [{"role": "assistant", "content": "부가가치세에 대해 어떤 것이 궁금하신가요?"}]
    
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None
    
for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])
    
@st.cache_resource
def load_and_split_pdf(file_path):
    loader = PyPDFLoader(file_path)
    return loader.load_and_split()

#텍스트 청크들을 Chroma 안에 임베딩 벡터로 저장
@st.cache_resource
def create_vector_store(_docs):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    split_docs = text_splitter.split_documents(_docs)
    persist_directory = "./chroma_db"
    vectorstore = Chroma.from_documents(
        split_docs, 
        OpenAIEmbeddings(model='text-embedding-3-small'),
        persist_directory=persist_directory
    )
    return vectorstore

@st.cache_resource
def get_vectorstore(_docs):
    persist_directory = "./chroma_db"
    if os.path.exists(persist_directory):
        return Chroma(
            persist_directory=persist_directory,
            embedding_function=OpenAIEmbeddings(model='text-embedding-3-small')
        )
    else:
        return create_vector_store(_docs)

def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)
    
@st.cache_resource
def chaining():
    file_path = "\\Users\\USER\\Downloads\\2024년 제2기 확정 부가가치세 신고안내.pdf"
    pages = load_and_split_pdf(file_path)
    vectorstore = get_vectorstore(pages)
    retriever = vectorstore.as_retriever()

    # Define the answer question prompt
    qa_system_prompt = """
[지시사항]
당신은 대한민국 부가가치세법을 기반으로 한 전문 Q&A 봇입니다. 사용자의 질문에 대해 정확하고, 상세하며, 법률적 근거를 바탕으로 답변해야 합니다. 답변은 이해하기 쉬운 언어로 작성하되, 필요한 경우 관련 법령 조항이나 세무 용어를 명확하게 설명해야 합니다.

[사용자 질문 분석 및 의도 파악]
1. 사용자의 질문에서 핵심 키워드(예: 간이과세자, 면세, 세금계산서, 매입세액 공제, 신고 기한, 업종, 매출액 등)를 추출합니다.
2. 사용자가 어떤 정보를 얻고자 하는지 질문의 의도를 파악합니다. (예: 특정 상황에서의 부가세 납부 의무, 공제 가능 여부, 신고 방법, 세금 계산 방식 등)

[정보 검색 및 추출 (RAG 시스템 활용)]
3. 추출된 키워드와 파악된 의도를 바탕으로, 귀하의 지식 기반 또는 외부 데이터베이스(부가가치세법, 관련 시행령, 국세청 예규, 판례, 세무 가이드 등)에서 가장 관련성이 높은 정보를 검색하고 추출합니다.
4. 검색된 정보 중에서 사용자의 질문에 직접적으로 답변할 수 있는 핵심 내용, 관련 법조문, 계산식, 예외 사항 등을 선별합니다.

[답변 생성 원칙]
5. **정확성:** 반드시 대한민국 부가가치세법 및 관련 규정에 부합하는 내용만을 답변합니다. 불확실한 정보나 추측성 답변은 금지합니다.
6. **구체성:** 추상적인 답변 대신 구체적인 상황과 연결하여 설명합니다. 필요한 경우 예시를 들어 이해를 돕습니다.
7. **상세함:** 단순히 "네/아니오"로 끝나는 답변이 아닌, 왜 그런지 이유와 근거를 함께 제시합니다.
8. **명확성:** 복잡한 세무 용어는 쉽게 풀어서 설명하되, 정확한 용어를 사용합니다.
9. **적절한 경고/안내:** 세법 해석이나 적용은 개별 상황에 따라 달라질 수 있음을 안내하고, 필요한 경우 전문가(세무사)와 상담을 권장합니다. 신고 기한, 가산세 등 중요한 정보는 강조하여 안내합니다.
10. **계산식 제공:** 세액 계산과 관련된 질문의 경우, 관련 계산식을 명확히 제시합니다.

[답변 구조 (예시)]
* **핵심 요약:** 질문에 대한 간략하고 직접적인 답변.
* **상세 설명:** 핵심 요약에 대한 구체적인 근거와 추가 정보.
* **관련 법규/조항 (선택 사항):** 답변의 법률적 근거 제시.
* **주의사항/참고:** 추가적으로 알아야 할 점, 예외 사항, 전문가 상담 권유 등.
* **(만약 질문이 불명확할 경우) 추가 정보 요청:** 답변을 위해 필요한 정보가 있다면 구체적으로 질문.

[예시 질문과 답변 프로세스]
**사용자 질문:** "디자인 외주로 매출 3000만 원이에요. 간이과세자인데 부가세 신고 대상인가요?"

**봇의 내부 처리 과정:**
1. **키워드:** 디자인 외주, 매출 3000만원, 간이과세자, 부가세 신고 대상
2. **의도:** 간이과세자로서 해당 매출액일 때 부가세 신고 의무가 있는지 확인.
3. **정보 검색:**
    * 간이과세자의 부가가치세 신고 의무
    * 간이과세자 과세기간 및 신고 기한
    * 간이과세자 부가가치세 계산 방법 (매출액 * 업종별 부가가치율 * 10%)
    * 간이과세자 납부 면제 기준 (연 매출 4,800만원 미만)
4. **답변 생성:** 위 원칙에 따라 답변 구성.

**봇의 예상 답변:**
네, 디자인 외주로 매출 3,000만 원인 간이과세자이시라면 부가가치세 신고 대상입니다.
---
### 간이과세자의 부가가치세 신고 의무

간이과세자도 일반과세자와 마찬가지로 부가가치세 신고 의무가 있습니다. 다만, 신고 기한과 세금 계산 방식에서 차이가 있습니다.

* **신고 기한:** 간이과세자는 매년 1월 1일부터 12월 31일까지의 과세기간에 대해 **다음 해 1월 25일**까지 부가가치세를 신고·납부해야 합니다.
* **납부세액 계산:** 간이과세자의 부가가치세는 다음과 같이 계산됩니다.
    * **납부세액 = (매출액 × 업종별 부가가치율) × 10%**
    * 디자인 업종은 대개 서비스업으로 분류되며, 부가가치세법 시행령에 따라 해당 업종의 부가가치율이 적용됩니다. (현재 디자인 서비스업의 부가가치율은 일반적으로 30%입니다.)
* **납부 의무 면제:** 간이과세자는 연 매출액이 4,800만 원 미만일 경우 부가가치세 납부 의무가 면제됩니다(부가가치세법 제44조). 귀하의 경우 매출액이 3,000만 원이므로 납부 의무는 면제되지만, **신고는 반드시 하셔야 합니다.** 신고를 하지 않을 경우 가산세가 부과될 수 있습니다.

---
### 주의사항
세금 신고는 개별 상황에 따라 달라질 수 있으므로, 정확한 신고를 위해서는 세무 전문가(세무사)와 상담하시는 것을 권장합니다.
    {context}
    """

    qa_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", qa_system_prompt),
            ("human", "{input}"),
        ]
    )

    llm = ChatOpenAI(model="gpt-4o-mini")
    rag_chain = (
        {"context": retriever | format_docs, "input": RunnablePassthrough()}
        | qa_prompt
        | llm
        | StrOutputParser()
    )
    return rag_chain

rag_chain = chaining()

if "messages" not in st.session_state:
    st.session_state["messages"] = [{"role": "assistant", "content": "헌법에 대해 무엇이든 물어보세요!"}]

for msg in st.session_state.messages:
    st.chat_message(msg['role']).write(msg['content'])

if prompt_message := st.chat_input("질문을 입력해주세요 :)"):
    st.chat_message("human").write(prompt_message)
    st.session_state.messages.append({"role": "user", "content": prompt_message})
    with st.chat_message("ai"):
        with st.spinner("Thinking..."):
            response = rag_chain.invoke(prompt_message)
            st.session_state.messages.append({"role": "assistant", "content": response})
            st.write(response)
