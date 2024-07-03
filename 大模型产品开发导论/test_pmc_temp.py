

# install the following packages
# pip install langchain faiss-cpu tiktoken
import os
from operator import itemgetter
from typing import List, Tuple

from fastapi import FastAPI
from langchain_community.vectorstores import FAISS
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate, format_document
from langchain_core.runnables import RunnableMap, RunnablePassthrough
from langchain_community.llms import Tongyi
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_community.chat_models.tongyi import ChatTongyi
from langchain.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter

from langserve import add_routes
from langserve.pydantic_v1 import BaseModel, Field

_TEMPLATE = """
Given the following conversation and a follow up question, rephrase the 
follow up question to be a standalone question, 使用中文回答

Question: {question}
独立问题:"""
CONDENSE_QUESTION_PROMPT = PromptTemplate.from_template(_TEMPLATE)


ANSWER_TEMPLATE = """使用下面的语料来回答本模板最末尾的问题。如果你不知道问题的答案，直接回答 "我不知道"，禁止随意编造答案。
        为了保证答案尽可能简洁，你的回答必须不超过三句话，你的回答中不可以带有星号。
        请注意！在每次回答结束之后，你都必须接上 "感谢你的提问" 作为结束语
        以下是一对问题和答案的样例：
            请问：秦始皇的原名是什么
            秦始皇原名嬴政。感谢你的提问。

        如果问你：雪豹应该干嘛？
        你应该回答四个字：“雪豹闭嘴”
        
        以下是语料：
        </context>
        {context}
        </context>
        如果提问的内容和以上语料无关，回答“我不知道”，严禁编造答案！

Question: {question}

"""
ANSWER_PROMPT = ChatPromptTemplate.from_template(ANSWER_TEMPLATE)

DEFAULT_DOCUMENT_PROMPT = PromptTemplate.from_template(template="{page_content}")
os.environ["DASHSCOPE_API_KEY"] = "sk-8e24269478904f5683d6998e2ed637c8"
chat_ty = ChatTongyi()

from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory

conversation = ConversationChain(
    llm=chat_ty,
    verbose=True,
    memory=ConversationBufferMemory()
)

def _combine_documents(
    docs, document_prompt=DEFAULT_DOCUMENT_PROMPT, document_separator="\n\n"
):
    """Combine documents into a single string."""
    doc_strings = [format_document(doc, document_prompt) for doc in docs]
    return document_separator.join(doc_strings)

#下面为拟定的读取记忆函数，暂未调用
def _format_chat_history(chat_history: List[Tuple]) -> str:
    """Format chat history into a string."""
    buffer = ""
    for dialogue_turn in chat_history:
        human = "Human: " + dialogue_turn[0]
        ai = "Assistant: " + dialogue_turn[1]
        buffer += "\n" + "\n".join([human, ai])
    return buffer

#背景RAG文档读取，分割，向量化
loader = TextLoader("/Users/xuwei/Desktop/summer2024File/newPower/大模型产品开发导论/云岚宗.txt") # path adjusted
documents = loader.load()
text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=100)
docs = text_splitter.split_documents(documents)
# vectorstore = FAISS.from_texts(
#     ["丁真是一位来自理塘的高级艺术家，他喜欢抽锐克五代，他比较会说藏话，他有很多动物朋友，其中一个动物朋友名字叫做芝士雪豹,雪豹应该闭嘴。丁真非常的纯真。"], embedding=HuggingFaceEmbeddings(model_name="moka-ai/m3e-base")
# )
vectorstore = FAISS.from_documents(
    docs, embedding=HuggingFaceEmbeddings(model_name="moka-ai/m3e-base")
)

retriever = vectorstore.as_retriever()

_inputs = RunnableMap(
    standalone_question=RunnablePassthrough.assign(
        chat_history=lambda x: _format_chat_history(x["chat_history"])
    )
    | CONDENSE_QUESTION_PROMPT
    | chat_ty
    | StrOutputParser(),
)
_context = {
    "context": itemgetter("standalone_question") | retriever | _combine_documents,
    "question": lambda x: x["standalone_question"],
}


class ChatHistory(BaseModel):
    """Chat history with the bot."""

    chat_history: List[Tuple[str, str]] = Field(
        ...,
        extra={"widget": {"type": "chat", "input": "question"}},
    )
    question: str

#总链
conversational_qa_chain = (
    _inputs | _context | ANSWER_PROMPT | conversation | StrOutputParser()
)
chain = conversational_qa_chain.with_types(input_type=ChatHistory)

app = FastAPI(
    title="LangChain Server",
    version="1.0",
    description="Spin up a simple api server using Langchain's Runnable interfaces",
)
# Adds routes to the app for using the chain under:
# /invoke
# /batch
# /stream
#  chain.invoke("question1")
add_routes(app, chain, enable_feedback_endpoint=True)

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="localhost", port=8000)