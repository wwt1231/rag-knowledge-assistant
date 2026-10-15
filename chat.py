"""Prompt 与对话引擎"""

from llama_index.core import PromptTemplate
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.chat_engine import CondenseQuestionChatEngine
from llama_index.core.response_synthesizers import get_response_synthesizer
from llama_index.core.llms import ChatMessage, MessageRole


QA_PROMPT = PromptTemplate(
    "以下是从知识库检索到的资料片段：\n"
    "---------------------\n"
    "{context_str}\n"
    "---------------------\n\n"
    "【问题】{query_str}\n\n"
    "【输出格式 - 必须严格遵守】\n"
    "每条信息后面紧跟来源编号，编号格式是方括号数字。\n"
    "编号的作用是标注「这句话来自第几个片段」，不是列表符号。\n\n"
    "正确示范：\n"
    "智慧港口通过物联网实现感知 [1]。宁波港实现效率提升 [2][3]。\n\n"
    "错误示范（不要这样）：\n"
    "[1] 智慧港口通过物联网实现感知。\n"
    "[2] 宁波港实现效率提升。\n\n"
    "【其他要求】\n"
    "1. 用中文回答\n"
    "2. 尽可能全面：资料涉及多个方面时，要全部覆盖，不要遗漏\n"
    "3. 复杂问题用分点、小标题或表格\n"
    "4. 忠于原文：数字、公式、术语、单位不要改动\n"
    "5. 资料里没有的信息，直接说「根据现有资料无法回答」\n\n"
    "【回答模板】\n"
    "- 总结类：研究问题 → 方法 → 主要发现 → 结论\n"
    "- 方法类：具体方法 → 数据来源 → 实验设置 → 参数\n"
    "- 对比类：分点列出异同\n"
)


def build_chat_history(messages):
    history = []
    for m in messages:
        role = m.get("role")
        content = m.get("content")
        if role not in ("user", "assistant") or not content:
            continue
        if not history and role == "assistant":
            continue
        history.append(
            ChatMessage(
                role=MessageRole.USER if role == "user" else MessageRole.ASSISTANT,
                content=content,
            )
        )
    return history


def build_chat_engine(retriever, rerank, messages):
    response_synthesizer = get_response_synthesizer(text_qa_template=QA_PROMPT)
    query_engine = RetrieverQueryEngine(
        retriever=retriever,
        response_synthesizer=response_synthesizer,
        node_postprocessors=[rerank],
    )
    history = build_chat_history(messages)
    return CondenseQuestionChatEngine.from_defaults(
        query_engine=query_engine,
        chat_history=history,
        verbose=False,
    )