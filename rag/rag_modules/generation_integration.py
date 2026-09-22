"""
生成集成模块
"""

# import os
import logging
from typing import List

from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
# from langchain_community.chat_models.moonshot import MoonshotChat
from langchain_deepseek import ChatDeepSeek
from langchain_core.documents import Document
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

logger = logging.getLogger(__name__)

class GenerationIntegrationModule:
    """生成集成模块 - 负责LLM集成和回答生成"""
    
    def __init__(self, model_name: str = "kimi-k2-0711-preview", temperature: float = 0.1, max_tokens: int = 2048):
        """
        初始化生成集成模块
        
        Args:
            model_name: 模型名称
            temperature: 生成温度
            max_tokens: 最大token数
        """
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.llm = None
        self.setup_llm()
    
    def setup_llm(self):
        """初始化大语言模型"""
        logger.info(f"正在初始化LLM: {self.model_name}")

        api_key = "sk-4f4fdb5581e045bc9426add277a90735"
        if not api_key:
            raise ValueError("请设置 MOONSHOT_API_KEY 环境变量")

        self.llm = ChatDeepSeek(
            model=self.model_name,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            api_key=api_key
        )
        
        logger.info("LLM初始化完成")
    
    def generate_basic_answer(self, query: str, context_docs: dict) -> str:
        """
        生成基础回答

        Args:
            query: 用户查询
            context_docs: 上下文文档列表

        Returns:
            生成的回答
        """
        context = self._build_context(context_docs)

        prompt = ChatPromptTemplate.from_template("""
你是一位专业的骨科专家。请根据以下疾病信息回答用户的问题。

用户问题: {question}

相关治疗方案:
{context}

请提供详细、实用的回答。如果信息不足，请诚实说明。

回答:""")

        # 使用LCEL构建链
        chain = (
            {"question": RunnablePassthrough(), "context": lambda _: context}
            | prompt
            | self.llm
            | StrOutputParser()
        )

        response = chain.invoke(query)
        return response
    
    def generate_step_by_step_answer(self, query: str, context_docs: dict) -> str:
        """
        生成分步骤回答

        Args:
            query: 用户查询
            context_docs: 上下文文档列表

        Returns:
            分步骤的详细回答
        """
        context = self._build_context(context_docs)

        prompt = ChatPromptTemplate.from_template("""
你是一位资深的脊柱外科与康复医学专家。请根据提供的医学知识库内容，为用户提供专业的疾病解析与治疗建议。

用户问题: {question}

相关医学背景/知识图谱信息:
{context}

请灵活组织回答，建议包含以下部分（可根据实际内容调整）：

## 📋 建议治疗方案
[基于知识库，列出分阶段的治疗建议，如：保守治疗（药物、物理）、微创干预或手术方案]

## 🧘‍♀️ 康复指导与锻炼
[详细的操作说明，包含具体的动作名称、频率、持续时间以及禁忌事项。如原文包含“康复动作”，请务必详细罗列]

## ⚠️ 专家提醒
[仅在有关键风险点或生活注意事项时包含。优先使用原文中的风险提示。如果没有额外的注意事项，可以基于临床经验总结关键要点，例如“何时需要立即就医”或“生活姿势矫正”，或者完全省略此部分]

注意：
- 保持医学术语的专业性，同时确保普通用户易于理解。
- 严禁强行提供知识库中未提及的医疗诊断建议。
- 重点突出方案的安全性与可操作性（例如：明确标注“请在专业人员指导下进行”）。
- 如果没有具体的康复动作或注意事项，可以省略相应部分。
- 请使用英文进行输出

回答:""")

        chain = (
            {"question": RunnablePassthrough(), "context": lambda _: context}
            | prompt
            | self.llm
            | StrOutputParser()
        )

        response = chain.invoke(query)
        return response

    def query_rewrite(self, query: str) -> str:
        """
        智能查询重写 - 让大模型判断是否需要重写查询

        Args:
            query: 原始查询

        Returns:
            重写后的查询或原查询
        """
        prompt = PromptTemplate(
            template="""
你是一个专业的医学查询分析助手。请分析用户关于脊柱健康的查询，判断是否需要重写，以优化在医学知识库中的检索效果。

原始查询: {query}

分析规则：
1. **具体明确的查询**（直接返回原查询）：
   - 包含具体疾病名称或解剖位：如"腰椎间盘突出怎么治疗"、"颈椎C4-C5节段突出"
   - 明确的症状描述：如"下肢放射性麻木的原因"、"腰椎术后伤口疼痛"
   - 具体的检查/术语询问：如"核磁共振MRI如何看脱出"、"腰椎融合术的禁忌症"

2. **模糊不清或过于口语化的查询**（需要重写）：
   - 过于宽泛：如"腰痛"、"脖子难受"、"脊柱有问题"
   - 缺乏临床信息：如"推荐个药"、"怎么锻炼"、"该看哪个科"
   - 口语化表达：如"腰快断了怎么办"、"脖子转不动了"

重写原则：
- **术语化**：将口语转换为规范的医学描述（如"脖子难受" → "颈椎不适感"）。
- **具象化**：增加“病因分析”、“治疗方案”或“康复锻炼”等引导词。
- **保持原意**：严禁改变用户描述的部位或症状性质。
- **简洁性**：重写后的短语应利于检索。

示例：
- "腰痛" → "腰痛的常见病因与治疗建议"
- "脖子难受" → "颈椎不适的缓解方法与康复锻炼"
- "推荐个药" → "脊柱相关疾病的常用药物指导"
- "腰快断了" → "急性腰部剧烈疼痛的处理方案"
- "腰椎间盘突出怎么治" → "腰椎间盘突出怎么治"（保持原查询）
- "颈椎病吃什么药" → "颈椎病吃什么药"（保持原查询）

请输出最终查询（如果不需要重写就返回原查询）:""",
            input_variables=["query"]
        )

        chain = (
            {"query": RunnablePassthrough()}
            | prompt
            | self.llm
            | StrOutputParser()
        )

        response = chain.invoke(query).strip()

        # 记录重写结果
        if response != query:
            logger.info(f"查询已重写: '{query}' → '{response}'")
        else:
            logger.info(f"查询无需重写: '{query}'")

        return response



    def query_router(self, query: str) -> str:
        """
        查询路由 - 根据查询类型选择不同的处理方式

        Args:
            query: 用户查询

        Returns:
            路由类型 ('list', 'detail', 'general')
        """
        prompt = ChatPromptTemplate.from_template("""
根据用户关于脊柱健康的问题，将其准确分类为以下两种类型之一：

1. 'list' - 用户想要获取各种疾病的治疗方案、科室推荐或药品。
   例如：腰椎间盘突出应该怎么治疗、推荐几种缓解颈椎痛的膏药。

2. 'detail' - 用户询问具体的治疗操作、康复锻炼步骤、手术细节或用药指导。
   例如：小燕飞怎么做、腰椎微创手术的过程是怎样的、这种药一天吃几次、术后如何翻身。

请只返回分类结果：list或detail

用户问题: {query}

分类结果:""")

        chain = (
            {"query": RunnablePassthrough()}
            | prompt
            | self.llm
            | StrOutputParser()
        )

        result = chain.invoke(query).strip().lower()

        # 确保返回有效的路由类型
        if result in ['list', 'detail']:
            return result
        else:
            return 'list'  # 默认类型

    def generate_basic_answer_stream(self, query: str, context_docs: dict):
        """
        生成基础回答 - 流式输出

        Args:
            query: 用户查询
            context_docs: 上下文文档列表

        Yields:
            生成的回答片段
        """
        context = self._build_context(context_docs)

        prompt = ChatPromptTemplate.from_template("""
你是一位专业的骨科专家。请根据以下疾病信息回答用户的问题。

用户问题: {question}

相关治疗方案:
{context}

请提供详细、实用的回答。如果信息不足，请诚实说明。

回答:""")

        chain = (
            {"question": RunnablePassthrough(), "context": lambda _: context}
            | prompt
            | self.llm
            | StrOutputParser()
        )

        for chunk in chain.stream(query):
            yield chunk

    def generate_step_by_step_answer_stream(self, query: str, context_docs: List[Document]):
        """
        生成详细步骤回答 - 流式输出

        Args:
            query: 用户查询
            context_docs: 上下文文档列表

        Yields:
            详细步骤回答片段
        """
        context = self._build_context(context_docs)

        prompt = ChatPromptTemplate.from_template("""
你是一位资深的脊柱外科与康复医学专家。请根据提供的医学知识库内容，为用户提供专业的疾病解析与治疗建议。

用户问题: {question}

相关医学背景/知识图谱信息:
{context}

请灵活组织回答，建议包含以下部分（可根据实际内容调整）：

## 📋 建议治疗方案
[基于知识库，列出分阶段的治疗建议，如：保守治疗（药物、物理）、微创干预或手术方案]

## 🧘‍♀️ 康复指导与锻炼
[详细的操作说明，包含具体的动作名称、频率、持续时间以及禁忌事项。如原文包含“康复动作”，请务必详细罗列]

## ⚠️ 专家提醒
[仅在有关键风险点或生活注意事项时包含。优先使用原文中的风险提示。如果没有额外的注意事项，可以基于临床经验总结关键要点，例如“何时需要立即就医”或“生活姿势矫正”，或者完全省略此部分]

注意：
- 保持医学术语的专业性，同时确保普通用户易于理解。
- 重点突出方案的安全性与可操作性（例如：明确标注“请在专业人员指导下进行”）。
- 如果没有具体的康复动作或注意事项，可以省略相应部分。
- 请使用英文进行输出

回答:""")

        chain = (
            {"question": RunnablePassthrough(), "context": lambda _: context}
            | prompt
            | self.llm
            | StrOutputParser()
        )

        for chunk in chain.stream(query):
            yield chunk

    def _build_context(self, dual_context: dict, max_length: int = 4000) -> str:
        """
        构建双库上下文字符串

        Args:
            dual_context: 包含指南和病例的字典 {"guidelines": List[Document], "case_reports": List[Document]}
            max_length: 最大长度

        Returns:
            格式化的上下文字符串
        """
        # 兼容旧版本的列表传入
        if isinstance(dual_context, list):
            dual_context = {"case_reports": dual_context, "guidelines": []}

        context_parts = []
        current_length = 0

        # 1. 拼接临床指南信息 (Standard of Care)
        guidelines = dual_context.get("guidelines", [])
        if guidelines:
            context_parts.append("【权威临床指南 (Guidelines)】")
            for i, doc in enumerate(guidelines, 1):
                citation = doc.metadata.get('citation_source', '未知指南章节')
                doc_text = f"指南出处 {i}: {citation}\n内容:\n{doc.page_content}\n"
                if current_length + len(doc_text) > max_length / 2:  # 给病例留一半空间
                    break
                context_parts.append(doc_text)
                current_length += len(doc_text)

        # 2. 拼接临床病例信息 (Clinical Practice)
        case_reports = dual_context.get("case_reports", [])
        if case_reports:
            context_parts.append("\n【相关临床病例与实践 (Case Reports)】")
            for i, doc in enumerate(case_reports, 1):
                case_id = doc.metadata.get('case_report_id', '未知病例')
                category = doc.metadata.get('category', '未知分类')
                doc_text = f"病例 {i}: {case_id} | 分类: {category}\n内容:\n{doc.page_content}\n"
                if current_length + len(doc_text) > max_length:
                    break
                context_parts.append(doc_text)
                current_length += len(doc_text)

        if not context_parts:
            return "暂无相关脊柱疾病的指南或病例信息。"

        return "\n" + "=" * 50 + "\n" + "\n".join(context_parts)
