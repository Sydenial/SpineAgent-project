"""
RAG系统主程序
"""

import os
import sys
import logging
from pathlib import Path
from typing import List

# 添加模块路径
sys.path.append(str(Path(__file__).parent))

from dotenv import load_dotenv
from config import DEFAULT_CONFIG, RAGConfig
from rag_modules import (
    DataPreparationModule,
    GuidelineDataPreparationModule,
    IndexConstructionModule,
    RetrievalOptimizationModule,
    GenerationIntegrationModule
)


# 加载环境变量
load_dotenv()

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class SpinalSpineRAGSystem:
    """食谱RAG系统主类"""

    def __init__(self, config: RAGConfig = None):
        """
        初始化RAG系统

        Args:
            config: RAG系统配置，默认使用DEFAULT_CONFIG
        """
        self.last_context = None
        self.config = config or DEFAULT_CONFIG
        self.case_report_data_module = None
        self.guideline_data_module = None
        self.index_module = None
        self.guideline_index_module = None
        self.case_report_retrieval_module = None
        self.guideline_retrieval_module = None
        self.generation_module = None

        # 检查数据路径
        if not Path(self.config.case_report_data_path).exists():
            raise FileNotFoundError(f"数据路径不存在: {self.config.case_report_data_path}")

        if not Path(self.config.guidelines_data_path).exists():
            raise FileNotFoundError(f"数据路径不存在: {self.config.guidelines_data_path}")

    
    def initialize_system(self):
        """初始化所有模块"""
        print("🚀 正在初始化RAG系统...")

        # 1. 初始化数据准备模块
        print("初始化数据准备模块...")
        self.case_report_data_module = DataPreparationModule(self.config.case_report_data_path)
        self.guideline_data_module = GuidelineDataPreparationModule(self.config.guidelines_data_path)

        # 2. 初始化索引构建模块
        print("初始化索引构建模块...")
        self.index_module = IndexConstructionModule(
            model_name=self.config.embedding_model,
            index_save_path=self.config.index_save_path
        )
        self.guideline_index_module = IndexConstructionModule(
            model_name=self.config.embedding_model,
            index_save_path=self.config.guidelines_index_save_path
        )

        # 3. 初始化生成集成模块
        print("🤖 初始化生成集成模块...")
        self.generation_module = GenerationIntegrationModule(
            model_name=self.config.llm_model,
            temperature=self.config.temperature,
            max_tokens=self.config.max_tokens
        )

        print("✅ 系统初始化完成！")
    
    def build_knowledge_base(self):
        """构建知识库"""
        print("\n正在构建知识库...")

        # 1. 尝试加载已保存的索引
        vectorstore = self.index_module.load_index()
        guidelines_vectorstore = self.guideline_index_module.load_index()

        if vectorstore is not None:
            print("✅ 成功加载已保存的case report向量索引！")
            # 仍需要加载文档和分块用于检索模块
            print("加载case_report文档...")
            self.case_report_data_module.load_documents()
            self.guideline_data_module.load_documents()
            print("进行文本分块...")
            chunks = self.case_report_data_module.chunk_documents()
        else:
            print("未找到已保存的索引，开始构建新索引...")

            # 2. 加载文档
            print("加载case_report文档...")
            self.case_report_data_module.load_documents()

            # 3. 文本分块
            print("进行文本分块...")
            chunks = self.case_report_data_module.chunk_documents()

            # 4. 构建向量索引
            print("构建向量索引...")
            vectorstore = self.index_module.build_vector_index(chunks)

            # 5. 保存索引
            print("保存向量索引...")
            self.index_module.save_index()

        if guidelines_vectorstore is not None:
            print("✅ 成功加载已保存的指南向量索引！")
            print("加载指南文档...")
            self.guideline_data_module.load_documents()
            print("进行文本分块...")
            guidelines_chunks = self.guideline_data_module.chunk_documents()
        else:
            print("未找到已保存的索引，开始构建新索引...")

            # 2. 加载文档
            print("加载指南文档...")
            self.guideline_data_module.load_documents()

            # 3. 文本分块
            print("进行文本分块...")
            guidelines_chunks = self.guideline_data_module.chunk_documents()

            # 4. 构建向量索引
            print("构建向量索引...")
            guidelines_vectorstore = self.guideline_index_module.build_vector_index(guidelines_chunks)

            # 5. 保存索引
            print("保存向量索引...")
            self.guideline_index_module.save_index()


        # 6. 初始化检索优化模块
        print("初始化检索优化...")
        self.case_report_retrieval_module = RetrievalOptimizationModule(vectorstore, chunks)
        self.guideline_retrieval_module = RetrievalOptimizationModule(guidelines_vectorstore, guidelines_chunks)

        # 7. 显示统计信息
        stats = self.case_report_data_module.get_statistics()
        guidelines_stats = self.guideline_data_module.get_statistics()
        print(f"\n📊 知识库统计:")
        print(f"   case report文档总数: {stats['total_documents']}")
        print(f"   case report文本块数: {stats['total_chunks']}")
        print(f"   指南文档总数: {guidelines_stats['total_documents']}")
        print(f"   指南文本块数: {guidelines_stats['total_chunks']}")

        print("✅ 知识库构建完成！")
    
    def ask_question(self, question: str, stream: bool = False):
        """
        回答用户问题

        Args:
            question: 用户问题
            stream: 是否使用流式输出

        Returns:
            生成的回答或生成器
        """
        if not all([self.case_report_retrieval_module, self.generation_module, self.guideline_retrieval_module]):
            raise ValueError("请先构建知识库")
        
        print(f"\n❓ 用户问题: {question}")

        # 1. 查询路由
        route_type = self.generation_module.query_router(question)
        print(f"🎯 查询类型: {route_type}")

        # 2. 智能查询重写（根据路由类型）
        if route_type == 'list':
            # 列表查询保持原查询
            rewritten_query = question
            print(f"📝 列表查询保持原样: {question}")
        else:
            # 详细查询和一般查询使用智能重写
            print("🤖 智能分析查询...")
            rewritten_query = self.generation_module.query_rewrite(question)

        # 3. 检索相关子块（自动应用元数据过滤）
        print("🔍 检索相关文档...")
        guideline_chunks = self.guideline_retrieval_module.hybrid_search(
            rewritten_query, top_k=self.config.top_k // 2 + 1  # 合理分配top_k数量
        )
        filters = self._extract_filters_from_query(question)
        if filters:
            print(f"应用过滤条件: {filters}")

            case_chunks = self.case_report_retrieval_module.metadata_filtered_search(
                rewritten_query, filters, top_k=self.config.top_k // 2 + 1
            )
        else:
            case_chunks = self.case_report_retrieval_module.hybrid_search(
                rewritten_query, top_k=self.config.top_k // 2 + 1
            )
        print(f"找到 {len(guideline_chunks)} 个指南文档块, {len(case_chunks)} 个病例文档块")

        if not guideline_chunks and not case_chunks:
            return "抱歉，在指南和病例库中均没有找到相关的脊柱疾病信息。请尝试其他关键词。"

        print("获取完整文档...")
        guideline_docs = self.guideline_data_module.get_parent_documents(guideline_chunks) if guideline_chunks else []
        case_docs = self.case_report_data_module.get_parent_documents(case_chunks) if case_chunks else []

        print(f"对应 {len(guideline_docs)} 个指南完整文档, {len(case_docs)} 个病例完整文档")

        # 6. 生成回答方式
        print("✍️ 融合双库知识生成详细回答...")

        # 这里需要将两类文档传递给生成模块
        dual_context = {
            "guidelines": guideline_docs,
            "case_reports": case_docs
        }
        
        self.last_context = dual_context

        # 根据路由类型自动选择回答模式 (你需要同步修改生成模块的方法入参)
        if stream:
            return self.generation_module.generate_step_by_step_answer_stream(question, dual_context)
        else:
            return self.generation_module.generate_step_by_step_answer(question, dual_context)


    def _extract_filters_from_query(self, query: str) -> dict:
        """
        从用户问题中提取元数据过滤条件（支持多分类匹配）
        """
        filters = {}
        matched_categories = set()  # 使用集合避免重复添加相同的分类
        query_lower = query.lower()

        # 1. 自定义关键词匹配规则
        if "椎间盘" in query_lower:
            matched_categories.add('椎间盘问题')

        if any(keyword in query_lower for keyword in ["转移瘤", "多发性骨髓瘤", "mm"]):
            matched_categories.add('恶性肿瘤')

        # 2. 动态加载所有支持的分类关键词进行补充匹配
        # 这里需要从 DataPreparationModule 中获取映射关系
        from rag_modules.data_preparation import DataPreparationModule
        category_labels = DataPreparationModule.get_supported_categories()

        for cat in category_labels:
            if cat in query:
                matched_categories.add(cat)

        # 3. 如果匹配到了任何分类，将集合转为列表放入 filters
        if matched_categories:
            filters['category'] = list(matched_categories)
            # 例如返回: {'category': ['椎间盘问题', '恶性肿瘤']}

        return filters
    
    def search_by_category(self, category: str, query: str = "") -> List[str]:
        """
        按分类搜索菜品
        
        Args:
            category: 疾病类型
            query: 可选的额外查询条件
            
        Returns:
            菜品名称列表
        """
        if not self.case_report_retrieval_module:
            raise ValueError("请先构建知识库")
        
        # 使用元数据过滤搜索
        search_query = query if query else category
        filters = {"category": category}
        
        docs = self.case_report_retrieval_module.metadata_filtered_search(search_query, filters, top_k=10)
        
        # 提取菜品名称
        disease_names = []
        for doc in docs:
            disease_name = doc.metadata.get('case_report_id', '未知疾病')
            if disease_name not in disease_names:
                disease_names.append(disease_name)
        
        return disease_names
    
    def get_treatment_plan(self, disease_name: str) -> str:
        """
        获取指定疾病

        Args:
            disease_name: 疾病名称

        Returns:
            食材信息
        """
        if not all([self.case_report_retrieval_module, self.generation_module]):
            raise ValueError("请先构建知识库")

        # 搜索相关文档
        docs = self.case_report_retrieval_module.hybrid_search(disease_name, top_k=3)

        # 生成食材信息
        answer = self.generation_module.generate_basic_answer(f"{disease_name}需要什么食材？", docs)

        return answer
    
    def run_interactive(self):
        """运行交互式问答"""
        print("=" * 60)
        print("🍽️  脊柱疾病治疗方案推荐RAG系统  🍽️")
        
        # 初始化系统
        self.initialize_system()
        
        # 构建知识库
        self.build_knowledge_base()
        
        print("\n交互式问答 (输入'退出'结束):")
        
        while True:
            try:
                user_input = input("\n您的问题: ").strip()
                if user_input.lower() in ['退出', 'quit', 'exit', '']:
                    break
                
                # 询问是否使用流式输出
                stream_choice = input("是否使用流式输出? (y/n, 默认y): ").strip().lower()
                use_stream = stream_choice != 'n'

                print("\n回答:")
                if use_stream:
                    # 流式输出
                    for chunk in self.ask_question(user_input, stream=True):
                        print(chunk, end="", flush=True)
                    print("\n")
                else:
                    # 普通输出
                    answer = self.ask_question(user_input, stream=False)
                    print(f"{answer}\n")
                
            except KeyboardInterrupt:
                break
            except Exception as e:
                print(f"处理问题时出错: {e}")
        
        print("\n感谢使用脊柱疾病治疗方案推荐RAG系统！")

