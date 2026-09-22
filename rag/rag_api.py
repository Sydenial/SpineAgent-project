# rag_api.py
from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn
from spinal_system import SpinalSpineRAGSystem  # 替换成你实际的 py 文件名

app = FastAPI()

# 1. 启动时自动初始化 RAG 系统并常驻内存
print("正在启动 RAG API 服务...")
rag_system = SpinalSpineRAGSystem()
rag_system.initialize_system()
rag_system.build_knowledge_base()


class TreatmentRequest(BaseModel):
    diagnosis_report: str  # 接收来自 Streamlit 的诊断报告


@app.post("/api/get_treatment")
def get_treatment(req: TreatmentRequest):
    print(f"收到诊断报告，正在生成治疗方案...")

    # 2. 调用 RAG 生成治疗方案 (使用非流式，方便通过网络传输)
    # 把收到的诊断报告作为问题丢给 RAG
    treatment_plan = rag_system.ask_question(req.diagnosis_report, stream=False)

    # 3. 提取刚刚保存的检索文档
    context = getattr(rag_system, 'last_context', {"guidelines": [], "case_reports": []})

    # 提取文档内容 (适配 LangChain 的 Document 对象格式)
    guidelines_text = [getattr(doc, 'page_content', str(doc)) for doc in context.get("guidelines", [])]
    cases_text = [getattr(doc, 'page_content', str(doc)) for doc in context.get("case_reports", [])]

    # 4. 统一打包返回给 Streamlit 前端
    return {
        "treatment_plan": treatment_plan,
        "guidelines": guidelines_text,
        "case_reports": cases_text
    }


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8002)