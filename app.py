import streamlit as st
import base64
from util.dcm2base64 import process_uploaded_dicoms_to_base64
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage


st.set_page_config(page_title="AI-Assisted Multimodal Spinal Diagnosis System", layout="wide")

st.title("🏥 AI-Assisted Multimodal Spinal Diagnosis System")
st.markdown("Supports uploading multiple MRI DICOM files for assisted analysis of degenerative diseases, tumors, inflammation, and trauma.")

# 划分左右两栏，左边输入，右边展示
col1, col2 = st.columns([1, 1])

with col1:
    # st.subheader("1. Enter Clinical Information")
    # # 预设一段典型的输入提示
    # clinical_text = st.text_area(
    #     "Please enter patient information (e.g., gender, age). If unavailable, please enter N/A",
    #     value="Example: Gender: Female; Age: 49.",
    #     height=150
    # )
    #
    # st.subheader("2. Upload Imaging Data")
    # uploaded_files = st.file_uploader(
    #     "📁 Drag and drop the entire folder containing the MRI DICOM sequences here.",
    #     accept_multiple_files=True
    # )
    #
    # submit_button = st.button("Submit to Diagnostic Agent", type="primary")

    st.subheader("1. Enter Clinical Information")

    # 使用子列将性别和年龄并排显示，使得UI更加紧凑美观
    col_gender, col_age = st.columns(2)

    with col_gender:
        gender = st.selectbox("Gender", options=["Female", "Male", "N/A"], index=0)

    with col_age:
        # 使用 number_input 限制只能输入数字，并标明单位
        age = st.number_input("Age (years)", min_value=0, max_value=150, value=49, step=1)

    # 新增补充信息的多行文本框
    supp_info = st.text_area(
        "Supplementary Information",
        placeholder="e.g., family history, past medical history, symptoms. Enter N/A if unavailable.",
        value="",
        height=100
    )
    if gender == "Male":
        gender = "男"
    elif gender == "Female":
        gender = "女"
    else:
        gender = "N/A"

    # 动态将分散的字段拼接成一段完整的文本，供后续大模型分析使用
    clinical_text_1 = f"性别: {gender}; 年龄: {age} 岁."
    if supp_info.strip():
        clinical_text_2 = clinical_text_1 + f" Supplementary Info: {supp_info}"

    st.subheader("2. Upload Imaging Data")
    uploaded_files = st.file_uploader(
        "📁 Drag and drop the entire folder containing the MRI DICOM sequences here.",
        accept_multiple_files=True
    )

    submit_button = st.button("Submit to Diagnostic Agent", type="primary")

with col2:
    st.subheader("3. Image Preview and Analysis Results")

    if submit_button:
        if not uploaded_files:
            st.warning("⚠️ Please upload DICOM imaging data first.")
        elif not clinical_text_1:
            st.warning("⚠️ Please enter clinical information.")
        else:
            try:
                # --- 第一步：处理 DICOM 文件，获取二维列表 ---
                with st.spinner("Parsing DICOM files and extracting core sagittal sequences..."):
                    # 这里的返回值必须是你修改后的 util/dcm2base64.py
                    # grouped_base64_images 结构: [[切片1的T1, 切片1的T2...], [切片2的T1...], ...]
                    grouped_base64_images, grouped_sequence_names = process_uploaded_dicoms_to_base64(uploaded_files)

                num_positions = len(grouped_base64_images)
                st.success(f"Parsing successful! Extracted sequence groups for {num_positions} core slice positions.")

                st.markdown("---")
                st.markdown("### Agent Slice-by-Slice Analysis Results")

                collected_results = []

                # 提前初始化大模型，避免在循环里反复初始化
                llm_diagnosis = ChatOpenAI(
                    model="qwen-spine-vlm",
                    base_url="http://localhost:8000/v1",
                    api_key="empty",
                    temperature=0.2,
                    max_tokens=4096
                )

                llm_summary = ChatOpenAI(
                    model="deepseek-chat",
                    base_url="https://api.deepseek.com",
                    max_tokens=8192,  # 替换为对应厂商的 API 地址
                    api_key="sk-4f4fdb5581e045bc9426add277a90735",
                    temperature=0.1  
                )

                # --- 第二步：循环 5 次，逐个位置进行推理 ---
                # 使用 zip 将图片组和对应的序列名组一一对应遍历
                for pos_idx, (b64_group, seq_group) in enumerate(zip(grouped_base64_images, grouped_sequence_names)):

                    # 1. 为当前切片位置生成专属的 UI 面板
                    with st.expander(f"Sagittal Slice {pos_idx + 1} Analysis Report", expanded=True):

                        # 先在面板里展示这组切片的预览图
                        preview_cols = st.columns(len(b64_group))
                        for img_idx, (b64_str, seq_name) in enumerate(zip(b64_group, seq_group)):
                            with preview_cols[img_idx]:
                                img_data = base64.b64decode(b64_str)
                                st.image(img_data, caption=f"Figure {img_idx + 1}: {seq_name}", width=300)

                        # 2. 动态组装当前这组图的 Prompt 序列信息
                        seq_descriptions = []
                        for i, seq_name in enumerate(seq_group):
                            seq_descriptions.append(f"第 {i + 1} 张图片是 {seq_name} 序列下的MRI医学影像")
                        seq_info_str = "，".join(seq_descriptions) + "。"

                        # 3. 构造当前这 1 次推理的完整文本
                        dynamic_input = f"""【角色设定】
你是一位脊柱影像诊断专家。请根据MRI影像和患者临床信息进行诊断。

【患者信息】
{clinical_text_1}
当前切片影像序列说明：{seq_info_str}

【诊断参考指南】
1. 血管瘤：T1/T2高信号（栅栏状/点状），属良性病变。
2. 转移瘤：T1低信号，STIR高信号，形态弥漫/膨胀，常伴软组织肿块，边界不清。
3. 骨髓瘤：T1低/等信号，STIR高信号，需结合多发性及实验室检查，影像似转移瘤。
4. 压缩性骨折：
    - 信号：T1WI片状低信号，STIR/T2显著高信号（提示骨髓水肿）。
    - 形态：椎体变扁/楔形变。可见线样低信号骨折线。
    - 鉴别：若保留部分正常骨髓信号或见'双线征'提示良性；若椎体膨胀、软组织肿块提示病理性。
5. 锥体骨折：椎体变扁，但STIR序列无高信号（无水肿），T1信号已恢复或硬化。
6. 退行性变：椎间盘高度降低、终板骨赘形成，终板炎。
【分析要求】
1. 首先观察STIR/T2序列：是否有异常高信号？（区分急慢性/有无病变）。
2. 对应T1序列：观察是否为低信号（水肿/肿瘤）还是高信号（脂肪/血管瘤）。
3. 观察形态：椎体高度是否丢失？是否有骨折线？是否有膨胀性改变？
4. 综合判断：结合指南排除干扰项。

【输出格式】
请严格输出一个 JSON 对象，格式如下：
[{{ "type": "疾病类型1", "loc": "病变部位1" }},{{ "type": "疾病类型2", "loc": "病变部位2" }},...]
示例：[{{ "type": "新鲜骨折", "loc": "L1" }},{{ "type": "转移性肿瘤", "loc": "C4椎体" }},...]"""

                        # 4. 组装 LangChain 消息
                        message_content = [{"type": "text", "text": dynamic_input}]
                        for b64 in b64_group:
                            message_content.append({
                                "type": "image_url",
                                "image_url": {"url": f"data:image/jpeg;base64,{b64}"}
                            })
                        messages = [HumanMessage(content=message_content)]

                        # 5. 发送请求给模型 (使用流式输出)
                        with st.spinner(f"Analyzing slice {pos_idx + 1}..."):
                            # 【注意】：这里使用的是视觉模型 llm_vision
                            response = llm_diagnosis.invoke(messages)
                            CN_layer_json_str = response.content
                            trans2EN_prompt = f"""**Role:**
Act as an expert medical translator and a board-certified musculoskeletal radiologist. Your task is to translate structured Chinese radiological reasoning and diagnostic conclusions into formal, academic medical English.

**Instructions:**
1. **Preserve Structure:** You must strictly preserve all XML tags (e.g., `<analysis>`, `<segmental_reasoning>`, `<step>`, `<loc>`, `<observation>`, `<logic>`, `<conclusion>`) and the exact JSON formatting for the diagnostic conclusions. Do not translate the XML tag names or the JSON keys (`type`, `loc`). 
2. **Academic Terminology:** Use standard, precise international radiological terminology. 
3. **Tone and Style:** Maintain an objective, concise, and professional clinical tone standard in English radiology reports. Avoid conversational language.

**Input Format:**
{response.content}

**Expected Output:**
Provide ONLY the translated XML and JSON. Do not include any introductory or concluding remarks.

**Example Translation:**

Input:
<step>
<loc>L5/S1</loc>
<observation>椎间盘向后上方脱出，造成椎管局部狭窄并压迫马尾神经；L5/S1相对终板显示长T1长T2异常信号。</observation>
<logic>髓核突破后纵韧带向上游离脱出，直接导致椎管容积减小及马尾神经受压；终板区域的长T1长T2信号提示局部水肿，符合Modic I型改变（终板炎）。</logic>
<conclusion>椎间盘脱出、椎管狭窄、马尾受压、椎体终板炎、椎间盘变性</conclusion>
</step>

Output:
<step>
<loc>L5/S1</loc>
<observation>The intervertebral disc is extruded posterosuperiorly, causing focal spinal canal stenosis and compression of the cauda equina; the opposing endplates of L5/S1 demonstrate abnormal long T1 and long T2 signals.</observation>
<logic>The nucleus pulposus has breached the posterior longitudinal ligament with superior upward extrusion, directly reducing the spinal canal volume and compressing the cauda equina; the long T1 and long T2 signals in the endplate regions indicate local edema, consistent with Modic type I changes (endplate inflammation).</logic>
<conclusion>Disc extrusion, spinal canal stenosis, cauda equina compression, endplate inflammation (Modic changes), disc degeneration</conclusion>
</step>
"""
                            trans_messages = [HumanMessage(content=trans2EN_prompt)]
                            EN_response = llm_summary.invoke(trans_messages)
                            layer_json_str = EN_response.content
                            st.write(layer_json_str)

                            collected_results.append(
                                f"切片 {pos_idx + 1}:\n{CN_layer_json_str}"
                            )

                # --- 循环结束，开始执行【空间连续性汇总与过滤】 ---
                st.markdown("---")
                st.markdown("### 📝 Final Comprehensive Diagnostic Report (Spatial Continuity Verified)")

                with st.spinner("Comparing adjacent slices to filter out isolated false positives and generating the final report..."):
                    all_layers_text = "\n\n".join(collected_results)

                    summary_prompt = f"""【角色设定】
你是一位严谨的资深脊柱外科主任医师。现在有一台 AI 视觉诊断系统对患者的脊柱 MRI 进行了逐层扫描（共5层核心矢状位，按人体从左至右的真实物理空间坐标排序），并生成了初步的病灶 JSON 列表。

【任务目标】
请你审核这 5 层的初步诊断结果，利用“三维空间分布”法则剔除 AI 的误判（幻觉），综合研判病灶的具体位置，并用英文输出最终的结构化数据与连贯的综合诊断报告。

【病灶判定与空间剔除法则】（极其重要，请严格执行！）
脊柱的真实病变（如肿瘤、骨折、退行性变等）具有物理体积。请按以下逻辑进行过滤和确认：
1. 真实病灶确认（含隔层现象）：只要同一种疾病类型在 5 层扫描中出现 **两次或两次以上**（无论是在连续切片如1和2中，还是在**隔层**如1和3、2和4中出现），均判定为**真实存在该病灶**。
2. 孤立病灶剔除：如果某种疾病类型**仅仅单独出现**在中间的某单一层（如仅第2层，或仅第3层，或仅第4层），且其他所有层（含隔层）均未提及，请判定为 AI 视觉误判，必须将其**完全剔除**。
3. 边缘层校验（疑似病灶）：对于处于边缘的第1层或第5层，如果其中报告了某病灶，但其他任何层（包括隔层）都没有该病灶，判定为边缘**可能**存在该病灶，不直接剔除，需列入“医生关注”范围。
4. 病灶位置综合研判（交集法则）：当确定存在某病灶后，请结合各层对该病灶位置的描述进行综合判断：
   - **位置有交集**：如果不同层中提到了相同的位置，则该交集位置判定为【确定的病变位置】；其余未形成交集的位置表述为【可能存在病变的位置】。
   - **位置无交集**：如果各层对该病灶的位置描述完全不同（无交集），则所有被提及的位置均判定为【可能存在病变的位置】（提醒医生仔细判断），该病灶无绝对确定的位置。

【输入数据：按物理空间 X 坐标排序的单层诊断结果】
{all_layers_text}

【输出要求】
请在认真执行上述法则后，输出最终结果，必须严格包含以下两部分：

### 1. 结构化 JSON 输出
请严格输出一个 JSON 对象数组，将确定位置与可能位置分开表述。格式如下：
[
  {{ 
    "type": "疾病类型1", 
    "confirmed_loc": "确定的病变部位（若无交集则填'无'）", 
    "possible_loc": "可能存在病灶的部位（若无则填'无'）" 
  }},
  ...
]
示例：
[
  {{ "type": "新鲜骨折", "confirmed_loc": "L1椎体", "possible_loc": "L3椎体" }},
  {{ "type": "转移性肿瘤", "confirmed_loc": "无", "possible_loc": "C4椎体,C5椎体" }}
]

### 2. 最终临床诊断结论
将判断出来的内容表示成符合医生书写习惯的影像学诊断报告，主要包含两部分结构：
- **【确定病灶及位置】**：详细汇总经过判定真实存在，且有确定位置（交集位置）的病变。
- **【医生重点关注（疑似及可能病灶）】**：汇总边缘层孤立出现的疑似病灶，以及虽确认有病灶但属于“可能位置（无交集或非交集部分）”的情况，明确提醒医生在阅片时仔细复核这些位置。

注意：
- 请使用英文进行输出
"""

                    # 纯文本请求不需要图片，直接包装 Prompt
                    summary_messages = [HumanMessage(content=summary_prompt)]

                    summary_placeholder = st.empty()
                    final_report = ""

                    # 【注意】：这里调用的变成了你的新模型 llm_summary
                    for chunk in llm_summary.stream(summary_messages):
                        final_report += chunk.content
                        summary_placeholder.markdown(final_report + "▌")
                    summary_placeholder.markdown(final_report)

                # ==========================================
                # 新增模块：调用 RAG 治疗系统
                # ==========================================
                import requests
                import time

                st.markdown("---")
                st.markdown("### 💊 Personalized Treatment Plan Recommendation (RAG-Enhanced Retrieval)")

                # 满足要求 (1)：将 final_report（诊断输出）作为治疗模块的输入
                with st.spinner("Querying RAG knowledge base for relevant medical guidelines and similar cases..."):
                    try:
                        # 向环境 B 中的 RAG API 发送请求
                        response = requests.post(
                            "http://localhost:8002/api/get_treatment",
                            json={"diagnosis_report": final_report},
                            timeout=120  # 给大模型留出足够的生成时间
                        )

                        if response.status_code == 200:
                            rag_data = response.json()

                            # 满足要求 (2)：折叠显示检索到的指南和病历文档
                            col_g, col_c = st.columns(2)
                            with col_g:
                                with st.expander("📚 Referenced Medical Guidelines (Click to expand)", expanded=False):
                                    if rag_data["guidelines"]:
                                        for idx, g_text in enumerate(rag_data["guidelines"]):
                                            st.info(f"**指南 {idx + 1}**:\n\n{g_text}")
                                    else:
                                        st.write("No relevant guidelines found.")

                            with col_c:
                                with st.expander("🩺 Similar Cases (Click to expand)", expanded=False):
                                    if rag_data["case_reports"]:
                                        for idx, c_text in enumerate(rag_data["case_reports"]):
                                            st.success(f"**病例 {idx + 1}**:\n\n{c_text}")
                                    else:
                                        st.write("No similar cases found.")

                            # 满足要求 (3)：输出对应的治疗方案
                            st.markdown("#### Comprehensive Treatment Plan")

                            # 伪流式输出
                            treatment_placeholder = st.empty()
                            displayed_treatment = ""

                            for char in rag_data["treatment_plan"]:
                                displayed_treatment += char
                                treatment_placeholder.markdown(displayed_treatment + "▌")
                                time.sleep(0.005)  # 控制打字速度，越小越快

                            treatment_placeholder.markdown(displayed_treatment)

                        else:
                            st.error(f"RAG System Error: {response.status_code}")

                    except requests.exceptions.ConnectionError:
                        st.error(
                            "无法连接到 RAG 治疗系统！请确保你已经在另一个终端运行了 `python rag_api.py` 并且 8002 端口已开放。")

            except Exception as e:
                st.error(f"处理过程中发生错误: {str(e)}")
