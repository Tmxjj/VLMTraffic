import streamlit as st
import pandas as pd
import json
import os
import argparse
import re
import time
import altair as alt
# 引入翻译库
try:
    from deep_translator import GoogleTranslator
except ImportError:
    pass # 下面有检查逻辑

# --- 1. 核心工具函数 ---

def format_vlm_response(text):
    if not text: return ""
    # --- 深度清洗 ---
    text = text.replace('\xa0', ' ')
    text = text.replace("Thought: [", "").strip()
    if text.endswith("]"):
        text = text[:-1]
        
    # --- 强制拍平文本 ---
    # 先将所有文本拍平成一行，后面再通过关键词触发换行来重构格式
    clean_text = " ".join([line.strip() for line in text.split('\n') if line.strip()])
    
    # --- 一级标题 ---
    primary_keywords = ["Scene Understanding", "Scene Analysis", "Selection Logic"]
    for key in primary_keywords:
        if key in clean_text:
            clean_text = clean_text.replace(key, f"\n\n**{key}**")
            
    # --- 二级标题 ---
    secondary_keywords = [
        "- [Phase 0]", "- [Phase 1]", "- [Phase 2]", "- [Phase 3]", "- [Phase 4]",
        "- Lane Analysis (Mandatory)", "- Phase Mapping",
        "- Emergency Check", "- Final Condition", 
        "- Rule Identification", "- Conclusion", "- Reasoning"
    ]
    for tag in secondary_keywords:
        if tag in clean_text:
            label = tag.replace("- ", "").strip()
            clean_text = clean_text.replace(tag, f"\n- **{label}**")

    # --- 三级标题 (新增逻辑) ---
    tertiary_keywords = [
        "North Approach", "South Approach", "East Approach", "West Approach",
        "Phase 0", "Phase 1", "Phase 2", "Phase 3"
    ]
    for tag in tertiary_keywords:
        if tag in clean_text:
            label = tag.replace("- ", "").strip()
            # 增加 4 个空格的缩进，使其在 Markdown 渲染中成为子列表
            clean_text = clean_text.replace(tag, f"\n    - **{label}**")

    # --- 处理 Action ---
    if "Action:" in clean_text:
        clean_text = clean_text.replace("Action:", "\n\n---\n### 🏁 Action:")
        
    return clean_text.strip()
@st.cache_data(show_spinner=False)
def translate_text(text):
    if not text: return ""
    try:
        translator = GoogleTranslator(source='auto', target='zh-CN')
        return translator.translate(text)
    except Exception as e:
        return f"翻译失败: {e}"

@st.cache_data
def load_data(file_path):
    data = []
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
    except FileNotFoundError:
        return pd.DataFrame()
    raw_samples = content.split('-----')
    for sample_str in raw_samples:
        clean_str = sample_str.strip()
        if not clean_str: continue
        try:
            obj = json.loads(clean_str)
            data.append(obj)
        except json.JSONDecodeError:
            continue
    return pd.DataFrame(data)

def get_annotated_path(input_path):
    base, ext = os.path.splitext(input_path)
    return f"{base}_annotated{ext}"

# --- 数据加载与保存逻辑升级 ---
def load_existing_annotations(anno_path):
    """
    读取支持缩进格式 (Pretty-printed) 的堆叠 JSON 文件
    """
    annotations = {}
    if not os.path.exists(anno_path):
        return annotations
    
    with open(anno_path, 'r', encoding='utf-8') as f:
        content = f.read().strip()
        if not content:
            return annotations
        
        # 使用 raw_decode 循环解析堆叠的 JSON 对象
        decoder = json.JSONDecoder()
        pos = 0
        while pos < len(content):
            # 跳过空白字符
            try:
                while pos < len(content) and content[pos].isspace():
                    pos += 1
                if pos >= len(content):
                    break
                
                # 解析一个完整的 JSON 对象
                obj, idx = decoder.raw_decode(content[pos:])
                pos += idx
                
                # 存入字典
                if isinstance(obj, dict):
                    uid = f"{obj.get('junction_id')}_{obj.get('step')}"
                    annotations[uid] = obj
            except json.JSONDecodeError:
                # 遇到解析错误跳过或停止
                print(f"解析警告: 在位置 {pos} 附近发现无法解析的内容")
                break
                
    return annotations

def save_or_update_annotation(anno_path, new_record):
    """
    保存标注：
    1. 读取现有文件所有记录。
    2. 检查是否存在相同的 (junction_id, step)。
    3. 如果存在 -> 覆盖；如果不存 -> 追加。
    4. 重写整个文件。
    
    Returns:
        bool: True 表示是覆盖更新 (Update), False 表示是新创建 (Create)
    """
    # 确保目录存在
    os.makedirs(os.path.dirname(os.path.abspath(anno_path)) or ".", exist_ok=True)
    
    # 1. 构造当前记录的唯一标识 (UID)
    current_uid = f"{new_record.get('junction_id')}_{new_record.get('step')}"
    
    all_records = []
    
    # 2. 读取现有所有数据 (解析堆叠的 JSON)
    if os.path.exists(anno_path):
        with open(anno_path, 'r', encoding='utf-8') as f:
            content = f.read().strip()
            if content:
                decoder = json.JSONDecoder()
                pos = 0
                while pos < len(content):
                    try:
                        while pos < len(content) and content[pos].isspace():
                            pos += 1
                        if pos >= len(content):
                            break
                        obj, idx = decoder.raw_decode(content[pos:])
                        pos += idx
                        all_records.append(obj)
                    except json.JSONDecodeError:
                        break # 容错处理

    # 3. 查找并替换，或者追加
    is_update = False
    updated_index = -1
    
    for i, record in enumerate(all_records):
        # 检查 UID
        rec_uid = f"{record.get('junction_id')}_{record.get('step')}"
        if rec_uid == current_uid:
            updated_index = i
            is_update = True
            break
    
    if is_update:
        all_records[updated_index] = new_record # 覆盖旧记录
    else:
        all_records.append(new_record) # 追加新记录

    # 4. 全量覆盖写入文件
    with open(anno_path, 'w', encoding='utf-8') as f:
        for record in all_records:
            # 保持 indent=4 的可读性
            json_str = json.dumps(record, ensure_ascii=False, indent=4)
            f.write(json_str + "\n")
            
    return is_update

# --- 2. 页面与侧边栏配置 ---
def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=str, default="dataset.jsonl")
    args, _ = parser.parse_known_args()
    return args

args = get_args()
default_data_path = args.path

st.set_page_config(layout="wide", page_title="Step 3 Inference Viewer")

st.sidebar.title("🛠️ 设置与筛选")

# 加载数据
df = None
if default_data_path and os.path.exists(default_data_path):
    df = load_data(default_data_path)
else:
    st.error(f"❌ 找不到文件: `{default_data_path}`")
    st.stop()

# 加载标注路径
annotated_file_path = get_annotated_path(default_data_path)
existing_annos = load_existing_annotations(annotated_file_path)

st.sidebar.info(f"已标注: {len(existing_annos)} 条 | 保存至: {os.path.basename(annotated_file_path)}")
st.sidebar.markdown("---")

# 路径前缀设置
path_prefix_to_replace = st.sidebar.text_input("数据路径前缀 (Old)", "/home/jyf/code/trafficVLM/code/VLMTraffic/")
local_path_prefix = st.sidebar.text_input("本地路径前缀 (New)", "./")

# 筛选逻辑
if df is not None and not df.empty:
    st.sidebar.markdown("---")
    j_ids = ["All"] + sorted(df['junction_id'].astype(str).unique().tolist()) if 'junction_id' in df else []
    labels = ["All"] + sorted(df['label'].astype(str).unique().tolist()) if 'label' in df else []
    
    sel_jid = st.sidebar.selectbox("Junction ID", j_ids)
    sel_label = st.sidebar.selectbox("Label (Auto)", labels)
    
    if 'step' in df:
        min_s, max_s = int(df['step'].min()), int(df['step'].max())
        sel_step = st.sidebar.slider("Step Range", min_s, max_s, (min_s, max_s))
    
    filtered_df = df.copy()
    if sel_jid != "All": filtered_df = filtered_df[filtered_df['junction_id'].astype(str) == sel_jid]
    if sel_label != "All": filtered_df = filtered_df[filtered_df['label'] == sel_label]
    if 'step' in df: filtered_df = filtered_df[(filtered_df['step'] >= sel_step[0]) & (filtered_df['step'] <= sel_step[1])]

    if len(filtered_df) == 0:
        st.warning("无匹配数据")
        st.stop()
    
    # 翻页控制
    if 'current_index' not in st.session_state: st.session_state.current_index = 0
    if st.session_state.current_index >= len(filtered_df): st.session_state.current_index = 0

    c1, c2, c3 = st.sidebar.columns([1, 2, 1])
    if c1.button("⬅️"): st.session_state.current_index = max(0, st.session_state.current_index - 1)
    if c3.button("➡️"): st.session_state.current_index = min(len(filtered_df) - 1, st.session_state.current_index + 1)
    c2.markdown(f"<center>{st.session_state.current_index + 1} / {len(filtered_df)}</center>", unsafe_allow_html=True)

    row = filtered_df.iloc[st.session_state.current_index]
    current_uid = f"{row.get('junction_id')}_{row.get('step')}"
    prev_anno = existing_annos.get(current_uid, None)

    # --- 主界面 ---
    lbl = row.get('label', 'N/A')
    lbl_color = ":green" if lbl == 'accepted' else ":red"
    anno_status = "✅ 已人工标注" if prev_anno else "⬜ 未标注"
    
    st.markdown(f"### 🚦 JID: `{row.get('junction_id')}` | Step: `{row.get('step')}` | {lbl_color}[**{lbl}**] | {anno_status}")

    # 指标栏
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Current Phase", row.get('current_phase', 'N/A'))
    m2.metric("VLM Action", row.get('vlm_action', 'N/A'))
    opt_act = row.get('optimal_action', 'N/A')
    vlm_act = row.get('vlm_action', 'N/A')
    is_correct = (str(opt_act) == str(vlm_act))
    m3.metric("Optimal Action", opt_act, delta="Correct" if is_correct else "Incorrect", delta_color="normal" if is_correct else "inverse")
    m4.metric("Best Metric Val", round(float(row.get('metric_val', 0)), 2))

    st.divider()

    # ==========================================
    # 📐 布局修改：三栏布局
    # [Visuals (40%)]  [VLM Analysis (30%)]  [Annotation (30%)]
    # ==========================================
    col_visual, col_analysis, col_anno = st.columns([4, 3, 3])

    # --- 第1栏：Visual Input ---
    with col_visual:
        st.subheader("🖼️ Visual Input")
        raw_path = row.get('image_path', '')
        if path_prefix_to_replace and local_path_prefix and raw_path:
            image_path = raw_path.replace(path_prefix_to_replace, local_path_prefix)
        else:
            image_path = raw_path
            
        if image_path and os.path.exists(image_path):
            st.image(image_path, use_column_width=True)
            st.caption(f"Path: {image_path}")
        else:
            st.warning(f"Image not found: {image_path}")

        with st.expander("System Prompt"):
            st.text(row.get('prompt', ''))
        with st.expander("Think Process (Chain of Thought)"):
            st.write(row.get('vlm_think_process', ''))

    # --- 第2栏：VLM Analysis & Charts ---
    with col_analysis:
        st.subheader("🤖 Original VLM Analysis")
        raw_response = row.get('vlm_response_raw', '')
        # display_text = raw_response
        display_text = format_vlm_response(raw_response)

        # 翻译
        enable_trans = st.toggle("🇨🇳 中文翻译 (Original)", value=False)
        if enable_trans:
            translated_text = translate_text(display_text)
            st.markdown(translated_text, unsafe_allow_html=True)
            with st.expander("Show Original"):
                st.markdown(display_text, unsafe_allow_html=True)
        else:
            st.markdown(display_text, unsafe_allow_html=True)

        # --- GT Vehicle Counts ---
        st.markdown("---")
        with st.expander("✅ Ground Truth Vehicle Counts"):
            gt_counts = row.get('gt_vehicle_counts', {})
            if gt_counts:
                st.json(gt_counts)
            else:
                st.warning("No GT Vehicle Counts found in this record.")
                
        # --- Auto Annotated Raw Response ---
        st.markdown("---")
        with st.expander("🤖 Auto-Annotated Valid Response (GT Corrected)"):
            auto_response = row.get('vlm_response_raw_auto_annotated', '')
            if auto_response:
                display_auto_text = format_vlm_response(auto_response)
                st.markdown(display_auto_text, unsafe_allow_html=True)
            else:
                st.warning("No auto-annotated response found.")

        st.markdown("---")
        st.caption("📊 Reward Metrics")
        metrics_dict = row.get('all_metrics', {})
        if metrics_dict:
            try:
                metrics_df = pd.DataFrame(list(metrics_dict.items()), columns=['Phase', 'Reward'])
                metrics_df['Phase'] = metrics_df['Phase'].astype(str)
                
                # 基础图表：柱状图
                bars = alt.Chart(metrics_df).mark_bar().encode(
                    x=alt.X('Phase:N', sort=None, title='Phase'),
                    y=alt.Y('Reward:Q', title='Reward Value'),
                    # 根据您的逻辑定义颜色（这里简单示例，也可根据 vlm_act 逻辑上色）
                    color=alt.condition(
                        alt.datum.Phase == str(vlm_act),
                        alt.value('#ff4b4b'), # VLM 选择的颜色
                        alt.value('#e6e9ef')  # 默认颜色
                    )
                )

                # 核心：添加数值文本层
                text = bars.mark_text(
                    align='center',
                    baseline='bottom',
                    dy=-5  # 将文字向上偏移 5 像素，避免压在柱子上
                ).encode(
                    text=alt.Text('Reward:Q', format='.2f') # 显示两位小数
                )

                # 叠加显示
                st.altair_chart(bars + text, use_container_width=True)
                
            except Exception as e:
                st.error(f"图表渲染失败: {e}")

        # --- 第3栏：Step 3 Inference Result (放在右侧) ---
        with col_anno:
            st.subheader("🚀 Step 3 Inference")
            
            st.markdown(f"**Step 3 Action:** `{row.get('step3_vlm_action', 'N/A')}`")
            step3_lbl = row.get('step3_label', 'N/A')
            step3_color = ":green" if step3_lbl == 'accepted' else ":red"
            st.markdown(f"**Step 3 Result:** {step3_color}[**{step3_lbl}**]")
            
            st.markdown("---")
            with st.expander("Step 3 Prompt"):
                st.text(row.get('step3_prompt', ''))
                
            st.markdown("---")
            st.subheader("Step 3 Response")
            step3_raw = row.get('step3_vlm_response_raw', '')
            if step3_raw:
                display_step3_text = format_vlm_response(step3_raw)
                # 翻译
                enable_trans_step3 = st.toggle("🇨🇳 中文翻译 (Step 3)", value=False)
                if enable_trans_step3:
                    translated_step3 = translate_text(display_step3_text)
                    st.markdown(translated_step3, unsafe_allow_html=True)
                    with st.expander("Show Original (Step 3)"):
                        st.markdown(display_step3_text, unsafe_allow_html=True)
                else:
                    st.markdown(display_step3_text, unsafe_allow_html=True)
            else:
                if row.get('step3_error'):
                    st.error(f"Inference Error: {row.get('step3_error')}")
                else:
                    st.warning("No Step 3 response generated.")

else:
    st.info("请加载数据。")