import streamlit as st
import pandas as pd
import json
import os
import argparse
import re
import time
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
    clean_text = " ".join([line.strip() for line in text.split('\n') if line.strip()])
    # --- 一级标题 ---
    primary_keywords = ["1. Scene Understanding", "2. Scene Analysis", "3. Selection Logic"]
    for key in primary_keywords:
        if key in clean_text:
            clean_text = clean_text.replace(key, f"\n\n**{key}**")
    # --- 二级标题 ---
    secondary_keywords = [
        "- [Phase 0]", "- [Phase 1]", "- [Phase 2]", "- [Phase 3]", "- [Phase 4]",
        "- Emergency Check", "- Final Condition", 
        "- Rule Identification", "- Conclusion", "- Reasoning"
    ]
    for tag in secondary_keywords:
        if tag in clean_text:
            label = tag.replace("- ", "").strip()
            clean_text = clean_text.replace(tag, f"\n- **{label}**")
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

def save_annotation_line(anno_path, record):
    """
    以缩进格式 (Indent=4) 追加保存 JSON
    """
    # 确保目录存在
    os.makedirs(os.path.dirname(os.path.abspath(anno_path)) or ".", exist_ok=True)
    
    with open(anno_path, 'a', encoding='utf-8') as f:
        # 使用 ensure_ascii=False 保证中文正常显示
        # 使用 indent=4 保证可读性
        json_str = json.dumps(record, ensure_ascii=False, indent=4)
        f.write(json_str + "\n")

# --- 2. 页面与侧边栏配置 ---
def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=str, default="dataset.jsonl")
    args, _ = parser.parse_known_args()
    return args

args = get_args()
default_data_path = args.path

st.set_page_config(layout="wide", page_title="TrafficVLM Case Viewer")

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
        st.subheader("🤖 VLM Analysis")
        raw_response = row.get('vlm_response_raw', '')
        display_text = format_vlm_response(raw_response)

        # 翻译
        enable_trans = st.toggle("🇨🇳 中文翻译", value=False)
        if enable_trans:
            translated_text = translate_text(display_text)
            st.markdown(translated_text, unsafe_allow_html=True)
            with st.expander("Show Original"):
                st.markdown(display_text, unsafe_allow_html=True)
        else:
            st.markdown(display_text, unsafe_allow_html=True)

        st.markdown("---")
        st.caption("📊 Reward Metrics")
        metrics_dict = row.get('all_metrics', {})
        if metrics_dict:
            try:
                metrics_df = pd.DataFrame(list(metrics_dict.items()), columns=['Phase', 'Reward'])
                metrics_df['Phase'] = metrics_df['Phase'].astype(str)
                colors = []
                for p in metrics_df['Phase']:
                    if str(p) == str(vlm_act): colors.append("#ff4b4b")
                    elif str(p) == str(opt_act): colors.append("#09ab3b")
                    else: colors.append("#e6e9ef")
                st.bar_chart(metrics_df.set_index('Phase')['Reward'], color=colors if len(colors)==len(metrics_df) else None)
            except: pass

    # --- 第3栏：Human Annotation (放在右侧) ---
    with col_anno:
        st.subheader("✍️ Human Annotation")
        
        # --- 1. 准备默认值 ---
        default_tag = "无误"
        default_text = display_text
        default_remark = "" # 新增备注默认值

        if prev_anno:
            default_tag = prev_anno.get('human_label', '无误')
            default_remark = prev_anno.get('error_reason', "") # 读取历史备注
            
            # 如果是"无误"，默认文本重置为当前生成的 display_text
            if prev_anno.get('human_label') == '无误':
                 default_text = display_text
            else:
                 default_text = prev_anno.get('corrected_response', display_text)
        
        st.info(f"当前状态: **{default_tag}**")

        # --- 2. 标签选择 (移出 Form 以支持交互) ---
        tag_options = ["无误", "视觉理解有误", "决策推理有误"]
        try:
            idx = tag_options.index(default_tag)
        except ValueError:
            idx = 0
        
        selected_tag = st.radio(
            "评估标签 (Select Label):", 
            tag_options, 
            index=idx, 
            horizontal=True,
            key=f"radio_{current_uid}" 
        )

        # 逻辑判断
        is_error = (selected_tag != "无误") # 是否为错误类型
        is_disabled_edit = (not is_error)   # 是否禁用编辑 (仅无误时禁用)

        # --- 3. 错误原因备注 (新增功能) ---
        error_remark = ""
        if is_error:
            st.markdown("---")
            st.caption("📝 **错误原因说明 (Error Explanation):**")
            error_remark = st.text_input(
                "简要说明错误点 (例如: 未识别出救护车 / 拥堵判断错误)",
                value=default_remark,
                key=f"remark_{current_uid}",
                help="请简要描述模型具体哪里错了"
            )
        
        # --- 4. 修正回复区域 ---
        st.markdown("---")
        st.caption("📝 **修正回复 (Corrected Response):**")
        
        if is_disabled_edit:
            st.warning("🔒 标签为“无误”时，内容不可编辑。")

        # 使用 Tabs 分离编辑和预览
        tab_edit, tab_preview = st.tabs(["✏️ 编辑 (Edit)", "👁️ 实时预览 (Preview)"])
        
        with tab_edit:
            corrected_text = st.text_area(
                "Markdown Source", 
                value=default_text, 
                height=500,
                label_visibility="collapsed",
                disabled=is_disabled_edit, 
                key=f"text_{current_uid}"
            )
        
        with tab_preview:
            preview_content = display_text if is_disabled_edit else corrected_text
            if preview_content:
                st.markdown(preview_content, unsafe_allow_html=True)
            else:
                st.caption("暂无内容")

        st.markdown("---")

        # --- 5. 保存提交 ---
        with st.form(key=f"save_form_{current_uid}"):
            submitted = st.form_submit_button("💾 保存标注 (Save)", use_container_width=True)

            if submitted:
                # 校验逻辑：如果是错误类型，建议填写备注（可选，这里不做强制拦截，只做数据处理）
                
                final_saved_text = ""
                final_remark = ""

                if selected_tag == "无误":
                    final_saved_text = display_text 
                    final_remark = "" # 无误时清空备注
                else:
                    final_saved_text = corrected_text.strip()
                    final_remark = error_remark.strip()

                # 构建保存对象
                save_record = row.to_dict()
                save_record['human_label'] = selected_tag
                save_record['corrected_response'] = final_saved_text
                save_record['error_reason'] = final_remark # [新增] 保存备注字段
                
                if 'index' in save_record: del save_record['index']

                try:
                    save_annotation_line(annotated_file_path, save_record)
                    existing_annos[current_uid] = save_record
                    
                    st.success(f"✅ 已保存! ({selected_tag})")
                    time.sleep(0.5)
                    st.rerun()
                except Exception as e:
                    st.error(f"保存失败: {e}")

else:
    st.info("请加载数据。")