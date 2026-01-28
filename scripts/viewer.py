import streamlit as st
import pandas as pd
import json
import os
import argparse
from PIL import Image

# 引入翻译库
try:
    from deep_translator import GoogleTranslator
except ImportError:
    st.error("请先安装翻译库: pip install deep-translator")
    st.stop()

# --- 命令行参数解析 ---
def get_args():
    parser = argparse.ArgumentParser(description="TrafficVLM Case Viewer")
    parser.add_argument("--path", type=str, default="dataset.jsonl", help="数据文件的路径")
    args, _ = parser.parse_known_args()
    return args

args = get_args()
default_data_path = args.path

st.set_page_config(layout="wide", page_title="TrafficVLM Case Viewer")

# --- 翻译辅助函数 (带缓存) ---
@st.cache_data(show_spinner=False)
def translate_text(text):
    if not text:
        return ""
    try:
        # 使用 Google 翻译源，目标语言为简体中文
        translator = GoogleTranslator(source='auto', target='zh-CN')
        # 如果文本太长（超过5000字符），Google接口可能会报错，这里做个简单的截断或分段处理建议
        # 这里直接翻译，通常 response 不会太长
        return translator.translate(text)
    except Exception as e:
        return f"翻译失败: {e} (请检查网络连接)"

# --- 数据加载函数 ---
@st.cache_data
def load_data(file_path):
    data = []
    content = ""
    try:
        if hasattr(file_path, 'read'):
            content = file_path.read().decode('utf-8')
        else:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
    except FileNotFoundError:
        return pd.DataFrame()

    raw_samples = content.split('-----')

    for sample_str in raw_samples:
        clean_str = sample_str.strip()
        if not clean_str:
            continue
        try:
            obj = json.loads(clean_str)
            data.append(obj)
        except json.JSONDecodeError:
            continue

    return pd.DataFrame(data)

# --- 侧边栏 ---
st.sidebar.title("🛠️ 设置与筛选")
st.sidebar.subheader("📂 数据来源")
uploaded_file = st.sidebar.file_uploader("上传文件", type=["jsonl", "txt"])
df = None

if uploaded_file is not None:
    df = load_data(uploaded_file)
elif default_data_path and os.path.exists(default_data_path):
    st.sidebar.info(f"读取: `{os.path.basename(default_data_path)}`")
    df = load_data(default_data_path)
else:
    st.error(f"❌ 找不到文件: `{default_data_path}`")
    st.stop()

# 路径映射
st.sidebar.markdown("---")
path_prefix_to_replace = st.sidebar.text_input("数据路径前缀 (Old)", "/home/jyf/code/trafficVLM/code/VLMTraffic/")
local_path_prefix = st.sidebar.text_input("本地路径前缀 (New)", "./")

# 筛选逻辑
if df is not None and not df.empty:
    st.sidebar.markdown("---")
    
    # 获取筛选选项
    j_ids = ["All"] + sorted(df['junction_id'].astype(str).unique().tolist()) if 'junction_id' in df else []
    labels = ["All"] + sorted(df['label'].astype(str).unique().tolist()) if 'label' in df else []
    
    sel_jid = st.sidebar.selectbox("Junction ID", j_ids)
    sel_label = st.sidebar.selectbox("Label", labels)
    
    if 'step' in df:
        min_s, max_s = int(df['step'].min()), int(df['step'].max())
        sel_step = st.sidebar.slider("Step Range", min_s, max_s, (min_s, max_s))
    
    # 应用筛选
    filtered_df = df.copy()
    if sel_jid != "All": filtered_df = filtered_df[filtered_df['junction_id'].astype(str) == sel_jid]
    if sel_label != "All": filtered_df = filtered_df[filtered_df['label'] == sel_label]
    if 'step' in df: filtered_df = filtered_df[(filtered_df['step'] >= sel_step[0]) & (filtered_df['step'] <= sel_step[1])]

    if len(filtered_df) == 0:
        st.warning("无匹配数据")
        st.stop()

    st.sidebar.markdown(f"**找到 {len(filtered_df)} 条数据**")
    
    if 'current_index' not in st.session_state: st.session_state.current_index = 0
    if st.session_state.current_index >= len(filtered_df): st.session_state.current_index = 0

    c1, c2, c3 = st.sidebar.columns([1, 2, 1])
    if c1.button("⬅️"): st.session_state.current_index = max(0, st.session_state.current_index - 1)
    if c3.button("➡️"): st.session_state.current_index = min(len(filtered_df) - 1, st.session_state.current_index + 1)
    c2.markdown(f"<center>{st.session_state.current_index + 1} / {len(filtered_df)}</center>", unsafe_allow_html=True)

    row = filtered_df.iloc[st.session_state.current_index]

    # --- 主界面布局优化 ---
    
    # 1. 标题行
    lbl = row.get('label', 'N/A')
    lbl_color = ":green" if lbl == 'accepted' else ":red"
    st.markdown(f"### 🚦 JID: `{row.get('junction_id')}` | Step: `{row.get('step')}` | Label: {lbl_color}[**{lbl}**]")

    # 2. 关键指标行 (放在顶部，不挤占下方空间)
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Current Phase", row.get('current_phase', 'N/A'))
    m2.metric("VLM Action", row.get('vlm_action', 'N/A'))
    
    opt_act = row.get('optimal_action', 'N/A')
    vlm_act = row.get('vlm_action', 'N/A')
    is_correct = (str(opt_act) == str(vlm_act))
    m3.metric("Optimal Action", opt_act, delta="Correct" if is_correct else "Incorrect", delta_color="normal" if is_correct else "inverse")
    m4.metric("Best Metric Val", round(float(row.get('metric_val', 0)), 2))

    st.divider()

    # 3. 左右分栏：左图，右文
    col_img, col_text = st.columns([1, 1]) # 1:1 比例，或者 [5,4]

    # --- 左侧：图片 ---
    with col_img:
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

        # 将 Prompt 和 Think Process 放在图片下方
        with st.expander("System Prompt"):
            st.text(row.get('prompt', ''))
        with st.expander("Think Process (Chain of Thought)"):
            st.write(row.get('vlm_think_process', ''))

    # --- 右侧：翻译后的回复 & 图表 ---
    with col_text:
        st.subheader("🤖 VLM Analysis (CN/EN)")
        
        raw_response = row.get('vlm_response_raw', '')
        
        # 翻译开关
        show_trans = st.toggle("启用中文翻译 (Translate to Chinese)", value=True)
        
        if show_trans and raw_response:
            with st.spinner("正在翻译..."):
                translated_text = translate_text(raw_response)
            
            # 使用 info 框高亮显示翻译内容
            st.success(f"**中文回复:**\n\n{translated_text}")
            
            # 在折叠框中保留原文，方便对照
            with st.expander("查看英文原文 (Original English)"):
                st.code(raw_response, language="text")
        else:
            # 不翻译时直接显示
            st.info(f"**Raw Response:**\n\n{raw_response}")

        st.divider()

        # 图表放在文字下方
        st.subheader("📊 Reward Metrics")
        metrics_dict = row.get('all_metrics', {})
        if metrics_dict:
            try:
                metrics_df = pd.DataFrame(list(metrics_dict.items()), columns=['Phase', 'Reward'])
                metrics_df['Phase'] = metrics_df['Phase'].astype(str)
                # 简单高亮 VLM 选择的 Phase
                colors = []
                for p in metrics_df['Phase']:
                    if str(p) == str(vlm_act):
                        colors.append("#ff4b4b") # 红色高亮选中的
                    elif str(p) == str(opt_act):
                         colors.append("#09ab3b") # 绿色高亮最优的(如果不重合)
                    else:
                        colors.append("#e6e9ef") # 灰色
                
                st.bar_chart(metrics_df.set_index('Phase')['Reward'], color=colors if len(colors)==len(metrics_df) else None)
                
                # 同时也显示表格，方便看具体数值
                st.dataframe(metrics_df.set_index('Phase').T)
            except Exception as e:
                st.write(metrics_dict)
else:
    st.info("请加载数据。")

    # 运行脚本：streamlit run scripts/viewer.py -- --path data/sft_dataset/JiNan_test/dataset.jsonl