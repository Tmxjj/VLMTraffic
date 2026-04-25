"""
5_data_filter.py - 01_dataset_raw.jsonl 手工过滤工具

用法：
    streamlit run scripts/5_data_filter.py -- --path data/sft_dataset/JiNan/anon_3_4_jinan_real_2000/01_dataset_raw.jsonl

功能：
    - 逐条浏览原始数据，展示 4 张进口道图像 + Rollout 奖励分布 + GT 排队数
    - 每条数据可标记为「保留」「丢弃」「跳过」
    - 实时统计保留 / 丢弃数量
    - 「导出」按钮将保留条目以原始 JSONL 格式（-----分隔，indent=4）写入新文件
"""

import streamlit as st
import json
import os
import argparse
import altair as alt
import pandas as pd

# ──────────────────────────────────────────────
# 工具函数
# ──────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=str, default="")
    args, _ = parser.parse_known_args()
    return args


def load_raw_jsonl(file_path: str) -> list[dict]:
    """解析「JSON对象 + -----分隔符」格式的 JSONL 文件，返回记录列表。"""
    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()

    records = []
    decoder = json.JSONDecoder()
    pos = 0
    while pos < len(content):
        # 跳过空白和分隔符 -----
        while pos < len(content) and (content[pos].isspace() or content[pos] == "-"):
            pos += 1
        if pos >= len(content):
            break
        try:
            obj, idx = decoder.raw_decode(content[pos:])
            pos += idx
            records.append(obj)
        except json.JSONDecodeError:
            pos += 1  # 跳过无法解析的字符

    return records


def save_filtered_jsonl(records: list[dict], out_path: str):
    """将记录列表以原始格式写出：每条 indent=4 的 JSON，记录间用 -----\\n 分隔。"""
    os.makedirs(os.path.dirname(os.path.abspath(out_path)) or ".", exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for i, rec in enumerate(records):
            f.write(json.dumps(rec, ensure_ascii=False, indent=4))
            f.write("\n")
            if i < len(records) - 1:
                f.write("-----\n")


def get_output_path(input_path: str) -> str:
    base, ext = os.path.splitext(input_path)
    return f"{base}_filtered{ext}"


def make_uid(rec: dict) -> str:
    return f"{rec.get('junction_id', '')}_{rec.get('sumo_step', '')}"


def replace_prefix(path: str, old: str, new: str) -> str:
    if old and new and path:
        return path.replace(old, new, 1)
    return path


# ──────────────────────────────────────────────
# 奖励图表
# ──────────────────────────────────────────────

def render_reward_chart(all_rollout_rewards: dict, best_action: dict):
    """将 all_rollout_rewards 渲染为分组柱状图，高亮最优动作。"""
    if not all_rollout_rewards:
        st.caption("无 Rollout 奖励数据")
        return

    rows = []
    best_key = f"{best_action.get('phase_id', -1)}_{best_action.get('duration', -1)}"
    for key, val in all_rollout_rewards.items():
        parts = key.split("_")
        phase = parts[0] if parts else "?"
        dur = parts[1] if len(parts) > 1 else "?"
        rows.append({"key": key, "phase": f"P{phase}", "duration": f"{dur}s",
                     "reward": float(val), "is_best": (key == best_key)})

    df = pd.DataFrame(rows)

    chart = alt.Chart(df).mark_bar().encode(
        x=alt.X("duration:N", sort=None, title="绿灯时长"),
        y=alt.Y("reward:Q", title="Reward"),
        color=alt.condition(
            alt.datum.is_best,
            alt.value("#2ecc71"),   # 最优动作：绿色
            alt.value("#bdc3c7")    # 其余：灰色
        ),
        column=alt.Column("phase:N", title="相位"),
        tooltip=["key", "reward"]
    ).properties(width=80)

    st.altair_chart(chart)


# ──────────────────────────────────────────────
# 会话状态初始化
# ──────────────────────────────────────────────

def init_session(records: list[dict]):
    if "decisions" not in st.session_state:
        # decisions: uid -> "keep" | "discard" | None
        st.session_state.decisions = {make_uid(r): None for r in records}
    if "current_idx" not in st.session_state:
        st.session_state.current_idx = 0


# ──────────────────────────────────────────────
# 主界面
# ──────────────────────────────────────────────

st.set_page_config(layout="wide", page_title="Golden Data Filter")

args = parse_args()
default_path = args.path

# ── 侧边栏：文件选择 & 路径前缀 ──
st.sidebar.title("⚙️ 设置")

data_path = st.sidebar.text_input("数据文件路径", value=default_path)
if not data_path or not os.path.exists(data_path):
    st.error(f"❌ 文件不存在：`{data_path}`")
    st.stop()

old_prefix = st.sidebar.text_input(
    "图片路径前缀 (服务器端)",
    value="/home/jyf/code/trafficVLM/code/VLMTraffic/"
)
new_prefix = st.sidebar.text_input("图片路径前缀 (本地端)", value="./")

# ── 加载数据 ──
@st.cache_data(show_spinner="正在加载数据…")
def cached_load(path):
    return load_raw_jsonl(path)

all_records = cached_load(data_path)
if not all_records:
    st.error("文件中未找到有效记录。")
    st.stop()

init_session(all_records)

# ── 侧边栏：筛选 & 统计 ──
st.sidebar.markdown("---")
st.sidebar.subheader("🔎 筛选")

all_jids = sorted(set(r.get("junction_id", "") for r in all_records))
sel_jid = st.sidebar.selectbox("Junction ID", ["All"] + all_jids)

all_labels = sorted(set(r.get("label", "") for r in all_records))
sel_label = st.sidebar.selectbox("原始标签 (label)", ["All"] + all_labels)

all_steps = [r.get("sumo_step", 0) for r in all_records]
min_s, max_s = int(min(all_steps)), int(max(all_steps))
sel_step = st.sidebar.slider("SUMO Step 范围", min_s, max_s, (min_s, max_s))

# 「仅显示未决策」开关
show_undecided = st.sidebar.checkbox("仅显示未决策条目", value=False)

# ── 应用筛选 ──
filtered = [r for r in all_records
            if (sel_jid == "All" or r.get("junction_id") == sel_jid)
            and (sel_label == "All" or r.get("label") == sel_label)
            and (sel_step[0] <= r.get("sumo_step", 0) <= sel_step[1])
            and (not show_undecided or st.session_state.decisions.get(make_uid(r)) is None)]

if not filtered:
    st.warning("当前筛选条件下无匹配数据")
    st.stop()

# ── 翻页 ──
if st.session_state.current_idx >= len(filtered):
    st.session_state.current_idx = 0

st.sidebar.markdown("---")
nav_col1, nav_col2, nav_col3 = st.sidebar.columns([1, 2, 1])
if nav_col1.button("⬅️"):
    st.session_state.current_idx = max(0, st.session_state.current_idx - 1)
if nav_col3.button("➡️"):
    st.session_state.current_idx = min(len(filtered) - 1, st.session_state.current_idx + 1)
nav_col2.markdown(
    f"<div style='text-align:center;padding-top:6px'>{st.session_state.current_idx + 1} / {len(filtered)}</div>",
    unsafe_allow_html=True
)

# ── 统计 ──
decisions = st.session_state.decisions
kept   = sum(1 for v in decisions.values() if v == "keep")
disc   = sum(1 for v in decisions.values() if v == "discard")
undec  = sum(1 for v in decisions.values() if v is None)

st.sidebar.markdown("---")
st.sidebar.subheader("📊 进度")
st.sidebar.markdown(
    f"✅ 保留：**{kept}**　｜　❌ 丢弃：**{disc}**　｜　⬜ 未决策：**{undec}**"
)

# ── 导出 ──
st.sidebar.markdown("---")
st.sidebar.subheader("💾 导出")

out_path = get_output_path(data_path)
out_path_input = st.sidebar.text_input("导出路径", value=out_path)

if st.sidebar.button("🚀 导出保留数据", use_container_width=True):
    kept_records = [r for r in all_records
                    if decisions.get(make_uid(r)) == "keep"]
    if not kept_records:
        st.sidebar.error("没有标记为「保留」的数据！")
    else:
        save_filtered_jsonl(kept_records, out_path_input)
        st.sidebar.success(f"✅ 已导出 {len(kept_records)} 条记录\n→ `{out_path_input}`")

# ──────────────────────────────────────────────
# 主体：当前条目
# ──────────────────────────────────────────────

rec = filtered[st.session_state.current_idx]
uid = make_uid(rec)
current_decision = decisions.get(uid)

jid        = rec.get("junction_id", "N/A")
sumo_step  = rec.get("sumo_step", "N/A")
label      = rec.get("label", "N/A")
cur_phase  = rec.get("current_phase_id", "N/A")
best_act   = rec.get("best_action", {})
vlm_act    = rec.get("vlm_action", {})
all_rewards = rec.get("all_rollout_rewards", {})
gt_counts  = rec.get("gt_jam_counts", {})
image_paths = rec.get("image_paths", [])
rollout_steps = rec.get("rollout_follow_steps", "N/A")

# 标题行
label_color = "#2ecc71" if label == "accepted" else "#e74c3c"
dec_badge = {"keep": "✅ 保留", "discard": "❌ 丢弃", None: "⬜ 未决策"}.get(current_decision, "⬜")

st.markdown(
    f"### 🚦 `{jid}` &nbsp;|&nbsp; Step: `{sumo_step}` &nbsp;|&nbsp; "
    f"<span style='color:{label_color}'>{label}</span> &nbsp;|&nbsp; {dec_badge}",
    unsafe_allow_html=True
)

# 指标行
m1, m2, m3, m4, m5 = st.columns(5)
m1.metric("当前相位", cur_phase)
m2.metric("最优动作", f"P{best_act.get('phase_id','?')} / {best_act.get('duration','?')}s")
m3.metric("最优 Reward", f"{rec.get('best_reward', 0):.1f}")
m4.metric("VLM 动作", f"P{vlm_act.get('phase_id','?')} / {vlm_act.get('duration','?')}s")
m5.metric("Rollout Steps", rollout_steps)

st.divider()

# ──────────────────────────────────────────────
# 三栏布局：图像 | 数据详情 | 决策
# ──────────────────────────────────────────────

col_img, col_data, col_action = st.columns([5, 3, 2])

# ── 左栏：4 张进口道图像 ──
with col_img:
    st.subheader("🖼️ 进口道图像（N / E / S / W）")
    dirs = ["N", "E", "S", "W"]

    if len(image_paths) >= 4:
        img_cols = st.columns(2)
        for i in range(4):
            raw = image_paths[i]
            local = replace_prefix(raw, old_prefix, new_prefix)
            with img_cols[i % 2]:
                if os.path.exists(local):
                    st.image(local, caption=f"{dirs[i]} 进口道", use_column_width=True)
                else:
                    st.warning(f"图像未找到\n`{local}`")
    else:
        st.warning("image_paths 不足 4 张")

# ── 中栏：GT 排队 + Rollout 奖励 ──
with col_data:
    st.subheader("📋 GT 排队车辆数")
    if gt_counts:
        # 转成易读的表格
        gt_rows = [{"road": k, "count": v} for k, v in gt_counts.items()]
        gt_df = pd.DataFrame(gt_rows)
        st.dataframe(gt_df, use_container_width=True, hide_index=True)
    else:
        st.caption("无 GT 数据")

    st.markdown("---")
    st.subheader("📊 Rollout 奖励分布")
    render_reward_chart(all_rewards, best_act)

    with st.expander("🔍 查看完整 Prompt"):
        st.text(rec.get("prompt", ""))

    with st.expander("🤖 VLM 原始响应"):
        st.text(rec.get("vlm_response", "") or rec.get("vlm_thought", "") or "（无响应）")

# ── 右栏：过滤决策 ──
with col_action:
    st.subheader("✍️ 过滤决策")

    # 显示当前状态
    if current_decision == "keep":
        st.success("当前标记：**保留** ✅")
    elif current_decision == "discard":
        st.error("当前标记：**丢弃** ❌")
    else:
        st.info("当前标记：**未决策** ⬜")

    st.markdown("---")

    # 决策按钮
    b1, b2 = st.columns(2)

    if b1.button("✅ 保留", use_container_width=True, type="primary"):
        st.session_state.decisions[uid] = "keep"
        # 自动跳到下一条
        next_idx = st.session_state.current_idx + 1
        if next_idx < len(filtered):
            st.session_state.current_idx = next_idx
        st.rerun()

    if b2.button("❌ 丢弃", use_container_width=True):
        st.session_state.decisions[uid] = "discard"
        next_idx = st.session_state.current_idx + 1
        if next_idx < len(filtered):
            st.session_state.current_idx = next_idx
        st.rerun()

    if st.button("⬜ 撤销 / 跳过", use_container_width=True):
        st.session_state.decisions[uid] = None
        st.rerun()

    st.markdown("---")
    st.caption("**快捷操作**")

    # 批量操作
    with st.expander("⚡ 批量操作"):
        if st.button("将当前筛选结果全部「保留」", use_container_width=True):
            for r in filtered:
                st.session_state.decisions[make_uid(r)] = "keep"
            st.rerun()

        if st.button("将当前筛选结果全部「丢弃」", use_container_width=True):
            for r in filtered:
                st.session_state.decisions[make_uid(r)] = "discard"
            st.rerun()

        if st.button("清空所有决策", use_container_width=True):
            for r in all_records:
                st.session_state.decisions[make_uid(r)] = None
            st.rerun()

        st.markdown("---")
        st.caption("按 label 批量操作（全局）")
        col_ba1, col_ba2 = st.columns(2)
        if col_ba1.button("保留全部 accepted", use_container_width=True):
            for r in all_records:
                if r.get("label") == "accepted":
                    st.session_state.decisions[make_uid(r)] = "keep"
            st.rerun()
        if col_ba2.button("丢弃全部 rejected", use_container_width=True):
            for r in all_records:
                if r.get("label") == "rejected":
                    st.session_state.decisions[make_uid(r)] = "discard"
            st.rerun()

    st.markdown("---")
    st.caption("**数据摘要**")
    st.json({
        "scenario":        rec.get("scenario"),
        "junction_id":     jid,
        "sumo_step":       sumo_step,
        "current_phase":   cur_phase,
        "best_action":     best_act,
        "vlm_action":      vlm_act,
        "best_reward":     round(rec.get("best_reward", 0), 2),
        "label":           label,
        "rollout_steps":   rollout_steps,
    })
