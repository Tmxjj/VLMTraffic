"""
5_data_filter.py - 01_dataset_raw.jsonl 手工过滤工具

用法：
    streamlit run scripts/5_data_filter.py -- --path data/sft_dataset/JiNan/anon_3_4_jinan_real_2000/01_dataset_raw.jsonl

文件分工：
    - 01_dataset_raw.jsonl  : 只读，原始数据，永不修改
    - 03_final_dataset.jsonl : 读写，保存所有已决策条目（keep/edit/discard）
                              刷新后从此文件恢复操作记录

导出字段：
    - filter_decision : "keep" | "edit" | "discard"
    - golden_response : keep → 原始 vlm_response；edit → 人工修改内容；discard → 不写
    - vlm_response    : 原样保留
"""

import streamlit as st
import json
import os
import argparse
import re
import altair as alt
import pandas as pd
from PIL import Image

# ──────────────────────────────────────────────
# 常量
# ──────────────────────────────────────────────

SRC_RESPONSE_KEY = "vlm_response"
DST_RESPONSE_KEY = "golden_response"
FINAL_FILENAME   = "03_final_dataset.jsonl"

# ──────────────────────────────────────────────
# 工具函数
# ──────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=str, default="")
    args, _ = parser.parse_known_args()
    return args


def get_final_path(raw_path: str) -> str:
    """与原始文件同目录，固定文件名 03_final_dataset.jsonl。"""
    return os.path.join(os.path.dirname(os.path.abspath(raw_path)), FINAL_FILENAME)


def make_uid(rec: dict) -> str:
    return f"{rec.get('junction_id', '')}_{rec.get('sumo_step', '')}"


def replace_prefix(path: str, old: str, new: str) -> str:
    if old and new and path:
        return path.replace(old, new, 1)
    return path


def parse_jsonl(file_path: str) -> list[dict]:
    """解析「JSON对象 + -----分隔符」格式文件，返回记录列表。"""
    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()
    records = []
    decoder = json.JSONDecoder()
    pos = 0
    while pos < len(content):
        while pos < len(content) and (content[pos].isspace() or content[pos] == "-"):
            pos += 1
        if pos >= len(content):
            break
        try:
            obj, idx = decoder.raw_decode(content[pos:])
            pos += idx
            records.append(obj)
        except json.JSONDecodeError:
            pos += 1
    return records


def write_jsonl(records: list[dict], out_path: str):
    """写出 indent=4 格式，记录间用 -----\n 分隔。"""
    os.makedirs(os.path.dirname(os.path.abspath(out_path)) or ".", exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for i, rec in enumerate(records):
            f.write(json.dumps(rec, ensure_ascii=False, indent=4))
            f.write("\n")
            if i < len(records) - 1:
                f.write("-----\n")


def build_final_record(raw_rec: dict, decision: str, edited_text: str | None) -> dict:
    """构建写入 03_final_dataset.jsonl 的记录。"""
    out = dict(raw_rec)                    # 保留全部原始字段（含 vlm_response）
    out["filter_decision"] = decision
    if decision == "keep":
        out[DST_RESPONSE_KEY] = raw_rec.get(SRC_RESPONSE_KEY, "") or raw_rec.get("vlm_thought", "")
    elif decision == "edit":
        out[DST_RESPONSE_KEY] = edited_text or ""
    # discard：不写 golden_response
    return out


def load_final_index(final_path: str) -> tuple[dict, dict]:
    """
    从 03_final_dataset.jsonl 读取已有决策，返回：
      decisions    : uid -> "keep" | "edit" | "discard"
      edited_texts : uid -> str（仅 edit 条目）
    """
    decisions: dict = {}
    edited_texts: dict = {}
    if not os.path.exists(final_path):
        return decisions, edited_texts
    try:
        records = parse_jsonl(final_path)
    except Exception:
        return decisions, edited_texts
    for r in records:
        uid = make_uid(r)
        dec = r.get("filter_decision")
        if dec in ("keep", "edit", "discard"):
            decisions[uid] = dec
        if dec == "edit":
            edited_texts[uid] = r.get(DST_RESPONSE_KEY, "")
    return decisions, edited_texts


def auto_save(all_raw: list[dict], final_path: str, decisions: dict, edited_texts: dict):
    """将所有已决策条目写入 03_final_dataset.jsonl，未决策条目不写入。"""
    export = []
    for r in all_raw:
        uid = make_uid(r)
        dec = decisions.get(uid)
        if dec in ("keep", "edit", "discard"):
            export.append(build_final_record(r, dec, edited_texts.get(uid)))
    write_jsonl(export, final_path)


# ──────────────────────────────────────────────
# 会话状态初始化
# ──────────────────────────────────────────────

def init_session(all_raw: list[dict], final_path: str):
    """
    冷启动（页面真正刷新）时从 03_final_dataset.jsonl 恢复决策状态。
    st.rerun() 触发的重渲染跳过，避免覆盖内存中刚写入的决策。
    """
    if "current_idx" not in st.session_state:
        st.session_state.current_idx = 0
    if "editing_uid" not in st.session_state:
        st.session_state.editing_uid = None

    # 用 final_path 作为冷启动标志：rerun 不改变此值，故跳过
    if st.session_state.get("_init_final_path") == final_path:
        return

    decisions, edited_texts = load_final_index(final_path)
    # 对原始数据中尚未出现在 final 文件里的 uid，默认为 None
    for r in all_raw:
        uid = make_uid(r)
        if uid not in decisions:
            decisions[uid] = None

    st.session_state.decisions    = decisions
    st.session_state.edited_texts = edited_texts
    st.session_state._init_final_path = final_path


# ──────────────────────────────────────────────
# 图像工具
# ──────────────────────────────────────────────

def load_rotated(path: str, rotate_deg):
    img = Image.open(path)
    if rotate_deg is not None:
        img = img.rotate(rotate_deg, expand=True)
    return img


DIR_CONFIG = [
    {"dir": "N", "idx": 0, "label": "↑ N", "rotate": None},
    {"dir": "E", "idx": 1, "label": "→ E", "rotate": -90},
    {"dir": "S", "idx": 2, "label": "↓ S", "rotate": 180},
    {"dir": "W", "idx": 3, "label": "← W", "rotate": 90},
]


def extract_dir_from_path(path: str) -> str | None:
    """从图片文件名解析方向，支持 xxx_N.png / upstream_xxx_W.png 等格式。"""
    if not path:
        return None
    stem = os.path.splitext(os.path.basename(path))[0]
    m = re.search(r"_([NESW])$", stem, re.IGNORECASE)
    if not m:
        return None
    return m.group(1).upper()


def build_dir_image_map(image_paths: list[str]) -> dict[str, str | None]:
    """
    将 image_paths 映射到 N/E/S/W。
    1) 优先使用文件名后缀识别方向；
    2) 无法识别时，回退到旧版索引顺序 N,E,S,W。
    """
    dir_map: dict[str, str | None] = {"N": None, "E": None, "S": None, "W": None}
    unresolved: list[tuple[int, str]] = []

    for idx, p in enumerate(image_paths):
        d = extract_dir_from_path(p)
        if d in dir_map and dir_map[d] is None:
            dir_map[d] = p
        else:
            unresolved.append((idx, p))

    legacy_order = ["N", "E", "S", "W"]
    for idx, p in unresolved:
        if idx < len(legacy_order):
            d = legacy_order[idx]
            if dir_map[d] is None:
                dir_map[d] = p

    return dir_map


# ──────────────────────────────────────────────
# 奖励图表
# ──────────────────────────────────────────────

def render_reward_chart(all_rollout_rewards: dict, best_action: dict):
    if not all_rollout_rewards:
        st.caption("无 Rollout 奖励数据")
        return
    rows = []
    best_key = f"{best_action.get('phase_id', -1)}_{best_action.get('duration', -1)}"
    for key, val in all_rollout_rewards.items():
        parts = key.split("_")
        phase = parts[0] if parts else "?"
        dur   = parts[1] if len(parts) > 1 else "?"
        rows.append({"key": key, "phase": f"P{phase}", "duration": f"{dur}s",
                     "reward": float(val), "is_best": (key == best_key)})
    df = pd.DataFrame(rows)
    chart = alt.Chart(df).mark_bar().encode(
        x=alt.X("duration:N", sort=None, title="绿灯时长"),
        y=alt.Y("reward:Q", title="Reward"),
        color=alt.condition(alt.datum.is_best, alt.value("#2ecc71"), alt.value("#bdc3c7")),
        column=alt.Column("phase:N", title="相位"),
        tooltip=["key", "reward"]
    ).properties(width=80)
    st.altair_chart(chart)


# ══════════════════════════════════════════════
# 主界面
# ══════════════════════════════════════════════

st.set_page_config(layout="wide", page_title="Golden Data Filter")

args        = parse_args()
default_path = args.path

# ── 侧边栏：文件选择 & 路径前缀 ──
st.sidebar.title("⚙️ 设置")

data_path = st.sidebar.text_input("原始数据文件路径", value=default_path)
if not data_path or not os.path.exists(data_path):
    st.error(f"❌ 文件不存在：`{data_path}`")
    st.stop()

final_path = get_final_path(data_path)

old_prefix = st.sidebar.text_input(
    "图片路径前缀 (服务器端)",
    value="/home/jyf/code/trafficVLM/code/VLMTraffic/"
)
new_prefix = st.sidebar.text_input("图片路径前缀 (本地端)", value="./")

# ── 加载原始数据（只读，永久缓存）──
@st.cache_data(show_spinner="正在加载原始数据…")
def cached_load(path):
    return parse_jsonl(path)

all_records = cached_load(data_path)
if not all_records:
    st.error("文件中未找到有效记录。")
    st.stop()

total = len(all_records)

init_session(all_records, final_path)

# ── 侧边栏：筛选 ──
st.sidebar.markdown("---")
st.sidebar.subheader("🔎 筛选")

all_jids   = sorted(set(r.get("junction_id", "") for r in all_records))
sel_jid    = st.sidebar.selectbox("Junction ID", ["All"] + all_jids)

all_labels = sorted(set(r.get("label", "") for r in all_records))
sel_label  = st.sidebar.selectbox("原始标签 (label)", ["All"] + all_labels)

all_steps  = [r.get("sumo_step", 0) for r in all_records]
min_s, max_s = int(min(all_steps)), int(max(all_steps))
sel_step   = st.sidebar.slider("SUMO Step 范围", min_s, max_s, (min_s, max_s))

show_undecided = st.sidebar.checkbox("仅显示未决策条目", value=False)

# ── 应用筛选 ──
decisions = st.session_state.decisions
filtered  = [r for r in all_records
             if (sel_jid == "All" or r.get("junction_id") == sel_jid)
             and (sel_label == "All" or r.get("label") == sel_label)
             and (sel_step[0] <= r.get("sumo_step", 0) <= sel_step[1])
             and (not show_undecided or decisions.get(make_uid(r)) is None)]

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
    st.session_state.editing_uid = None
if nav_col3.button("➡️"):
    st.session_state.current_idx = min(len(filtered) - 1, st.session_state.current_idx + 1)
    st.session_state.editing_uid = None
nav_col2.markdown(
    f"<div style='text-align:center;padding-top:6px'>"
    f"{st.session_state.current_idx + 1} / {len(filtered)}</div>",
    unsafe_allow_html=True
)

# ── 统计 ──
kept   = sum(1 for v in decisions.values() if v == "keep")
edited = sum(1 for v in decisions.values() if v == "edit")
disc   = sum(1 for v in decisions.values() if v == "discard")
undec  = sum(1 for v in decisions.values() if v is None)
processed = kept + edited + disc

st.sidebar.markdown("---")
st.sidebar.subheader("📊 进度")
st.sidebar.progress(processed / total if total else 0,
                    text=f"已处理 {processed} / {total} 条")
st.sidebar.markdown(
    f"✅ 保留：**{kept}**　｜　✏️ 修改：**{edited}**　｜　"
    f"❌ 丢弃：**{disc}**　｜　⬜ 未决策：**{undec}**"
)
st.sidebar.caption(f"💾 决策保存至：`{final_path}`")

# ══════════════════════════════════════════════
# 主体：当前条目
# ══════════════════════════════════════════════

rec = filtered[st.session_state.current_idx]
uid = make_uid(rec)
current_decision = decisions.get(uid)

jid           = rec.get("junction_id", "N/A")
sumo_step     = rec.get("sumo_step", "N/A")
label         = rec.get("label", "N/A")
cur_phase     = rec.get("current_phase_id", "N/A")
best_act      = rec.get("best_action", {})
vlm_act       = rec.get("vlm_action", {})
all_rewards   = rec.get("all_rollout_rewards", {})
gt_counts     = rec.get("gt_jam_counts", {})
image_paths   = rec.get("image_paths", [])
rollout_steps = rec.get("rollout_follow_steps", "N/A")
vlm_resp_orig = rec.get(SRC_RESPONSE_KEY, "") or rec.get("vlm_thought", "") or "（无响应）"

label_color = "#2ecc71" if label == "accepted" else "#e74c3c"
dec_badge   = {
    "keep": "✅ 保留", "edit": "✏️ 修改", "discard": "❌ 丢弃", None: "⬜ 未决策"
}.get(current_decision, "⬜")

st.markdown(
    f"### 🚦 `{jid}` &nbsp;|&nbsp; Step: `{sumo_step}` &nbsp;|&nbsp; "
    f"<span style='color:{label_color}'>{label}</span> &nbsp;|&nbsp; {dec_badge}",
    unsafe_allow_html=True
)

m1, m2, m3, m4, m5 = st.columns(5)
m1.metric("当前相位", cur_phase)
m2.metric("最优动作", f"P{best_act.get('phase_id','?')} / {best_act.get('duration','?')}s")
m3.metric("最优 Reward", f"{rec.get('best_reward', 0):.1f}")
m4.metric("VLM 动作", f"P{vlm_act.get('phase_id','?')} / {vlm_act.get('duration','?')}s")
m5.metric("Rollout Steps", rollout_steps)

st.divider()

# ──────────────────────────────────────────────
# 三栏布局：图像 | 响应+数据 | 决策
# ──────────────────────────────────────────────

col_img, col_data, col_action = st.columns([6, 2.2, 1.3])

# ── 左栏：4 张进口道图像 ──
with col_img:
    st.subheader("🖼️ 进口道图像")

    dir_map = build_dir_image_map(image_paths)
    if not any(dir_map.values()):
        st.warning("image_paths 中未找到可显示的方向图片")
    else:
        IMG      = 1
        MID_GAP  = 0.25
        NS_SIDE  = (IMG + MID_GAP) / 2
        W_PX     = 400

        cfg_by_dir = {c["dir"]: c for c in DIR_CONFIG}

        def render_direction(direction: str):
            cfg = cfg_by_dir[direction]
            raw_path = dir_map.get(direction)
            if not raw_path:
                st.caption(f"{cfg['label']}：无该方向图片")
                return
            local = replace_prefix(raw_path, old_prefix, new_prefix)
            if os.path.exists(local):
                st.image(load_rotated(local, cfg["rotate"]), caption=cfg["label"], width=W_PX)
            else:
                st.warning(f"未找到\n`{local}`")

        # 行1：北（居中）
        _, c_n, _ = st.columns([NS_SIDE, IMG, NS_SIDE])
        with c_n:
            render_direction("N")

        # 行2：西（左）、东（右）
        c_w, _, c_e = st.columns([IMG, MID_GAP, IMG])
        with c_w:
            render_direction("W")
        with c_e:
            render_direction("E")

        # 行3：南（居中）
        _, c_s, _ = st.columns([NS_SIDE, IMG, NS_SIDE])
        with c_s:
            render_direction("S")

# ── 中栏：VLM 响应 + GT + 奖励 ──
with col_data:
    st.subheader("🤖 VLM 原始响应")
    display_resp = (st.session_state.edited_texts.get(uid, vlm_resp_orig)
                    if current_decision == "edit" else vlm_resp_orig)
    st.text_area("", value=display_resp, height=380, disabled=True,
                 label_visibility="collapsed", key=f"view_resp_{uid}")

    with st.expander("🔍 查看完整 Prompt"):
        st.text(rec.get("prompt", ""))

    st.markdown("---")
    st.subheader("📋 GT 排队车辆数")
    if gt_counts:
        st.dataframe(pd.DataFrame([{"road": k, "count": v} for k, v in gt_counts.items()]),
                     use_container_width=True, hide_index=True)
    else:
        st.caption("无 GT 数据")

    st.markdown("---")
    st.subheader("📊 Rollout 奖励分布")
    render_reward_chart(all_rewards, best_act)

# ── 右栏：决策操作 ──
with col_action:
    st.subheader("✍️ 过滤决策")

    if current_decision == "keep":
        st.success("当前标记：**保留** ✅")
    elif current_decision == "edit":
        st.warning("当前标记：**修改** ✏️")
    elif current_decision == "discard":
        st.error("当前标记：**丢弃** ❌")
    else:
        st.info("当前标记：**未决策** ⬜")

    st.markdown("---")

    b_keep, b_discard = st.columns(2)

    if b_keep.button("✅ 保留", use_container_width=True, type="primary"):
        st.session_state.decisions[uid] = "keep"
        st.session_state.editing_uid    = None
        auto_save(all_records, final_path, st.session_state.decisions, st.session_state.edited_texts)
        next_idx = st.session_state.current_idx + 1
        if next_idx < len(filtered):
            st.session_state.current_idx = next_idx
        st.rerun()

    if b_discard.button("❌ 丢弃", use_container_width=True):
        st.session_state.decisions[uid] = "discard"
        st.session_state.editing_uid    = None
        auto_save(all_records, final_path, st.session_state.decisions, st.session_state.edited_texts)
        next_idx = st.session_state.current_idx + 1
        if next_idx < len(filtered):
            st.session_state.current_idx = next_idx
        st.rerun()

    if st.button("✏️ 修改响应", use_container_width=True):
        st.session_state.editing_uid = uid
        if uid not in st.session_state.edited_texts:
            st.session_state.edited_texts[uid] = vlm_resp_orig
        st.rerun()

    if st.button("⬜ 撤销 / 跳过", use_container_width=True):
        st.session_state.decisions[uid] = None
        st.session_state.editing_uid    = None
        auto_save(all_records, final_path, st.session_state.decisions, st.session_state.edited_texts)
        st.rerun()

    # ── 内联编辑区 ──
    if st.session_state.editing_uid == uid:
        st.markdown("---")
        st.caption("📝 编辑后点击「保存修改」")

        draft_init  = st.session_state.edited_texts.get(uid, vlm_resp_orig)
        edited_draft = st.text_area("编辑区", value=draft_init, height=320,
                                    key=f"draft_{uid}", label_visibility="collapsed")

        save_col, cancel_col = st.columns(2)
        if save_col.button("💾 保存修改", use_container_width=True, type="primary"):
            st.session_state.edited_texts[uid] = edited_draft
            st.session_state.decisions[uid]    = "edit"
            st.session_state.editing_uid       = None
            auto_save(all_records, final_path, st.session_state.decisions, st.session_state.edited_texts)
            next_idx = st.session_state.current_idx + 1
            if next_idx < len(filtered):
                st.session_state.current_idx = next_idx
            st.rerun()

        if cancel_col.button("✖ 取消", use_container_width=True):
            st.session_state.editing_uid = None
            st.rerun()

    st.markdown("---")
    st.caption("**快捷操作**")

    with st.expander("⚡ 批量操作"):
        if st.button("将当前筛选结果全部「保留」", use_container_width=True):
            for r in filtered:
                st.session_state.decisions[make_uid(r)] = "keep"
            auto_save(all_records, final_path, st.session_state.decisions, st.session_state.edited_texts)
            st.rerun()

        if st.button("将当前筛选结果全部「丢弃」", use_container_width=True):
            for r in filtered:
                st.session_state.decisions[make_uid(r)] = "discard"
            auto_save(all_records, final_path, st.session_state.decisions, st.session_state.edited_texts)
            st.rerun()

        if st.button("清空所有决策", use_container_width=True):
            for r in all_records:
                st.session_state.decisions[make_uid(r)] = None
            auto_save(all_records, final_path, st.session_state.decisions, st.session_state.edited_texts)
            st.rerun()

        st.markdown("---")
        st.caption("按 label 批量操作（全局）")
        col_ba1, col_ba2 = st.columns(2)
        if col_ba1.button("保留全部 accepted", use_container_width=True):
            for r in all_records:
                if r.get("label") == "accepted":
                    st.session_state.decisions[make_uid(r)] = "keep"
            auto_save(all_records, final_path, st.session_state.decisions, st.session_state.edited_texts)
            st.rerun()
        if col_ba2.button("丢弃全部 rejected", use_container_width=True):
            for r in all_records:
                if r.get("label") == "rejected":
                    st.session_state.decisions[make_uid(r)] = "discard"
            auto_save(all_records, final_path, st.session_state.decisions, st.session_state.edited_texts)
            st.rerun()

    st.markdown("---")
    st.caption("**数据摘要**")
    st.json({
        "scenario":      rec.get("scenario"),
        "junction_id":   jid,
        "sumo_step":     sumo_step,
        "current_phase": cur_phase,
        "best_action":   best_act,
        "vlm_action":    vlm_act,
        "best_reward":   round(rec.get("best_reward", 0), 2),
        "label":         label,
        "rollout_steps": rollout_steps,
    })
