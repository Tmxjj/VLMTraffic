import os
from PIL import Image, ImageDraw, ImageFont
import pathlib
import re

# ─────────────────────────────────────────────────────────────────
# 场景到水印配置键的映射
# ─────────────────────────────────────────────────────────────────
SCENARIO_TO_WATERMARK_KEY = {
    "JiNan":             "4_WAY_RHT_3LANE",
    "Hangzhou":          "4_WAY_RHT_3LANE",
    "JiNan_test":        "4_WAY_RHT_3LANE",
    "NewYork":           "4_WAY_RHT_3LANE",
    "France_Massy":      "T_JUNCTION_RHT_2LANE",
    "SouthKorea_Songdo": "SONGDO_4WAY_RHT_ASYM",
    "Hongkong_YMT":      "YMT_4WAY_LHT_3LANE",
}

# ─────────────────────────────────────────────────────────────────
# 水印布局配置
# ─────────────────────────────────────────────────────────────────
WATERMARK_CONFIGS = {
    "4_WAY_RHT_3LANE": {
        "North": {
            "lanes": 3,
            "lane_w_ratio": 0.1,
            "x_offset_ratio": 0,
            "y_offset_ratio": 0.155,
            "text_rotation": 0,
            "text_pos_x": 0.594,
            "text_pos_y": 0.885,
            "lane_font_size_ratio": 0.08,
            "text_font_size_ratio": 0.047
        },
        "upstream_North": {
            "lanes": 3,
            "lane_w_ratio": 0.1,
            "x_offset_ratio": -0.00469529085872575,
            "y_offset_ratio": 0.18208033240997226,
            "text_rotation": 0,
            "text_pos_x": 0.779595567867036,
            "text_pos_y": 0.08390027700831038,
            "lane_font_size_ratio": 0.08,
            "text_font_size_ratio": 0.047
        },
        "South": {
            "lanes": 3,
            "lane_w_ratio": 0.1,
            "x_offset_ratio": 0,
            "y_offset_ratio": 0.155,
            "text_rotation": 0,
            "text_pos_x": 0.594,
            "text_pos_y": 0.908,
            "lane_font_size_ratio": 0.08,
            "text_font_size_ratio": 0.047
        },
        "upstream_South": {
            "lanes": 3,
            "lane_w_ratio": 0.1,
            "x_offset_ratio": -0.00469529085872575,
            "y_offset_ratio": 0.18208033240997226,
            "text_rotation": 0,
            "text_pos_x": 0.779595567867036,
            "text_pos_y": 0.08390027700831038,
            "lane_font_size_ratio": 0.08,
            "text_font_size_ratio": 0.047
        },
        "West": {
            "lanes": 3,
            "lane_w_ratio": 0.1,
            "x_offset_ratio": 0,
            "y_offset_ratio": 0.155,
            "text_rotation": 0,
            "text_pos_x": 0.594,
            "text_pos_y": 0.908,
            "lane_font_size_ratio": 0.08,
            "text_font_size_ratio": 0.047
        },
        "upstream_West": {
            "lanes": 3,
            "lane_w_ratio": 0.1,
            "x_offset_ratio": -0.00469529085872575,
            "y_offset_ratio": 0.18208033240997226,
            "text_rotation": 0,
            "text_pos_x": 0.779595567867036,
            "text_pos_y": 0.08390027700831038,
            "lane_font_size_ratio": 0.08,
            "text_font_size_ratio": 0.047
        },
        "East": {
            "lanes": 3,
            "lane_w_ratio": 0.1,
            "x_offset_ratio": 0,
            "y_offset_ratio": 0.155,
            "text_rotation": 0,
            "text_pos_x": 0.594,
            "text_pos_y": 0.908,
            "lane_font_size_ratio": 0.08,
            "text_font_size_ratio": 0.047
        },
        "upstream_East": {
            "lanes": 3,
            "lane_w_ratio": 0.1,
            "x_offset_ratio": -0.00469529085872575,
            "y_offset_ratio": 0.18208033240997226,
            "text_rotation": 0,
            "text_pos_x": 0.779595567867036,
            "text_pos_y": 0.08390027700831038,
            "lane_font_size_ratio": 0.08,
            "text_font_size_ratio": 0.047
        }
    },
    "T_JUNCTION_RHT_2LANE": {
        "North": {
            "lanes": 2,
            "lane_w_ratio": 0.103,
            "x_offset_ratio": -0.0041551246537396645,
            "y_offset_ratio": 0.175,
            "text_rotation": 5,
            "text_pos_x": 0.5,
            "text_pos_y": 0.882,
            "lane_font_size_ratio": 0.08,
            "text_font_size_ratio": 0.047
        },
        "upstream_North": {
            "lanes": 2,
            "lane_w_ratio": 0.131,
            "x_offset_ratio": -0.019390581717451498,
            "y_offset_ratio": 0.2285318559556787,
            "text_rotation": 11,
            "text_pos_x": 0.8033240997229917,
            "text_pos_y": 0.18711911357340721,
            "lane_font_size_ratio": 0.08,
            "text_font_size_ratio": 0.047
        },
        "South": {
            "lanes": 2,
            "lane_w_ratio": 0.091,
            "x_offset_ratio": 0.00831024930747909,
            "y_offset_ratio": 0.18,
            "text_rotation": 5,
            "text_pos_x": 0.5,
            "text_pos_y": 0.882,
            "lane_font_size_ratio": 0.08,
            "text_font_size_ratio": 0.047
        },
        "upstream_South": {
            "lanes": 2,
            "lane_w_ratio": 0.05,
            "x_offset_ratio": 0,
            "y_offset_ratio": 0,
            "text_rotation": 0,
            "text_pos_x": 0.5,
            "text_pos_y": 0.05,
            "lane_font_size_ratio": 0.08,
            "text_font_size_ratio": 0.047
        },
        "West": {
            "lanes": 2,
            "lane_w_ratio": 0.091,
            "x_offset_ratio": -0.011080332409972414,
            "y_offset_ratio": 0.18,
            "text_rotation": 1,
            "text_pos_x": 0.6055235457063711,
            "text_pos_y": 0.8817867036011079,
            "lane_font_size_ratio": 0.08,
            "text_font_size_ratio": 0.047
        },
        "upstream_West": {
            "lanes": 2,
            "lane_w_ratio": 0.098,
            "x_offset_ratio": -0.004155124653739666,
            "y_offset_ratio": 0.19390581717451524,
            "text_rotation": 0,
            "text_pos_x": 0.7894736842105264,
            "text_pos_y": 0.11925207756232686,
            "lane_font_size_ratio": 0.08,
            "text_font_size_ratio": 0.047
        }
    },
    "SONGDO_4WAY_RHT_ASYM": {
        "North": {
            "lanes": 6,
            "lane_w_ratio": 0.111,
            "x_offset_ratio": -0.011,
            "y_offset_ratio": 0.165,
            "text_rotation": -1,
            "text_pos_x": 0.46458725761772857,
            "text_pos_y": 0.8727146814404434,
            "lane_font_size_ratio": 0.08,
            "text_font_size_ratio": 0.047
        },
        "upstream_North": {
            "lanes": 6,
            "lane_w_ratio": 0.102,
            "x_offset_ratio": -0.0013850415512465727,
            "y_offset_ratio": 0.17451523545706368,
            "text_rotation": 0,
            "text_pos_x": 0.4930747922437673,
            "text_pos_y": 0.8519390581717452,
            "lane_font_size_ratio": 0.08,
            "text_font_size_ratio": 0.047
        },
        "South": {
            "lanes": 5,
            "lane_w_ratio": 0.103,
            "x_offset_ratio": -0.024930747922437657,
            "y_offset_ratio": 0.165,
            "text_rotation": 2,
            "text_pos_x": 0.49168975069252085,
            "text_pos_y": 0.8727146814404432,
            "lane_font_size_ratio": 0.08,
            "text_font_size_ratio": 0.047
        },
        "upstream_South": {
            "lanes": 5,
            "lane_w_ratio": 0.103,
            "x_offset_ratio": 0.011080332409972329,
            "y_offset_ratio": 0.18144044321329633,
            "text_rotation": 0,
            "text_pos_x": 0.4986149584487534,
            "text_pos_y": 0.8768698060941831,
            "lane_font_size_ratio": 0.08,
            "text_font_size_ratio": 0.047
        },
        "West": {
            "lanes": 6,
            "lane_w_ratio": 0.102,
            "x_offset_ratio": 0.02631578947368421,
            "y_offset_ratio": 0.165,
            "text_rotation": 0,
            "text_pos_x": 0.5152354570637119,
            "text_pos_y": 0.885180055401662,
            "lane_font_size_ratio": 0.08,
            "text_font_size_ratio": 0.047
        },
        "upstream_West": {
            "lanes": 6,
            "lane_w_ratio": 0.109,
            "x_offset_ratio": -0.03462603878116338,
            "y_offset_ratio": 0.17867036011080323,
            "text_rotation": 0,
            "text_pos_x": 0.3905817174515235,
            "text_pos_y": 0.8560941828254849,
            "lane_font_size_ratio": 0.08,
            "text_font_size_ratio": 0.047
        },
        "East": {
            "lanes": 5,
            "lane_w_ratio": 0.115,
            "x_offset_ratio": 0.06648199445983376,
            "y_offset_ratio": 0.155,
            "text_rotation": 0,
            "text_pos_x": 0.5,
            "text_pos_y": 0.869,
            "lane_font_size_ratio": 0.08,
            "text_font_size_ratio": 0.047
        },
        "upstream_East": {
            "lanes": 5,
            "lane_w_ratio": 0.107,
            "x_offset_ratio": -0.06786703601108034,
            "y_offset_ratio": 0.17590027700831032,
            "text_rotation": 0,
            "text_pos_x": 0.4626038781163435,
            "text_pos_y": 0.8768698060941829,
            "lane_font_size_ratio": 0.08,
            "text_font_size_ratio": 0.047
        }
    },
    "YMT_4WAY_LHT_3LANE": {
        "North": {
            "lanes": 3,
            "lane_w_ratio": 0.094,
            "x_offset_ratio": 0.005240997229916916,
            "y_offset_ratio": 0.165,
            "text_rotation": 0,
            "text_pos_x": 0.47506925207756234,
            "text_pos_y": 0.8796398891966761,
            "lane_font_size_ratio": 0.08,
            "text_font_size_ratio": 0.047
        },
        "upstream_North": {
            "lanes": 3,
            "lane_w_ratio": 0.098,
            "x_offset_ratio": 0.015235457063711835,
            "y_offset_ratio": 0.18975069252077573,
            "text_rotation": 0,
            "text_pos_x": 0.4861495844875346,
            "text_pos_y": 0.858864265927978,
            "lane_font_size_ratio": 0.08,
            "text_font_size_ratio": 0.047
        },
        "South": {
            "lanes": 3,
            "lane_w_ratio": 0.1,
            "x_offset_ratio": 0.019390581717451498,
            "y_offset_ratio": 0.170,
            "text_rotation": 0,
            "text_pos_x": 0.4626038781163434,
            "text_pos_y": 0.8893351800554017,
            "lane_font_size_ratio": 0.08,
            "text_font_size_ratio": 0.047
        },
        "upstream_South": {
            "lanes": 3,
            "lane_w_ratio": 0.101,
            "x_offset_ratio": -0.02493074792243767,
            "y_offset_ratio": 0.18698060941828248,
            "text_rotation": 0,
            "text_pos_x": 0.41412742382271467,
            "text_pos_y": 0.8671745152354572,
            "lane_font_size_ratio": 0.08,
            "text_font_size_ratio": 0.047
        },
        "West": {
            "lanes": 3,
            "lane_w_ratio": 0.1,
            "x_offset_ratio": -0.005540166204986163,
            "y_offset_ratio": 0.160,
            "text_rotation": 0,
            "text_pos_x": 0.4986149584487535,
            "text_pos_y": 0.8907202216066482,
            "lane_font_size_ratio": 0.08,
            "text_font_size_ratio": 0.047
        },
        "upstream_West": {
            "lanes": 3,
            "lane_w_ratio": 0.112,
            "x_offset_ratio": -0.012465373961218836,
            "y_offset_ratio": 0.2036011080332409,
            "text_rotation": 0,
            "text_pos_x": 0.5,
            "text_pos_y": 0.845,
            "lane_font_size_ratio": 0.08,
            "text_font_size_ratio": 0.047
        },
        "East": {
            "lanes": 3,
            "lane_w_ratio": 0.093,
            "x_offset_ratio": -0.029085872576177244,
            "y_offset_ratio": 0.175,
            "text_rotation": 0,
            "text_pos_x": 0.443213296398892,
            "text_pos_y": 0.8782548476454294,
            "lane_font_size_ratio": 0.08,
            "text_font_size_ratio": 0.047
        },
        "upstream_East": {
            "lanes": 3,
            "lane_w_ratio": 0.12,
            "x_offset_ratio": 0.042936288088642666,
            "y_offset_ratio": 0.203601108033241,
            "text_rotation": -6,
            "text_pos_x": 0.5,
            "text_pos_y": 0.88,
            "lane_font_size_ratio": 0.08,
            "text_font_size_ratio": 0.047
        }
    }
}

def extract_info_from_path(input_path: str):
    """
    通过父文件路径推断场景名、提取方向和是否为上游图片。
    """
    parts = pathlib.Path(input_path).parts
    scenario_name = None
    if "eval" in parts:
        eval_idx = parts.index("eval")
        if eval_idx + 1 < len(parts):
            scenario_name = parts[eval_idx + 1]
    
    if scenario_name is None:
        scenario_name = "JiNan"
        
    filename = os.path.basename(input_path)
    filename_no_ext = os.path.splitext(filename)[0]
    
    is_upstream = filename_no_ext.startswith("upstream_")
    
    match = re.search(r'_([NESW])$', filename_no_ext)
    dir_short = match.group(1) if match else "N"
    
    DIR_MAP = {"N": "North", "E": "East", "S": "South", "W": "West"}
    dir_long = DIR_MAP.get(dir_short, "North")
    
    approach_key = f"upstream_{dir_long}" if is_upstream else dir_long
    return scenario_name, approach_key, is_upstream, dir_long

def _draw_approach_rigid(
    txt_layer: Image.Image,
    group_cx: float, group_cy: float,
    lanes: list,
    font,
    text_color: tuple,
    outline_color: tuple,
    rotation: int = 0,
) -> None:
    if not lanes:
        return

    stroke_width = 2
    tmp_measure = ImageDraw.Draw(Image.new("RGBA", (1, 1)))

    metrics = []
    for text, rx, ry in lanes:
        bbox = tmp_measure.textbbox((0, 0), text, font=font, stroke_width=stroke_width)
        tw = bbox[2] - bbox[0]
        th = bbox[3] - bbox[1]
        metrics.append((text, rx, ry, tw, th))

    pad = 12
    max_half_w = max(abs(rx) + tw / 2 for _, rx, _, tw, _ in metrics)
    max_half_h = max(abs(ry) + th / 2 for _, _, ry, _, th in metrics)
    canvas_w = int(max_half_w * 2 + pad * 2)
    canvas_h = int(max_half_h * 2 + pad * 2)

    group_canvas = Image.new("RGBA", (canvas_w, canvas_h), (0, 0, 0, 0))
    d = ImageDraw.Draw(group_canvas)

    cx_local = canvas_w / 2.0
    cy_local = canvas_h / 2.0

    for text, rx, ry, tw, th in metrics:
        tx = int(round(cx_local + rx - tw / 2))
        ty = int(round(cy_local + ry - th / 2))
        d.text(
            (tx, ty),
            text,
            font=font,
            fill=text_color,
            stroke_width=stroke_width,
            stroke_fill=outline_color,
        )

    if rotation != 0:
        group_canvas = group_canvas.rotate(-rotation, expand=True, resample=Image.BICUBIC)

    paste_x = int(round(group_cx - group_canvas.width / 2))
    paste_y = int(round(group_cy - group_canvas.height / 2))

    layer_canvas = Image.new("RGBA", txt_layer.size, (0, 0, 0, 0))
    layer_canvas.paste(group_canvas, (paste_x, paste_y))
    txt_layer.alpha_composite(layer_canvas)

def add_lane_watermarks(input_path: str, output_path: str = None, scenario_name: str = None):
    if not os.path.exists(input_path):
        print(f"❌ 找不到输入图片: {input_path}")
        return

    extracted_scenario, approach_key, is_upstream, dir_long = extract_info_from_path(input_path)
    if not scenario_name:
        scenario_name = extracted_scenario

    base_img = Image.open(input_path).convert("RGBA")
    W, H = base_img.size

    txt_layer = Image.new("RGBA", base_img.size, (255, 255, 255, 0))

    text_color    = (255, 255, 255, 180)
    outline_color = (0,   0,   0,   100)

    config_key = SCENARIO_TO_WATERMARK_KEY.get(scenario_name, "4_WAY_RHT_3LANE")
    config = WATERMARK_CONFIGS.get(config_key, {}).get(approach_key)

    if config:
        lane_font_ratio = config.get("lane_font_size_ratio", 0.05) if scenario_name != 'SouthKorea_Songdo' else config.get("lane_font_size_ratio", 0.04)
        text_font_ratio = config.get("text_font_size_ratio", 0.05) if scenario_name != 'SouthKorea_Songdo' else config.get("text_font_size_ratio", 0.04)
    else:
        lane_font_ratio = 0.05
        text_font_ratio = 0.05

    try:
        lane_font = ImageFont.truetype("data/font/Arial.ttf", int(W * lane_font_ratio))
        text_font = ImageFont.truetype("data/font/Arial.ttf", int(W * text_font_ratio))
    except IOError:
        lane_font = ImageFont.load_default()
        text_font = ImageFont.load_default()
    
    if not config:
        print(f"⚠️ 找不到 {scenario_name} 的 {approach_key} 预设，略过水印绘制。")
        watermarked_img = base_img
    else:
        n_lanes      = config["lanes"]
        lane_w_ratio = config.get("lane_w_ratio", 0.05)
        x_offset     = W * config.get("x_offset_ratio", 0.0)
        y_offset     = H * config.get("y_offset_ratio", 0.0)
        text_rotation = config.get("text_rotation", 0)
        text_pos_x_ratio = config.get("text_pos_x", 0.5)
        text_pos_y_ratio = config.get("text_pos_y", 0.05)

        Cx, Cy = W / 2.0, H / 2.0
        group_cx = Cx + x_offset
        group_cy = Cy + y_offset

        lanes = []
        for lane_idx in range(1, n_lanes + 1):
            # 将车道标识横向展开，通过 x_offset/y_offset 以及整体的 text_rotation 来调节（y轴镜像）
            base_rx = (lane_idx - (n_lanes + 1) / 2.0) * W * lane_w_ratio
             # 根据要求：downstream 翻转，upstream 不翻转
            if is_upstream:
                rx = base_rx
            else:
                rx = -base_rx
            if scenario_name == 'Hongkong_YMT':
                rx = -rx
            ry = 0.0
            lanes.append((str(lane_idx), rx, ry))

        _draw_approach_rigid(
            txt_layer, group_cx, group_cy, lanes,
            lane_font, text_color, outline_color, text_rotation
        )

        d = ImageDraw.Draw(txt_layer)
        # info_text = f"Approach: {dir_long}\nType: {'Upstream' if is_upstream else 'Downstream'}"
        info_text = f"Approach: {dir_long}"
        stroke_width = 2
        tx = int(W * text_pos_x_ratio)
        ty = int(H * text_pos_y_ratio)
        d.text(
            (tx, ty),
            info_text,
            font=text_font,
            fill=text_color,
            anchor="mm",
            align="center",
            stroke_width=stroke_width,
            stroke_fill=outline_color,
        )

        watermarked_img = Image.alpha_composite(base_img, txt_layer)

    final_img = watermarked_img.convert("RGB")

    if not output_path:
        base, ext = os.path.splitext(input_path)
        output_path = f"{base}_watermarked{ext}"

    final_img.save(output_path)
    print(f"✅ 水印添加成功 [{scenario_name} - {approach_key}]! 已保存至: {output_path}")

if __name__ == "__main__":
    input_data_dict ={
        # "JiNan":             "data/eval/JiNan/anon_3_4_jinan_real/qwen3-vl-4b/intersection_1_1/28/intersection_1_1_E.png",
        # "Hangzhou":          "data/eval/Hangzhou/anon_4_4_hangzhou_real/qwen3-vl-4b/intersection_1_3/28/upstream_intersection_1_3_E.png",
        # "NewYork":           "data/eval/Hangzhou/anon_4_4_hangzhou_real/qwen3-vl-4b/intersection_1_3/28/upstream_intersection_1_3_W.png",
        "France_Massy":      "data/sft_dataset/France_Massy/massy_bus/INT1/53/INT1_S.png",
    #     "SouthKorea_Songdo": "data/eval/SouthKorea_Songdo/songdo/qwen3-vl-4b/J2/28/J2_W.png",
    #     "Hongkong_YMT":      "data/eval/Hongkong_YMT/YMT/qwen3-vl-4b/J1/54/J1_S.png",
    }

    for scenario, input_path in input_data_dict.items():
        add_lane_watermarks(input_path, scenario_name=scenario)
        