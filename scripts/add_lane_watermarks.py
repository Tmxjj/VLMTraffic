'''
Author: yufei Ji
Date: 2026-03-03 20:25:07
LastEditTime: 2026-04-13 20:24:58
Description: 为路口 BEV 图像在各进口道车道上添加淡色数字水印，支持多种场景拓扑：
             - 标准四叉路口 RHT 3 车道 (JiNan, Hangzhou, NewYork)
             - T 型三叉路口 RHT 2 车道 (France_Massy)
             - Songdo 四叉路口 RHT 非对称 5/6 车道 (SouthKorea_Songdo)
             - 香港 YMT 四叉路口 LHT 3 车道 (Hongkong_YMT)
FilePath: /VLMTraffic/scripts/add_lane_watermarks.py
'''
import os
from PIL import Image, ImageDraw, ImageFont


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
# 水印布局配置）
#
# 每个 approach 包含：
#   type             : "vertical"   → 沿相对水平展开车道，"horizontal" → 沿相对垂直展开车道
#   lanes            : 该进口道的车道数量
#   side_sign        : 车道偏移方向（+1 增加，-1 减少）
#   lane_w_ratio     : 单车道像素宽度 = W * ratio
#   x_offset_ratio   : 整个车道组的中心原点的水平偏移 = W * ratio
#   y_offset_ratio   : 整个车道组的中心原点的垂直偏移 = H * ratio
#   text_rotation    : 整个组围绕其中心原点旋转的角度（以及文字自身的旋转）（度）
#
# 坐标计算：
#   原点 O = (Cx + W*x_offset_ratio, Cy + H*y_offset_ratio)
#   计算相对于 O 的车道偏移并执行整个组别的矩阵旋转
#
# 调参指南：
#   1. 先用 __main__ 或可视化前端对单张图片测试
#   2. 直接使用鼠标拖拽来直接调整 x_offset_ratio 和 y_offset_ratio
#   3. 若标签间距不等于车道宽度 → 调整 lane_w_ratio
#   4. 若整体车道斜向 → 调节 text_rotation
# ─────────────────────────────────────────────────────────────────
WATERMARK_CONFIGS = {

    # ════════════════════════════════════════════════════════════════
    # 标准四叉路口 RHT，每方向 3 车道
    # 适用场景：JiNan / Hangzhou / JiNan_test / NewYork
    # ════════════════════════════════════════════════════════════════

    # "4_WAY_RHT_3LANE": {
    #     "approaches": {
    #         "South": {
    #             "type": "vertical",
    #             "lanes": 3,
    #             "side_sign": 1,
    #             "lane_w_ratio": 0.025,
    #             "x_offset_ratio": 0.04190937353210589,
    #             "y_offset_ratio": 0.09456986960984713,
    #             "text_rotation": 0
    #         },
    #         "North": {
    #             "type": "vertical",
    #             "lanes": 3,
    #             "side_sign": -1,
    #             "lane_w_ratio": 0.025,
    #             "x_offset_ratio": -0.04190937353210594,
    #             "y_offset_ratio": -0.101687954595681,
    #             "text_rotation": 0
    #         },
    #         "West": {
    #             "type": "horizontal",
    #             "lanes": 3,
    #             "side_sign": 1,
    #             "lane_w_ratio": 0.025,
    #             "x_offset_ratio": -0.10101672046341671,
    #             "y_offset_ratio": 0.037036558111082196,
    #             "text_rotation": 0
    #         },
    #         "East": {
    #             "type": "horizontal",
    #             "lanes": 3,
    #             "side_sign": -1,
    #             "lane_w_ratio": 0.025,
    #             "x_offset_ratio": 0.10101528368555225,
    #             "y_offset_ratio": -0.040928967551251805,
    #             "text_rotation": 0
    #         }
    #     }
    # },
    "4_WAY_RHT_3LANE": {
        "approaches": {
            "South": {
                "type": "vertical",
                "lanes": 3,
                "side_sign": 1,
                "lane_w_ratio": 0.025,
                "x_offset_ratio": 0.03216278483814872,
                "y_offset_ratio": 0.07800066883012001,
                "text_rotation": 0
            },
            "North": {
                "type": "vertical",
                "lanes": 3,
                "side_sign": -1,
                "lane_w_ratio": 0.025,
                "x_offset_ratio": -0.03313744370754451,
                "y_offset_ratio": -0.08511875381595388,
                "text_rotation": 0
            },
            "West": {
                "type": "horizontal",
                "lanes": 3,
                "side_sign": 1,
                "lane_w_ratio": 0.025,
                "x_offset_ratio": -0.08542217855308533,
                "y_offset_ratio": 0.031188604894707872,
                "text_rotation": 0
            },
            "East": {
                "type": "horizontal",
                "lanes": 3,
                "side_sign": -1,
                "lane_w_ratio": 0.025,
                "x_offset_ratio": 0.08639540064461639,
                "y_offset_ratio": -0.035081014334877536,
                "text_rotation": 0
            }
        }
    },
    "T_JUNCTION_RHT_2LANE": {
        "approaches": {
            "North": {
                "type": "vertical",
                "lanes": 2,
                "side_sign": -1,
                "lane_w_ratio": 0.028,
                "x_offset_ratio": -0.03244105834884113,
                "y_offset_ratio": -0.08175329365732507,
                "text_rotation": -21
            },
            "South": {
                "type": "vertical",
                "lanes": 2,
                "side_sign": 1,
                "lane_w_ratio": 0.029,
                "x_offset_ratio": 0.08517260698576544,
                "y_offset_ratio": 0.031203108853164643,
                "text_rotation": -26
            },
            "West": {
                "type": "horizontal",
                "lanes": 2,
                "side_sign": 1,
                "lane_w_ratio": 0.029,
                "x_offset_ratio": -0.04447118040790983,
                "y_offset_ratio": 0.03772197011782709,
                "text_rotation": -25
            }
        }
    },

    "SONGDO_4WAY_RHT_ASYM": {
        "approaches": {
            "North": {
                "type": "vertical",
                "lanes": 6,
                "side_sign": -1,
                "lane_w_ratio": 0.019,
                "x_offset_ratio": -0.06292803792581353,
                "y_offset_ratio": -0.25317798365244215,
                "text_rotation": -3
            },
            "South": {
                "type": "vertical",
                "lanes": 5,
                "side_sign": 1,
                "lane_w_ratio": 0.0198,
                "x_offset_ratio": 0.07512859883587686,
                "y_offset_ratio": 0.2556187001632152,
                "text_rotation": 0
            },
            "West": {
                "type": "horizontal",
                "lanes": 6,
                "side_sign": 1,
                "lane_w_ratio": 0.0198,
                "x_offset_ratio": -0.24195242816971596,
                "y_offset_ratio": 0.09269259608943503,
                "text_rotation": 17
            },
            "East": {
                "type": "horizontal",
                "lanes": 5,
                "side_sign": -1,
                "lane_w_ratio": 0.022,
                "x_offset_ratio": 0.2307297734650399,
                "y_offset_ratio": -0.0922047912113863,
                "text_rotation": 17
            }
        }
    },

    "YMT_4WAY_LHT_3LANE": {
        "approaches": {
            "South": {
                "type": "vertical",
                "lanes": 3,
                "side_sign": -1,
                "lane_w_ratio": 0.025,
                "x_offset_ratio": -0.02211258150289498,
                "y_offset_ratio": 0.08975185687946056,
                "text_rotation": -8
            },
            "North": {
                "type": "vertical",
                "lanes": 3,
                "side_sign": 1,
                "lane_w_ratio": 0.025,
                "x_offset_ratio": 0.026012174832387613,
                "y_offset_ratio": -0.10547310444281342,
                "text_rotation": -8
            },
            "West": {
                "type": "horizontal",
                "lanes": 3,
                "side_sign": -1,
                "lane_w_ratio": 0.025,
                "x_offset_ratio": -0.10254960676058116,
                "y_offset_ratio": -0.025332988863055794,
                "text_rotation": -9
            },
            "East": {
                "type": "horizontal",
                "lanes": 3,
                "side_sign": 1,
                "lane_w_ratio": 0.025,
                "x_offset_ratio": 0.09377576123220015,
                "y_offset_ratio": 0.02533729919664962,
                "text_rotation": -7
            }
        }
    },
}


def _draw_approach_rigid(
    txt_layer: Image.Image,
    group_cx: float, group_cy: float,
    lanes: list,
    font,
    text_color: tuple,
    outline_color: tuple,
    rotation: int = 0,
) -> None:
    """
    将一个进口道的所有车道数字水印作为刚体整体渲染并旋转。

    先将所有数字按未旋转布局绘制到同一临时组画布，再将整个画布
    绕其中心旋转 rotation 度，最后以 (group_cx, group_cy) 为锚点
    贴回主画布。

    :param group_cx:  进口道组中心在主画布上的 x 坐标
    :param group_cy:  进口道组中心在主画布上的 y 坐标
    :param lanes:     [(text, rx, ry), ...]，rx/ry 为相对组中心的未旋转偏移
    """
    if not lanes:
        return

    stroke_width = 2
    tmp_measure = ImageDraw.Draw(Image.new("RGBA", (1, 1)))

    # 1. 测量每个数字的文字包围盒
    metrics = []
    for text, rx, ry in lanes:
        bbox = tmp_measure.textbbox((0, 0), text, font=font, stroke_width=stroke_width)
        tw = bbox[2] - bbox[0]
        th = bbox[3] - bbox[1]
        metrics.append((text, rx, ry, tw, th))

    # 2. 计算组画布尺寸：覆盖所有数字位置 + 旋转余量
    pad = 12
    max_half_w = max(abs(rx) + tw / 2 for _, rx, _, tw, _ in metrics)
    max_half_h = max(abs(ry) + th / 2 for _, _, ry, _, th in metrics)
    canvas_w = int(max_half_w * 2 + pad * 2)
    canvas_h = int(max_half_h * 2 + pad * 2)

    group_canvas = Image.new("RGBA", (canvas_w, canvas_h), (0, 0, 0, 0))
    d = ImageDraw.Draw(group_canvas)

    # 组画布逻辑中心（对应外部 group_cx / group_cy）
    cx_local = canvas_w / 2.0
    cy_local = canvas_h / 2.0

    # 3. 在组画布上按未旋转坐标绘制每条车道的数字（不作任何单独旋转）
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

    # 4. 将整个组画布作为刚体绕其中心旋转
    # 注意：Pillow rotate 正角度为逆时针，HTML Canvas rotate 正角度为顺时针，
    # 取负号保持与前端调参预览的旋转方向一致（均以顺时针为正）。
    if rotation != 0:
        group_canvas = group_canvas.rotate(-rotation, expand=True, resample=Image.BICUBIC)

    # 5. 以 group_cx / group_cy 为锚点贴回主画布
    paste_x = int(round(group_cx - group_canvas.width / 2))
    paste_y = int(round(group_cy - group_canvas.height / 2))

    layer_canvas = Image.new("RGBA", txt_layer.size, (0, 0, 0, 0))
    layer_canvas.paste(group_canvas, (paste_x, paste_y))
    txt_layer.alpha_composite(layer_canvas)


def add_lane_watermarks(input_path: str, output_path: str = None, scenario_name: str = "JiNan"):
    """
    为路口 BEV 图像在各进口道车道上添加淡色数字水印。

    :param input_path:    输入图片路径
    :param output_path:   输出图片路径（默认在原文件同目录下追加 _watermarked 后缀）
    :param scenario_name: 场景名称，用于从 WATERMARK_CONFIGS 中选取对应布局（默认 "JiNan"）
    """
    if not os.path.exists(input_path):
        print(f"❌ 找不到输入图片: {input_path}")
        return

    # 打开图像并转换为 RGBA 以支持半透明文字层
    base_img = Image.open(input_path).convert("RGBA")
    W, H = base_img.size

    # 全透明文字层
    txt_layer = Image.new("RGBA", base_img.size, (255, 255, 255, 0))

    # 字体（自适应图片宽度；找不到系统字体时降级使用默认字体）
    font_size = int(W * 0.02) if scenario_name == 'SouthKorea_Songdo' else int(W*0.03)
    try:
        font = ImageFont.truetype("data/font/Arial.ttf", font_size)
    except IOError:
        font = ImageFont.load_default()

    # 水印颜色：白色主体 + 黑色描边，均为半透明
    text_color    = (255, 255, 255, 180)  # A=120 约 47% 不透明度
    outline_color = (0,   0,   0,   100)  # A=100 约 39% 不透明度

    # ── 选取水印配置 ──────────────────────────────────────────────
    config_key = SCENARIO_TO_WATERMARK_KEY.get(scenario_name, "4_WAY_RHT_3LANE")
    config     = WATERMARK_CONFIGS[config_key]

    Cx, Cy      = W / 2.0, H / 2.0

    # ── 计算各进口道车道标记坐标并绘制 ──────────────────────────────
    for approach, cfg in config["approaches"].items():
        n_lanes      = cfg["lanes"]
        side_sign    = cfg["side_sign"]
        ap_type      = cfg["type"]
        lane_w_ratio = cfg.get("lane_w_ratio", 0.03)
        x_offset     = W * cfg.get("x_offset_ratio", 0.0)
        y_offset     = H * cfg.get("y_offset_ratio", 0.0)
        text_rotation = cfg.get("text_rotation", 0)

        # 进口道组中心在主画布上的坐标
        group_cx = Cx + x_offset
        group_cy = Cy + y_offset

        # 收集每条车道相对组中心的未旋转偏移 (rx, ry)
        lanes = []
        for lane_idx in range(1, n_lanes + 1):
            lane_offset = (lane_idx - (n_lanes + 1) / 2.0) * W * lane_w_ratio
            if ap_type == "vertical":
                rx, ry = side_sign * lane_offset, 0.0
            else:
                rx, ry = 0.0, side_sign * lane_offset
            lanes.append((str(lane_idx), rx, ry))

        # 将整个进口道的数字作为刚体整体旋转后绘制
        _draw_approach_rigid(
            txt_layer, group_cx, group_cy, lanes,
            font, text_color, outline_color, text_rotation
        )

    # ── 合成并保存 ────────────────────────────────────────────────
    watermarked_img = Image.alpha_composite(base_img, txt_layer)
    final_img       = watermarked_img.convert("RGB")

    if not output_path:
        base, ext = os.path.splitext(input_path)
        output_path = f"{base}_watermarked{ext}"

    final_img.save(output_path)
    print(f"✅ 水印添加成功 [{scenario_name}]! 已保存至: {output_path}")


if __name__ == "__main__":
    # 使用示例
    input_image = "data/test/France_Massy/2/aircraft_INT1.png"
    add_lane_watermarks(input_image, scenario_name="France_Massy")

    input_image1 = "data/test/Hongkong_YMT/1/aircraft_J1.png"
    add_lane_watermarks(input_image1, scenario_name="Hongkong_YMT")

    input_image2 = "data/test/SouthKorea_Songdo/1/aircraft_J2.png"
    add_lane_watermarks(input_image2, scenario_name="SouthKorea_Songdo")

    input_image3 = "data/test/JiNan/8/aircraft_intersection_1_1.png"
    add_lane_watermarks(input_image3, scenario_name="JiNan")
