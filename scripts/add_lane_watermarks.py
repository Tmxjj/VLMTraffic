'''
Author: yufei Ji
Date: 2026-03-03 20:25:07
LastEditTime: 2026-03-05 23:36:33
Description: this script is used to 为十字路口的 BEV 图像在各个进口道的车道上添加极淡的数字水印 (1, 2, 3)，以帮助模型更好地理解车道位置和数量
FilePath: /VLMTraffic/scripts/add_lane_watermarks.py
'''
import os
from PIL import Image, ImageDraw, ImageFont

def add_lane_watermarks(input_path: str, output_path: str = None):
    """
    为十字路口的 BEV 图像在各个进口道的车道上添加极淡的数字水印 (1, 2, 3)。
    
    :param input_path: 输入图片的路径
    :param output_path: 输出图片的路径 (默认在原文件同目录下追加 _watermarked)
    """
    if not os.path.exists(input_path):
        print(f"❌ 找不到输入图片: {input_path}")
        return

    # 打开图像并转换为 RGBA 以支持透明度层
    base_img = Image.open(input_path).convert("RGBA")
    W, H = base_img.size

    # 创建一个与原图大小相同的全透明图层
    txt_layer = Image.new("RGBA", base_img.size, (255, 255, 255, 0))
    draw = ImageDraw.Draw(txt_layer)

    # --- 参数配置区 ---
    # 尝试加载较大且清晰的字体，如果找不到则使用默认字体
    font_size = int(W * 0.03) # 字体大小自适应图片宽度
    try:
        # 常见操作系统的默认字体路径
        font = ImageFont.truetype("data/font/Arial.ttf", font_size)
    except IOError:
        print("⚠️ 未找到系统字体，使用默认字体 (可能较小)")
        font = ImageFont.load_default()

    # 水印颜色与透明度：RGBA格式，A=120 表示极淡 (取值0-255)
    text_color = (255, 255, 255, 120)  # 白色，透明度约 50%
    outline_color = (0, 0, 0, 100)      # 黑色描边，增加在浅色路面或白车上的对比度

    # --- 坐标计算 (基于标准的对称十字路口) ---
    # 假设中心点
    Cx, Cy = W / 2, H / 2
    
    # 估算车道宽度和停止线位置 (根据您的图片比例估算)
    lane_w = W * 0.03  # 单根车道的像素宽度
    stop_dist_y = H * 0.12 # 停止线距离中心的Y轴距离
    stop_dist_x = W * 0.12 # 停止线距离中心的X轴距离

    # 距中心黄线的偏移量：Lane 1 (0.5w), Lane 2 (1.5w), Lane 3 (2.5w)
    d1, d2, d3 = lane_w * 0.5, lane_w * 1.5, lane_w * 2.5

    

    # 定义四个进口道的坐标 (基于 RHT 右侧通行规则)
    watermark_positions = {
        "South": [ # 位于下方，车流向上，右侧半幅路
            ("1", Cx + d1*0.94, Cy + stop_dist_y*0.78),
            ("2", Cx + d2*0.94, Cy + stop_dist_y*0.78),
            ("3", Cx + d3*0.94, Cy + stop_dist_y*0.78)
        ],
        "North": [ # 位于上方，车流向下，左侧半幅路
            ("1", Cx - d1*0.94, Cy - stop_dist_y*0.88),
            ("2", Cx - d2*0.94, Cy - stop_dist_y*0.88),
            ("3", Cx - d3*0.94, Cy - stop_dist_y*0.88)
        ],
        "West": [  # 位于左侧，车流向右，下方半幅路
            ("1", Cx - stop_dist_x*0.85, Cy + d1*0.8),
            ("2", Cx - stop_dist_x*0.85, Cy + d2*0.8),
            ("3", Cx - stop_dist_x*0.85, Cy + d3*0.8)
        ],
        "East": [  # 位于右侧，车流向左，上方半幅路
            ("1", Cx + stop_dist_x*0.85, Cy - d1),
            ("2", Cx + stop_dist_x*0.85, Cy - d2),
            ("3", Cx + stop_dist_x*0.85, Cy - d3)
        ]
    }

    # --- 绘制文本 ---
    for approach, lanes in watermark_positions.items():
        for text, x, y in lanes:
            # 居中对齐坐标
            bbox = draw.textbbox((0, 0), text, font=font)
            text_w = bbox[2] - bbox[0]
            text_h = bbox[3] - bbox[1]
            pos = (x - text_w / 2, y - text_h / 2)

            # 绘制极淡的黑色描边 (上下左右偏移1像素)
            for offset_x, offset_y in [(-1,-1), (1,-1), (-1,1), (1,1)]:
                draw.text((pos[0]+offset_x, pos[1]+offset_y), text, font=font, fill=outline_color)
            
            # 绘制白色主体数字
            draw.text(pos, text, font=font, fill=text_color)

    # --- 合成与保存 ---
    # 将包含半透明字体的层与原图合并
    watermarked_img = Image.alpha_composite(base_img, txt_layer)
    
    # 转换回 RGB 并保存
    final_img = watermarked_img.convert("RGB")
    
    if not output_path:
        base, ext = os.path.splitext(input_path)
        output_path = f"{base}_watermarked{ext}"
        
    final_img.save(output_path)
    print(f"✅ 水印添加成功! 已保存至: {output_path}")

if __name__ == "__main__":
    # 使用示例：替换为您的实际图片路径
    input_image1 = "data/eval/JiNan/anon_3_4_jinan_real.rou/qwen3-vl-8b/step_0/aircraft_intersection_1_1_bev_raw.png"
    add_lane_watermarks(input_image1)
    # input_image2 = "data/sft_dataset/JiNan/step_3/intersection_4_1_bev.jpg"
    # add_lane_watermarks(input_image2)
    # input_image3 = "data/sft_dataset/JiNan/step_3/intersection_3_1_bev.jpg"
    # add_lane_watermarks(input_image3)    