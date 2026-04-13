'''
Author: yufei Ji
Description: 数字水印可视化调参 GUI（Tkinter）
             与 add_lane_watermarks.py 的刚体旋转逻辑严格对齐：
             调用后端 _draw_approach_rigid 直接渲染预览，
             支持滑块调参和鼠标拖拽调节进口道组中心偏移。
'''
import copy
import json
import os
import sys

import tkinter as tk
from tkinter import ttk, messagebox

from PIL import Image, ImageTk, ImageFont

# 确保能从 scripts 目录导入
sys.path.insert(0, os.path.dirname(__file__))
from add_lane_watermarks import (
    WATERMARK_CONFIGS,
    SCENARIO_TO_WATERMARK_KEY,
    _draw_approach_rigid,
)


class WatermarkTuner:
    def __init__(self, root):
        self.root = root
        self.root.title("Watermark Parameter Tuner")

        self.scenarios = list(SCENARIO_TO_WATERMARK_KEY.keys())
        self.current_scenario = tk.StringVar(value=self.scenarios[0])

        # 各场景的测试图片路径
        self.image_paths = {
            "France_Massy":      "data/test/France_Massy/2/aircraft_INT1.png",
            "Hongkong_YMT":      "data/test/Hongkong_YMT/1/aircraft_J1.png",
            "SouthKorea_Songdo": "data/test/SouthKorea_Songdo/1/aircraft_J2.png",
            "JiNan":             "data/test/JiNan/1/aircraft_intersection_1_1.png",
        }

        self.img      = None
        self.tk_img   = None
        self.W        = 800
        self.H        = 600
        self.config   = {}
        self.sliders  = {}

        # 拖拽状态
        self._drag_approach  = None
        self._drag_start     = (0, 0)
        self._drag_off_start = (0.0, 0.0)

        self._setup_ui()
        self.load_scenario()

    # ─────────────────────────────────────────────────────────────────
    # UI 搭建
    # ─────────────────────────────────────────────────────────────────
    def _setup_ui(self):
        # 顶部工具栏
        top = ttk.Frame(self.root)
        top.pack(side=tk.TOP, fill=tk.X, padx=5, pady=5)

        ttk.Label(top, text="Scenario:").pack(side=tk.LEFT)
        cb = ttk.Combobox(top, textvariable=self.current_scenario,
                          values=self.scenarios, state="readonly", width=22)
        cb.pack(side=tk.LEFT, padx=5)
        cb.bind("<<ComboboxSelected>>", lambda _: self.load_scenario())

        ttk.Button(top, text="Save Config", command=self.save_config).pack(side=tk.LEFT, padx=5)

        # 主区域
        main = ttk.Frame(self.root)
        main.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        # ── 左侧滚动控制面板 ──────────────────────────────────────────
        ctrl_outer = ttk.Frame(main, width=320)
        ctrl_outer.pack(side=tk.LEFT, fill=tk.Y)
        ctrl_outer.pack_propagate(False)

        ctrl_scroll = tk.Canvas(ctrl_outer, width=310)
        sb = ttk.Scrollbar(ctrl_outer, orient=tk.VERTICAL, command=ctrl_scroll.yview)
        ctrl_scroll.configure(yscrollcommand=sb.set)
        sb.pack(side=tk.RIGHT, fill=tk.Y)
        ctrl_scroll.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self.controls_frame = ttk.Frame(ctrl_scroll)
        ctrl_scroll.create_window((0, 0), window=self.controls_frame, anchor=tk.NW)
        self.controls_frame.bind(
            "<Configure>",
            lambda e: ctrl_scroll.configure(scrollregion=ctrl_scroll.bbox("all"))
        )

        # ── 右侧图像画布 ──────────────────────────────────────────────
        self.canvas = tk.Canvas(main, bg="gray")
        self.canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self.canvas.bind("<ButtonPress-1>",  self.on_mouse_down)
        self.canvas.bind("<B1-Motion>",      self.on_mouse_move)
        self.canvas.bind("<ButtonRelease-1>", self.on_mouse_up)

    # ─────────────────────────────────────────────────────────────────
    # 场景加载
    # ─────────────────────────────────────────────────────────────────
    def load_scenario(self):
        scenario  = self.current_scenario.get()
        img_path  = self.image_paths.get(scenario, "")

        if img_path and os.path.exists(img_path):
            self.img = Image.open(img_path).convert("RGBA")
        else:
            self.img = Image.new("RGBA", (800, 600), (70, 70, 70, 255))

        self.W, self.H = self.img.size

        config_key  = SCENARIO_TO_WATERMARK_KEY.get(scenario, "4_WAY_RHT_3LANE")
        self.config = copy.deepcopy(WATERMARK_CONFIGS[config_key])

        self.build_controls()
        self.draw_watermarks()

    # ─────────────────────────────────────────────────────────────────
    # 左侧控制面板
    # ─────────────────────────────────────────────────────────────────
    def build_controls(self):
        for w in self.controls_frame.winfo_children():
            w.destroy()
        self.sliders = {}

        # 每个进口道的四个可调参数
        param_defs = [
            ("lane_w_ratio",   0.005, 0.10, 0.001),
            ("x_offset_ratio", -0.5,  0.5,  0.001),
            ("y_offset_ratio", -0.5,  0.5,  0.001),
            ("text_rotation",  -180,  180,  1),
        ]

        for approach, ap_cfg in self.config["approaches"].items():
            ttk.Label(self.controls_frame, text=f"── {approach} ──",
                      font=("Arial", 10, "bold")).pack(pady=(8, 2), anchor=tk.W, padx=5)
            self.sliders[approach] = {}

            for param, lo, hi, step in param_defs:
                row = ttk.Frame(self.controls_frame)
                row.pack(fill=tk.X, padx=5, pady=1)

                ttk.Label(row, text=param, width=16,
                          font=("Arial", 8)).pack(side=tk.LEFT)

                val0 = ap_cfg.get(param, 0.0)
                var  = tk.DoubleVar(value=val0)

                slider = ttk.Scale(
                    row, from_=lo, to=hi, variable=var,
                    command=lambda v, a=approach, p=param: self._on_slider(a, p, v)
                )
                slider.pack(side=tk.LEFT, fill=tk.X, expand=True)

                lbl = ttk.Label(row, text=self._fmt(param, val0),
                                width=8, font=("Arial", 8))
                lbl.pack(side=tk.LEFT)

                self.sliders[approach][param] = (var, lbl)

    def _fmt(self, param, value):
        if param == "text_rotation":
            return f"{int(round(float(value)))}"
        return f"{float(value):.5f}"

    def _on_slider(self, approach, param, value):
        v = float(value)
        self.config["approaches"][approach][param] = v
        var, lbl = self.sliders[approach][param]
        lbl.config(text=self._fmt(param, v))
        self.draw_watermarks()

    # ─────────────────────────────────────────────────────────────────
    # 渲染预览（直接调用后端 _draw_approach_rigid，与实际输出完全一致）
    # ─────────────────────────────────────────────────────────────────
    def draw_watermarks(self):
        W, H = self.W, self.H
        Cx, Cy = W / 2.0, H / 2.0

        base      = self.img.copy()
        txt_layer = Image.new("RGBA", (W, H), (0, 0, 0, 0))

        scenario  = self.current_scenario.get()
        font_size = int(W * 0.02) if scenario == "SouthKorea_Songdo" else int(W * 0.03)
        try:
            font = ImageFont.truetype("data/font/Arial.ttf", font_size)
        except IOError:
            font = ImageFont.load_default()

        text_color    = (255, 255, 255, 180)
        outline_color = (0,   0,   0,   100)

        for approach, cfg in self.config["approaches"].items():
            n_lanes       = cfg["lanes"]
            side_sign     = cfg["side_sign"]
            ap_type       = cfg["type"]
            lane_w_ratio  = cfg.get("lane_w_ratio", 0.03)
            x_offset      = W * cfg.get("x_offset_ratio", 0.0)
            y_offset      = H * cfg.get("y_offset_ratio", 0.0)
            text_rotation = cfg.get("text_rotation", 0)

            group_cx = Cx + x_offset
            group_cy = Cy + y_offset

            lanes = []
            for lane_idx in range(1, n_lanes + 1):
                lane_offset = (lane_idx - (n_lanes + 1) / 2.0) * W * lane_w_ratio
                if ap_type == "vertical":
                    rx, ry = side_sign * lane_offset, 0.0
                else:
                    rx, ry = 0.0, side_sign * lane_offset
                lanes.append((str(lane_idx), rx, ry))

            _draw_approach_rigid(txt_layer, group_cx, group_cy, lanes,
                                  font, text_color, outline_color, text_rotation)

        result = Image.alpha_composite(base, txt_layer)

        # 缩放到画布尺寸
        max_w = 1200
        disp_w = min(W, max_w)
        disp_h = int(H * disp_w / W)
        result_disp = result.resize((disp_w, disp_h), Image.LANCZOS)

        self.tk_img = ImageTk.PhotoImage(result_disp)
        self.canvas.config(width=disp_w, height=disp_h)
        self.canvas.delete("all")
        self.canvas.create_image(0, 0, anchor=tk.NW, image=self.tk_img)

        # 记录显示缩放比例，供拖拽使用
        self._disp_scale = disp_w / W

    # ─────────────────────────────────────────────────────────────────
    # 鼠标拖拽：拖拽进口道组中心（x_offset_ratio / y_offset_ratio）
    # ─────────────────────────────────────────────────────────────────
    def _group_center_disp(self, approach):
        """返回进口道组中心在显示画布上的像素坐标。"""
        cfg   = self.config["approaches"][approach]
        scale = getattr(self, "_disp_scale", 1.0)
        Cx    = self.W / 2.0 * scale
        Cy    = self.H / 2.0 * scale
        gx    = Cx + self.W * cfg.get("x_offset_ratio", 0.0) * scale
        gy    = Cy + self.H * cfg.get("y_offset_ratio", 0.0) * scale
        return gx, gy

    def on_mouse_down(self, event):
        mx, my = event.x, event.y
        best, best_d = None, float("inf")
        threshold = max(40, self.W * getattr(self, "_disp_scale", 1.0) * 0.06)

        for approach in self.config["approaches"]:
            gx, gy = self._group_center_disp(approach)
            d = ((mx - gx) ** 2 + (my - gy) ** 2) ** 0.5
            if d < threshold and d < best_d:
                best_d, best = d, approach

        self._drag_approach = best
        if best:
            cfg = self.config["approaches"][best]
            self._drag_start     = (mx, my)
            self._drag_off_start = (cfg.get("x_offset_ratio", 0.0),
                                     cfg.get("y_offset_ratio", 0.0))

    def on_mouse_move(self, event):
        if not self._drag_approach:
            return
        scale = getattr(self, "_disp_scale", 1.0)
        dx    = (event.x - self._drag_start[0]) / (self.W * scale)
        dy    = (event.y - self._drag_start[1]) / (self.H * scale)

        approach = self._drag_approach
        cfg      = self.config["approaches"][approach]
        new_x    = self._drag_off_start[0] + dx
        new_y    = self._drag_off_start[1] + dy
        cfg["x_offset_ratio"] = new_x
        cfg["y_offset_ratio"] = new_y

        # 同步滑块显示
        for param, val in [("x_offset_ratio", new_x), ("y_offset_ratio", new_y)]:
            if param in self.sliders.get(approach, {}):
                var, lbl = self.sliders[approach][param]
                var.set(val)
                lbl.config(text=self._fmt(param, val))

        self.draw_watermarks()

    def on_mouse_up(self, event):
        self._drag_approach = None

    # ─────────────────────────────────────────────────────────────────
    # 保存：复用 watermark_server.update_config_in_file 写回 Python 源文件
    # ─────────────────────────────────────────────────────────────────
    def save_config(self):
        scenario   = self.current_scenario.get()
        config_key = SCENARIO_TO_WATERMARK_KEY.get(scenario)
        script_path = os.path.join(os.path.dirname(__file__), "add_lane_watermarks.py")

        try:
            from watermark_server import update_config_in_file
            ok = update_config_in_file(script_path, config_key, self.config)
            if ok:
                messagebox.showinfo("Saved",
                    f"Config [{config_key}] 已成功写回 add_lane_watermarks.py")
            else:
                messagebox.showerror("Error", f"未能在文件中定位配置键 [{config_key}]")
        except Exception as e:
            # 降级：打印到控制台
            print(f"\n[{config_key}]")
            print(json.dumps(self.config, indent=4))
            messagebox.showinfo("Fallback",
                "无法直接写文件，配置已打印到控制台，请手动粘贴。\n" + str(e))


if __name__ == "__main__":
    root = tk.Tk()
    app  = WatermarkTuner(root)
    root.mainloop()
