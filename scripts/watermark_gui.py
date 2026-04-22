import copy
import json
import os
import sys
import tkinter as tk
from tkinter import ttk, messagebox

from PIL import Image, ImageTk, ImageFont

sys.path.insert(0, os.path.dirname(__file__))
from add_lane_watermarks import (
    WATERMARK_CONFIGS,
    SCENARIO_TO_WATERMARK_KEY,
    _draw_approach_rigid,
    extract_info_from_path
)

class WatermarkTuner:
    def __init__(self, root):
        self.root = root
        self.root.title("Watermark Parameter Tuner")

        self.scenarios = list(SCENARIO_TO_WATERMARK_KEY.keys())
        self.current_scenario = tk.StringVar(value=self.scenarios[0])
        self.current_approach = tk.StringVar(value="North")

        self.img = None
        self.tk_img = None
        self.W = 800
        self.H = 600
        self.config = {}
        self.sliders = {}

        self._drag_target = None
        self._drag_start = (0, 0)
        self._drag_off_start = (0.0, 0.0)

        self._setup_ui()
        self.load_scenario()

    def _setup_ui(self):
        top = ttk.Frame(self.root)
        top.pack(side=tk.TOP, fill=tk.X, padx=5, pady=5)

        ttk.Label(top, text="Scenario:").pack(side=tk.LEFT)
        cb = ttk.Combobox(top, textvariable=self.current_scenario,
                          values=self.scenarios, state="readonly", width=20)
        cb.pack(side=tk.LEFT, padx=5)
        cb.bind("<<ComboboxSelected>>", lambda _: self.load_scenario())

        ttk.Label(top, text="Approach:").pack(side=tk.LEFT)
        self.approach_cb = ttk.Combobox(top, textvariable=self.current_approach,
                                        state="readonly", width=20)
        self.approach_cb.pack(side=tk.LEFT, padx=5)
        self.approach_cb.bind("<<ComboboxSelected>>", lambda _: self.change_approach())

        ttk.Button(top, text="Save Config", command=self.save_config).pack(side=tk.LEFT, padx=5)

        main = ttk.Frame(self.root)
        main.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

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

        self.canvas = tk.Canvas(main, bg="gray")
        self.canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self.canvas.bind("<ButtonPress-1>",  self.on_mouse_down)
        self.canvas.bind("<B1-Motion>",      self.on_mouse_move)
        self.canvas.bind("<ButtonRelease-1>", self.on_mouse_up)
        
        # Load custom image
        ttk.Button(top, text="Load Custom Image", command=self.load_image).pack(side=tk.LEFT, padx=5)
        self.image_path = None

    def load_scenario(self):
        scenario = self.current_scenario.get()
        config_key = SCENARIO_TO_WATERMARK_KEY.get(scenario, "4_WAY_RHT_3LANE")
        if config_key not in WATERMARK_CONFIGS:
            return
            
        self.config = copy.deepcopy(WATERMARK_CONFIGS[config_key])
        approaches = list(self.config.keys())
        self.approach_cb["values"] = approaches
        
        if self.current_approach.get() not in approaches:
            self.current_approach.set(approaches[0])
            
        self.change_approach()
        
    def load_image(self):
        from tkinter import filedialog
        path = filedialog.askopenfilename()
        if path:
            self.image_path = path
            # auto detect
            scenario, approach_key, is_upstream, dir_long = extract_info_from_path(self.image_path)
            self.current_scenario.set(scenario)
            self.load_scenario()
            if approach_key in self.config:
                self.current_approach.set(approach_key)
            self.change_approach()

    def change_approach(self):
        if self.image_path and os.path.exists(self.image_path):
            self.img = Image.open(self.image_path).convert("RGBA")
        else:
            self.img = Image.new("RGBA", (800, 600), (70, 70, 70, 255))
            
        self.W, self.H = self.img.size
        self.build_controls()
        self.draw_watermarks()

    def build_controls(self):
        for w in self.controls_frame.winfo_children():
            w.destroy()
        self.sliders = {}

        approach = self.current_approach.get()
        if approach not in self.config:
            return
            
        ap_cfg = self.config[approach]
        
        ttk.Label(self.controls_frame, text=f"── {approach} ──",
                  font=("Arial", 10, "bold")).pack(pady=(8, 2), anchor=tk.W, padx=5)
        
        param_defs = [
            ("lane_w_ratio",   0.005, 0.20, 0.001),
            ("x_offset_ratio", -0.5,  0.5,  0.001),
            ("y_offset_ratio", -0.5,  0.5,  0.001),
            ("text_rotation",  -180,  180,  1),
            ("text_pos_x",     0.0,   1.0,  0.001),
            ("text_pos_y",     0.0,   1.0,  0.001),
        ]

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
                command=lambda v, p=param: self._on_slider(p, v)
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

    def _on_slider(self, param, value):
        approach = self.current_approach.get()
        if not approach: return
        v = float(value)
        self.config[approach][param] = v
        var, lbl = self.sliders[approach][param]
        lbl.config(text=self._fmt(param, v))
        self.draw_watermarks()

    def draw_watermarks(self):
        W, H = self.W, self.H
        Cx, Cy = W / 2.0, H / 2.0

        base = self.img.copy()
        txt_layer = Image.new("RGBA", (W, H), (0, 0, 0, 0))

        scenario = self.current_scenario.get()
        font_size = int(W * 0.02) if scenario == "SouthKorea_Songdo" else int(W * 0.03)
        try:
            font = ImageFont.truetype("data/font/Arial.ttf", font_size)
        except IOError:
            font = ImageFont.load_default()

        text_color = (255, 255, 255, 180)
        outline_color = (0, 0, 0, 100)

        approach = self.current_approach.get()
        cfg = self.config.get(approach)
        if not cfg:
            return

        n_lanes = cfg["lanes"]
        lane_w_ratio = cfg.get("lane_w_ratio", 0.05)
        x_offset = W * cfg.get("x_offset_ratio", 0.0)
        y_offset = H * cfg.get("y_offset_ratio", 0.0)
        text_rotation = cfg.get("text_rotation", 0)
        text_pos_x = cfg.get("text_pos_x", 0.5)
        text_pos_y = cfg.get("text_pos_y", 0.05)

        group_cx = Cx + x_offset
        group_cy = Cy + y_offset

        lanes = []
        for lane_idx in range(1, n_lanes + 1):
            rx = (lane_idx - (n_lanes + 1) / 2.0) * W * lane_w_ratio
            ry = 0.0
            lanes.append((str(lane_idx), rx, ry))

        _draw_approach_rigid(txt_layer, group_cx, group_cy, lanes,
                              font, text_color, outline_color, text_rotation)

        d = ImageDraw.Draw(txt_layer)
        is_upstream = approach.startswith("upstream_")
        dir_long = approach.replace("upstream_", "")
        info_text = f"Approach: {dir_long}, Type: {'Upstream' if is_upstream else 'Normal'}"
        
        tx = int(W * text_pos_x)
        ty = int(H * text_pos_y)
        d.text(
            (tx, ty),
            info_text,
            font=font,
            fill=text_color,
            anchor="mm",
            stroke_width=2,
            stroke_fill=outline_color,
        )

        result = Image.alpha_composite(base, txt_layer)

        max_w = 1200
        disp_w = min(W, max_w)
        disp_h = int(H * disp_w / W)
        result_disp = result.resize((disp_w, disp_h), Image.LANCZOS)

        self.tk_img = ImageTk.PhotoImage(result_disp)
        self.canvas.config(width=disp_w, height=disp_h)
        self.canvas.delete("all")
        self.canvas.create_image(0, 0, anchor=tk.NW, image=self.tk_img)

        self._disp_scale = disp_w / W

    def _group_center_disp(self):
        approach = self.current_approach.get()
        if not approach: return 0, 0
        cfg = self.config[approach]
        scale = getattr(self, "_disp_scale", 1.0)
        Cx = self.W / 2.0 * scale
        Cy = self.H / 2.0 * scale
        gx = Cx + self.W * cfg.get("x_offset_ratio", 0.0) * scale
        gy = Cy + self.H * cfg.get("y_offset_ratio", 0.0) * scale
        return gx, gy
        
    def _text_center_disp(self):
        approach = self.current_approach.get()
        if not approach: return 0, 0
        cfg = self.config[approach]
        scale = getattr(self, "_disp_scale", 1.0)
        tx = self.W * cfg.get("text_pos_x", 0.5) * scale
        ty = self.H * cfg.get("text_pos_y", 0.05) * scale
        return tx, ty

    def on_mouse_down(self, event):
        mx, my = event.x, event.y
        approach = self.current_approach.get()
        if not approach: return
        
        gx, gy = self._group_center_disp()
        tx, ty = self._text_center_disp()
        
        d_group = ((mx - gx) ** 2 + (my - gy) ** 2) ** 0.5
        d_text = ((mx - tx) ** 2 + (my - ty) ** 2) ** 0.5
        
        threshold = max(40, self.W * getattr(self, "_disp_scale", 1.0) * 0.06)

        if d_group < threshold and d_group <= d_text:
            self._drag_target = "group"
            cfg = self.config[approach]
            self._drag_start = (mx, my)
            self._drag_off_start = (cfg.get("x_offset_ratio", 0.0), cfg.get("y_offset_ratio", 0.0))
        elif d_text < threshold:
            self._drag_target = "text"
            cfg = self.config[approach]
            self._drag_start = (mx, my)
            self._drag_off_start = (cfg.get("text_pos_x", 0.5), cfg.get("text_pos_y", 0.05))

    def on_mouse_move(self, event):
        if not self._drag_target:
            return
            
        approach = self.current_approach.get()
        scale = getattr(self, "_disp_scale", 1.0)
        dx = (event.x - self._drag_start[0]) / (self.W * scale)
        dy = (event.y - self._drag_start[1]) / (self.H * scale)

        cfg = self.config[approach]
        new_x = self._drag_off_start[0] + dx
        new_y = self._drag_off_start[1] + dy

        if self._drag_target == "group":
            cfg["x_offset_ratio"] = new_x
            cfg["y_offset_ratio"] = new_y
            sync_params = ["x_offset_ratio", "y_offset_ratio"]
            sync_vals = [new_x, new_y]
        else:
            cfg["text_pos_x"] = new_x
            cfg["text_pos_y"] = new_y
            sync_params = ["text_pos_x", "text_pos_y"]
            sync_vals = [new_x, new_y]

        for param, val in zip(sync_params, sync_vals):
            if param in self.sliders.get(approach, {}):
                var, lbl = self.sliders[approach][param]
                var.set(val)
                lbl.config(text=self._fmt(param, val))

        self.draw_watermarks()

    def on_mouse_up(self, event):
        self._drag_target = None

    def save_config(self):
        scenario = self.current_scenario.get()
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
            print(f"
[{config_key}]")
            print(json.dumps(self.config, indent=4))
            messagebox.showinfo("Fallback",
                "无法直接写文件，配置已打印到控制台，请手动粘贴。
" + str(e))


if __name__ == "__main__":
    root = tk.Tk()
    app = WatermarkTuner(root)
    root.mainloop()
