'''
Author: yufei Ji
Date: 2026-01-12 16:49:26
LastEditTime: 2026-04-18 19:49:41
Description: sample test for lights and shadows in Panda3D. 增加了离屏相机测试与开关。
FilePath: /VLMTraffic/test_lights.py
'''
from panda3d.core import loadPrcFileData

# 高分辨率渲染窗口（必须在 ShowBase 初始化前设置）
loadPrcFileData("", "win-size 1920 1080")

# ==========================================
# 全局控制参数：选择截图方式
# ==========================================
# True : 使用新建的离屏相机 (Offscreen Camera) 截取并拉取显存 Texture
# False: 使用原始的主窗口截图 (base.win.saveScreenshot)
USE_OFFSCREEN_CAMERA = True

from panda3d.core import *
from direct.gui.OnscreenText import OnscreenText
import sys, os
import direct.directbase.DirectStart
from direct.interval.IntervalGlobal import *
from direct.actor.Actor import Actor

import simplepbr

# 修改此路径指向要测试的 map.glb 和车辆模型
MAP_GLB_PATH = "data/raw/France_Massy/3d_assets/map.glb"
VEHICLE_GLB_PATH = "TransSimHub/tshub/tshub_env3d/_assets_3d/vehicles_high_poly/background/a.glb"

def srgb_to_linear(c):
    """与 scene_loader.py 中 srgb_to_linear 完全一致"""
    def _conv(x):
        return x / 12.92 if x <= 0.04045 else ((x + 0.055) / 1.055) ** 2.4
    r, g, b, a = c
    return (_conv(r), _conv(g), _conv(b), a)

def addTitle(text):
    return OnscreenText(text=text, style=1, fg=(1, 1, 1, 1),
                        pos=(1.3, -0.95), align=TextNode.ARight, scale=.07)

class World:
    def __init__(self):
    
        if (base.win.getGsg().getSupportsBasicShaders()==0):
            self.t=addTitle("Shadow Demo: Video driver reports that shaders are not supported.")
            return
        if (base.win.getGsg().getSupportsDepthTexture()==0):
            self.t=addTitle("Shadow Demo: Video driver reports that depth textures are not supported.")
            return       
   
        base.setBackgroundColor(0.1, 0.1, 0.3, 1)
        base.camLens.setNearFar(1.0, 10000)
        base.camLens.setFov(75)
        base.disableMouse()
  
        # -------------------------------------------------------
        # 加载 map.glb
        # -------------------------------------------------------
        print(f"[INFO] 加载 map.glb: {MAP_GLB_PATH}")
        map_np_raw = loader.loadModel(MAP_GLB_PATH, noCache=True)
        map_node = render.attachNewNode("road_map")
        map_np_raw.reparentTo(map_node)

        bounds = map_np_raw.getBounds()
        map_radius = bounds.getRadius()
        map_center = bounds.getCenter()
        print(f"[INFO] map_radius={map_radius:.2f}, map_center={map_center}")

        map_node.setColor(*srgb_to_linear((1.0, 1.0, 1.0, 1.0)))
        map_node.setTextureOff(1)

        map_node.clearShader()
        for child in map_node.findAllMatches("**"):
            child.clearShader()

        map_node.setTransparency(False)
        map_node.set_depth_write(True)

        road_mat = Material("road_material")
        road_mat.setBaseColor(Vec4(1.0, 1.0, 1.0, 1.0))
        road_mat.setMetallic(0.0)
        road_mat.setRoughness(0.85)
        map_node.setMaterial(road_mat, 1)
 
        # ==========================================
        # 2. PBR 物理灯光设置
        # ==========================================
        self.light = render.attachNewNode(DirectionalLight("Directional"))
        self.light.node().setScene(render)
        self.light.node().setShadowCaster(True, 8192, 8192)
        self.light.node().setColor(Vec4(4.5, 4.2, 3.8, 1))

        lens = self.light.node().getLens()
        shadow_film = min(map_radius * 2, 4096)
        lens.setFilmSize(shadow_film, shadow_film)
        shadow_near = -max(map_radius * 0.6, 250)
        shadow_far  =  max(map_radius * 1.8, 500)
        lens.setNearFar(shadow_near, shadow_far)

        light_dir = Vec3(-1, -1, -0.5)
        light_dir.normalize()
        light_pos = map_center - light_dir * map_radius
        light_pos.z = 300
        self.light.setPos(light_pos)
        self.light.lookAt(map_center)
        render.setLight(self.light)
 
        self.alight = render.attachNewNode(AmbientLight("Ambient"))
        self.alight.node().setColor(Vec4(0.08, 0.08, 0.10, 1))
        render.setLight(self.alight)

        # ==========================================
        # 3. 激活 PBR 渲染管线
        # ==========================================
        shadow_range = shadow_far - shadow_near
        shadow_bias  = max(0.00005, min(0.001, 0.3 / shadow_range))
        print(f"[INFO] shadow_film={shadow_film:.1f}m, nearFar=({shadow_near:.0f}, {shadow_far:.0f}), range={shadow_range:.0f}m, bias={shadow_bias:.6f}")
        print("SIM: 正在加载 simplepbr 着色器管线...")
        self.pbr_pipeline = simplepbr.init(
            render_node=render,
            msaa_samples=16,
            use_hardware_skinning=True,
            use_normal_maps=True,
            use_occlusion_maps=True,
            use_emission_maps=True,
            use_330=True,
            enable_shadows=True,
            shadow_bias=shadow_bias,
            exposure=0,
        )

        print(f"[INFO] 加载车辆: {VEHICLE_GLB_PATH}")
        self.teapot = loader.loadModel(VEHICLE_GLB_PATH, noCache=True)
        self.teapot.reparentTo(render)
        self.teapot.setPos(map_center.x+40, map_center.y, map_center.z+1)
        
        # 【关键修复】：为了让载入的车辆投射物理阴影，必须清理原生Shader并允许写入深度
        self.teapot.clearShader()
        for child in self.teapot.findAllMatches("**"):
            child.clearShader()
        self.teapot.set_depth_write(True)

        self.teapotMovement = self.teapot.hprInterval(50, Point3(0, 360, 360))
        self.teapotMovement.loop()

        # 主摄像机位置
        self.cameraSelection = 0
        cam_dist = map_radius * 1.5
        base.cam.setPos(
            map_center.x + cam_dist * 0.5,
            map_center.y - cam_dist,
            map_center.z + cam_dist * 0.7,
        )
        base.cam.lookAt(map_center)
        
        # ==========================================
        # 4. 根据参数路由不同的截图任务
        # ==========================================
        if USE_OFFSCREEN_CAMERA:
            self.setup_test_offscreen_camera(map_center, map_radius)
            base.taskMgr.doMethodLater(1.5, self.save_offscreen_image, "SaveOffscreenTask")
        else:
            base.taskMgr.doMethodLater(1.5, self.take_screenshot_and_exit, "TakeScreenshotTask")

    # -------------------------------------------------------
    # 原版截图方式
    # -------------------------------------------------------
    def take_screenshot_and_exit(self, task):
        from panda3d.core import Filename
        out_path = Filename("test_map_shadow.png")
        base.win.saveScreenshot(out_path)
        print(f"\n[完成] 主窗口截图已保存: test_map_shadow.png")
        sys.exit()
        return task.done

    # -------------------------------------------------------
    # 离屏相机方式 (集成你的 offscreen_camera 逻辑)
    # -------------------------------------------------------
    def setup_test_offscreen_camera(self, map_center, map_radius):
        print("SIM: 初始化离屏相机...")
        self.offscreen_tex = Texture()
        
        win_props = WindowProperties.size(1920, 1080)
        fb_props = FrameBufferProperties()
        fb_props.setRgbColor(True)
        fb_props.setRgbaBits(8, 8, 8, 8)
        fb_props.setDepthBits(24)
        fb_props.setAuxRgba(1)
        fb_props.setStencilBits(8)
        fb_props.setSrgbColor(True) # 强制sRGB，防止阴影消失

        buffer = base.win.engine.makeOutput(
            base.pipe, "TestSensor-buffer", 1, fb_props, win_props, # sort=1确保在PBR后渲染
            GraphicsPipe.BFRefuseWindow, base.win.getGsg(), base.win
        )
        buffer.setClearColor((0, 0, 0, 1))

        region = buffer.getDisplayRegion(0)
        region.window.addRenderTexture(
            self.offscreen_tex, GraphicsOutput.RTM_copy_ram, GraphicsOutput.RTP_color
        )

        lens = PerspectiveLens()
        lens.setFilmSize(1920, 1080)
        lens.setFov(75) # 匹配主相机的 FOV

        self.offscreen_cam_np = base.makeCamera(buffer, camName="TestSensor", scene=render, lens=lens)
        # 核心：继承主相机的 PBR 渲染状态
        self.offscreen_cam_np.node().setInitialState(base.cam.node().getInitialState())

        # 将离屏相机摆放到与主摄像机完全相同的位置，方便对比效果
        cam_dist = map_radius * 1.5
        self.offscreen_cam_np.setPos(
            map_center.x + cam_dist * 0.5,
            map_center.y - cam_dist,
            map_center.z + cam_dist * 0.7,
        )
        self.offscreen_cam_np.lookAt(map_center)

    def save_offscreen_image(self, task):
        out_path = "test_offscreen_shadow.png"
        if self.offscreen_tex.hasRamImage():
            self.offscreen_tex.write(out_path)
            print(f"\n[完成] 离屏相机截图已保存: {out_path}")
        else:
            print("[错误] 离屏纹理未捕获到图像数据")
        sys.exit()
        return task.done

World()
run()