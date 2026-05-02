'''
Author: yufei Ji
Date: 2026-01-12 16:48:24
LastEditTime: 2026-05-02 22:41:20
Description: 在线 BEV 渲染脚本。
             支持一次性循环渲染多种交通事件类型（共 6 种：normal / emergency / bus /
             accident / debris / pedestrian），每种事件类型独立运行一次仿真。

             核心机制：
               tshub_env3d.py 的事件加载逻辑始终从 .sumocfg 的 <route-files> 节点
               读取路由文件路径，因此对于非 normal 场景，本脚本在渲染前动态生成
               {NETFILE}_{event_type}.sumocfg，将 route-files 替换为对应的事件路由
               文件路径，再以新 sumocfg 启动仿真，确保事件对象（crash_vehicle /
               debris / pedestrian 等）能被 EmergencyManager3D 正确渲染。
FilePath: /VLMTraffic/src/bev_generation/online_bev_render.py
'''

import copy
import os
import time
import xml.etree.ElementTree as ET

import cv2
from tshub.utils.get_abs_path import get_abs_path
from tshub.utils.init_log import set_logger

from configs.scenairo_config import SCENARIO_CONFIGS
from configs.env_config import TSHUB_ENV_CONFIG
from utils.make_tsc_env import make_env
from utils.tools import save_to_json, create_folder

# ──────────────────────────────────────────────────────────────────────────────
# ① 全局场景配置
# ──────────────────────────────────────────────────────────────────────────────
scenario_key = "Syn-Train"  # 可选: Hongkong_YMT, France_Massy, SouthKorea_Songdo,
                                #       Hangzhou, NewYork, JiNan

# ──────────────────────────────────────────────────────────────────────────────
# ② 事件类型列表（按需修改，顺序即渲染顺序）
#    6 种可选类型：
#      'normal'     - 正常车流，使用 sumocfg 默认路由文件
#      'emergency'  - 紧急车辆（ambulance / police / fire_engine）
#      'bus'        - 公交车 / 校车
#      'accident'   - 交通事故（碰撞残骸 + 倒地行人）
#      'debris'     - 路面碎片（路障 / 树枝）
#      'pedestrian' - 行人过街
# ──────────────────────────────────────────────────────────────────────────────
RENDER_EVENT_TYPES = [
    'normal',
    # 'emergency',
    # 'bus',
    # 'accident',
    # 'debris',
    # 'pedestrian',
]

# 每种事件类型对应的路由文件后缀（与 batch_generate_all_scenes.sh 生成的文件名保持一致）
_EVENT_ROUTE_SUFFIX = {
    'normal':     None,          # 不覆盖，使用 sumocfg 默认值
    'emergency':  '_emergy',     # {BASE}_emergy.rou.xml
    'bus':        '_bus',        # {BASE}_bus.rou.xml
    'accident':   '_accident_1',   # {BASE}_accident.rou.xml
    'debris':     '_debris',     # {BASE}_debris.rou.xml
    'pedestrian': '_pedestrian', # {BASE}_pedestrian.rou.xml
}

# ──────────────────────────────────────────────────────────────────────────────
path_convert = get_abs_path(__file__)

config = SCENARIO_CONFIGS.get(scenario_key)
SCENARIO_NAME               = config["SCENARIO_NAME"]
NETFILE                     = config["NETFILE"]
JUNCTION_NAME               = config["JUNCTION_NAME"]
PHASE_NUMBER                = config["PHASE_NUMBER"]
SENSOR_INDEX_2_PHASE_INDEX  = config["SENSOR_INDEX_2_PHASE_INDEX"]
RENDERER_CFG                = config.get("RENDERER_CFG")
SENSOR_CFG                  = config.get("SENSOR_CFG")


def _get_event_route_file(sumocfg_path: str, event_type: str) -> str | None:
    """
    根据事件类型返回对应的路由文件绝对路径。
    通过解析 sumocfg 获得基础路由文件路径，再拼接事件后缀。

    Returns:
        路由文件绝对路径（存在时），或 None（normal / 文件不存在时）。
    """
    suffix = _EVENT_ROUTE_SUFFIX.get(event_type)
    if suffix is None:  # normal：不需要覆盖
        return None

    try:
        tree = ET.parse(sumocfg_path)
        root = tree.getroot()
        node = root.find('.//route-files')
        if node is None:
            print(f"[EventRoute] sumocfg 中未找到 route-files 节点: {sumocfg_path}")
            return None

        # 取第一个路由文件（逗号分隔时取首项）
        rel_route = node.get('value', '').split(',')[0].strip()
        sumocfg_dir = os.path.dirname(os.path.abspath(sumocfg_path))
        base_route_abs = os.path.normpath(os.path.join(sumocfg_dir, rel_route))

        # 构造事件路由文件路径：替换文件名中的 .rou.xml 后缀
        base_name   = os.path.basename(base_route_abs)   # e.g., YMT.rou.xml
        stem        = base_name.replace('.rou.xml', '')   # e.g., YMT
        env_dir     = os.path.dirname(base_route_abs)
        event_route = os.path.join(env_dir, f"{stem}{suffix}.rou.xml")

        if not os.path.exists(event_route):
            print(f"[EventRoute] 事件路由文件不存在（已跳过）: {event_route}")
            return None

        return event_route

    except Exception as e:
        print(f"[EventRoute] 解析 sumocfg 失败: {e}")
        return None


def _make_event_sumocfg(sumocfg_path: str, event_route_abs: str, event_type: str) -> str:
    """
    基于原始 sumocfg 生成事件专用 sumocfg，将 <route-files value="..."> 替换为事件路由
    文件的相对路径，写入同目录下的 {stem}_{event_type}.sumocfg。

    tshub_env3d.py 始终从 .sumocfg 的 route-files 节点读取路由文件来加载事件对象，
    因此必须在此处生成正确的 sumocfg，而不能仅靠 route_file 参数覆盖。

    Args:
        sumocfg_path    : 原始 .sumocfg 的绝对路径（如 .../YMT.sumocfg）
        event_route_abs : 事件路由文件的绝对路径（如 .../env/YMT_accident.rou.xml）
        event_type      : 事件类型字符串，用于命名新文件（如 'accident'）

    Returns:
        新生成的事件 sumocfg 绝对路径（如 .../YMT_accident.sumocfg）
    """
    tree = ET.parse(sumocfg_path)
    root = tree.getroot()

    node = root.find('.//route-files')
    if node is None:
        raise RuntimeError(f"[EventSumocfg] sumocfg 中无 route-files 节点: {sumocfg_path}")

    # 计算事件路由文件相对于 sumocfg 所在目录的相对路径
    sumocfg_dir = os.path.dirname(os.path.abspath(sumocfg_path))
    event_route_rel = os.path.relpath(event_route_abs, sumocfg_dir)
    # 统一使用正斜杠，保证 SUMO 跨平台解析正常
    event_route_rel = event_route_rel.replace(os.sep, '/')
    node.set('value', event_route_rel)

    # 新文件命名：{原始stem}_{event_type}.sumocfg，写入同目录
    sumocfg_stem     = os.path.splitext(os.path.basename(sumocfg_path))[0]
    new_sumocfg_name = f"{sumocfg_stem}_{event_type}.sumocfg"
    new_sumocfg_path = os.path.join(sumocfg_dir, new_sumocfg_name)

    if hasattr(ET, 'indent'):
        ET.indent(root, space='    ', level=0)
    tree.write(new_sumocfg_path, encoding='utf-8', xml_declaration=True)

    print(f"[EventSumocfg] 已生成: {new_sumocfg_name}  (route-files → {event_route_rel})")
    return new_sumocfg_path


def _convert_rgb_to_bgr(image):
    return image[:, :, ::-1]


def _run_event_render(event_type: str, sumo_cfg: str, scenario_glb_dir: str) -> None:
    """
    针对单一事件类型运行完整的仿真渲染流程，将图像保存到
    data/test/{SCENARIO_NAME}/{event_type}/{time_step}/ 目录。
    """
    print(f"\n{'='*60}")
    print(f"  渲染事件类型: [{event_type.upper()}]")
    print(f"{'='*60}")

    # ── 确定路由文件 & 事件专用 sumocfg ────────────────────────────────────────
    route_file = _get_event_route_file(sumo_cfg, event_type)

    if route_file:
        # 非 normal 场景：生成事件专用 sumocfg，使 tshub_env3d 能从 sumocfg 读到正确
        # 的路由文件路径并加载事件对象（EmergencyManager3D 读取 param 标签）
        active_sumo_cfg = _make_event_sumocfg(sumo_cfg, route_file, event_type)
        print(f"[EventRoute] 事件路由: {os.path.basename(route_file)}")
        print(f"[EventRoute] 事件 sumocfg: {os.path.basename(active_sumo_cfg)}")
    else:
        # normal 场景：直接使用原始 sumocfg
        active_sumo_cfg = sumo_cfg
        print(f"[EventRoute] 使用原始 sumocfg（正常车流）")

    # ── 构建环境参数 ────────────────────────────────────────────────────────────
    output_base = path_convert(f"../../data/test/{SCENARIO_NAME}/{event_type}/")
    trip_info   = os.path.join(output_base, "tripinfo.out.xml")
    create_folder(output_base)

    tls_add = [
        path_convert(f'../../data/raw/{SCENARIO_NAME}/add/e2.add.xml'),
        path_convert(f'../../data/raw/{SCENARIO_NAME}/add/tls_programs.add.xml'),
    ]

    # event_env_cfg 不再需要注入 route_file：路由已由 active_sumo_cfg 指定
    event_env_cfg = copy.deepcopy(TSHUB_ENV_CONFIG)

    params = {
        'tls_id':           JUNCTION_NAME,
        'number_phases':    PHASE_NUMBER,
        'sumo_cfg':         active_sumo_cfg,   # ← 使用事件专用 sumocfg
        'scenario_glb_dir': scenario_glb_dir,
        'trip_info':        trip_info,
        'tls_state_add':    tls_add,
        'renderer_cfg':     RENDERER_CFG,
        'sensor_cfg':       SENSOR_CFG,
        'tshub_env_cfg':    event_env_cfg,
    }

    print("  初始化仿真环境...")
    env = make_env(**params)()

    # ── GPU 渲染器信息（仅首次打印）────────────────────────────────────────────
    try:
        from direct.showbase.ShowBaseGlobal import base
        if hasattr(base, 'win') and base.win:
            gsg = base.win.getGsg()
            if gsg:
                print(f"  GPU: {gsg.getDriverVendor()} - {gsg.getDriverRenderer()}")
    except Exception:
        pass

    # ── 仿真主循环 ──────────────────────────────────────────────────────────────
    obs, _info = env.reset()
    time_step  = 0

    prof_env_step    = 0.0
    prof_img_save    = 0.0
    prof_loop_total  = 0.0

    while True:
        loop_start = time.perf_counter()

        # 每一步切换一个相位，按照相位的总数循环
        if isinstance(JUNCTION_NAME, list):
            env_action = {jid: time_step % PHASE_NUMBER for jid in JUNCTION_NAME}
        else:
            env_action = time_step % PHASE_NUMBER

        t0 = time.perf_counter()
        obs, rewards, truncated, dones, infos, render_json = env.step(env_action)
        prof_env_step += time.perf_counter() - t0

        time_step += 1
        step_folder = os.path.join(output_base, str(time_step))
        create_folder(step_folder)

        # 保存车辆 JSON
        save_to_json(render_json, os.path.join(step_folder, 'data.json'))

        # 保存传感器图像
        t2 = time.perf_counter()
        sensor_data_imgs = infos.get('3d_data', {}).get('image')
        if sensor_data_imgs:
            for jid, jid_imgs in sensor_data_imgs.items():
                front_img = jid_imgs.get('junction_front_all')
                if front_img is not None:
                    cv2.imwrite(
                        os.path.join(step_folder, f"{jid}.png"),
                        _convert_rgb_to_bgr(front_img)
                    )
        prof_img_save   += time.perf_counter() - t2
        prof_loop_total += time.perf_counter() - loop_start

        if dones or truncated:
            break
        if time_step >= 20:
            break

    # ── 性能摘要 ────────────────────────────────────────────────────────────────
    print(f"\n  [{event_type}] 完成 {time_step} 步")
    if time_step > 0:
        print(f"  平均每步: env={prof_env_step/time_step:.3f}s  "
              f"img={prof_img_save/time_step:.3f}s  "
              f"total={prof_loop_total/time_step:.3f}s")
    print(f"  图像输出目录: {output_base}")

    env.close()


# ──────────────────────────────────────────────────────────────────────────────
# 主入口
# ──────────────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    set_logger(path_convert(f'../../log/{scenario_key}/'))

    sumo_cfg         = path_convert(f"../../data/raw/{SCENARIO_NAME}/{NETFILE}.sumocfg")
    scenario_glb_dir = path_convert(f"../../data/raw/{SCENARIO_NAME}/3d_assets/")

    print(f"\n场景: {SCENARIO_NAME}")
    print(f"待渲染事件类型: {RENDER_EVENT_TYPES}")
    print(f"共 {len(RENDER_EVENT_TYPES)} 种，依次执行...\n")

    for idx, event_type in enumerate(RENDER_EVENT_TYPES, start=1):
        print(f"\n[{idx}/{len(RENDER_EVENT_TYPES)}] 开始渲染: {event_type}")
        _run_event_render(event_type, sumo_cfg, scenario_glb_dir)

    print(f"\n{'='*60}")
    print(f"  全部事件类型渲染完毕！")
    print(f"  输出根目录: data/test/{SCENARIO_NAME}/")
    print(f"  子目录结构: {{event_type}}/{{time_step}}/{{element_id}}.png")
    print(f"{'='*60}\n")
