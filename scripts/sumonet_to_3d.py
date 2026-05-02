'''
Author: Maonan Wang
Date: 2025-04-21 19:12:20
LastEditTime: 2026-05-02 22:20:26
LastEditors: Please set LastEditors
Description: 将 SUMO Net 转换为 3D 场景
'''
from tshub.utils.init_log import set_logger
from tshub.utils.get_abs_path import get_abs_path

from tshub.tshub_env3d.vis3d_sumonet_convert.sumonet_to_tshub3d import SumoNet3D

path_convert = get_abs_path(__file__)
# set_logger(path_convert('./'), terminal_log_level='INFO')

if __name__ == '__main__':
    print(f"Converting SUMO Net:  to 3D assets...")
    netxml = path_convert("../data/raw/Syn-Train/env/roadnet_4_4.net.xml")
    print(f"Converting SUMO Net: {netxml} to 3D assets...")
    sumonet_to_3d = SumoNet3D(net_file=netxml)
    sumonet_to_3d.to_glb(glb_dir=path_convert(f"../data/raw/Syn-Train/3d_assets/"))