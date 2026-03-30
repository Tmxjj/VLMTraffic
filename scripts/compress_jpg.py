import os
import zipfile
import shutil
from pathlib import Path
from tqdm import tqdm  # 导入进度条库

# 将文件储存中的非xml、txt文件压缩为zip，减少文件数量
def compress_and_cleanup(base_path="autodl-fs/data/eval"):
    base_dir = Path(base_path)
    
    if not base_dir.exists():
        print(f"❌ 错误: 找不到基础路径 {base_dir}")
        return

    # 遍历 Scene (e.g., Hangzhou)
    for scene_dir in base_dir.iterdir():
        if not scene_dir.is_dir(): continue

        # 遍历 Route (e.g., anon_4_4_hangzhou_real.rou)
        for route_dir in scene_dir.iterdir():
            if not route_dir.is_dir(): continue

            # 遍历 Model (e.g., fixed_time)
            for model_dir in route_dir.iterdir():
                if not model_dir.is_dir(): continue

                print(f"\n⏳ 正在处理: {scene_dir.name} / {route_dir.name} / {model_dir.name}")
                
                # 定义压缩包的名称，放在当前 Model 目录下
                zip_filename = model_dir / f"{model_dir.name}_images_archive.zip"
                
                files_to_compress = []
                
                # 遍历当前 Model 目录下的所有文件（包括子文件夹 step_0 等）
                for root, _, files in os.walk(model_dir):
                    for file in files:
                        # 排除 xml, txt 和 已经打包好的 zip 文件
                        if not file.endswith(('.xml', '.txt', '.zip')):
                            files_to_compress.append(Path(root) / file)
                
                if not files_to_compress:
                    print(f"   ↳ 跳过: 没有找到需要压缩的图片/日志文件。")
                    continue

                # 1. 将文件写入 Zip 压缩包（加上进度条）
                with zipfile.ZipFile(zip_filename, 'w', zipfile.ZIP_DEFLATED) as zipf:
                    # 使用 tqdm 包装 files_to_compress
                    for file_path in tqdm(files_to_compress, desc="   ↳ 📦 压缩中", unit="文件", leave=False):
                        # 计算文件在 zip 中的相对路径，保持原有的目录结构 (如 step_0/**.png)
                        arcname = file_path.relative_to(model_dir)
                        zipf.write(file_path, arcname)
                
                # 2. 压缩完成后，安全删除原文件（加上进度条）
                # 使用 tqdm 包装，leave=False 表示执行完后进度条会自动消失，保持终端整洁
                for file_path in tqdm(files_to_compress, desc="   ↳ 🗑️ 删除中", unit="文件", leave=False):
                    try:
                        file_path.unlink()
                    except OSError as e:
                        print(f"\n   ↳ ⚠️ 删除文件失败 {file_path}: {e}")

                # 3. 可选：清理变成了空的子文件夹
                # topdown=False 确保从最深层的文件夹开始删
                for root, dirs, _ in os.walk(model_dir, topdown=False):
                    for d in dirs:
                        dir_path = Path(root) / d
                        if not os.listdir(dir_path):  # 如果文件夹为空
                            dir_path.rmdir()
                
                print(f"   ↳ ✅ 完成: 成功压缩并清理了 {len(files_to_compress)} 个文件。")

if __name__ == "__main__":
    # 如果你的路径不同，可以在这里修改
    compress_and_cleanup("/root/autodl-fs/data/eval")