import os
from OpenGL.EGL import *
from OpenGL.GL import *

def test_egl_support():
    print("🔍 正在启动 EGL 硬件加速检测 (无显示器模式)...")
    
    # 1. 获取默认显示设备 (EGL_DEFAULT_DISPLAY)
    egl_display = eglGetDisplay(EGL_DEFAULT_DISPLAY)
    if egl_display == EGL_NO_DISPLAY:
        print("❌ 错误: 无法获取 EGL Display。")
        return

    # 2. 初始化 EGL
    major, minor = EGLint(), EGLint()
    if not eglInitialize(egl_display, major, minor):
        print("❌ 错误: EGL 初始化失败。")
        return
    print(f"✅ EGL 初始化成功! 版本: {major.value}.{minor.value}")

    # 3. 配置渲染属性
    config_attributes = [
        EGL_SURFACE_TYPE, EGL_PBUFFER_BIT,
        EGL_BLUE_SIZE, 8,
        EGL_GREEN_SIZE, 8,
        EGL_RED_SIZE, 8,
        EGL_DEPTH_SIZE, 24,
        EGL_RENDERABLE_TYPE, EGL_OPENGL_BIT,
        EGL_NONE
    ]
    
    num_configs = EGLint()
    configs = (EGLConfig * 1)()
    eglChooseConfig(egl_display, config_attributes, configs, 1, num_configs)

    # 4. 创建无窗口渲染上下文 (Pbuffer)
    pbuffer_attributes = [
        EGL_WIDTH, 640,
        EGL_HEIGHT, 480,
        EGL_NONE
    ]
    egl_surface = eglCreatePbufferSurface(egl_display, configs[0], pbuffer_attributes)
    
    eglBindAPI(EGL_OPENGL_API)
    egl_context = eglCreateContext(egl_display, configs[0], EGL_NO_CONTEXT, None)
    
    # 5. 激活上下文并读取 GPU 信息
    eglMakeCurrent(egl_display, egl_surface, egl_surface, egl_context)
    
    vendor = glGetString(GL_VENDOR).decode('utf-8')
    renderer = glGetString(GL_RENDERER).decode('utf-8')
    version = glGetString(GL_VERSION).decode('utf-8')

    print("\n--- 渲染器详细信息 ---")
    print(f"🏢 供应商 (Vendor):   {vendor}")
    print(f"🚀 渲染器 (Renderer): {renderer}")
    print(f"📜 OpenGL 版本:      {version}")
    print("----------------------\n")

    if "NVIDIA" in renderer.upper():
        print("🎉 恭喜！服务器支持 EGL 硬件加速 (NVIDIA GPU)。")
    elif "LLVMPIPE" in renderer.upper() or "SOFTWARE" in renderer.upper():
        print("⚠️ 警告：当前使用的是 CPU 软件渲染 (llvmpipe)，硬件加速未开启。")
    else:
        print("ℹ️ 检测到非 NVIDIA 渲染器，请根据输出确认是否为预期 GPU。")

    # 6. 清理
    eglTerminate(egl_display)

if __name__ == "__main__":
    # 强制在 Linux 下使用 device 平台（防止寻找 X11）
    os.environ['EGL_PLATFORM'] = 'device'
    try:
        test_egl_support()
    except Exception as e:
        print(f"💥 运行出错: {e}")
        print("\n💡 提示: 如果报错找不到共享库，请检查是否安装了 NVIDIA 驱动以及 libegl1。")