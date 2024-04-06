#refer:https://pab47.github.io/mujoco.html
import mujoco
import glfw
import os

#mujoco Abstract visualization
cam = mujoco.MjvCamera() #摄像机
pert = mujoco.MjvPerturb() #扰动    
opt = mujoco.MjvOption() #选项
con = mujoco.MjrContext()

# 从XML中读取模型
path = "model.xml"
dirname = os.path.dirname(__file__)
abspath = os.path.join(dirname + "/" + path)
model = mujoco.MjModel.from_xml_path(abspath)

# 数据
data = mujoco.MjData(model)

# 创建窗口
glfw.init()
window = glfw.create_window(640, 480, "MuJoCo", None, None)
glfw.make_context_current(window)
glfw.swap_interval(1)

# initialize visualization data structures
mujoco.mjv_defaultCamera(cam)
mujoco.mjv_defaultPerturb(pert)
mujoco.mjv_defaultOption(opt)

# create scene and context
scn = mujoco.MjvScene(model, maxgeom=1000)
con = mujoco.MjrContext(model, mujoco.mjtFontScale.mjFONTSCALE_100)

# 渲染器
# renderer = mujoco.Renderer(model)

# 仿真循环
while not glfw.window_should_close(window):
    starttime = data.time
    while(data.time - starttime < 1/60):
        mujoco.mj_step(model, data)

    # get framebuffer viewport
    viewport = mujoco.MjrRect(0, 0, 0, 0)
    viewport.width, viewport.height = glfw.get_framebuffer_size(window)

    #update scene and render
    mujoco.mjv_updateScene(model, data, opt, pert, cam, mujoco.mjtCatBit.mjCAT_ALL ,scn)
    mujoco.mjr_render(viewport, scn, con)

    # 交换缓冲
    glfw.swap_buffers(window)

    # 检测是否要退出
    glfw.poll_events()

# 释放资源
glfw.destroy_window(window)
glfw.terminate()
mujoco.mjv_freeScene(scn)
mujoco.mjr_freeContext(con)