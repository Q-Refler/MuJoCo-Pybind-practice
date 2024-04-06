#refer:https://pab47.github.io/mujoco.html
import mujoco
import glfw
import os
# For callback functions
button_left = False
button_middle = False
button_right = False
lastx = 0
lasty = 0

# mouse click callback
def mouse_button(window, button, act, mods):
    global button_left
    global button_middle
    global button_right
    # update button state
    button_left = (glfw.get_mouse_button(
        window, glfw.MOUSE_BUTTON_LEFT) == glfw.PRESS)
    button_middle = (glfw.get_mouse_button(
        window, glfw.MOUSE_BUTTON_MIDDLE) == glfw.PRESS)
    button_right = (glfw.get_mouse_button(
        window, glfw.MOUSE_BUTTON_RIGHT) == glfw.PRESS)

    # update mouse position
    glfw.get_cursor_pos(window)

def mouse_move_data(window, xpos, ypos, data):
    # compute mouse displacement, save
    global lastx
    global lasty
    dx = xpos - lastx
    dy = ypos - lasty
    lastx = xpos
    lasty = ypos

    # no buttons down: nothing to do
    if (not button_left) and (not button_middle) and (not button_right):
        return

    # get current window size
    width, height = glfw.get_window_size(window)

    # move mocap in z plane if button_left is down
    if button_left:
        data.mocap_pos[0] += [dx / width, -dy / height, 0]
        print("left button")
        print(data.mocap_pos)
    # move mocap in x plane if button_right is down
    if button_right:
        data.mocap_pos[0] += [0, dx / width, dy / height]

def keyboard_data(window, key, scancode, act, mods, data):
    if act == glfw.PRESS and key == glfw.KEY_W:
        data.mocap_pos[0] += [0, 0, 1]
        print("press w")
        print(data.mocap_pos)
    if act == glfw.PRESS and key == glfw.KEY_S:
        data.mocap_pos[0] += [0, 0, -1]
        print("press s")
        print(data.mocap_pos)
def main(path):
    #mujoco Abstract visualization
    cam = mujoco.MjvCamera() #摄像机
    # pert = mujoco.MjvPerturb() #扰动    
    opt = mujoco.MjvOption() #选项
    con = mujoco.MjrContext()

    # 从XML中读取模型
    dirname = os.path.dirname(__file__)
    abspath = os.path.join(dirname + "/" + path)
    model = mujoco.MjModel.from_xml_path(abspath)

    # 数据
    data = mujoco.MjData(model)
    print(data.mocap_pos)

    # 创建窗口
    glfw.init()
    window = glfw.create_window(640, 480, "MuJoCo", None, None)
    glfw.make_context_current(window)
    glfw.swap_interval(1)
    
    # initialize visualization data structures
    mujoco.mjv_defaultCamera(cam)
    # mujoco.mjv_defaultPerturb(pert)
    mujoco.mjv_defaultOption(opt)

    # create scene and context
    scn = mujoco.MjvScene(model, maxgeom=1000)
    con = mujoco.MjrContext(model, mujoco.mjtFontScale.mjFONTSCALE_100)
    def keyboard(window, key, scancode, act, mods):
        keyboard_data(window, key, scancode, act, mods, data)
    def mouse_move(window, xpos, ypos):
        mouse_move_data(window, xpos, ypos, data)
    # set mouse button callback for glfw
    glfw.set_key_callback(window, keyboard)
    glfw.set_cursor_pos_callback(window, mouse_move)
    glfw.set_mouse_button_callback(window, mouse_button)
    # glfw.set_scroll_callback(window, scroll)
    
    # 仿真循环
    while not glfw.window_should_close(window):
        starttime = data.time
        while(data.time - starttime < 1/60):
            cam.lookat[2] = data.mocap_pos[0,2]
            mujoco.mj_step(model, data)
        
        # get framebuffer viewport
        viewport = mujoco.MjrRect(0, 0, 0, 0)
        viewport.width, viewport.height = glfw.get_framebuffer_size(window)

        #update scene and render
        # mujoco.mjv_updateScene(model, data, opt, pert, cam, mujoco.mjtCatBit.mjCAT_ALL ,scn)
        mujoco.mjv_updateScene(model, data, opt, None, cam, mujoco.mjtCatBit.mjCAT_ALL ,scn)
        mujoco.mjr_render(viewport, scn, con)

        # 交换缓冲
        glfw.swap_buffers(window)

        # 检测是否要退出
        glfw.poll_events()

    # 释放资源
    glfw.destroy_window(window)
    glfw.terminate()

if __name__ == "__main__":
    relative_model_path = "ball.xml"
    main(relative_model_path)