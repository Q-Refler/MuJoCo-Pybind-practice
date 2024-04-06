#refer:https://pab47.github.io/mujoco.html
import mujoco
from moviepy.editor import ImageSequenceClip #pip install moviepy

#从xml中读取数据
model = mujoco.MjModel.from_xml_path("01_xml/model.xml")
data = mujoco.MjData(model)
renderer = mujoco.Renderer(model)


# 多次保存图片，形成视频
fps = 30
duration = 7
frames=[]
while data.time < duration:
    mujoco.mj_step(model, data)
    if len(frames) < data.time * fps:
        renderer.update_scene(data)
        frames.append(renderer.render())

# 从frames生成视频
clip = ImageSequenceClip(frames, fps=fps)
clip.write_videofile('./01_xml/video.mp4')
