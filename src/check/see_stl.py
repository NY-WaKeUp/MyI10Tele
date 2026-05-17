#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import trimesh

mesh_dir = "/Users/ningyu/code_before_paper/mujoco-learning/model/aubo_i10_inspire/meshes"
mesh = trimesh.load_mesh(os.path.join(mesh_dir, "gripper_base_link.STL"))
# 带坐标系
# Create a scene with coordinate axes and add the mesh
scene = trimesh.Scene()
scene.add_geometry(mesh)
# Add coordinate axes to the scene for visualization
axes = trimesh.creation.axis(origin_size=0.01, axis_length=0.1)
scene.add_geometry(axes)
scene.show()


import os
import trimesh
import plotly.graph_objects as go
import numpy as np

# 载入整个gripper相关mesh
mesh_dir = "/Users/ningyu/code_before_paper/mujoco-learning/model/aubo_i10_inspire/meshes"
# 假设gripper由多个STL文件组成，拼接成一个整体
gripper_mesh_files = [
    "gripper_base_link.STL",
    # "gripper_Link1.STL",
    # "gripper_Link2.STL",
    # "gripper_Link3.STL",
    # "gripper_Link4.STL",
    # 如有其它part，继续添加
]

mesh_list = []
for fname in gripper_mesh_files:
    path = os.path.join(mesh_dir, fname)
    if os.path.exists(path):
        part = trimesh.load_mesh(path)
        # 若某些mesh不是trimesh.Trimesh对象而是Scene，需统一为Trimesh列表
        if isinstance(part, trimesh.Scene):
            part = trimesh.util.concatenate(part.dump())
        mesh_list.append(part)
    else:
        print(f"Warning: {path} not found")

# 拼成一个整体mesh
if len(mesh_list) == 0:
    raise RuntimeError("No gripper mesh loaded!")
elif len(mesh_list) == 1:
    mesh = mesh_list[0]
else:
    mesh = trimesh.util.concatenate(mesh_list)

# 坐标轴长度
axis_length = 0.1

# 获取mesh的三角面和点
vertices = mesh.vertices
faces = mesh.faces

# Plotly mesh3d
mesh_trace = go.Mesh3d(
    x=vertices[:, 0],
    y=vertices[:, 1],
    z=vertices[:, 2],
    i=faces[:, 0],
    j=faces[:, 1],
    k=faces[:, 2],
    color='lightblue',
    opacity=0.8,
    name="Gripper Mesh"
)

# 添加坐标轴
origin = np.array([0, 0, 0])
axes_lines = [
    # X axis (red)
    go.Scatter3d(x=[0, axis_length], y=[0, 0], z=[0, 0], mode='lines', line=dict(color='red', width=5), name='X'),
    # Y axis (green)
    go.Scatter3d(x=[0, 0], y=[0, axis_length], z=[0, 0], mode='lines', line=dict(color='green', width=5), name='Y'),
    # Z axis (blue)
    go.Scatter3d(x=[0, 0], y=[0, 0], z=[0, axis_length], mode='lines', line=dict(color='blue', width=5), name='Z')
]

# 计算AABB bounding box
bbox = mesh.bounds  # shape (2, 3): min, max
mins, maxs = bbox
bbox_corners = np.array([
    [mins[0], mins[1], mins[2]],
    [mins[0], mins[1], maxs[2]],
    [mins[0], maxs[1], mins[2]],
    [mins[0], maxs[1], maxs[2]],
    [maxs[0], mins[1], mins[2]],
    [maxs[0], mins[1], maxs[2]],
    [maxs[0], maxs[1], mins[2]],
    [maxs[0], maxs[1], maxs[2]],
])

# 定义bounding box线的连接（前后、左右、上下12条边）
lines = [
    (0, 1), (0, 2), (0, 4),
    (1, 3), (1, 5),
    (2, 3), (2, 6),
    (3, 7),
    (4, 5), (4, 6),
    (5, 7),
    (6, 7)
]
bbox_line_traces = []
for i, j in lines:
    bbox_line_traces.append(
        go.Scatter3d(
            x=[bbox_corners[i, 0], bbox_corners[j, 0]],
            y=[bbox_corners[i, 1], bbox_corners[j, 1]],
            z=[bbox_corners[i, 2], bbox_corners[j, 2]],
            mode='lines',
            line=dict(color='orange', width=4),
            name='bounding box' if i == 0 and j == 1 else None,
            showlegend=(i == 0 and j == 1)
        )
    )

# 添加文本注释显示尺寸
size_x = maxs[0] - mins[0]
size_y = maxs[1] - mins[1]
size_z = maxs[2] - mins[2]

# 中心点
center = (mins + maxs) / 2

# 在三个轴方向分别添加尺寸文字
bbox_text = [
    go.Scatter3d(
        x=[center[0]], y=[mins[1]-0.01], z=[mins[2]-0.01],
        mode="text",
        text=[f"Lx={size_x:.4f} m"],
        textposition="bottom center",
        showlegend=False
    ),
    go.Scatter3d(
        x=[mins[0]-0.01], y=[center[1]], z=[mins[2]-0.01],
        mode="text",
        text=[f"Ly={size_y:.4f} m"],
        textposition="bottom center",
        showlegend=False
    ),
    go.Scatter3d(
        x=[mins[0]-0.01], y=[mins[1]-0.01], z=[center[2]],
        mode="text",
        text=[f"Lz={size_z:.4f} m"],
        textposition="bottom center",
        showlegend=False
    ),
]

fig = go.Figure(data=[mesh_trace] + axes_lines + bbox_line_traces + bbox_text)
fig.update_layout(
    scene=dict(
        xaxis_title='X',
        yaxis_title='Y',
        zaxis_title='Z',
        aspectmode='data'
    ),
    margin=dict(r=10, l=10, b=10, t=10)
)
fig.show()

