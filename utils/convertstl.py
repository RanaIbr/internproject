import stl
from stl import mesh
import vtkplotlib as vpl
import open3d as o3d
from vtkmodules import*
import copy
import numpy as np

import aspose.threed as a3d

body = o3d.io.read_point_cloud("Clean body.ply")
mesh_f = copy.deepcopy(body)
R1 = body.get_rotation_matrix_from_xyz((np.radians(90), 0, np.radians(-270)))
mesh_f.rotate(R1, center=(0, 0, 0))

vis = o3d.visualization.VisualizerWithEditing()
vis.create_window(width=800, height=800, left=800, top=200)
vis.add_geometry(mesh_f)
vis.run()  # user picks points
vis.capture_screen_image("image-front.png", do_render=True)
vis.destroy_window()




scene = a3d.Scene.from_file("Clean body.ply")
scene.save("Output.stl")

mesh = mesh.Mesh.from_file('Output.stl')
plot = vpl.mesh_plot(mesh)

vpl.text("Profil View 1")
vpl.show()
