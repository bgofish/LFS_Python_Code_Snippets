import lichtfeld as lf
import numpy as np

scene = lf.get_scene()
nodes = scene.get_visible_nodes()
node = nodes[0]

mn, mx = scene.get_node_bounds(node.name)
mn = np.array(mn)
mx = np.array(mx)

centre_x = (mn[0] + mx[0]) / 2
centre_z = (mn[2] + mx[2]) / 2# + mx[2]) / 2
floor_y  = mx[1] 

print(f"translating by: x={-centre_x:.4f}  y={-floor_y:.4f}  z={-centre_z:.4f}")

transform = np.eye(4, dtype=np.float32)
transform[0, 3] = -centre_x
transform[1, 3] = -floor_y
transform[2, 3] = -centre_z

scene.set_node_transform(node.name, transform)
print("done — model floored at Y=0, centred on X and Z")

mn2, mx2 = scene.get_node_bounds(node.name)
mn2 = np.array(mn2)
mx2 = np.array(mx2)
