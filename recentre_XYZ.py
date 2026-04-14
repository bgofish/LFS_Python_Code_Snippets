import lichtfeld as lf
import numpy as np

scene = lf.get_scene()
nodes = scene.get_visible_nodes()
node = nodes[0]

mn, mx = scene.get_node_bounds(node.name)
mn = np.array(mn)
mx = np.array(mx)
centroid = (mn + mx) / 2

print(f"bounds centre: {centroid}")
print(f"will translate by: {-centroid}")

# Build 4x4 translation matrix to move centroid to origin
transform = np.eye(4, dtype=np.float32)
transform[0, 3] = -centroid[0]
transform[1, 3] = -centroid[1]
transform[2, 3] = -centroid[2]

scene.set_node_transform(node.name, transform)
print("done — centroid moved to 0,0,0")

# Verify
mn2, mx2 = scene.get_node_bounds(node.name)
mn2 = np.array(mn2)
mx2 = np.array(mx2)
print(f"new bounds centre: {(mn2+mx2)/2}")