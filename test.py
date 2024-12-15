import cdd as pcdd
import matplotlib.pyplot as plt
import numpy as np

points = np.array(
    [(2.348076211353316, 2.309401076758503, -3.3306690738754696e-16),
     (2.1056624327025935, 1.5773502691896233, 4.440892098500626e-16),
     (2.133974596215561, 1.9999999999999996, -1.1102230246251565e-16),
     (3.1419797543432466, 2.522639672457664, 2.886579864025407e-15),
     (2.457033892914712, 1.4140677858294244, 3.3306690738754696e-16),
     (2.646410161513775, 1.9999999999999996, -1.1102230246251565e-16)]

)

# to get the convex hull with cdd, one has to prepend a column of ones
vertices = np.hstack((np.ones((points.shape[0], 1)), points))

# do the polyhedron
mat = pcdd.Matrix(vertices.tolist(), linear=False, number_type="fraction")
mat.rep_type = pcdd.RepType.GENERATOR
poly = pcdd.Polyhedron(mat)

# get the adjacent vertices of each vertex
adjacencies = [list(x) for x in poly.get_input_adjacency()]

# store the edges in a matrix (giving the indices of the points)
edges = []
for i, indices in enumerate(adjacencies[:-1]):
    indices = list(filter(lambda x: x > i, indices))
    col1 = np.full((len(indices), 1), i)
    indices = np.reshape(indices, (len(indices), 1))
    if len(indices) > 0:
        edges.append(np.hstack((col1, indices)))
Edges = np.vstack(edges)

# plot
fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")

start = points[Edges[:, 0]]
end = points[Edges[:, 1]]

for i in range(Edges.shape[0]):
    ax.plot(
        [start[i, 0], end[i, 0]],
        [start[i, 1], end[i, 1]],
        [start[i, 2], end[i, 2]],
        "blue"
    )

ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_zlabel("z")

ax.set_xlim3d(-5, 5)
ax.set_ylim3d(-5, 5)
ax.set_zlim3d(-5, 5)

plt.show()
