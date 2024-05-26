import math

from v2.helper import calculate_intersection, convert_to_3d_coordinates, make_4d_point

a = [0, 0, -1, -1]
b = [1, 0, 0.5, 1]
c = [-0.5, 1.5, 1, 1]
d = [0, 1, 0, 1]
e = [0, -0.25, 0.5, 1]

hyperplane = {'point': [-1, 2, -2], 'angle': -math.pi / 3}

total = []
for a, b in [(a, b), (b, c), (c, a), (a, d), (b, d), (c, d), (a, e), (b, e), (c, e), (d, e)]:
    intersection = calculate_intersection(a, b, hyperplane)
    if intersection:
        converted = convert_to_3d_coordinates(intersection, make_4d_point(hyperplane['point']), hyperplane['angle'])
        print(intersection)
        total.append(converted)

print(total)
