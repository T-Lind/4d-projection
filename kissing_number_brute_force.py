import numpy as np


def generate_random_possible_point(n_dims):
    # This should generate a random point on a hypersphere of radius 2 in n dimensions
    vec = np.random.normal(size=n_dims)

    # Scale the vector to the correct radius
    vec /= np.linalg.norm(vec)
    vec *= 2
    return vec


def distance(point1, point2):
    # Calculate the Euclidean distance between two points
    return np.linalg.norm(point1 - point2)


def check_no_overlap(points):
    # Check if any two points are closer than the diameter (2 units)
    n_points = len(points)
    for i in range(n_points):
        for j in range(i + 1, n_points):
            if distance(points[i], points[j]) < 2:
                return False
    return True


def find_kissing_number(n_dims, max_attempts=1000):
    max_spheres = 0
    current_spheres = 1

    while True:
        points = []
        for _ in range(current_spheres):
            valid_point = False
            attempts = 0
            while not valid_point and attempts < max_attempts:
                new_point = generate_random_possible_point(n_dims)
                points.append(new_point)
                if check_no_overlap(points):
                    valid_point = True
                else:
                    points.pop()  # Remove the last added point
                attempts += 1

            if attempts >= max_attempts:
                # If we can't find a valid point after max_attempts, break
                break

        if len(points) < current_spheres:
            break

        max_spheres = current_spheres
        current_spheres += 1

    return max_spheres


n_dimensions = 3
n_trials = 1000
max_kissing_number = -1
for _ in range(n_trials):
    kissing_number = find_kissing_number(n_dimensions)
    max_kissing_number = max(max_kissing_number, kissing_number)

print(f"Kissing number in {n_dimensions} dimensions: {kissing_number}")
