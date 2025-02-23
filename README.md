# 4D Shape Intersection Visualizer

This project visualizes the intersection of 4D shapes with a dynamic 3D hyperplane. The program loads one or more 4D shapes from a JSON file, computes the intersection with a hyperplane that rotates about the W–Z plane, projects the resulting intersection into 3D, and renders it using OpenGL with proper lighting, face shading, and edge drawing. In addition, it implements FPS-style controls so you can freely navigate the 3D world.

![4D Example Demo](4d-example-full.gif)

## Features

- **Multi-Shape Support:** Load multiple 4D shapes from a JSON file. Each shape can have its own color and an optional position offset.
- **Dynamic Hyperplane:** The 3D hyperplane is defined in 4D and rotates about the W–Z plane. Use the mouse wheel to adjust the rotation.
- **Projection & Intersection:** For each shape, the program computes the intersection of its edges with a hyperplane, projects the 4D intersection points to 3D, and then creates a convex hull.
- **Rendering:** The 3D shape is rendered with ambient/diffuse lighting, proper face shading, and edge highlighting.
- **FPS-Style Controls:** Use keyboard controls for smooth continuous movement (translation and rotation) and zooming.
- **Demo:** A demo file (`4d-example-full.gif`) is provided to showcase an example of the visualization.

## Installation

This project requires Python 3.11+ and the following packages:

- `numpy`
- `scipy`
- `PyQt5`
- `pyqtgraph`
- `PyOpenGL`

You can install all dependencies using pip:

```bash
pip install -r requirements.txt
```
