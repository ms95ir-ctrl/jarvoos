---
name: build123d
description: Model parametric 3D parts in Python with build123d, with a watertight check before export
status: tested
---
Used for the ESP32-C6 Tamagotchi case (ten versions).
1. Create a venv on the machine that does the CAD and `pip install build123d`. Keep parameters (mm) in one block at the top; version the model file (`body_v8.py`) instead of overwriting.
2. Build with Part/Sketch builders, split shells with planes, export STL with `export_stl`.
3. Two OCCT traps seen in practice: coincident faces make booleans fail silently (extend one body past the other, e.g. `extend_back`); holes tangent to a face edge drop that face from the exported STL.
4. Run a watertight and volume check on the final mesh before slicing (for example with trimesh: `mesh.is_watertight`, `mesh.volume > 0`). Do not print a mesh that fails it.
5. Show ray-cast renders for review; matplotlib mesh plots hide problems.
6. When the user gives a measured dimension, it overrides the datasheet. Record assumed values and say which are assumed.
