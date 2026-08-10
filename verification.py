import trimesh
import numpy as np

# Load the dental mold you designed
mold = trimesh.load('lower_teeth_hollow.stl')

print("--- RUNNING VERIFICATION AUDIT ---")

# Verification 1: Is it physically printable?
if mold.is_watertight:
    print("PASS: The mold is watertight.")
else:
    print("FAIL: The mesh has open holes! The 3D printer software will crash.")

# Verification 2: Is the base perfectly flat?
# Find all vertices at the very bottom of the model
lowest_z = mold.bounds[0][2]
bottom_vertices = [v for v in mold.vertices if abs(v[2] - lowest_z) < 0.01]

# If a large percentage of the mesh is at the lowest Z point, the base is flat
if len(bottom_vertices) > 100:
    print("PASS: The base is flat and ready for the build plate.")
else:
    print("FAIL: The base is uneven. The print will fail to adhere.")