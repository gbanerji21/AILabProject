import os
import slicer
import vtk
import sys

def convert_mha_to_stl_via_segmentation(input_mha_path, output_stl_path, lower_threshold=1, upper_threshold=300):
    print(f"[*] Starting segmentation-based STL export pipeline...")

    # Clear previous runs out of the scene
    slicer.mrmlScene.Clear(0)

    # 1. Load the volume
    print("[*] Loading .mha volume...")
    volume_node = slicer.util.loadVolume(input_mha_path)
    if not volume_node:
        print("[!] Error: Failed to load the input volume.")
        return False

    # 2. Create a segmentation node
    print("[*] Creating segmentation...")
    segmentation_node = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLSegmentationNode")
    segmentation_node.CreateDefaultDisplayNodes()

    # 3. Create a segment
    print("[*] Creating segment...")
    segment_id = segmentation_node.GetSegmentation().AddEmptySegment("Dental_Segment")

    # 4. Apply threshold to create a binary labelmap and extract geometry
    print(f"[*] Applying threshold ({lower_threshold}-{upper_threshold} HU)...")

    # Get the image data from the volume
    image_data = volume_node.GetImageData()
    if not image_data:
        print("[!] Error: Failed to get image data from volume.")
        return False

    # Apply threshold to create binary mask
    threshold_filter = vtk.vtkImageThreshold()
    threshold_filter.SetInputData(image_data)
    threshold_filter.ThresholdBetween(lower_threshold, upper_threshold)
    threshold_filter.SetInValue(1)
    threshold_filter.SetOutValue(0)
    threshold_filter.Update()
    thresholded_image = threshold_filter.GetOutput()

    # First, try standard marching cubes to generate a mesh
    print("[*] Extracting 3D surface geometry with marching cubes...")
    marching_cubes = vtk.vtkMarchingCubes()
    marching_cubes.SetInputData(thresholded_image)
    marching_cubes.SetValue(0, 1)
    marching_cubes.Update()
    mesh = marching_cubes.GetOutput()
    print(f"[*] Initial mesh: {mesh.GetNumberOfPoints()} points, {mesh.GetNumberOfCells()} cells")

    # Check if mesh was generated
    if mesh is None or mesh.GetNumberOfCells() == 0:
        print("[!] Error: No mesh generated from segmentation.")
        return False

    # Apply coordinate transformation to match original CT space
    print("[*] Applying coordinate transformation...")
    matrix = vtk.vtkMatrix4x4()
    volume_node.GetIJKToRASMatrix(matrix)

    transform = vtk.vtkTransform()
    transform.SetMatrix(matrix)

    transform_filter = vtk.vtkTransformPolyDataFilter()
    transform_filter.SetInputData(mesh)
    transform_filter.SetTransform(transform)
    transform_filter.Update()
    transformed_mesh = transform_filter.GetOutput()

    # Fill any small holes in the mesh
    print("[*] Filling small holes...")
    hole_filler = vtk.vtkFillHolesFilter()
    hole_filler.SetInputData(transformed_mesh)
    hole_filler.SetHoleSize(10.0)
    hole_filler.Update()
    clipped_mesh = hole_filler.GetOutput()

    # Apply smoothing filter (Windowed Sinc) to match GUI export quality
    smoother = vtk.vtkWindowedSincPolyDataFilter()
    smoother.SetInputData(clipped_mesh)
    smoother.SetNumberOfIterations(60)
    smoother.SetPassBand(0.06)
    smoother.SetFeatureAngle(80)
    smoother.SetEdgeAngle(80)
    smoother.SetNonManifoldSmoothing(True)
    smoother.FeatureEdgeSmoothingOff()
    smoother.BoundarySmoothingOn()
    smoother.Update()
    final_mesh = smoother.GetOutput()

    # 5. Write to STL file
    print("[*] Writing STL file...")
    output_dir = os.path.dirname(output_stl_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    stl_writer = vtk.vtkSTLWriter()
    stl_writer.SetFileName(output_stl_path)
    stl_writer.SetFileTypeToBinary()
    stl_writer.SetInputData(final_mesh)
    stl_writer.Write()

    if os.path.exists(output_stl_path):
        final_size_mb = os.path.getsize(output_stl_path) / (1024 * 1024)
        print(f"[+] Success! STL exported ({final_size_mb:.2f} MB) -> {output_stl_path}")
        return True
    else:
        print("[!] Error: Failed to write STL file.")
        return False

# Numbers to skip: 17, 19, 35, 42, 43
# File Execution Settings
INPUT_MHA = "/home/garvb/Downloads/ToothFairy3/labelsTr/ToothFairy3P_020.nii.gz"
OUTPUT_STL = "/home/garvb/Downloads/GNN Training Data/Before/scan70_before.stl"
LOWER_THRESHOLD = 1
UPPER_THRESHOLD = 300

# Run the updated pipeline
convert_mha_to_stl_via_segmentation(INPUT_MHA, OUTPUT_STL, LOWER_THRESHOLD, UPPER_THRESHOLD)
sys.exit(0)