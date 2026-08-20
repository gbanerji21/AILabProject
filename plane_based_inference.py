"""
Inference Pipeline: Measurement Prediction + Blender Cuts
- Multi-Stage ML: ArchPointNet (Classification) -> AnglePredictor (Upper Angle Correction) -> Regional DGCNNs (Trim Distances)
- Applies trim cuts using Blender Python API
- Outputs trimmed STL files and measurement JSON
"""

import os
import json
import subprocess
import torch
import torch.nn as nn
import torch.nn.functional as F
import trimesh
import numpy as np

# ==================== PATHS ====================
BEFORE_MESH = '/home/garvb/Downloads/Inference Data/scan61_before.stl'
CHECKPOINT_DIR = '/home/garvb/AILabProject/checkpoints'
OUTPUT_DIR = '/home/garvb/AILabProject/inference_results'
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# ==================== SAMPLING & INTERPOLATION HELPERS ====================

def sample_point_cloud(vertices, num_points=1000):
    """Uniformly samples mesh vertices down to num_points."""
    num_verts = len(vertices)
    if num_verts == 0:
        return np.zeros((num_points, 3), dtype=np.float32), np.arange(num_points)

    if num_verts >= num_points:
        idx = np.random.choice(num_verts, num_points, replace=False)
    else:
        idx = np.random.choice(num_verts, num_points, replace=True)

    return vertices[idx], idx


def map_predictions_to_full_mesh(sampled_verts, sampled_preds, full_verts):
    """Uses Nearest Neighbor search to project predictions from 1000 points back to all N vertices."""
    from scipy.spatial import KDTree
    tree = KDTree(sampled_verts)
    _, nearest_idx = tree.query(full_verts)
    return sampled_preds[nearest_idx]

# ==================== ARCH POINTNET CLASSIFIER ====================

class ArchPointNet(nn.Module):
    """DGCNN-based arch classifier for 3-class segmentation (upper/lower/discard)."""

    def __init__(self, num_points=1000, num_classes=3, k=20):
        super().__init__()
        self.num_points = num_points
        self.num_classes = num_classes
        self.k = k

        self.edge_conv1 = self._edge_conv_layer(6, 64)
        self.edge_conv2 = self._edge_conv_layer(128, 128)
        self.edge_conv3 = self._edge_conv_layer(256, 256)

        self.fc1 = nn.Linear(256, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc_points = nn.Linear(128 + 256, 128)
        self.fc_out = nn.Linear(128, num_classes)

    def _edge_conv_layer(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def get_edge_features(self, x, k):
        batch_size, num_points, _ = x.shape

        # Memory-safe k-NN computation
        dist = torch.cdist(x, x, p=2) ** 2
        _, knn_idx = torch.topk(dist, k + 1, dim=-1, largest=False)
        knn_idx = knn_idx[:, :, 1:]  # Exclude self-neighbor

        batch_idx = torch.arange(batch_size, device=x.device).view(batch_size, 1, 1).expand(-1, num_points, k)
        neighbors = x[batch_idx, knn_idx]

        point_expanded = x.unsqueeze(2).expand(-1, -1, k, -1)
        edge_feature = torch.cat([neighbors - point_expanded, neighbors], dim=3)

        return edge_feature

    def edge_conv(self, x, edge_conv_layer, k):
        edge_feat = self.get_edge_features(x, k)
        edge_feat = edge_feat.permute(0, 3, 1, 2)
        out = edge_conv_layer(edge_feat)
        out = torch.max(out, dim=3)[0]
        out = out.permute(0, 2, 1)
        return out

    def forward(self, x):
        ec1 = self.edge_conv(x, self.edge_conv1, self.k)
        ec2 = self.edge_conv(ec1, self.edge_conv2, self.k)
        ec3 = self.edge_conv(ec2, self.edge_conv3, self.k)

        global_feat = torch.max(ec3, dim=1)[0]
        global_feat = F.relu(self.fc1(global_feat))
        global_feat = F.relu(self.fc2(global_feat))

        global_feat_expanded = global_feat.unsqueeze(1).expand(-1, self.num_points, -1)
        combined = torch.cat([ec3, global_feat_expanded], dim=2)

        logits = F.relu(self.fc_points(combined))
        logits = self.fc_out(logits)

        return logits


# ==================== DGCNN DISTANCE REGRESSOR ====================

class DGCNNDistanceRegressor(nn.Module):
    """DGCNN-based distance regressor using dynamic graph convolutions."""

    def __init__(self, num_points=500, k=20, num_outputs=3):
        super().__init__()
        self.num_points = num_points
        self.k = k
        self.num_outputs = num_outputs

        self.edge_conv1 = self._edge_conv_layer(6, 64)
        self.edge_conv2 = self._edge_conv_layer(128, 128)
        self.edge_conv3 = self._edge_conv_layer(256, 256)

        self.fc_global1 = nn.Linear(256, 128)
        self.fc_global2 = nn.Linear(128, 64)

        self.fc_region1 = nn.Linear(256 + 64, 128)
        self.fc_region2 = nn.Linear(128, 64)
        self.fc_region_out = nn.Linear(64, 1)

        self.fc_out = nn.Linear(64, num_outputs - 1)
        self.dropout = nn.Dropout(0.3)

    def _edge_conv_layer(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def get_edge_features(self, x, k):
        batch_size, num_points, _ = x.shape

        dist = torch.cdist(x, x, p=2) ** 2
        _, knn_idx = torch.topk(dist, k + 1, dim=-1, largest=False)
        knn_idx = knn_idx[:, :, 1:]

        batch_idx = torch.arange(batch_size, device=x.device).view(batch_size, 1, 1).expand(-1, num_points, k)
        neighbors = x[batch_idx, knn_idx]

        point_expanded = x.unsqueeze(2).expand(-1, -1, k, -1)
        edge_feature = torch.cat([neighbors - point_expanded, neighbors], dim=3)

        return edge_feature

    def edge_conv(self, x, edge_conv_layer, k):
        edge_feat = self.get_edge_features(x, k)
        edge_feat = edge_feat.permute(0, 3, 1, 2)
        out = edge_conv_layer(edge_feat)
        out = torch.max(out, dim=3)[0]
        out = out.permute(0, 2, 1)
        return out

    def forward(self, x, region='left'):
        ec1 = self.edge_conv(x, self.edge_conv1, self.k)
        ec2 = self.edge_conv(ec1, self.edge_conv2, self.k)
        ec3 = self.edge_conv(ec2, self.edge_conv3, self.k)

        global_feat = torch.max(ec3, dim=1)[0]
        global_feat = F.relu(self.fc_global1(global_feat))
        global_feat = self.dropout(global_feat)
        global_feat = F.relu(self.fc_global2(global_feat))

        global_out = F.relu(self.fc_out(global_feat))

        region_feat = torch.max(ec3, dim=1)[0]
        region_combined = torch.cat([region_feat, global_feat], dim=1)
        region_out = F.relu(self.fc_region1(region_combined))
        region_out = self.dropout(region_out)
        region_out = F.relu(self.fc_region2(region_out))
        y_back = F.relu(self.fc_region_out(region_out))

        distances = torch.cat([global_out, y_back], dim=1)

        return distances


# ==================== ANGLE PREDICTOR ====================

class AnglePredictor(nn.Module):
    """PointNet-based Euler angle predictor."""

    def __init__(self, num_points=1000):
        super().__init__()
        self.num_points = num_points

        self.conv1 = nn.Conv1d(3, 64, 1)
        self.conv2 = nn.Conv1d(64, 128, 1)
        self.conv3 = nn.Conv1d(128, 256, 1)

        self.fc1 = nn.Linear(256, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc_out = nn.Linear(64, 3)

    def forward(self, x):
        x = x.transpose(2, 1)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = torch.max(x, dim=2)[0]
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        logits = self.fc_out(x)
        return logits


def euler_angles_to_rotation_matrix(angle_x, angle_y, angle_z):
    Rx = np.array([
        [1, 0, 0],
        [0, np.cos(angle_x), -np.sin(angle_x)],
        [0, np.sin(angle_x), np.cos(angle_x)]
    ])

    Ry = np.array([
        [np.cos(angle_y), 0, np.sin(angle_y)],
        [0, 1, 0],
        [-np.sin(angle_y), 0, np.cos(angle_y)]
    ])

    Rz = np.array([
        [np.cos(angle_z), -np.sin(angle_z), 0],
        [np.sin(angle_z), np.cos(angle_z), 0],
        [0, 0, 1]
    ])

    return Rz @ Ry @ Rx


def predict_and_correct_angles(mesh_vertices, angle_model, device='cuda'):
    verts_sampled, _ = sample_point_cloud(mesh_vertices, num_points=1000)

    centroid = verts_sampled.mean(axis=0)
    verts_centered = verts_sampled - centroid
    max_dist = np.max(np.linalg.norm(verts_centered, axis=1))
    verts_norm = verts_centered / max_dist if max_dist > 0 else verts_centered

    with torch.no_grad():
        points = torch.from_numpy(verts_norm).float().unsqueeze(0).to(device)
        angles = angle_model(points)
        angles_np = angles[0].cpu().numpy()

    angle_x, angle_y, angle_z = angles_np
    rotation_matrix = euler_angles_to_rotation_matrix(angle_x, angle_y, angle_z)

    return rotation_matrix, angle_x, angle_y, angle_z


# ==================== MEASUREMENT PREDICTION ====================

def predict_upper_measurements(mesh_vertices, left_model, right_model, device='cuda'):
    x_center = mesh_vertices[:, 0].mean()

    left_mask = mesh_vertices[:, 0] <= x_center
    right_mask = mesh_vertices[:, 0] > x_center

    left_verts = mesh_vertices[left_mask]
    right_verts = mesh_vertices[right_mask]

    z_cut = x_left = x_right = y_back_left = y_back_right = 0.0

    with torch.no_grad():
        if len(left_verts) > 10:
            left_verts_sampled, _ = sample_point_cloud(left_verts, num_points=500)

            centroid = left_verts_sampled.mean(axis=0)
            verts_centered = left_verts_sampled - centroid
            max_dist = np.max(np.linalg.norm(verts_centered, axis=1))
            verts_norm = verts_centered / max_dist if max_dist > 0 else verts_centered

            points = torch.from_numpy(verts_norm).float().unsqueeze(0).to(device)
            pred = left_model(points, region='left')
            pred_np = pred[0].cpu().numpy()

            z_cut = float(pred_np[0])
            x_left = float(pred_np[1])
            y_back_left = float(pred_np[2])

        if len(right_verts) > 10:
            right_verts_sampled, _ = sample_point_cloud(right_verts, num_points=500)

            centroid = right_verts_sampled.mean(axis=0)
            verts_centered = right_verts_sampled - centroid
            max_dist = np.max(np.linalg.norm(verts_centered, axis=1))
            verts_norm = verts_centered / max_dist if max_dist > 0 else verts_centered

            points = torch.from_numpy(verts_norm).float().unsqueeze(0).to(device)
            pred = right_model(points, region='right')
            pred_np = pred[0].cpu().numpy()

            x_right = float(pred_np[1])
            y_back_right = float(pred_np[2])

    return z_cut, x_left, x_right, y_back_left, y_back_right


def predict_lower_measurements(mesh_vertices, lower_model, device='cuda'):
    z_cut = 0.0
    if len(mesh_vertices) > 10:
        verts_sampled, _ = sample_point_cloud(mesh_vertices, num_points=500)

        centroid = verts_sampled.mean(axis=0)
        verts_centered = verts_sampled - centroid
        max_dist = np.max(np.linalg.norm(verts_centered, axis=1))
        verts_norm = verts_centered / max_dist if max_dist > 0 else verts_centered

        with torch.no_grad():
            points = torch.from_numpy(verts_norm).float().unsqueeze(0).to(device)
            pred = lower_model(points, region='left')
            pred_np = pred[0].cpu().numpy()
            z_cut = float(pred_np[0])

    return z_cut


# ==================== BLENDER CUTTING EXECUTION ====================

def run_blender_cuts(upper_stl_path, lower_stl_path, measurements_json_path, output_dir, base_name):
    """Executes precision hollow planar cuts, applies a Solidify modifier, and repositions meshes to ground level."""
    blender_script = f"""
import bpy
import json
import os
import math

with open(r'{measurements_json_path}', 'r') as f:
    measurements = json.load(f)

def clear_scene():
    bpy.ops.wm.read_factory_settings(use_empty=True)

def import_stl(filepath):
    if hasattr(bpy.ops.wm, 'stl_import'):
        bpy.ops.wm.stl_import(filepath=filepath)
    else:
        bpy.ops.import_mesh.stl(filepath=filepath)

def export_stl(filepath):
    if hasattr(bpy.ops.wm, 'stl_export'):
        bpy.ops.wm.stl_export(filepath=filepath)
    else:
        bpy.ops.export_mesh.stl(filepath=filepath)

def apply_cut(stl_path, m, mode='upper', out_path=''):
    clear_scene()
    import_stl(stl_path)
    obj = bpy.context.selected_objects[0]
    bpy.context.view_layer.objects.active = obj

    bbox = obj.bound_box
    min_b = [min(b[i] for b in bbox) for i in range(3)]
    max_b = [max(b[i] for b in bbox) for i in range(3)]
    x_center = (min_b[0] + max_b[0]) / 2.0

    bpy.ops.object.mode_set(mode='EDIT')

    if mode == 'upper' and m:
        # 1. Z-Cut (Top Plane) - Hollow
        bpy.ops.mesh.select_all(action='SELECT')
        bpy.ops.mesh.bisect(
            plane_co=(0, 0, max_b[2] - m['z_cut']), 
            plane_no=(0, 0, 1), 
            clear_outer=True, 
            use_fill=False
        )

        # 2. X-Left Trim - Hollow
        bpy.ops.mesh.select_all(action='SELECT')
        bpy.ops.mesh.bisect(
            plane_co=(min_b[0] + m['x_left'], 0, 0), 
            plane_no=(-1, 0, 0), 
            clear_outer=True, 
            use_fill=False
        )

        # 3. X-Right Trim - Hollow
        bpy.ops.mesh.select_all(action='SELECT')
        bpy.ops.mesh.bisect(
            plane_co=(max_b[0] - m['x_right'], 0, 0), 
            plane_no=(1, 0, 0), 
            clear_outer=True, 
            use_fill=False
        )

        # 4. Regional Y-Back Cut (Left Arch) - Hollow
        if 'y_back_left' in m and m['y_back_left'] > 0:
            bpy.ops.mesh.select_all(action='DESELECT')
            bpy.ops.object.mode_set(mode='OBJECT')
            for v in obj.data.vertices:
                if v.co.x <= x_center:
                    v.select = True
            bpy.ops.object.mode_set(mode='EDIT')

            y_cut_left = min_b[1] + m['y_back_left']
            bpy.ops.mesh.bisect(
                plane_co=(0, y_cut_left, 0), 
                plane_no=(0, -1, 0), 
                clear_outer=True, 
                use_fill=False
            )

        # 5. Regional Y-Back Cut (Right Arch) - Hollow
        if 'y_back_right' in m and m['y_back_right'] > 0:
            bpy.ops.mesh.select_all(action='DESELECT')
            bpy.ops.object.mode_set(mode='OBJECT')
            for v in obj.data.vertices:
                if v.co.x > x_center:
                    v.select = True
            bpy.ops.object.mode_set(mode='EDIT')

            y_cut_right = min_b[1] + m['y_back_right']
            bpy.ops.mesh.bisect(
                plane_co=(0, y_cut_right, 0), 
                plane_no=(0, -1, 0), 
                clear_outer=True, 
                use_fill=False
            )

    elif mode == 'lower' and m:
        # Z-Cut for lower arch - Hollow
        bpy.ops.mesh.select_all(action='SELECT')
        bpy.ops.mesh.bisect(
            plane_co=(0, 0, min_b[2] + m['z_cut']), 
            plane_no=(0, 0, 1), 
            clear_outer=True, 
            use_fill=False
        )

    bpy.ops.object.mode_set(mode='OBJECT')

    # ==================== SOLIDIFY MODIFIER ====================
    mod = obj.modifiers.new(name="Solidify", type='SOLIDIFY')
    mod.thickness = 0.15          # 0.15mm
    mod.offset = 1.000            # Outward shell growth
    mod.use_even_offset = False   # Even thickness
    mod.use_rim = True            # Rim fill enabled

    # Apply modifier permanent to geometry before export
    bpy.ops.object.modifier_apply(modifier=mod.name)

    # ==================== REPOSITIONING SECTION ====================
    if mode == 'upper':
        # Rotate ONLY upper teeth 180 degrees in the Y direction
        obj.rotation_euler[1] += math.radians(180)
        bpy.ops.object.transform_apply(location=False, rotation=True, scale=False)

    # Recalculate bounding box to make base flush with ground (Z = 0)
    bpy.context.view_layer.update()
    updated_bbox = obj.bound_box
    min_z = min(b[2] for b in updated_bbox)

    # Translate mesh vertically down so lowest Z boundary rests exactly on Z = 0
    obj.location.z -= min_z
    bpy.ops.object.transform_apply(location=True, rotation=False, scale=False)

    export_stl(out_path)

if os.path.exists(r'{upper_stl_path}') and 'upper' in measurements:
    apply_cut(r'{upper_stl_path}', measurements['upper'], mode='upper', out_path=r'{os.path.join(output_dir, f"{base_name}_upper_trimmed.stl")}')

if os.path.exists(r'{lower_stl_path}') and 'lower' in measurements:
    apply_cut(r'{lower_stl_path}', measurements['lower'], mode='lower', out_path=r'{os.path.join(output_dir, f"{base_name}_lower_trimmed.stl")}')
"""
    temp_script = os.path.join(output_dir, "_temp_blender_script.py")
    with open(temp_script, "w") as f:
        f.write(blender_script)

    subprocess.run(["blender", "--background", "--python", temp_script], check=True)
    os.remove(temp_script)

# ==================== MAIN INFERENCE ====================

def run_inference(before_mesh_path=BEFORE_MESH,
                  checkpoint_dir=CHECKPOINT_DIR,
                  output_dir=OUTPUT_DIR,
                  device=DEVICE):

    filename = os.path.basename(before_mesh_path)
    base_name = filename.replace('_before.stl', '').replace('before.stl', '').replace('.stl', '')
    os.makedirs(output_dir, exist_ok=True)

    print(f"\nINFERENCE PIPELINE: {base_name}")

    before_mesh = trimesh.load(before_mesh_path, force='mesh', process=False)
    measurements = {}

    # ==================== PHASE 1: ARCH CLASSIFICATION & SEPARATION ====================
    print("\n[PHASE 1] Segmenting arches into upper, lower, and discard files...")
    arch_model_path = os.path.join(checkpoint_dir, 'arch_classifier.pt')

    arch_model = ArchPointNet(num_points=1000, num_classes=3, k=20)
    arch_model.load_state_dict(torch.load(arch_model_path, map_location=device))
    arch_model.to(device)
    arch_model.eval()

    # Downsample dense mesh vertices to 1000 points for neural network evaluation
    verts_full = before_mesh.vertices
    verts_sampled, sampled_indices = sample_point_cloud(verts_full, num_points=1000)

    centroid = verts_sampled.mean(axis=0)
    verts_norm = (verts_sampled - centroid) / np.max(np.linalg.norm(verts_sampled - centroid, axis=1))

    points_tensor = torch.from_numpy(verts_norm).float().unsqueeze(0).to(device)
    with torch.no_grad():
        logits = arch_model(points_tensor)
        sampled_preds = torch.argmax(logits[0], dim=1).cpu().numpy()

    # Map predictions back to the full-density mesh vertices via Nearest Neighbors
    preds = map_predictions_to_full_mesh(verts_sampled, sampled_preds, verts_full)

    # Separate mesh faces based on interpolated full-vertex predictions
    upper_faces = [f for f in before_mesh.faces if all(preds[v] == 0 for v in f)]
    lower_faces = [f for f in before_mesh.faces if all(preds[v] == 1 for v in f)]

    upper_mesh = trimesh.Trimesh(vertices=before_mesh.vertices, faces=upper_faces, process=False)
    lower_mesh = trimesh.Trimesh(vertices=before_mesh.vertices, faces=lower_faces, process=False)

    # Repair holes & unreferenced vertices
    upper_mesh.remove_unreferenced_vertices()
    trimesh.repair.fill_holes(upper_mesh)
    lower_mesh.remove_unreferenced_vertices()
    trimesh.repair.fill_holes(lower_mesh)

    upper_stl_path = os.path.join(output_dir, f'{base_name}_upper.stl')
    lower_stl_path = os.path.join(output_dir, f'{base_name}_lower.stl')

    upper_mesh.export(upper_stl_path)
    lower_mesh.export(lower_stl_path)

    # ==================== STAGE 1: ANGLE CORRECTION (UPPER TEETH ONLY) ====================
    print("\n[STAGE 1] Predicting angle corrections on upper arch...")
    angle_model_path = os.path.join(checkpoint_dir, 'angle_predictor.pt')
    if os.path.exists(angle_model_path) and len(upper_mesh.vertices) > 0:
        angle_model = AnglePredictor(num_points=1000)
        angle_model.load_state_dict(torch.load(angle_model_path, map_location=device))
        angle_model.to(device)
        angle_model.eval()

        R, ax, ay, az = predict_and_correct_angles(upper_mesh.vertices, angle_model, device=device)
        upper_mesh.vertices = upper_mesh.vertices @ R.T
        upper_mesh.export(upper_stl_path)
        print(f"  Rotation Corrected: Rx={ax:.3f}, Ry={ay:.3f}, Rz={az:.3f}")

    # ==================== PHASE 2: REGIONAL DISTANCE PREDICTIONS ====================
    print("\n[PHASE 2] Predicting trim measurements via DGCNN models...")

    # Upper Arch Predictions
    left_model_path = os.path.join(checkpoint_dir, 'left_distance_regressor.pt')
    right_model_path = os.path.join(checkpoint_dir, 'right_distance_regressor.pt')

    if os.path.exists(left_model_path) and os.path.exists(right_model_path) and len(upper_mesh.vertices) > 0:
        left_model = DGCNNDistanceRegressor(num_points=500, k=20, num_outputs=3).to(device)
        right_model = DGCNNDistanceRegressor(num_points=500, k=20, num_outputs=3).to(device)

        left_model.load_state_dict(torch.load(left_model_path, map_location=device))
        right_model.load_state_dict(torch.load(right_model_path, map_location=device))

        left_model.eval()
        right_model.eval()

        z_cut, x_left, x_right, y_back_left, y_back_right = predict_upper_measurements(
            upper_mesh.vertices, left_model, right_model, device=device
        )

        measurements['upper'] = {
            'z_cut': round(z_cut, 2),
            'x_left': round(x_left, 2),
            'x_right': round(x_right, 2),
            'y_back_left': round(y_back_left, 2),
            'y_back_right': round(y_back_right, 2),
        }

    # Lower Arch Predictions
    lower_model_path = os.path.join(checkpoint_dir, 'lower_distance_regressor.pt')
    if os.path.exists(lower_model_path) and len(lower_mesh.vertices) > 0:
        lower_model = DGCNNDistanceRegressor(num_points=500, k=20, num_outputs=3).to(device)
        lower_model.load_state_dict(torch.load(lower_model_path, map_location=device))
        lower_model.eval()

        z_cut_lower = predict_lower_measurements(lower_mesh.vertices, lower_model, device=device)
        measurements['lower'] = {'z_cut': round(z_cut_lower, 2)}

    # Save Measurements
    measurements_path = os.path.join(output_dir, f'{base_name}_measurements.json')
    with open(measurements_path, 'w') as f:
        json.dump(measurements, f, indent=2)

    # ==================== PHASE 3: BLENDER TRIMMING ====================
    print("\n[PHASE 3] Executing trimming via Blender API...")
    run_blender_cuts(upper_stl_path, lower_stl_path, measurements_path, output_dir, base_name)
    print("\nWorkflow Execution Complete.")


if __name__ == '__main__':
    run_inference(BEFORE_MESH, checkpoint_dir=CHECKPOINT_DIR, output_dir=OUTPUT_DIR, device=DEVICE)
if __name__ == "__main__":
    main()
