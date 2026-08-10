"""
Unified Training Pipeline: Stage 1 (Angle Prediction) + Phase 1 (PointNet) + Phase 2 (Vertex Classifier)
Stage 1: Predicts rotation angle for arch alignment
Phase 1: Classifies mesh vertices as upper/lower/discard
Phase 2: Per-vertex classification (keep/remove) with plane fitting post-processing
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import trimesh
import numpy as np
import os
from tqdm import tqdm
import matplotlib.pyplot as plt
import gc
from scipy.spatial import cKDTree
import argparse


# ==================== SCALAR DISTANCE REGRESSOR ====================

class DistanceRegressor(nn.Module):
    """Simple PointNet-style regressor for predicting 4 cutting distances: [Z, X_L, X_R, Y_B]"""

    def __init__(self, num_points=1000):
        super().__init__()
        self.num_points = num_points

        # Conv layers for feature extraction
        self.conv1 = nn.Conv1d(3, 64, 1)
        self.conv2 = nn.Conv1d(64, 128, 1)
        self.conv3 = nn.Conv1d(128, 256, 1)
        self.conv4 = nn.Conv1d(256, 512, 1)

        # Global MLP
        self.fc1 = nn.Linear(512, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc_out = nn.Linear(64, 4)  # Output: [Z_cut, X_left, X_right, Y_back]

        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        """
        Args:
            x: (batch, num_points, 3)
        Returns:
            distances: (batch, 4)
        """
        # Conv features
        x = x.transpose(2, 1)  # (batch, 3, num_points)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))

        # Global max pooling
        x = torch.max(x, dim=2)[0]  # (batch, 512)

        # MLP regressor
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = F.relu(self.fc3(x))
        distances = self.fc_out(x)  # (batch, 4)

        return distances


# ==================== DGCNN: Dynamic Graph CNN for Point Clouds ====================

class DGCNNVertexClassifier(nn.Module):
    """DGCNN: Dynamic Graph Convolutional Neural Network for per-vertex classification.

    Uses k-nearest neighbors to build dynamic graphs, enabling better local feature learning
    than standard PointNet. Edge convolutions capture local geometric relationships.
    """

    def __init__(self, num_points=1000, k=20, num_classes=2):
        super().__init__()
        self.num_points = num_points
        self.k = k
        self.num_classes = num_classes

        # EdgeConv layers: (in_channels) -> (out_channels)
        self.edge_conv1 = self._edge_conv_layer(6, 64)     # (x, x_neighbor) -> 64
        self.edge_conv2 = self._edge_conv_layer(128, 128)  # (64+64, 64_neighbor) -> 128
        self.edge_conv3 = self._edge_conv_layer(256, 256)  # (128+128, 128_neighbor) -> 256

        # Global feature
        self.fc1 = nn.Linear(256, 256)
        self.fc2 = nn.Linear(256, 128)

        # Per-point classification
        self.fc_points = nn.Linear(128 + 256, 128)
        self.fc_out = nn.Linear(128, num_classes)

    def _edge_conv_layer(self, in_channels, out_channels):
        """Edge convolution: combines point and neighbor features."""
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def get_edge_features(self, x, k):
        """Extract edge features using GPU-accelerated k-NN with chunking for memory efficiency.

        Computes distances in chunks to avoid OOM while keeping computation on GPU.

        Args:
            x: (batch, num_points, 3) point cloud
            k: number of nearest neighbors

        Returns:
            edge_feature: (batch, num_points, k, 6) concatenated point and neighbor coords
        """
        batch_size, num_points, _ = x.shape
        device = x.device

        # Compute k-NN on GPU with chunking to avoid OOM
        all_knn_idx = []

        for b in range(batch_size):
            points = x[b]  # (num_points, 3)

            # Compute distances in chunks to save memory
            chunk_size = 256
            all_distances = []

            for i in range(0, num_points, chunk_size):
                chunk_end = min(i + chunk_size, num_points)
                chunk = points[i:chunk_end]  # (chunk_size, 3)

                # Compute distances: (chunk_size, num_points)
                diff = chunk.unsqueeze(1) - points.unsqueeze(0)  # (chunk_size, num_points, 3)
                distances = torch.sum(diff ** 2, dim=2)  # (chunk_size, num_points)
                all_distances.append(distances)

            distances = torch.cat(all_distances, dim=0)  # (num_points, num_points)

            # Get k+1 nearest (includes self)
            _, knn = torch.topk(distances, k + 1, dim=1, largest=False)
            knn = knn[:, 1:]  # Remove self: (num_points, k)
            all_knn_idx.append(knn)

        knn_idx = torch.stack(all_knn_idx)  # (batch, num_points, k)

        # Gather neighbor coordinates
        batch_idx = torch.arange(batch_size, device=device).view(batch_size, 1, 1)
        neighbors = x[batch_idx, knn_idx]  # (batch, num_points, k, 3)

        # Combine point + neighbors
        point_expanded = x.unsqueeze(2).expand(-1, -1, k, -1)
        edge_feature = torch.cat([
            neighbors - point_expanded,
            neighbors
        ], dim=3)

        return edge_feature

    def edge_conv(self, x, edge_conv_layer, k):
        """Apply edge convolution with dynamic graph."""
        edge_feat = self.get_edge_features(x, k)  # (batch, num_points, k, 6)
        edge_feat = edge_feat.permute(0, 3, 1, 2)  # (batch, 6, num_points, k)

        out = edge_conv_layer(edge_feat)  # (batch, out_channels, num_points, k)
        out = torch.max(out, dim=3)[0]  # max pooling over neighbors: (batch, out_channels, num_points)
        out = out.permute(0, 2, 1)  # (batch, num_points, out_channels)

        return out

    def forward(self, x):
        """
        Args:
            x: (batch, num_points, 3)
        Returns:
            logits: (batch, num_points, num_classes)
        """
        batch_size = x.size(0)

        # Edge convolutions with skip connections
        ec1 = self.edge_conv(x, self.edge_conv1, self.k)  # (batch, num_points, 64)
        ec2 = self.edge_conv(ec1, self.edge_conv2, self.k)  # (batch, num_points, 128)
        ec3 = self.edge_conv(ec2, self.edge_conv3, self.k)  # (batch, num_points, 256)

        # Global feature: max pool over points
        global_feat = torch.max(ec3, dim=1)[0]  # (batch, 256)
        global_feat = F.relu(self.fc1(global_feat))  # (batch, 256)
        global_feat = F.relu(self.fc2(global_feat))  # (batch, 128)

        # Per-point classification
        global_feat_expanded = global_feat.unsqueeze(1).expand(-1, self.num_points, -1)  # (batch, num_points, 128)
        combined = torch.cat([ec3, global_feat_expanded], dim=2)  # (batch, num_points, 256+128)

        logits = F.relu(self.fc_points(combined))  # (batch, num_points, 128)
        logits = self.fc_out(logits)  # (batch, num_points, 2)

        return logits


# ==================== PLANE EXTRACTION (Geometric ground truth) ====================

def fit_plane_to_points(points):
    """Fit a plane to a set of points using SVD. Returns unit normal vector and offset."""
    if len(points) < 3:
        return None, None

    # Sample if too many points to avoid memory issues
    if len(points) > 1000:
        points = points[np.random.choice(len(points), 1000, replace=False)]

    centroid = points.mean(axis=0)
    shifted = points - centroid

    # Use full_matrices=False to avoid allocating huge matrices
    U, S, Vt = np.linalg.svd(shifted, full_matrices=False)
    normal = Vt[-1]

    # Normalize to unit vector
    normal_mag = np.linalg.norm(normal)
    if normal_mag > 0:
        normal = normal / normal_mag

    offset_dist = np.dot(centroid, normal)
    return normal, offset_dist


# ==================== ROTATION EXTRACTION (Stage 1 ground truth) ====================

def compute_rotation_matrix(before_verts, after_verts):
    """Compute best-fit rotation matrix using SVD on point clouds.
    Uses all vertices to capture the overall geometric relationship."""

    # Center both point clouds
    before_center = before_verts.mean(axis=0)
    after_center = after_verts.mean(axis=0)

    before_norm = before_verts - before_center
    after_norm = after_verts - after_center

    # If clouds have different sizes, resample to the same count
    min_points = min(len(before_norm), len(after_norm))
    sample_size = min(2000, min_points)

    if len(before_norm) != len(after_norm):
        before_indices = np.random.choice(len(before_norm), sample_size, replace=False)
        after_indices = np.random.choice(len(after_norm), sample_size, replace=False)
        before_norm = before_norm[before_indices]
        after_norm = after_norm[after_indices]

    # Compute covariance matrix for rotation
    H = before_norm.T @ after_norm
    U, S, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T

    # Ensure proper rotation (det = 1, not reflection)
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T

    return R


def rotation_matrix_to_euler_angles(R):
    """Convert rotation matrix to Euler angles (XYZ order, in radians).
    Returns angle_x (pitch), angle_y (roll), angle_z (yaw)."""
    # Extract Euler angles from rotation matrix using ZYX convention
    # This is more stable than angle-axis and gives independent rotations per axis

    sy = np.sqrt(R[0, 0] ** 2 + R[1, 0] ** 2)

    singular = sy < 1e-6

    if not singular:
        angle_x = np.arctan2(R[2, 1], R[2, 2])
        angle_y = np.arctan2(-R[2, 0], sy)
        angle_z = np.arctan2(R[1, 0], R[0, 0])
    else:
        angle_x = np.arctan2(-R[1, 2], R[1, 1])
        angle_y = np.arctan2(-R[2, 0], sy)
        angle_z = 0

    return np.array([angle_x, angle_y, angle_z], dtype=np.float32)


def euler_angles_to_rotation_matrix(angle_x, angle_y, angle_z):
    """Convert Euler angles (XYZ order) to rotation matrix."""
    # Rotation around X axis
    Rx = np.array([
        [1, 0, 0],
        [0, np.cos(angle_x), -np.sin(angle_x)],
        [0, np.sin(angle_x), np.cos(angle_x)]
    ])

    # Rotation around Y axis
    Ry = np.array([
        [np.cos(angle_y), 0, np.sin(angle_y)],
        [0, 1, 0],
        [-np.sin(angle_y), 0, np.cos(angle_y)]
    ])

    # Rotation around Z axis
    Rz = np.array([
        [np.cos(angle_z), -np.sin(angle_z), 0],
        [np.sin(angle_z), np.cos(angle_z), 0],
        [0, 0, 1]
    ])

    # Combined rotation (Z * Y * X order)
    return Rz @ Ry @ Rx


def angle_axis_to_rotation_matrix(axis, angle):
    """Convert angle-axis to rotation matrix using Rodrigues' formula."""
    axis = np.array(axis)
    axis = axis / np.linalg.norm(axis)

    K = np.array([
        [0, -axis[2], axis[1]],
        [axis[2], 0, -axis[0]],
        [-axis[1], axis[0], 0]
    ])

    R = np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * (K @ K)
    return R


def extract_vertex_labels(before_mesh, after_mesh):
    """Extract per-vertex keep/remove labels using adaptive percentile-based thresholding."""
    before_verts = before_mesh.vertices
    after_verts = after_mesh.vertices

    # Build KDTree from after mesh for efficient distance queries
    from scipy.spatial import cKDTree
    tree = cKDTree(after_verts)

    # Sample to determine adaptive threshold
    sample_size = min(10000, len(before_verts))
    sample_indices = np.random.choice(len(before_verts), sample_size, replace=False)
    sample_verts = before_verts[sample_indices]
    sample_distances, _ = tree.query(sample_verts)

    # Adaptive threshold: find the distance that approximately matches the after mesh vertex count
    target_ratio = len(after_verts) / len(before_verts)
    target_percentile = min(99, max(1, target_ratio * 100))
    distance_threshold = np.percentile(sample_distances, target_percentile)

    # Batch distance queries to avoid memory issues
    batch_size = 50000
    labels = np.zeros(len(before_verts), dtype=np.int32)

    for i in range(0, len(before_verts), batch_size):
        batch_end = min(i + batch_size, len(before_verts))
        batch_verts = before_verts[i:batch_end]

        # Query distance to nearest after vertex
        distances, _ = tree.query(batch_verts)

        # Keep if close to after mesh, remove if far away
        labels[i:batch_end] = np.where(distances < distance_threshold, 1, 0).astype(np.int32)

    return labels


def extract_cutting_plane(before_mesh, labels):
    """Extract cutting plane parameters from removed vertices in before mesh space."""
    before_verts = before_mesh.vertices
    removed_indices = np.where(labels == 0)[0]

    if len(removed_indices) < 10:
        return None, None

    # Fit plane to removed vertices in before space
    removed_verts = before_verts[removed_indices]
    plane_normal, plane_offset = fit_plane_to_points(removed_verts)

    if plane_normal is None:
        return None, None

    # Ensure normal points away from kept vertices (in a consistent direction)
    kept_indices = np.where(labels == 1)[0]
    if len(kept_indices) > 0:
        kept_verts = before_verts[kept_indices]
        kept_mean = kept_verts.mean(axis=0)
        # If normal points toward kept material, flip it
        if np.dot(plane_normal, kept_mean) > np.dot(plane_normal, removed_verts.mean(axis=0)):
            plane_normal = -plane_normal
            plane_offset = -plane_offset

    return plane_normal, plane_offset


def extract_cutting_distances(before_mesh, after_mesh):
    """Extract 4 cutting distances using bbox comparison: [Z_cut, X_left, X_right, Y_back]"""
    before_bounds = before_mesh.bounds
    after_bounds = after_mesh.bounds

    # Z cut: bottom difference
    z_cut = max(0, before_bounds[1][2] - after_bounds[1][2])

    # X cuts (left and right) - approximate as symmetric
    before_x_min, before_x_max = before_bounds[0][0], before_bounds[1][0]
    after_x_min, after_x_max = after_bounds[0][0], after_bounds[1][0]
    x_left = max(0, after_x_min - before_x_min)
    x_right = max(0, before_x_max - after_x_max)

    # Y cut (back) - single cut at one end
    before_y_min, before_y_max = before_bounds[0][1], before_bounds[1][1]
    after_y_min, after_y_max = after_bounds[0][1], after_bounds[1][1]
    y_back = max(0, before_y_max - after_y_max)

    return np.array([z_cut, x_left, x_right, y_back], dtype=np.float32)


def extract_distance_dataset(before_dir, after_dir, output_npz_path):
    """Extract cutting distances and before mesh vertices for regression training."""
    before_files = [f for f in os.listdir(before_dir) if f.endswith('upper_before.stl')]
    scan_ids = [f.replace('upper_before.stl', '') for f in before_files]
    scan_ids = sorted(scan_ids, key=lambda x: int(x.replace('scan', '')))

    results = []
    print(f"\n[*] Extracting cutting distances from {len(scan_ids)} samples...")
    print(f"{'='*60}\n")

    for i, scan_id in enumerate(scan_ids):
        before_path = os.path.join(before_dir, f"{scan_id}upper_before.stl")
        after_path = os.path.join(after_dir, f"{scan_id}upper_after.stl")

        try:
            before_mesh = trimesh.load(before_path, force='mesh', process=False)
            after_mesh = trimesh.load(after_path, force='mesh', process=False)

            distances = extract_cutting_distances(before_mesh, after_mesh)

            results.append({
                'scan_id': scan_id,
                'distances': distances,
                'vertices': before_mesh.vertices.copy(),
            })
            print(f"[{i+1}/{len(scan_ids)}] {scan_id}: Z={distances[0]:.1f}mm, X_L={distances[1]:.1f}mm, X_R={distances[2]:.1f}mm, Y_B={distances[3]:.1f}mm")
        except Exception as e:
            print(f"[{i+1}/{len(scan_ids)}] {scan_id} - Error: {e}")
        finally:
            del before_mesh, after_mesh
            gc.collect()

    if results:
        np.savez(
            output_npz_path,
            scan_ids=np.array([r['scan_id'] for r in results], dtype=object),
            distances=np.array([r['distances'] for r in results], dtype=np.float32),
            vertices_list=np.array([r['vertices'] for r in results], dtype=object),
        )
        print(f"\n[+] Extracted {len(results)} samples → {output_npz_path}\n")
        return True
    else:
        print(f"[!] No valid samples extracted")
        return False


def extract_vertex_labels_dataset(before_dir, after_dir, output_npz_path):
    """Extract per-vertex keep/remove labels from dataset."""
    before_files = [f for f in os.listdir(before_dir) if f.endswith('upper_before.stl')]
    scan_ids = [f.replace('upper_before.stl', '') for f in before_files]
    scan_ids = sorted(scan_ids, key=lambda x: int(x.replace('scan', '')))

    results = []
    print(f"\n[*] Extracting vertex labels from {len(scan_ids)} samples...")
    print(f"{'='*60}\n")

    for i, scan_id in enumerate(scan_ids):
        before_path = os.path.join(before_dir, f"{scan_id}upper_before.stl")
        after_path = os.path.join(after_dir, f"{scan_id}upper_after.stl")

        try:
            before_mesh = trimesh.load(before_path, force='mesh', process=False)
            after_mesh = trimesh.load(after_path, force='mesh', process=False)

            # Check alignment for scan1 (known misalignment issue)
            if scan_id == "scan1" or scan_id.startswith("scan1"):
                before_center = before_mesh.vertices.mean(axis=0)
                after_center = after_mesh.vertices.mean(axis=0)
                center_dist = np.linalg.norm(before_center - after_center)
                if center_dist > 40:
                    print(f"[!] {scan_id} WARNING: meshes are {center_dist:.1f}mm apart (likely misaligned data)")

            labels = extract_vertex_labels(before_mesh, after_mesh)

            removed_count = np.sum(labels == 0)
            kept_count = np.sum(labels == 1)

            if removed_count > 0 and kept_count > 0:
                # Extract cutting plane parameters
                plane_normal, plane_offset = extract_cutting_plane(before_mesh, labels)

                results.append({
                    'scan_id': scan_id,
                    'labels': labels,
                    'vertices': before_mesh.vertices.copy(),
                    'plane_normal': plane_normal,
                    'plane_offset': plane_offset,
                })
                print(f"[{i+1}/{len(scan_ids)}] {scan_id}: {kept_count} kept, {removed_count} removed")
            else:
                print(f"[{i+1}/{len(scan_ids)}] {scan_id} - Invalid split (all kept or all removed)")
        except Exception as e:
            print(f"[{i+1}/{len(scan_ids)}] {scan_id} - Error: {e}")
        finally:
            del before_mesh, after_mesh
            gc.collect()

    if results:
        # Build proper arrays with correct dtypes
        plane_normals_array = np.array([r['plane_normal'] for r in results], dtype=np.float32)
        plane_offsets_array = np.array([r['plane_offset'] for r in results], dtype=np.float32)

        np.savez(
            output_npz_path,
            scan_ids=np.array([r['scan_id'] for r in results], dtype=object),
            labels_list=np.array([r['labels'] for r in results], dtype=object),
            vertices_list=np.array([r['vertices'] for r in results], dtype=object),
            plane_normals=plane_normals_array,
            plane_offsets=plane_offsets_array,
        )
        print(f"\n[+] Extracted {len(results)} valid samples → {output_npz_path}\n")
        return True
    else:
        print(f"[!] No valid samples extracted")
        return False


# ==================== STAGE 1: ANGLE PREDICTION (Ground Truth Extraction) ====================

def find_tooth_cusps_robust(vertices, num_cusps=48):
    """Find stable tooth cusps using highest Z points across X regions.

    For zero-angle-change validation, uses topmost points which should be
    invariant to cutting (cutting only affects lower parts of mesh).

    Args:
        vertices: Mesh vertices (N, 3)
        num_cusps: Number of cusps to extract

    Returns:
        cusp_vertices: (num_cusps, 3) array of stable tooth peak positions
    """
    # Use only top 30% of vertices (should include all cusps, exclude cutting plane)
    z_threshold = np.percentile(vertices[:, 2], 70)
    upper_verts = vertices[vertices[:, 2] > z_threshold]

    if len(upper_verts) < num_cusps:
        return upper_verts

    # Sort by X coordinate and find highest point in each region
    upper_sorted = upper_verts[np.argsort(upper_verts[:, 0])]
    cluster_size = max(1, len(upper_sorted) // num_cusps)

    cusps = []
    for i in range(num_cusps):
        start_idx = i * cluster_size
        end_idx = start_idx + cluster_size if i < num_cusps - 1 else len(upper_sorted)

        if end_idx > start_idx:
            cluster = upper_sorted[start_idx:end_idx]
            peak_idx = np.argmax(cluster[:, 2])
            cusps.append(cluster[peak_idx])

    return np.array(cusps)


def cusp_icp_registration(src_cusps, dst_cusps, constrain_z=True, max_iter=20):
    """ICP algorithm on tooth cusps with Z-constraint.

    Args:
        src_cusps: Source cusps (N, 3)
        dst_cusps: Destination cusps (M, 3)
        constrain_z: Force z-rotation to 0
        max_iter: Maximum ICP iterations

    Returns:
        R: Rotation matrix (3x3)
        t: Translation vector (3,)
        match_pct: Percentage of matched cusps
        mean_error: Mean registration error
    """
    # Center cusps
    src_center = src_cusps.mean(axis=0)
    dst_center = dst_cusps.mean(axis=0)
    src = src_cusps - src_center
    dst = dst_cusps - dst_center

    R = np.eye(3)
    tree_dst = cKDTree(dst)

    for iteration in range(max_iter):
        # Transform source with current rotation
        src_transformed = (src @ R.T)

        # Find nearest neighbors
        distances, indices = tree_dst.query(src_transformed)

        # Outlier rejection: keep matches with distance < 90th percentile
        threshold = np.percentile(distances, 90)
        inliers = distances < threshold

        if np.sum(inliers) < 3:
            break

        src_inliers = src_transformed[inliers]
        dst_inliers = dst[indices[inliers]]

        # Compute rotation from SVD
        H = src_inliers.T @ dst_inliers
        U, S, Vt = np.linalg.svd(H)
        R_delta = Vt.T @ U.T

        if np.linalg.det(R_delta) < 0:
            Vt[-1, :] *= -1
            R_delta = Vt.T @ U.T

        # Update rotation
        R = R_delta @ R

        # Apply Z-constraint every iteration
        if constrain_z:
            angles = rotation_matrix_to_euler_angles(R)
            angles[2] = 0
            R = euler_angles_to_rotation_matrix(angles[0], angles[1], angles[2])

        # Check convergence
        if len(distances) > 0 and np.mean(distances[inliers]) < 1.0:
            break

    match_pct = (np.sum(inliers) / len(src_cusps)) * 100 if 'inliers' in locals() else 0
    mean_error = np.mean(distances[inliers]) if 'inliers' in locals() and np.sum(inliers) > 0 else 0

    t = dst_center - src_center @ R.T

    return R, t, match_pct, mean_error


def tooth_landmark_icp(src_pts, dst_pts, constrain_z=True):
    """Tooth landmark-based angle extraction using ICP on robust cusps.

    Extracts tooth cusps at multiple Z levels and uses ICP with iterative
    outlier rejection for maximum robustness to cutting and mesh variations.

    Args:
        src_pts: Before mesh vertices
        dst_pts: After mesh vertices
        constrain_z: Force z-rotation to 0

    Returns:
        R: Rotation matrix (3x3)
        t: Translation vector (3,)
        match_pct: Match percentage (cusps with valid correspondences)
        mean_error: Mean error of correspondences (mm)
    """
    # Extract tooth cusps at multiple levels from both meshes
    src_cusps = find_tooth_cusps_robust(src_pts, num_cusps=48)
    dst_cusps = find_tooth_cusps_robust(dst_pts, num_cusps=48)

    # Use ICP on cusps for robust registration
    R, t, match_pct, mean_error = cusp_icp_registration(
        src_cusps, dst_cusps, constrain_z=constrain_z, max_iter=20
    )

    return R, t, match_pct, mean_error


def fast_icp(src_pts, dst_pts, max_iterations=15, sample_size=5000, convergence_threshold=0.05, constrain_z=True, warm_start_R=None):
    """Iterative Closest Point with Z-rotation constraint and optional warm start.

    Args:
        src_pts: Source point cloud (before mesh vertices)
        dst_pts: Target point cloud (after mesh vertices)
        max_iterations: Maximum iterations (15 for better convergence, 5 for coarse)
        sample_size: Points to sample (5000 fine, 500 coarse)
        convergence_threshold: Stop if error improvement < threshold
        constrain_z: Force z-rotation to ~0
        warm_start_R: Optional rotation matrix to initialize from (for multi-scale)

    Returns:
        R: Rotation matrix (3x3)
        t: Translation vector (3,)
        match_pct: Percentage of points within 1mm after alignment
        mean_error: Mean distance after alignment
    """
    # Downsample for speed
    if len(src_pts) > sample_size:
        src_indices = np.random.choice(len(src_pts), sample_size, replace=False)
        src_pts = src_pts[src_indices]

    if len(dst_pts) > sample_size:
        dst_indices = np.random.choice(len(dst_pts), sample_size, replace=False)
        dst_pts = dst_pts[dst_indices]

    # Center point clouds
    src_center = src_pts.mean(axis=0)
    dst_center = dst_pts.mean(axis=0)
    src_centered = src_pts - src_center
    dst_centered = dst_pts - dst_center

    # Build KDTree for target
    tree_dst = cKDTree(dst_centered)

    # Initialize rotation from warm start or identity
    R = warm_start_R if warm_start_R is not None else np.eye(3)
    prev_error = float('inf')

    for iteration in range(max_iterations):
        # Transform source by current rotation
        src_transformed = src_centered @ R.T

        # Find nearest neighbors
        distances, indices = tree_dst.query(src_transformed)
        dst_matched = dst_centered[indices]

        # Mean error for this iteration
        mean_error = np.mean(distances)

        # Early stopping if converged
        error_improvement = prev_error - mean_error
        if error_improvement < convergence_threshold:
            break

        prev_error = mean_error

        # Compute new rotation from ICP correspondence
        H = src_transformed.T @ dst_matched
        U, S, Vt = np.linalg.svd(H)
        R_new = Vt.T @ U.T

        # Ensure proper rotation
        if np.linalg.det(R_new) < 0:
            Vt[-1, :] *= -1
            R_new = Vt.T @ U.T

        # Apply Z-constraint: force z-rotation to 0
        if constrain_z:
            angles = rotation_matrix_to_euler_angles(R_new)
            angles[2] = 0  # Set z-rotation to 0
            R_new = euler_angles_to_rotation_matrix(angles[0], angles[1], angles[2])

        R = R_new @ R

    # Final transformation and metrics
    src_final = src_centered @ R.T
    distances_final, _ = tree_dst.query(src_final)

    match_pct = (np.sum(distances_final < 1.0) / len(src_final)) * 100
    mean_error_final = np.mean(distances_final)

    # Apply final Z-constraint
    if constrain_z:
        angles = rotation_matrix_to_euler_angles(R)
        angles[2] = 0
        R = euler_angles_to_rotation_matrix(angles[0], angles[1], angles[2])

    # Compute translation in original coordinates
    t = dst_center - src_center @ R.T

    return R, t, match_pct, mean_error_final


def extract_angle_prediction_dataset(before_dir, after_dir, output_npz_path):
    """Extract rotation angles using fast ICP alignment.
    Aggressively tuned for speed: 5 iterations, 1000 samples, ~2-5 sec per scan."""
    before_files = [f for f in os.listdir(before_dir) if f.endswith('upper_before.stl')]
    scan_ids = [f.replace('upper_before.stl', '') for f in before_files]
    scan_ids = sorted(scan_ids, key=lambda x: int(x.replace('scan', '')))

    results = []
    print(f"\n[*] Extracting rotation angles using Tooth Landmarks...")
    print(f"[*] Matching 16 tooth cusps per mesh (z-constrained, fixed geometry)")
    print(f"{'='*60}\n")

    for i, scan_id in enumerate(scan_ids):
        before_path = os.path.join(before_dir, f"{scan_id}upper_before.stl")
        after_path = os.path.join(after_dir, f"{scan_id}upper_after.stl")

        try:
            before_mesh = trimesh.load(before_path, force='mesh', process=False)
            after_mesh = trimesh.load(after_path, force='mesh', process=False)

            before_verts = before_mesh.vertices
            after_verts = after_mesh.vertices

            if len(before_verts) < 10 or len(after_verts) < 10:
                print(f"[{i+1}/{len(scan_ids)}] {scan_id} - Error: Too few vertices")
                continue

            # Run tooth landmark-based extraction: matches tooth cusps for stable rotation
            R, t, match_pct, mean_error = tooth_landmark_icp(
                before_verts.copy(),
                after_verts.copy(),
                constrain_z=True
            )

            # Extract Euler angles from rotation matrix
            euler_angles = rotation_matrix_to_euler_angles(R)

            results.append({
                'scan_id': scan_id,
                'before_verts': before_verts.copy(),
                'euler_angles': euler_angles,
            })

            angle_x_deg = np.degrees(euler_angles[0])
            angle_y_deg = np.degrees(euler_angles[1])
            angle_z_deg = np.degrees(euler_angles[2])

            print(f"[{i+1}/{len(scan_ids)}] {scan_id}: x={angle_x_deg:7.2f}°, y={angle_y_deg:6.2f}°, z={angle_z_deg:6.2f}° | match={match_pct:5.1f}%, error={mean_error:.3f}mm")

        except Exception as e:
            print(f"[{i+1}/{len(scan_ids)}] {scan_id} - Error: {e}")
        finally:
            del before_mesh, after_mesh
            gc.collect()

    if results:
        # Build proper arrays with correct dtypes
        euler_angles_array = np.array([r['euler_angles'] for r in results], dtype=np.float32)  # (N, 3)

        np.savez(
            output_npz_path,
            scan_ids=np.array([r['scan_id'] for r in results], dtype=object),
            before_verts_list=np.array([r['before_verts'] for r in results], dtype=object),
            euler_angles=euler_angles_array,
        )
        print(f"\n[+] Extracted {len(results)} Euler angle predictions → {output_npz_path}\n")
        return True
    else:
        print(f"[!] No valid samples extracted")
        return False


# ==================== PHASE 1: PointNet Classes ====================

class ArchPointNet(nn.Module):
    """PointNet for 3-class arch classification."""

    def __init__(self, num_points=1000, num_classes=3):
        super().__init__()
        self.num_points = num_points

        self.conv1 = nn.Conv1d(3, 64, 1)
        self.conv2 = nn.Conv1d(64, 128, 1)
        self.conv3 = nn.Conv1d(128, 256, 1)

        self.fc1 = nn.Linear(256, 128)
        self.fc2 = nn.Linear(128, 64)

        self.fc_points = nn.Linear(64 + 256, 128)
        self.fc_out = nn.Linear(128, num_classes)

    def forward(self, x):
        batch_size = x.size(0)
        x = x.transpose(2, 1)

        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))

        global_feat = torch.max(x, dim=2)[0]
        global_feat = F.relu(self.fc1(global_feat))
        global_feat = F.relu(self.fc2(global_feat))

        x = x.transpose(2, 1)
        global_feat_expanded = global_feat.unsqueeze(1).expand(-1, self.num_points, -1)
        x_combined = torch.cat([x, global_feat_expanded], dim=2)

        x = F.relu(self.fc_points(x_combined))
        logits = self.fc_out(x)

        return logits


class ArchDataset(Dataset):
    """Dataset for PointNet arch classification."""

    def __init__(self, before_dir, after_dir, num_points=1000):
        self.before_dir = before_dir
        self.after_dir = after_dir
        self.num_points = num_points

        before_files = [f for f in os.listdir(before_dir) if f.endswith('_before.stl')]
        self.scan_ids = [f.replace('_before.stl', '') for f in before_files]
        # Sort numerically
        self.scan_ids = sorted(self.scan_ids, key=lambda x: int(x.replace('scan', '')))

        print(f"[*] PointNet Dataset: Found {len(self.scan_ids)} scans")

    def __len__(self):
        return len(self.scan_ids)

    def __getitem__(self, idx):
        scan_id = self.scan_ids[idx]
        before_path = os.path.join(self.before_dir, f"{scan_id}_before.stl")
        upper_path = os.path.join(self.after_dir, f"{scan_id}upper_after.stl")
        lower_path = os.path.join(self.after_dir, f"{scan_id}lower_after.stl")

        try:
            before_mesh = trimesh.load(before_path, force='mesh')
            upper_mesh = trimesh.load(upper_path, force='mesh')
            lower_mesh = trimesh.load(lower_path, force='mesh')
        except Exception as e:
            return torch.zeros((self.num_points, 3), dtype=torch.float32), torch.zeros(self.num_points, dtype=torch.long)

        try:
            points, face_indices = trimesh.sample.sample_surface(before_mesh, self.num_points)
        except:
            if len(before_mesh.vertices) >= self.num_points:
                points = before_mesh.vertices[np.random.choice(len(before_mesh.vertices), self.num_points, replace=False)]
            else:
                points = before_mesh.vertices

        centroid = points.mean(axis=0)
        points = points - centroid
        max_dist = np.max(np.linalg.norm(points, axis=1))
        if max_dist > 0:
            points = points / max_dist

        labels = np.zeros(len(points), dtype=np.int64)

        upper_verts = (upper_mesh.vertices - centroid) / max_dist if max_dist > 0 else upper_mesh.vertices - centroid
        lower_verts = (lower_mesh.vertices - centroid) / max_dist if max_dist > 0 else lower_mesh.vertices - centroid

        tree_upper = cKDTree(upper_verts)
        tree_lower = cKDTree(lower_verts)

        dist_upper, _ = tree_upper.query(points)
        dist_lower, _ = tree_lower.query(points)

        threshold = 0.1
        upper_mask = dist_upper < threshold
        lower_mask = dist_lower < threshold

        for i in range(len(points)):
            if upper_mask[i] and lower_mask[i]:
                labels[i] = 0 if dist_upper[i] < dist_lower[i] else 1
            elif upper_mask[i]:
                labels[i] = 0
            elif lower_mask[i]:
                labels[i] = 1
            else:
                labels[i] = 2

        return torch.from_numpy(points).float(), torch.from_numpy(labels).long()


class ArchTrainer:
    """Trainer for PointNet arch classification."""

    def __init__(self, model, device='cuda', lr=0.001, patience=10):
        self.model = model.to(device)
        self.device = device
        self.optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=10, gamma=0.5)
        self.criterion = nn.CrossEntropyLoss()
        self.history = {'train_loss': [], 'val_loss': [], 'val_acc': []}
        self.patience = patience
        self.best_val_acc = 0
        self.patience_counter = 0

    def train_epoch(self, train_loader):
        self.model.train()
        total_loss = 0
        num_batches = 0

        pbar = tqdm(train_loader, desc="PointNet Training")
        for points, labels in pbar:
            points = points.to(self.device)
            labels = labels.to(self.device)

            logits = self.model(points)
            loss = self.criterion(logits.reshape(-1, 3), labels.reshape(-1))

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
            num_batches += 1
            pbar.set_postfix(loss=f'{loss.item():.4f}')

        avg_loss = total_loss / num_batches
        self.history['train_loss'].append(avg_loss)
        return avg_loss

    def validate(self, val_loader):
        self.model.eval()
        total_loss = 0
        total_correct = 0
        total_points = 0

        with torch.no_grad():
            for points, labels in val_loader:
                points = points.to(self.device)
                labels = labels.to(self.device)

                logits = self.model(points)
                loss = self.criterion(logits.reshape(-1, 3), labels.reshape(-1))

                preds = torch.argmax(logits, dim=2)
                correct = (preds == labels).sum().item()

                total_loss += loss.item()
                total_correct += correct
                total_points += labels.numel()

        avg_loss = total_loss / len(val_loader)
        accuracy = total_correct / total_points if total_points > 0 else 0

        self.history['val_loss'].append(avg_loss)
        self.history['val_acc'].append(accuracy)

        return avg_loss, accuracy

    def train(self, train_loader, val_loader, epochs=50):
        print(f"[*] PointNet Training")
        print(f"[*] Epochs: {epochs}, Early stopping: {self.patience}\n")

        for epoch in range(epochs):
            train_loss = self.train_epoch(train_loader)
            val_loss, val_acc = self.validate(val_loader)
            self.scheduler.step()

            print(f"Epoch {epoch+1}/{epochs} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")

            if val_acc > self.best_val_acc:
                self.best_val_acc = val_acc
                self.patience_counter = 0
                print(f"  [+] Best model saved (acc: {val_acc:.4f})")
            else:
                self.patience_counter += 1
                if self.patience_counter >= self.patience:
                    print(f"[*] Early stopping at epoch {epoch+1}\n")
                    break

            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    def plot_history(self, save_path=None):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

        ax1.plot(self.history['train_loss'], label='Train Loss')
        ax1.plot(self.history['val_loss'], label='Val Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend()
        ax1.grid()

        ax2.plot(self.history['val_acc'], label='Val Accuracy')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy')
        ax2.legend()
        ax2.grid()

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path)
            print(f"[+] Plot saved to {save_path}")
        plt.close()


# ==================== STAGE 1: ANGLE PREDICTOR ====================

class AnglePredictor(nn.Module):
    """PointNet-based Euler angle predictor. Predicts 3D rotations: [angle_x, angle_y, angle_z]."""

    def __init__(self, num_points=1000):
        super().__init__()
        self.num_points = num_points

        self.conv1 = nn.Conv1d(3, 64, 1)
        self.conv2 = nn.Conv1d(64, 128, 1)
        self.conv3 = nn.Conv1d(128, 256, 1)

        self.fc1 = nn.Linear(256, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc_out = nn.Linear(64, 3)  # [angle_x, angle_y, angle_z] in radians

    def forward(self, x):
        # x: (batch, num_points, 3)
        x = x.transpose(2, 1)  # (batch, 3, num_points)

        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))

        x = torch.max(x, dim=2)[0]  # Global max pooling
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        logits = self.fc_out(x)  # (batch, 3)

        return logits


class AngleDataset(Dataset):
    """Dataset for Euler angle prediction."""

    def __init__(self, angle_npz_path, num_points=1000):
        self.num_points = num_points

        if not os.path.exists(angle_npz_path):
            raise FileNotFoundError(f"Angle dataset not found: {angle_npz_path}")

        data = np.load(angle_npz_path, allow_pickle=True)
        self.scan_ids = data['scan_ids']
        self.before_verts_list = data['before_verts_list']
        self.euler_angles = np.array(data['euler_angles'], dtype=np.float32)  # (N, 3)

        print(f"[*] Angle Dataset: Loaded {len(self.scan_ids)} samples from {angle_npz_path}")
        print(f"[*] Predicting 3D Euler angles: [angle_x, angle_y, angle_z]")

    def __len__(self):
        return len(self.scan_ids)

    def __getitem__(self, idx):
        try:
            verts = self.before_verts_list[idx]

            # Sample vertices
            if len(verts) > self.num_points:
                sample_indices = np.random.choice(len(verts), self.num_points, replace=False)
            else:
                sample_indices = np.arange(len(verts))

            sampled_verts = verts[sample_indices]

            # Normalize vertices
            centroid = sampled_verts.mean(axis=0)
            verts_centered = sampled_verts - centroid
            max_dist = np.max(np.linalg.norm(verts_centered, axis=1))
            if max_dist > 0:
                verts_norm = verts_centered / max_dist
            else:
                verts_norm = verts_centered

            # Get Euler angle targets (3D: angle_x, angle_y, angle_z)
            euler_angles = self.euler_angles[idx]  # (3,) in radians

            return torch.from_numpy(verts_norm).float(), torch.from_numpy(euler_angles).float()

        except Exception as e:
            print(f"[!] Error processing sample {idx}: {e}")
            return None, None


class AnglePredictionTrainer:
    """Trainer for angle prediction (Stage 1)."""

    def __init__(self, model, device='cuda', lr=0.001, patience=15):
        self.model = model.to(device)
        self.device = device
        self.optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=10, gamma=0.5)
        self.criterion = nn.MSELoss()
        self.history = {'train_loss': [], 'val_loss': []}
        self.patience = patience
        self.best_loss = float('inf')
        self.patience_counter = 0

    def train_epoch(self, train_loader):
        self.model.train()
        total_loss = 0

        for points, angles in tqdm(train_loader, desc="Angle Training"):
            if points is None:
                continue

            points = points.to(self.device)
            angles = angles.to(self.device)

            self.optimizer.zero_grad()
            predicted = self.model(points)  # (batch, 3)

            # Direct MSE loss on Euler angles (no normalization needed)
            loss = self.criterion(predicted, angles)
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()

        return total_loss / max(len(train_loader), 1)

    def validate(self, val_loader):
        self.model.eval()
        total_loss = 0

        with torch.no_grad():
            for points, angles in tqdm(val_loader, desc="Angle Validation"):
                if points is None:
                    continue

                points = points.to(self.device)
                angles = angles.to(self.device)

                predicted = self.model(points)  # (batch, 3)

                loss = self.criterion(predicted, angles)
                total_loss += loss.item()

        return total_loss / max(len(val_loader), 1)

    def train(self, train_loader, val_loader, epochs=50, checkpoint_dir='checkpoints'):
        os.makedirs(checkpoint_dir, exist_ok=True)

        for epoch in range(epochs):
            train_loss = self.train_epoch(train_loader)
            val_loss = self.validate(val_loader)

            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)

            print(f"Epoch {epoch+1}/{epochs} | Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f}")

            if val_loss < self.best_loss:
                self.best_loss = val_loss
                self.patience_counter = 0
                torch.save(self.model.state_dict(), os.path.join(checkpoint_dir, 'angle_predictor.pt'))
                print(f"  [+] Best model saved (loss: {val_loss:.6f})")
            else:
                self.patience_counter += 1
                if self.patience_counter >= self.patience:
                    print(f"[*] Early stopping at epoch {epoch+1}")
                    break

            self.scheduler.step()

        self.plot_history(os.path.join(checkpoint_dir, 'angle_training_history.png'))

    def plot_history(self, save_path=None):
        plt.figure(figsize=(10, 5))
        plt.plot(self.history['train_loss'], label='Train Loss', marker='o')
        plt.plot(self.history['val_loss'], label='Val Loss', marker='s')
        plt.xlabel('Epoch')
        plt.ylabel('MSE Loss')
        plt.title('Stage 1: Angle Prediction Training History')
        plt.legend()
        plt.grid()

        if save_path:
            plt.savefig(save_path)
            print(f"[+] Plot saved to {save_path}")
        plt.close()


# ==================== PHASE 2: CUTTING DISTANCE PREDICTOR ====================

def extract_cutting_distances_from_planes(before_mesh, after_mesh, num_back_cuts=1):
    """Extract cutting distances by comparing mesh bounds (bbox method).

    Simple and robust: directly compares before/after bounding boxes to extract
    how much material was removed from each direction. Immune to mesh rotation and
    vertex correspondence issues since it only examines final bounds.
    """
    before_bounds = before_mesh.bounds
    after_bounds = after_mesh.bounds

    z_cut = max(0, before_bounds[1][2] - after_bounds[1][2])
    x_left = max(0, after_bounds[0][0] - before_bounds[0][0])
    x_right = max(0, before_bounds[1][0] - after_bounds[1][0])
    y_back = max(0, before_bounds[1][1] - after_bounds[1][1])

    if num_back_cuts == 1:
        return z_cut, x_left, x_right, y_back
    else:
        return z_cut, x_left, x_right, y_back, y_back


def extract_cutting_dataset(before_dir, after_dir, output_npz_path, num_back_cuts=1):
    """Extract cutting distances using plane fitting on removed vertices.

    More accurate than bounding boxes - captures actual manual cuts by analyzing
    which vertices were removed and fitting planes to extract precise distances.
    """
    before_files = [f for f in os.listdir(before_dir) if f.endswith('upper_before.stl')]
    scan_ids = [f.replace('upper_before.stl', '') for f in before_files]
    scan_ids = sorted(scan_ids, key=lambda x: int(x.replace('scan', '')))

    results = []
    num_params = 4 if num_back_cuts == 1 else 5
    print(f"\n[*] Extracting {num_params} cutting distances using plane fitting...")
    print(f"[*] Analyzing removed vertices to extract accurate cut planes")
    print(f"{'='*60}\n")

    for i, scan_id in enumerate(scan_ids):
        before_path = os.path.join(before_dir, f"{scan_id}upper_before.stl")
        after_path = os.path.join(after_dir, f"{scan_id}upper_after.stl")

        try:
            # Load meshes without processing
            before_mesh = trimesh.load(before_path, force='mesh', process=False)
            after_mesh = trimesh.load(after_path, force='mesh', process=False)

            # Extract distances using plane fitting (more accurate than bounding boxes)
            if num_back_cuts == 1:
                z_cut, x_left, x_right, y_back = extract_cutting_distances_from_planes(
                    before_mesh, after_mesh, num_back_cuts=1
                )
                distances = np.array([z_cut, x_left, x_right, y_back], dtype=np.float32)
            else:
                z_cut, x_left, x_right, y_back_left, y_back_right = extract_cutting_distances_from_planes(
                    before_mesh, after_mesh, num_back_cuts=2
                )
                distances = np.array([z_cut, x_left, x_right, y_back_left, y_back_right], dtype=np.float32)

            results.append({
                'scan_id': scan_id,
                'before_verts': before_mesh.vertices,
                'distances': distances,
            })

            if num_back_cuts == 1:
                print(f"[{i+1}/{len(scan_ids)}] {scan_id}: Z={z_cut:6.2f}mm, X_L={x_left:6.2f}mm, X_R={x_right:6.2f}mm, Y_B={y_back:6.2f}mm")
            else:
                print(f"[{i+1}/{len(scan_ids)}] {scan_id}: Z={z_cut:6.2f}mm, X_L={x_left:6.2f}mm, X_R={x_right:6.2f}mm, Y_BL={y_back_left:6.2f}mm, Y_BR={y_back_right:6.2f}mm")

        except Exception as e:
            print(f"[{i+1}/{len(scan_ids)}] {scan_id} - Error: {e}")

    if results:
        # Build arrays
        scan_ids_result = np.array([r['scan_id'] for r in results], dtype=object)
        before_verts_list = np.array([r['before_verts'].copy() for r in results], dtype=object)
        distances_array = np.array([r['distances'] for r in results], dtype=np.float32)

        np.savez(
            output_npz_path,
            scan_ids=scan_ids_result,
            before_verts_list=before_verts_list,
            distances=distances_array,
        )
        print(f"\n[+] Extracted {len(results)} cutting distances (plane-fitted) → {output_npz_path}\n")
        return True
    else:
        print(f"[!] No valid samples extracted")
        return False


class CuttingDistancePredictor(nn.Module):
    """PointNet-based regression network predicting 4 (or 5) cutting distances."""

    def __init__(self, num_points=1000, num_outputs=4):
        super().__init__()
        self.num_points = num_points
        self.num_outputs = num_outputs

        self.conv1 = nn.Conv1d(3, 64, 1)
        self.conv2 = nn.Conv1d(64, 128, 1)
        self.conv3 = nn.Conv1d(128, 256, 1)

        self.fc1 = nn.Linear(256, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc_out = nn.Linear(64, num_outputs)

    def forward(self, x):
        x = x.transpose(2, 1)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = torch.max(x, dim=2)[0]
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc_out(x)


class CuttingDistanceDataset(Dataset):
    """Dataset for cutting distance prediction."""

    def __init__(self, distances_npz_path, num_points=1000):
        self.num_points = num_points

        if not os.path.exists(distances_npz_path):
            raise FileNotFoundError(f"Distances file not found: {distances_npz_path}")

        data = np.load(distances_npz_path, allow_pickle=True)
        self.scan_ids = data['scan_ids']
        self.before_verts_list = data['before_verts_list']
        self.distances = np.array(data['distances'], dtype=np.float32)
        self.num_outputs = self.distances.shape[1]

        print(f"[*] Cutting Distance Dataset: Loaded {len(self.scan_ids)} samples")
        print(f"[*] Predicting {self.num_outputs} distances: ", end="")
        if self.num_outputs == 4:
            print("[Z-depth, X-left, X-right, Y-back]")
        else:
            print("[Z-depth, X-left, X-right, Y-back-left, Y-back-right]")

    def __len__(self):
        return len(self.scan_ids)

    def __getitem__(self, idx):
        try:
            verts = self.before_verts_list[idx]

            if len(verts) > self.num_points:
                sample_indices = np.random.choice(len(verts), self.num_points, replace=False)
            else:
                sample_indices = np.arange(len(verts))

            sampled_verts = verts[sample_indices]

            centroid = sampled_verts.mean(axis=0)
            verts_centered = sampled_verts - centroid
            max_dist = np.max(np.linalg.norm(verts_centered, axis=1))
            if max_dist > 0:
                verts_norm = verts_centered / max_dist
            else:
                verts_norm = verts_centered

            distances = self.distances[idx]

            return torch.from_numpy(verts_norm).float(), torch.from_numpy(distances).float()

        except Exception as e:
            print(f"[!] Error processing sample {idx}: {e}")
            return None, None


class CuttingDistanceTrainer:
    """Trainer for cutting distance prediction."""

    def __init__(self, model, device='cuda', lr=0.001, patience=15):
        self.model = model.to(device)
        self.device = device
        self.optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=10, gamma=0.5)
        self.criterion = nn.MSELoss()
        self.history = {'train_loss': [], 'val_loss': [], 'val_mae': []}
        self.patience = patience
        self.best_loss = float('inf')
        self.patience_counter = 0

    def train_epoch(self, train_loader):
        self.model.train()
        total_loss = 0

        for points, distances in tqdm(train_loader, desc="Cutting Distance Training"):
            if points is None:
                continue

            points = points.to(self.device)
            distances = distances.to(self.device)

            self.optimizer.zero_grad()
            predicted = self.model(points)
            loss = self.criterion(predicted, distances)
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()

        return total_loss / max(len(train_loader), 1)

    def validate(self, val_loader):
        self.model.eval()
        total_loss = 0
        total_mae = 0
        num_batches = 0

        with torch.no_grad():
            for points, distances in tqdm(val_loader, desc="Cutting Distance Validation"):
                if points is None:
                    continue

                points = points.to(self.device)
                distances = distances.to(self.device)

                predicted = self.model(points)
                loss = self.criterion(predicted, distances)
                mae = torch.mean(torch.abs(predicted - distances))

                total_loss += loss.item()
                total_mae += mae.item()
                num_batches += 1

        return total_loss / max(num_batches, 1), total_mae / max(num_batches, 1)

    def train(self, train_loader, val_loader, epochs=50, checkpoint_dir='checkpoints'):
        os.makedirs(checkpoint_dir, exist_ok=True)

        for epoch in range(epochs):
            train_loss = self.train_epoch(train_loader)
            val_loss, val_mae = self.validate(val_loader)

            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['val_mae'].append(val_mae)

            print(f"Epoch {epoch+1}/{epochs} | Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f} | Val MAE: {val_mae:.6f}mm")

            if val_loss < self.best_loss:
                self.best_loss = val_loss
                self.patience_counter = 0
                torch.save(self.model.state_dict(), os.path.join(checkpoint_dir, 'cutting_distance_predictor.pt'))
                print(f"  [+] Best model saved (loss: {val_loss:.6f})")
            else:
                self.patience_counter += 1
                if self.patience_counter >= self.patience:
                    print(f"[*] Early stopping at epoch {epoch+1}")
                    break

            self.scheduler.step()

        self.plot_history(os.path.join(checkpoint_dir, 'cutting_distance_training_history.png'))

    def plot_history(self, save_path=None):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

        ax1.plot(self.history['train_loss'], label='Train Loss', marker='o')
        ax1.plot(self.history['val_loss'], label='Val Loss', marker='s')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('MSE Loss')
        ax1.set_title('Phase 2: Cutting Distance Training Loss')
        ax1.legend()
        ax1.grid()

        ax2.plot(self.history['val_mae'], label='Val MAE', marker='o', color='orange')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Mean Absolute Error (mm)')
        ax2.set_title('Phase 2: Cutting Distance Prediction Accuracy')
        ax2.legend()
        ax2.grid()

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path)
            print(f"[+] Plot saved to {save_path}")
        plt.close()


# ==================== PHASE 2 (OLD): Constrained Plane Predictor ====================

class VertexDataset(Dataset):
    """Dataset for per-vertex keep/remove classification."""

    def __init__(self, labels_npz_path, num_points=1000):
        self.num_points = num_points

        if not os.path.exists(labels_npz_path):
            raise FileNotFoundError(f"Labels file not found: {labels_npz_path}")

        data = np.load(labels_npz_path, allow_pickle=True)
        self.scan_ids = data['scan_ids']
        self.labels_list = data['labels_list']
        self.vertices_list = data['vertices_list']
        self.plane_normals = data.get('plane_normals', None)
        self.plane_offsets = data.get('plane_offsets', None)

        print(f"[*] Vertex Dataset: Loaded {len(self.scan_ids)} samples from {labels_npz_path}")
        if self.plane_normals is not None:
            print(f"[*] Plane parameters loaded for dual-output training")

    def __len__(self):
        return len(self.scan_ids)

    def __getitem__(self, idx):
        try:
            verts = self.vertices_list[idx]
            labels = self.labels_list[idx]

            # Sample vertices
            if len(verts) > self.num_points:
                sample_indices = np.random.choice(len(verts), self.num_points, replace=False)
            else:
                sample_indices = np.arange(len(verts))

            sampled_verts = verts[sample_indices]
            sampled_labels = labels[sample_indices]

            # Normalize vertices
            centroid = sampled_verts.mean(axis=0)
            verts_centered = sampled_verts - centroid
            max_dist = np.max(np.linalg.norm(verts_centered, axis=1))
            if max_dist > 0:
                verts_norm = verts_centered / max_dist
            else:
                verts_norm = verts_centered

            result = (torch.from_numpy(verts_norm).float(), torch.from_numpy(sampled_labels).long())

            # Include plane parameters if available
            if self.plane_normals is not None and self.plane_offsets is not None:
                plane_normal = np.array(self.plane_normals[idx], dtype=np.float32)
                plane_offset = np.array([self.plane_offsets[idx]], dtype=np.float32)
                if plane_normal is not None and plane_offset is not None:
                    plane_target = np.concatenate([plane_normal, plane_offset])
                    result = result + (torch.from_numpy(plane_target).float(),)

            return result

        except Exception as e:
            print(f"[!] Error processing sample {idx}: {e}")
            return None, None


class VertexClassifier(nn.Module):
    """Per-vertex keep/remove classifier using PointNet-style architecture."""

    def __init__(self, num_points=1000):
        super().__init__()
        self.num_points = num_points

        self.conv1 = nn.Conv1d(3, 64, 1)
        self.conv2 = nn.Conv1d(64, 128, 1)
        self.conv3 = nn.Conv1d(128, 256, 1)

        self.fc1 = nn.Linear(256, 128)
        self.fc2 = nn.Linear(128, 64)

        self.fc_points = nn.Linear(64 + 256, 128)
        self.fc_out = nn.Linear(128, 2)  # Binary classification: keep/remove

    def forward(self, x):
        # x shape: (batch, num_points, 3)
        x = x.transpose(2, 1)  # (batch, 3, num_points)

        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))

        # Global feature
        global_feat = torch.max(x, dim=2)[0]
        global_feat = F.relu(self.fc1(global_feat))
        global_feat = F.relu(self.fc2(global_feat))

        # Per-point features for classification
        x = x.transpose(2, 1)  # (batch, num_points, 256)
        global_feat_expanded = global_feat.unsqueeze(1).expand(-1, self.num_points, -1)
        x_combined = torch.cat([x, global_feat_expanded], dim=2)

        class_logits = F.relu(self.fc_points(x_combined))
        return self.fc_out(class_logits)  # (batch, num_points, 2)


class VertexTrainer:
    """Trainer for per-vertex vertex classifier."""

    def __init__(self, model, device='cuda', lr=0.001, patience=15, class_weights=None):
        self.model = model.to(device)
        self.device = device
        self.optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=10, gamma=0.7)
        self.criterion = nn.CrossEntropyLoss(weight=class_weights)
        self.history = {'train_loss': [], 'val_loss': [], 'val_acc': []}
        self.patience = patience
        self.best_val_loss = float('inf')
        self.patience_counter = 0

    def train_epoch(self, train_loader):
        self.model.train()
        total_loss = 0
        num_batches = 0

        pbar = tqdm(train_loader, desc="Vertex Classifier Training")
        for batch in pbar:
            if batch[0] is None or batch[1] is None:
                continue

            points = batch[0].to(self.device)
            labels = batch[1].to(self.device)

            output = self.model(points)

            # Handle dual-output (classification + plane prediction)
            if isinstance(output, tuple):
                class_logits, plane_logits = output
                class_loss = self.criterion(class_logits.reshape(-1, 2), labels.reshape(-1))

                # Plane regression loss (MSE)
                if len(batch) > 2 and batch[2] is not None:
                    plane_targets = batch[2].to(self.device)
                    plane_preds = plane_logits

                    # Normalize plane normal predictions
                    plane_norms = plane_preds[:, :3]
                    plane_norms = F.normalize(plane_norms, dim=1)
                    plane_preds = torch.cat([plane_norms, plane_preds[:, 3:4]], dim=1)

                    plane_loss = F.mse_loss(plane_preds, plane_targets)
                    loss = class_loss + 0.5 * plane_loss  # Weight plane loss equally
                else:
                    loss = class_loss
            else:
                class_logits = output
                loss = self.criterion(class_logits.reshape(-1, 2), labels.reshape(-1))

            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()

            total_loss += loss.item()
            num_batches += 1
            pbar.set_postfix(loss=f'{loss.item():.4f}')

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        avg_loss = total_loss / num_batches if num_batches > 0 else 0
        self.history['train_loss'].append(avg_loss)
        return avg_loss

    def validate(self, val_loader):
        self.model.eval()
        total_loss = 0
        total_correct = 0
        total_points = 0

        with torch.no_grad():
            for batch in val_loader:
                if batch[0] is None or batch[1] is None:
                    continue

                points = batch[0].to(self.device)
                labels = batch[1].to(self.device)

                output = self.model(points)

                # Handle dual-output
                if isinstance(output, tuple):
                    class_logits, plane_logits = output
                    class_loss = self.criterion(class_logits.reshape(-1, 2), labels.reshape(-1))

                    if len(batch) > 2 and batch[2] is not None:
                        plane_targets = batch[2].to(self.device)
                        plane_preds = plane_logits
                        plane_norms = plane_preds[:, :3]
                        plane_norms = F.normalize(plane_norms, dim=1)
                        plane_preds = torch.cat([plane_norms, plane_preds[:, 3:4]], dim=1)
                        plane_loss = F.mse_loss(plane_preds, plane_targets)
                        loss = class_loss + 0.5 * plane_loss
                    else:
                        loss = class_loss

                    preds = torch.argmax(class_logits, dim=2)
                else:
                    class_logits = output
                    loss = self.criterion(class_logits.reshape(-1, 2), labels.reshape(-1))
                    preds = torch.argmax(class_logits, dim=2)

                correct = (preds == labels).sum().item()
                total_loss += loss.item()
                total_correct += correct
                total_points += labels.numel()

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        avg_loss = total_loss / len(val_loader) if len(val_loader) > 0 else 0
        accuracy = total_correct / total_points if total_points > 0 else 0

        self.history['val_loss'].append(avg_loss)
        self.history['val_acc'].append(accuracy)
        return avg_loss, accuracy

    def train(self, train_loader, val_loader, epochs=50):
        print(f"[*] Vertex Classifier Training")
        print(f"[*] Epochs: {epochs}, Patience: {self.patience}\n")

        for epoch in range(epochs):
            train_loss = self.train_epoch(train_loader)
            val_loss, val_acc = self.validate(val_loader)
            self.scheduler.step()

            print(f"Epoch {epoch+1}/{epochs} | Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f} | Val Acc: {val_acc:.4f}")

            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.patience_counter = 0
                print(f"  [+] Best model saved (loss: {val_loss:.6f})")
            else:
                self.patience_counter += 1
                if self.patience_counter >= self.patience:
                    print(f"[*] Early stopping at epoch {epoch+1}\n")
                    break

    def plot_history(self, save_path=None):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

        ax1.plot(self.history['train_loss'], label='Train Loss')
        ax1.plot(self.history['val_loss'], label='Val Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('MSE Loss')
        ax1.legend()
        ax1.grid()

        if 'val_mae' in self.history:
            ax2.plot(self.history['val_mae'], label='Val MAE')
            ax2.set_ylabel('Mean Absolute Error')
        else:
            ax2.plot(self.history['val_acc'], label='Val Accuracy')
            ax2.set_ylabel('Accuracy')
        ax2.set_xlabel('Epoch')
        ax2.legend()
        ax2.grid()

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path)
            print(f"[+] Plot saved to {save_path}")
        plt.close()


def collate_fn(batch):
    """Custom collate that filters out None."""
    batch = [item for item in batch if item[0] is not None]
    if len(batch) == 0:
        return None, None

    features = torch.stack([item[0] for item in batch])
    targets = torch.stack([item[1] for item in batch])

    return features, targets


def main():
    parser = argparse.ArgumentParser(description="Unified Training: Phase 1 (PointNet) → Stage 1 (Angle) → Phase 2 (Cutting Distance)")
    parser.add_argument("--epochs", type=int, default=50, help="Number of epochs")
    parser.add_argument("--patience", type=int, default=15, help="Early stopping patience")
    parser.add_argument("--skip-pointnet", action="store_true", help="Skip Phase 1 (PointNet) training")
    parser.add_argument("--skip-angle", action="store_true", help="Skip Stage 1 (Angle) training")
    parser.add_argument("--skip-cutting", action="store_true", help="Skip Phase 2 (Cutting Distance) training")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--back-cuts", type=int, default=1, choices=[1, 2], help="Number of back cuts (1 or 2)")

    args = parser.parse_args()

    POINTNET_BEFORE = "/home/garvb/Downloads/Pointnet Training Data/Before"
    POINTNET_AFTER = "/home/garvb/Downloads/Pointnet Training Data/After"
    # Angle prediction uses clean rotation-only dataset (no cutting confounding)
    ANGLE_BEFORE = "/home/garvb/Downloads/Angle Predictor Training Data/Before"
    ANGLE_AFTER = "/home/garvb/Downloads/Angle Predictor Training Data/After"

    # Plane prediction uses full dataset with cutting
    PLANE_BEFORE = "/home/garvb/Downloads/Plane Predictor Training Data/Before"
    PLANE_AFTER = "/home/garvb/Downloads/Plane Predictor Training Data/After"
    ANGLES_NPZ = "/home/garvb/AILabProject/angle_predictions.npz"

    if args.back_cuts == 1:
        CUTTING_NPZ = "/home/garvb/AILabProject/cutting_distances_4cut.npz"
    else:
        CUTTING_NPZ = "/home/garvb/AILabProject/cutting_distances_5cut.npz"

    CHECKPOINT_DIR = "/home/garvb/AILabProject/checkpoints"
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"[*] Using device: {device}\n")

    print(f"{'='*60}")
    print(f"[*] Unified Training Pipeline: Phase 1 (PointNet) → Stage 1 (Angle) → Phase 2 (Vertex)")
    print(f"{'='*60}\n")

    # ==================== PHASE 1: Train PointNet ====================
    if not args.skip_pointnet:
        print(f"[*] PHASE 1: Training PointNet (Upper/Lower/Discard Classification)")
        print(f"{'='*60}\n")

        if not os.path.exists(POINTNET_BEFORE) or not os.path.exists(POINTNET_AFTER):
            print(f"[!] PointNet directories not found, skipping Phase 1")
        else:
            pointnet_dataset = ArchDataset(POINTNET_BEFORE, POINTNET_AFTER, num_points=1000)
            train_size = int(0.8 * len(pointnet_dataset))
            val_size = len(pointnet_dataset) - train_size
            train_dataset, val_dataset = torch.utils.data.random_split(pointnet_dataset, [train_size, val_size])

            train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=0)
            val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False, num_workers=0)

            pointnet_model = ArchPointNet(num_points=1000, num_classes=3)
            pointnet_trainer = ArchTrainer(pointnet_model, device=device, lr=args.lr, patience=args.patience)

            pointnet_trainer.train(train_loader, val_loader, epochs=args.epochs)

            pointnet_path = os.path.join(CHECKPOINT_DIR, "arch_classifier.pt")
            torch.save(pointnet_model.state_dict(), pointnet_path)
            print(f"[+] PointNet saved to {pointnet_path}\n")
            pointnet_trainer.plot_history(os.path.join(CHECKPOINT_DIR, "pointnet_training_history.png"))

    # ==================== STAGE 1: Train Angle Predictor ====================
    if not args.skip_angle:
        print(f"\n[*] STAGE 1: Training Angle Predictor (Rotation Estimation)")
        print(f"{'='*60}\n")

        # Extract angle predictions if not already extracted
        if not os.path.exists(ANGLES_NPZ):
            print(f"[*] Angles file not found. Extracting from dataset...")
            extraction_success = extract_angle_prediction_dataset(ANGLE_BEFORE, ANGLE_AFTER, ANGLES_NPZ)
            if not extraction_success:
                print(f"[!] Angle extraction failed")
                return
        else:
            print(f"[+] Using existing angles file: {ANGLES_NPZ}\n")
    else:
        print(f"\n[*] STAGE 1: Skipping Angle Predictor training\n")
        if not os.path.exists(ANGLES_NPZ):
            print(f"[!] Angles file not found. Cannot skip - must extract angles first")
            return
        else:
            print(f"[+] Using existing angles file: {ANGLES_NPZ}\n")

    if os.path.exists(ANGLES_NPZ):
        angle_dataset = AngleDataset(ANGLES_NPZ, num_points=1000)

        if len(angle_dataset) == 0:
            print(f"[!] No valid samples in angle dataset")
        else:
            train_size = int(0.8 * len(angle_dataset))
            val_size = len(angle_dataset) - train_size

            train_dataset, val_dataset = torch.utils.data.random_split(
                angle_dataset,
                [train_size, val_size],
                generator=torch.Generator().manual_seed(42)
            )

            def collate_angle_fn(batch):
                batch = [item for item in batch if item[0] is not None]
                if len(batch) == 0:
                    return None, None
                points = torch.stack([item[0] for item in batch])
                angles = torch.stack([item[1] for item in batch])
                return points, angles

            train_loader = DataLoader(
                train_dataset,
                batch_size=4,
                shuffle=True,
                num_workers=0,
                collate_fn=collate_angle_fn
            )
            val_loader = DataLoader(
                val_dataset,
                batch_size=4,
                shuffle=False,
                num_workers=0,
                collate_fn=collate_angle_fn
            )

            print(f"[*] Angle dataset split: {train_size} train, {val_size} val\n")

            angle_model = AnglePredictor(num_points=1000)
            angle_trainer = AnglePredictionTrainer(angle_model, device=device, lr=args.lr, patience=args.patience)

            angle_trainer.train(train_loader, val_loader, epochs=args.epochs, checkpoint_dir=CHECKPOINT_DIR)

            print(f"\n{'='*60}")
            print(f"[+] Stage 1 Training Complete!")
            print(f"[+] Angle predictor trained for rotation estimation")
            print(f"{'='*60}\n")

    # ==================== PHASE 2: Train Per-Vertex Classifier ====================
    if not args.skip_cutting:
        print(f"\n[*] PHASE 2: Training Distance Regressor (Scalar Cutting Distances)")
        print(f"{'='*60}\n")

        # Extract cutting distances if not already extracted
        DISTANCE_NPZ = os.path.join(CHECKPOINT_DIR, "distance_dataset.npz")
        if not os.path.exists(DISTANCE_NPZ):
            print(f"[*] Distance dataset not found. Extracting from dataset...")
            extraction_success = extract_distance_dataset(PLANE_BEFORE, PLANE_AFTER, DISTANCE_NPZ)
            if not extraction_success:
                print(f"[!] Distance extraction failed")
                return
        else:
            print(f"[+] Using existing distance dataset file: {DISTANCE_NPZ}\n")

        if os.path.exists(DISTANCE_NPZ):
            # Load distance dataset
            data = np.load(DISTANCE_NPZ, allow_pickle=True)
            num_samples = len(data['scan_ids'])

            if num_samples == 0:
                print(f"[!] No valid samples in distance dataset")
            else:
                train_size = max(1, int(0.8 * num_samples))
                val_size = num_samples - train_size
                train_indices = list(range(train_size))
                val_indices = list(range(train_size, num_samples))

                print(f"[*] Dataset split: {train_size} train, {val_size} val")
                print(f"[*] Training distance regressor for scalar cutting distances...\n")

                # Collate function for regression
                def collate_distance_fn(batch_indices):
                    batch_points = []
                    batch_distances = []

                    for idx in batch_indices:
                        try:
                            verts = data['vertices_list'][idx]
                            distances = data['distances'][idx]

                            if len(verts) < 10:
                                continue

                            # Sample 1000 points if more available
                            if len(verts) > 1000:
                                sample_idx = np.random.choice(len(verts), 1000, replace=False)
                                verts = verts[sample_idx]

                            points_tensor = torch.from_numpy(verts).float()
                            dist_tensor = torch.from_numpy(distances).float()

                            batch_points.append(points_tensor)
                            batch_distances.append(dist_tensor)
                        except:
                            continue

                    if len(batch_points) == 0:
                        return None, None

                    # Pad to same size
                    max_pts = max([p.shape[0] for p in batch_points])
                    padded_points = []
                    for pts in batch_points:
                        if pts.shape[0] < max_pts:
                            pad_size = max_pts - pts.shape[0]
                            pts = torch.cat([pts, torch.zeros(pad_size, 3)], dim=0)
                        padded_points.append(pts)

                    points = torch.stack(padded_points)[:, :1000, :]
                    distances = torch.stack(batch_distances)
                    return points, distances

                # Train distance regressor
                model = DistanceRegressor(num_points=1000)
                model.to(device)
                optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
                scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.7)
                criterion = nn.MSELoss()

                best_val_loss = float('inf')
                patience_counter = 0

                for epoch in range(args.epochs):
                    # Training
                    model.train()
                    train_loss_total = 0
                    train_count = 0

                    for batch_idx in tqdm(range(0, len(train_indices), 4), desc=f"Epoch {epoch+1}/{args.epochs} Train"):
                        batch_indices = train_indices[batch_idx:batch_idx+4]
                        points, distances = collate_distance_fn(batch_indices)
                        if points is None:
                            continue

                        points = points.to(device)
                        distances = distances.to(device)

                        optimizer.zero_grad()
                        pred_distances = model(points)
                        loss = criterion(pred_distances, distances)
                        loss.backward()
                        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                        optimizer.step()

                        train_loss_total += loss.item()
                        train_count += 1

                    # Validation
                    model.eval()
                    val_loss_total = 0
                    val_count = 0

                    with torch.no_grad():
                        for batch_idx in range(0, len(val_indices), 4):
                            batch_indices = val_indices[batch_idx:batch_idx+4]
                            points, distances = collate_distance_fn(batch_indices)
                            if points is None:
                                continue

                            points = points.to(device)
                            distances = distances.to(device)

                            pred_distances = model(points)
                            loss = criterion(pred_distances, distances)
                            val_loss_total += loss.item()
                            val_count += 1

                    avg_train_loss = train_loss_total / max(1, train_count)
                    avg_val_loss = val_loss_total / max(1, val_count)

                    print(f"Epoch {epoch+1}/{args.epochs} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")

                    if avg_val_loss < best_val_loss:
                        best_val_loss = avg_val_loss
                        patience_counter = 0
                        torch.save(model.state_dict(), os.path.join(CHECKPOINT_DIR, "distance_regressor.pt"))
                        print(f"  [+] Best model saved (MAE: {avg_val_loss:.4f}mm)")
                    else:
                        patience_counter += 1
                        if patience_counter >= args.patience:
                            print(f"[*] Early stopping at epoch {epoch+1}")
                            break

                    scheduler.step()

                print(f"\n{'='*60}")
                print(f"[+] Phase 2 Training Complete!")
                print(f"[+] Distance regressor trained for scalar cutting distances")
                print(f"[+] Model predicts: [Z_cut, X_left, X_right, Y_back] in mm")
                print(f"{'='*60}\n")

    print(f"{'='*60}")
    print(f"[+] Full Training Pipeline Complete!")
    print(f"[+] Order: Phase 1 (PointNet) → Stage 1 (Angle) → Phase 2 (Distance Regression)")
    print(f"[+] Models saved to {CHECKPOINT_DIR}")
    print(f"[+] Phase 2: Scalar regression for flat cutting distances")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
