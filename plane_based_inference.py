"""
Unified Inference Pipeline: Stage 1 (Angle Prediction) + Phase 1 (PointNet) + Phase 2 (Vertex Classifier)
Stage 1: Predicts rotation angle for arch alignment
Phase 1: Classifies and extracts upper/lower arch
Phase 2: Per-vertex classification with plane fitting post-processing for flat cuts
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import trimesh
import numpy as np
import os
from scipy.spatial import cKDTree


# ==================== DISTANCE REGRESSOR ====================

class DistanceRegressor(nn.Module):
    """Predicts 4 cutting distances: [Z_cut, X_left, X_right, Y_back]"""

    def __init__(self, num_points=1000):
        super().__init__()
        self.num_points = num_points

        self.conv1 = nn.Conv1d(3, 64, 1)
        self.conv2 = nn.Conv1d(64, 128, 1)
        self.conv3 = nn.Conv1d(128, 256, 1)
        self.conv4 = nn.Conv1d(256, 512, 1)

        self.fc1 = nn.Linear(512, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc_out = nn.Linear(64, 4)

        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        x = x.transpose(2, 1)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))

        x = torch.max(x, dim=2)[0]

        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = F.relu(self.fc3(x))
        distances = self.fc_out(x)

        return distances


# ==================== DGCNN: Dynamic Graph CNN ====================

class DGCNNVertexClassifier(nn.Module):
    """DGCNN: Dynamic Graph Convolutional Neural Network for per-vertex classification."""

    def __init__(self, num_points=1000, k=20, num_classes=2):
        super().__init__()
        self.num_points = num_points
        self.k = k
        self.num_classes = num_classes

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
        """Extract edge features using GPU-accelerated k-NN with chunking."""
        batch_size, num_points, _ = x.shape
        device = x.device

        all_knn_idx = []
        for b in range(batch_size):
            points = x[b]

            chunk_size = 256
            all_distances = []

            for i in range(0, num_points, chunk_size):
                chunk_end = min(i + chunk_size, num_points)
                chunk = points[i:chunk_end]

                diff = chunk.unsqueeze(1) - points.unsqueeze(0)
                distances = torch.sum(diff ** 2, dim=2)
                all_distances.append(distances)

            distances = torch.cat(all_distances, dim=0)
            _, knn = torch.topk(distances, k + 1, dim=1, largest=False)
            knn = knn[:, 1:]
            all_knn_idx.append(knn)

        knn_idx = torch.stack(all_knn_idx)

        batch_idx = torch.arange(batch_size, device=device).view(batch_size, 1, 1)
        neighbors = x[batch_idx, knn_idx]

        point_expanded = x.unsqueeze(2).expand(-1, -1, k, -1)
        edge_feature = torch.cat([
            neighbors - point_expanded,
            neighbors
        ], dim=3)

        return edge_feature

    def edge_conv(self, x, edge_conv_layer, k):
        edge_feat = self.get_edge_features(x, k)
        edge_feat = edge_feat.permute(0, 3, 1, 2)
        out = edge_conv_layer(edge_feat)
        out = torch.max(out, dim=3)[0]
        out = out.permute(0, 2, 1)
        return out

    def forward(self, x):
        batch_size = x.size(0)

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


# ==================== UTILITIES ====================

def fit_plane_to_points(points):
    """Fit a plane to points using SVD. Returns unit normal and offset."""
    if len(points) < 3:
        return None, None

    centroid = points.mean(axis=0)
    shifted = points - centroid

    U, S, Vt = np.linalg.svd(shifted, full_matrices=False)
    normal = Vt[-1]

    normal_mag = np.linalg.norm(normal)
    if normal_mag > 0:
        normal = normal / normal_mag

    offset_dist = np.dot(centroid, normal)
    return normal, offset_dist


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


# ==================== STAGE 1: Angle Predictor ====================

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
        self.fc_out = nn.Linear(64, 3)  # [angle_x, angle_y, angle_z]

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


# ==================== PHASE 2: Per-Vertex Classifier ====================

# ==================== PHASE 1: PointNet ====================

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


# ==================== PHASE 2: Plane Predictor ====================

class UnifiedInference:
    """Unified inference: Stage 1 (Angle) + Phase 1 (PointNet) + Phase 2 (Per-Vertex Classifier)."""

    def __init__(self, angle_checkpoint, pointnet_checkpoint, vertex_classifier_checkpoint, num_back_cuts=1, device='cuda'):
        self.device = device
        self.num_back_cuts = num_back_cuts

        # Load Angle Predictor (Stage 1)
        self.angle_predictor = AnglePredictor(num_points=1000).to(device)
        if not os.path.exists(angle_checkpoint):
            raise FileNotFoundError(f"Angle predictor checkpoint not found: {angle_checkpoint}")
        self.angle_predictor.load_state_dict(torch.load(angle_checkpoint, map_location=device))
        self.angle_predictor.eval()
        print(f"[+] Angle Predictor loaded from {angle_checkpoint}")

        # Load PointNet (Phase 1)
        self.pointnet = ArchPointNet(num_points=1000, num_classes=3).to(device)
        if not os.path.exists(pointnet_checkpoint):
            raise FileNotFoundError(f"PointNet checkpoint not found: {pointnet_checkpoint}")
        self.pointnet.load_state_dict(torch.load(pointnet_checkpoint, map_location=device))
        self.pointnet.eval()
        print(f"[+] PointNet loaded from {pointnet_checkpoint}")

        # Load Vertex Classifier (Phase 2) - per-vertex keep/remove classification only
        self.distance_regressor = DistanceRegressor(num_points=1000).to(device)
        if not os.path.exists(vertex_classifier_checkpoint):
            raise FileNotFoundError(f"Distance regressor checkpoint not found: {vertex_classifier_checkpoint}")
        self.distance_regressor.load_state_dict(torch.load(vertex_classifier_checkpoint, map_location=device))
        self.distance_regressor.eval()
        print(f"[+] Distance Regressor loaded from {vertex_classifier_checkpoint}")
        print(f"[+] Using device: {device}")

    def classify_mesh(self, mesh_path, num_points=1000):
        """Classify mesh using PointNet (loads from path)."""
        print(f"\n[*] Phase 1: Classifying mesh with PointNet...")
        mesh = trimesh.load(mesh_path, force='mesh', process=False)
        return self._classify_mesh_object(mesh, num_points)

    def _classify_mesh_object(self, mesh, num_points=1000):
        """Classify already-loaded mesh object using PointNet."""
        try:
            sampled_points, _ = trimesh.sample.sample_surface(mesh, num_points)
        except:
            sampled_points = mesh.vertices[:num_points] if len(mesh.vertices) >= num_points else mesh.vertices

        # IMPORTANT: Compute centroid and max_dist from ALL vertices (same as training ground truth extraction)
        # not from sampled points, to ensure coordinate space consistency
        centroid = mesh.vertices.mean(axis=0)
        all_verts_centered = mesh.vertices - centroid
        max_dist = np.max(np.linalg.norm(all_verts_centered, axis=1))

        # Normalize sampled points for PointNet input
        points = sampled_points - centroid
        if max_dist > 0:
            points = points / max_dist

        with torch.no_grad():
            points_tensor = torch.from_numpy(points).unsqueeze(0).float().to(self.device)
            logits = self.pointnet(points_tensor)
            probs = F.softmax(logits, dim=2)
            sampled_preds = torch.argmax(probs, dim=2)[0].cpu().numpy()

        # Normalize all mesh vertices using same centroid and max_dist
        points_normalized = (mesh.vertices - centroid) / max_dist if max_dist > 0 else mesh.vertices - centroid
        sampled_points_norm = (sampled_points - centroid) / max_dist if max_dist > 0 else sampled_points - centroid

        tree = cKDTree(sampled_points_norm)
        _, nearest_indices = tree.query(points_normalized)
        all_predictions = sampled_preds[nearest_indices]

        return all_predictions, mesh, centroid, max_dist

    def extract_upper_mesh(self, mesh, predictions):
        """Extract upper arch from predictions."""
        upper_mask = predictions == 0

        if not np.any(upper_mask):
            print(f"[!] No upper vertices found")
            return None

        upper_faces = []
        for face in mesh.faces:
            if np.all(upper_mask[face]):
                upper_faces.append(face)

        if len(upper_faces) == 0:
            print(f"[!] No complete upper faces")
            return None

        upper_faces = np.array(upper_faces, dtype=np.int64)
        unique_vertices = np.unique(upper_faces.flatten())

        vertex_map = {old_idx: new_idx for new_idx, old_idx in enumerate(unique_vertices)}
        remapped_faces = np.array([[vertex_map[v] for v in face] for face in upper_faces])

        upper_mesh = trimesh.Trimesh(
            vertices=mesh.vertices[unique_vertices],
            faces=remapped_faces,
            process=False
        )

        print(f"[+] Upper arch extracted: {len(upper_mesh.vertices)} vertices, {len(upper_mesh.faces)} faces")
        return upper_mesh

    def extract_lower_mesh(self, mesh, predictions):
        """Extract lower arch from predictions."""
        lower_mask = predictions == 1

        if not np.any(lower_mask):
            print(f"[!] No lower vertices found")
            return None

        lower_faces = []
        for face in mesh.faces:
            if np.all(lower_mask[face]):
                lower_faces.append(face)

        if len(lower_faces) == 0:
            print(f"[!] No complete lower faces")
            return None

        lower_faces = np.array(lower_faces, dtype=np.int64)
        unique_vertices = np.unique(lower_faces.flatten())

        vertex_map = {old_idx: new_idx for new_idx, old_idx in enumerate(unique_vertices)}
        remapped_faces = np.array([[vertex_map[v] for v in face] for face in lower_faces])

        lower_mesh = trimesh.Trimesh(
            vertices=mesh.vertices[unique_vertices],
            faces=remapped_faces,
            process=False
        )

        print(f"[+] Lower arch extracted: {len(lower_mesh.vertices)} vertices, {len(lower_mesh.faces)} faces")
        return lower_mesh

    def extract_mesh_features(self, mesh):
        """Extract mesh features for plane predictor input."""
        bounds = mesh.bounds
        verts = mesh.vertices

        features = np.array([
            len(mesh.vertices),
            len(mesh.faces),
            mesh.volume if hasattr(mesh, 'volume') else 0.0,
            mesh.area if hasattr(mesh, 'area') else 0.0,
            bounds[1][0] - bounds[0][0],
            bounds[1][1] - bounds[0][1],
            bounds[1][2] - bounds[0][2],
            np.std(verts[:, 0]),
            np.std(verts[:, 1]),
            np.std(verts[:, 2]),
        ], dtype=np.float32)

        features_normalized = (features - features.mean()) / (features.std() + 1e-8)
        return features_normalized

    def classify_vertices(self, mesh, centroid, max_dist):
        """Per-vertex keep/remove classification + plane parameter prediction."""
        print(f"[*] Phase 2: Classifying vertices and predicting cutting plane...")

        verts = mesh.vertices
        verts_centered = verts - centroid
        verts_norm = verts_centered / max_dist if max_dist > 0 else verts_centered

        # Classify all vertices in batches (must match training batch size of 1000)
        batch_size = 1000
        full_preds = np.zeros(len(verts), dtype=np.int32)
        plane_predictions = []

        with torch.no_grad():
            for i in range(0, len(verts), batch_size):
                batch_end = min(i + batch_size, len(verts))
                batch_verts = verts_norm[i:batch_end]

                # Pad batch to exactly 1000 if needed
                if len(batch_verts) < batch_size:
                    padding = np.zeros((batch_size - len(batch_verts), 3), dtype=np.float32)
                    batch_verts = np.vstack([batch_verts, padding])
                    actual_length = batch_end - i
                else:
                    actual_length = batch_size

                verts_tensor = torch.from_numpy(batch_verts).unsqueeze(0).float().to(self.device)
                output = self.vertex_classifier(verts_tensor)

                # Handle dual-output (classification + plane prediction)
                if isinstance(output, tuple):
                    class_logits, plane_logits = output
                    probs = F.softmax(class_logits, dim=2)
                    preds = torch.argmax(probs, dim=2)[0].cpu().numpy()  # (1000,)
                    plane_pred = plane_logits[0].cpu().numpy()  # (4,)
                    plane_predictions.append(plane_pred)
                else:
                    class_logits = output
                    probs = F.softmax(class_logits, dim=2)
                    preds = torch.argmax(probs, dim=2)[0].cpu().numpy()

                # Only keep predictions for actual vertices (discard padding)
                full_preds[i:batch_end] = preds[:actual_length]

        keep_mask = full_preds == 1
        remove_mask = full_preds == 0

        print(f"[+] Classification: {keep_mask.sum()} kept, {remove_mask.sum()} removed")

        # Average plane predictions across batches
        predicted_plane = None
        if plane_predictions:
            predicted_plane = np.mean(plane_predictions, axis=0)
            # Normalize the normal vector
            predicted_plane[:3] = predicted_plane[:3] / (np.linalg.norm(predicted_plane[:3]) + 1e-8)
            print(f"[+] Predicted plane normal: {predicted_plane[:3]}, offset: {predicted_plane[3]:.4f}")

        return keep_mask, predicted_plane

    def classify_vertices_for_trimming(self, mesh, centroid, max_dist):
        """Phase 2: Classify each vertex as keep (1) or remove (0) based on learned geometry.

        Returns: (keep_mask, vertex_probabilities)
        """
        print(f"\n[*] Phase 2: Classifying vertices for geometry-based trimming...")

        vertices = mesh.vertices

        batch_size = 1000
        keep_probs = np.zeros(len(vertices), dtype=np.float32)

        with torch.no_grad():
            for i in range(0, len(vertices), batch_size):
                batch_end = min(i + batch_size, len(vertices))
                batch_verts = vertices[i:batch_end]

                # Pad to 1000 if needed
                if len(batch_verts) < batch_size:
                    padding = np.zeros((batch_size - len(batch_verts), 3), dtype=np.float32)
                    batch_verts_padded = np.vstack([batch_verts, padding])
                    actual_length = batch_end - i
                else:
                    batch_verts_padded = batch_verts
                    actual_length = batch_size

                verts_tensor = torch.from_numpy(batch_verts_padded).unsqueeze(0).float().to(self.device)
                logits = self.vertex_classifier(verts_tensor)[0]  # (1000, 2)
                probs = F.softmax(logits, dim=1)  # (1000, 2)
                keep_prob = probs[:, 1].cpu().numpy()  # Probability of keeping

                keep_probs[i:batch_end] = keep_prob[:actual_length]

        # Threshold at 0.5
        keep_mask = keep_probs > 0.5

        kept_count = np.sum(keep_mask)
        removed_count = len(keep_mask) - kept_count
        print(f"[+] Vertex classification: {kept_count} kept, {removed_count} removed")
        print(f"[+] Average keep probability: {keep_probs.mean():.3f}")

        return keep_mask, keep_probs

    def fit_planes_to_trimming_regions(self, mesh, keep_mask, centroid, max_dist):
        """Fit planes to boundaries between kept and removed regions for clean, flat cuts."""
        print(f"\n[*] Fitting planes to trimming boundaries for flat cuts...")

        vertices = mesh.vertices
        faces = mesh.faces
        verts_norm = (vertices - centroid) / max_dist if max_dist > 0 else vertices - centroid

        removed_mask = ~keep_mask
        if np.sum(removed_mask) < 10:
            print(f"[!] Insufficient removed vertices for plane fitting")
            return None

        removed_verts = verts_norm[removed_mask]

        # Fit plane to removed vertices (this is the cutting surface)
        if len(removed_verts) >= 3:
            # Use SVD to fit plane to removed vertices
            centroid_removed = removed_verts.mean(axis=0)
            shifted = removed_verts - centroid_removed

            U, S, Vt = np.linalg.svd(shifted, full_matrices=False)
            plane_normal = Vt[-1]

            # Normalize
            plane_normal = plane_normal / (np.linalg.norm(plane_normal) + 1e-8)

            # Ensure normal points toward kept material
            kept_verts = verts_norm[keep_mask]
            kept_mean = kept_verts.mean(axis=0)
            if np.dot(plane_normal, kept_mean) < np.dot(plane_normal, removed_verts.mean(axis=0)):
                plane_normal = -plane_normal

            offset = np.dot(centroid_removed, plane_normal)
            print(f"[+] Fitted plane normal: [{plane_normal[0]:.3f}, {plane_normal[1]:.3f}, {plane_normal[2]:.3f}]")

            # Apply plane cut
            distances_from_plane = np.dot(verts_norm, plane_normal) - offset
            final_keep_mask = distances_from_plane >= -0.5  # Small tolerance for numerical stability

            return final_keep_mask
        else:
            return keep_mask

    def apply_vertex_classification_trimming(self, mesh, keep_mask):
        """Extract mesh containing only vertices marked as keep."""
        print(f"\n[*] Extracting trimmed mesh from vertex classification...")

        vertices = mesh.vertices
        faces = mesh.faces

        # Extract kept vertices and faces
        kept_indices = np.where(keep_mask)[0]
        if len(kept_indices) == 0:
            print(f"[!] No vertices kept after classification")
            return mesh.copy()

        # Remap face indices
        index_map = {old_idx: new_idx for new_idx, old_idx in enumerate(kept_indices)}
        kept_faces = []
        for face in faces:
            if all(v in index_map for v in face):
                kept_faces.append([index_map[v] for v in face])

        if len(kept_faces) == 0:
            print(f"[!] No faces remain after extraction")
            return mesh.copy()

        kept_faces = np.array(kept_faces, dtype=np.int64)

        # Create result mesh
        result_mesh = trimesh.Trimesh(
            vertices=vertices[kept_indices],
            faces=kept_faces,
            process=False
        )

        print(f"[+] Generated retainer: {len(result_mesh.vertices)} vertices, {len(result_mesh.faces)} faces")
        return result_mesh

    def fit_planes_to_boundary(self, mesh, keep_mask, centroid, max_dist, predicted_plane=None):
        """Apply cutting plane: use predicted plane if available, otherwise fit to boundary."""
        if predicted_plane is not None:
            print(f"[*] Using predicted cutting plane...")
            normal = predicted_plane[:3]
            offset = predicted_plane[3]
            print(f"[+] Predicted plane: [{normal[0]:.4f}, {normal[1]:.4f}, {normal[2]:.4f}] @ {offset:.4f}")
        else:
            print(f"[*] Fitting planes to boundary for flat cuts...")
            removed_indices = np.where(~keep_mask)[0]
            if len(removed_indices) < 15:
                print(f"[!] Not enough removed vertices for plane fitting")
                return None

            removed_verts = mesh.vertices[removed_indices]
            removed_verts_norm = (removed_verts - centroid) / max_dist if max_dist > 0 else removed_verts - centroid

            # Fit plane to removed vertices - this approximates the cutting surface
            normal, offset = fit_plane_to_points(removed_verts_norm)
            if normal is None:
                print(f"[!] Could not fit plane to boundary")
                return None

            # Ensure normal points toward removed material (typically downward/outward)
            removed_mean = removed_verts_norm.mean(axis=0)
            if np.dot(normal, removed_mean) < 0:
                normal = -normal
                offset = -offset

            print(f"[+] Fitted plane: [{normal[0]:.4f}, {normal[1]:.4f}, {normal[2]:.4f}] @ {offset:.4f}")

        # Apply the fitted plane cut
        vertices_normalized = (mesh.vertices - centroid) / max_dist if max_dist > 0 else mesh.vertices - centroid
        plane_mask = np.dot(vertices_normalized, normal) >= offset
        keep_count = plane_mask.sum()
        print(f"[+] Plane cut: {keep_count}/{len(mesh.vertices)} vertices kept")

        # Extract faces where all vertices are kept
        keep_faces = []
        for face in mesh.faces:
            if np.all(plane_mask[face]):
                keep_faces.append(face)

        if len(keep_faces) == 0:
            print(f"[!] No faces remain after plane fitting")
            return None

        keep_faces = np.array(keep_faces, dtype=np.int64)
        unique_vertices = np.unique(keep_faces.flatten())

        vertex_map = {old_idx: new_idx for new_idx, old_idx in enumerate(unique_vertices)}
        remapped_faces = np.array([[vertex_map[v] for v in face] for face in keep_faces])

        result_mesh = trimesh.Trimesh(
            vertices=mesh.vertices[unique_vertices],
            faces=remapped_faces,
            process=False
        )

        print(f"[+] Result mesh: {len(result_mesh.vertices)} vertices, {len(result_mesh.faces)} faces")
        return result_mesh

    def predict_angle(self, mesh, num_points=1000):
        """Stage 1: Predict Euler angles for the mesh."""
        print(f"\n[*] Stage 1: Predicting rotation angles...")

        # Sample points from mesh
        try:
            sampled_points, _ = trimesh.sample.sample_surface(mesh, num_points)
        except:
            sampled_points = mesh.vertices[:num_points] if len(mesh.vertices) >= num_points else mesh.vertices

        # Normalize using mesh centroid and max distance
        centroid = mesh.vertices.mean(axis=0)
        all_verts_centered = mesh.vertices - centroid
        max_dist = np.max(np.linalg.norm(all_verts_centered, axis=1))

        points = sampled_points - centroid
        if max_dist > 0:
            points = points / max_dist

        # Predict Euler angles
        with torch.no_grad():
            points_tensor = torch.from_numpy(points).unsqueeze(0).float().to(self.device)
            predictions = self.angle_predictor(points_tensor)[0].cpu().numpy()

        # Extract angles (in radians)
        angle_x = predictions[0]
        angle_y = predictions[1]
        angle_z = predictions[2]

        angle_x_deg = np.degrees(angle_x)
        angle_y_deg = np.degrees(angle_y)
        angle_z_deg = np.degrees(angle_z)

        print(f"[+] Predicted Euler angles:")
        print(f"    X (pitch): {angle_x_deg:7.2f}°")
        print(f"    Y (roll):  {angle_y_deg:7.2f}°")
        print(f"    Z (yaw):   {angle_z_deg:7.2f}°")

        return angle_x, angle_y, angle_z, angle_x_deg, angle_y_deg, angle_z_deg

    def apply_angle_to_mesh(self, mesh, angle_x, angle_y, angle_z):
        """Apply predicted Euler rotation to mesh vertices."""
        # Check if all angles are near-zero
        if abs(angle_x) < 0.01 and abs(angle_y) < 0.01 and abs(angle_z) < 0.01:
            print(f"[*] All angles near-zero, no rotation applied")
            return mesh

        print(f"\n[*] Applying rotation to mesh...")
        R = euler_angles_to_rotation_matrix(angle_x, angle_y, angle_z)

        rotated_verts = mesh.vertices @ R.T
        rotated_mesh = trimesh.Trimesh(vertices=rotated_verts, faces=mesh.faces, process=False)

        print(f"[+] Rotation applied successfully")
        return rotated_mesh

    def predict_cutting_distances(self, mesh, num_points=1000):
        """Predict 4 cutting distances using distance regressor."""
        try:
            sampled_points, _ = trimesh.sample.sample_surface(mesh, num_points)
        except:
            sampled_points = mesh.vertices[:num_points] if len(mesh.vertices) >= num_points else mesh.vertices

        points_tensor = torch.from_numpy(sampled_points).unsqueeze(0).float().to(self.device)

        with torch.no_grad():
            distances = self.distance_regressor(points_tensor)[0].cpu().numpy()  # (4,)

        return np.maximum(distances, 0)  # Ensure non-negative

    def apply_bbox_cuts(self, mesh, distances):
        """Apply 4 bbox-based flat cuts: [Z_cut, X_left, X_right, Y_back]"""
        z_cut, x_left, x_right, y_back = distances
        bounds = mesh.bounds

        # Keep vertices within bounds minus cuts
        x_min, y_min, z_min = bounds[0]
        x_max, y_max, z_max = bounds[1]

        keep_mask = (
            (mesh.vertices[:, 0] >= x_min + x_left) &
            (mesh.vertices[:, 0] <= x_max - x_right) &
            (mesh.vertices[:, 1] <= y_max - y_back) &
            (mesh.vertices[:, 2] <= z_max - z_cut)
        )

        if not np.any(keep_mask):
            print(f"[!] No vertices remain after cutting")
            return None

        # Extract faces where all vertices are kept
        kept_faces = []
        for face in mesh.faces:
            if np.all(keep_mask[face]):
                kept_faces.append(face)

        if len(kept_faces) == 0:
            print(f"[!] No faces remain after cutting")
            return None

        kept_faces = np.array(kept_faces, dtype=np.int64)
        unique_vertices = np.unique(kept_faces.flatten())

        vertex_map = {old_idx: new_idx for new_idx, old_idx in enumerate(unique_vertices)}
        remapped_faces = np.array([[vertex_map[v] for v in face] for face in kept_faces])

        result_mesh = trimesh.Trimesh(
            vertices=mesh.vertices[unique_vertices],
            faces=remapped_faces,
            process=False
        )

        print(f"[+] Applied cuts: {len(result_mesh.vertices)} vertices, {len(result_mesh.faces)} faces remain")
        return result_mesh

    def generate_retainer_from_combined(self, combined_mesh_path, output_dir):
        """Process combined mesh (upper + lower) using PointNet segmentation.

        Args:
            combined_mesh_path: Path to combined mesh file
            output_dir: Output directory

        Returns:
            (upper_result_path, lower_result_path)
        """
        print(f"\n{'='*60}")
        print(f"[*] Segmenting combined mesh with PointNet...")
        print(f"{'='*60}\n")

        # Load combined mesh
        mesh = trimesh.load(combined_mesh_path, force='mesh', process=False)

        # Phase 1: Classify with PointNet
        print(f"  [*] Phase 1: Classifying with PointNet...")
        predictions, mesh, _, _ = self._classify_mesh_object(mesh)

        upper_mesh = self.extract_upper_mesh(mesh, predictions)
        lower_mesh = self.extract_lower_mesh(mesh, predictions)

        upper_result = None
        lower_result = None

        # Process upper arch
        if upper_mesh is not None:
            print(f"\n[*] Processing upper arch:")
            upper_result = self._process_arch(upper_mesh, output_dir, arch_type="upper")

        # Process lower arch
        if lower_mesh is not None:
            print(f"\n[*] Processing lower arch:")
            lower_result = self._process_arch(lower_mesh, output_dir, arch_type="lower")

        return upper_result, lower_result

    def _process_arch(self, mesh, output_dir, arch_type="upper"):
        """Process single arch through angle prediction and distance regression.

        Args:
            mesh: Trimesh object
            output_dir: Output directory
            arch_type: "upper" or "lower"

        Returns:
            Output path or None
        """
        print(f"  [*] Predicting angles...")

        # Stage 1: Predict angles
        angle_x, angle_y, angle_z, x_deg, y_deg, z_deg = self.predict_angle(mesh)
        print(f"  [+] Angles: X={x_deg:7.2f}°, Y={y_deg:6.2f}°, Z={z_deg:6.2f}°")

        # Apply rotation first to align geometry
        aligned_mesh = self.apply_angle_to_mesh(mesh, angle_x, angle_y, angle_z)

        # Phase 2: Predict cutting distances on aligned mesh
        print(f"  [*] Predicting cutting distances on aligned mesh...")
        distances = self.predict_cutting_distances(aligned_mesh)
        z_cut, x_left, x_right, y_back = distances
        print(f"  [+] Predicted distances: Z={z_cut:.1f}mm, X_L={x_left:.1f}mm, X_R={x_right:.1f}mm, Y_B={y_back:.1f}mm")

        # Apply flat cuts to aligned mesh (produces flat top)
        retainer = self.apply_bbox_cuts(aligned_mesh, distances)

        # Save
        os.makedirs(output_dir, exist_ok=True)

        if retainer is not None:
            output_path = os.path.join(output_dir, f"{arch_type}_retainer.stl")
            retainer.export(output_path)
            print(f"  [+] {arch_type.capitalize()} retainer saved to {os.path.basename(output_path)}")
            return output_path
        else:
            print(f"  [!] Failed to generate {arch_type} retainer")
            return None

    def generate_retainer(self, input_mesh_path, output_dir, apply_rotation=True, arch_type="upper"):
        """Generate retainer from pre-separated upper or lower mesh (no PointNet needed).

        Args:
            input_mesh_path: Path to upper or lower arch STL
            output_dir: Output directory
            apply_rotation: Whether to apply angle prediction
            arch_type: "upper" or "lower"
        """
        # Load mesh directly (already separated)
        mesh = trimesh.load(input_mesh_path, force='mesh', process=False)

        # Stage 1: Predict angles and apply rotation first
        aligned_mesh = mesh
        if apply_rotation:
            print(f"  [*] Predicting angles...")
            angle_x, angle_y, angle_z, x_deg, y_deg, z_deg = self.predict_angle(mesh)
            print(f"  [+] Angles: X={x_deg:7.2f}°, Y={y_deg:6.2f}°, Z={z_deg:6.2f}°")
            aligned_mesh = self.apply_angle_to_mesh(mesh, angle_x, angle_y, angle_z)
        else:
            print(f"  [*] Skipping angle prediction")

        # Phase 2: Predict cutting distances on aligned mesh
        print(f"  [*] Predicting cutting distances...")
        distances = self.predict_cutting_distances(aligned_mesh)
        z_cut, x_left, x_right, y_back = distances
        print(f"  [+] Predicted distances: Z={z_cut:.1f}mm, X_L={x_left:.1f}mm, X_R={x_right:.1f}mm, Y_B={y_back:.1f}mm")

        # Apply flat cuts to aligned mesh
        retainer = self.apply_bbox_cuts(aligned_mesh, distances)

        # Save
        os.makedirs(output_dir, exist_ok=True)

        if retainer is not None:
            output_path = os.path.join(output_dir, f"{arch_type}_retainer.stl")
            retainer.export(output_path)
            print(f"  [+] {arch_type.capitalize()} retainer saved to {os.path.basename(output_path)}")
            return output_path
        else:
            print(f"  [!] Failed to generate {arch_type} retainer")
            return None


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Unified Inference Pipeline (Combined Mesh → Upper + Lower)")
    parser.add_argument("--input", type=str, default="/home/garvb/Downloads/Inference Data/scan62_before.stl", help="Combined mesh path (with upper + lower)")
    parser.add_argument("--output", type=str, default="/home/garvb/Downloads/plane_inference_results", help="Output directory")
    parser.add_argument("--checkpoint-dir", type=str, default="/home/garvb/AILabProject/checkpoints", help="Checkpoint directory")
    parser.add_argument("--back-cuts", type=int, default=1, choices=[1, 2], help="Number of back cuts")
    parser.add_argument("--device", type=str, default='cuda' if torch.cuda.is_available() else 'cpu', help="Device")

    args = parser.parse_args()

    # Verify input file exists
    if not os.path.exists(args.input):
        print(f"[!] Input mesh not found: {args.input}")
        return

    ANGLE_CHECKPOINT = os.path.join(args.checkpoint_dir, "angle_predictor.pt")
    POINTNET_CHECKPOINT = os.path.join(args.checkpoint_dir, "arch_classifier.pt")
    DISTANCE_REGRESSOR_CHECKPOINT = os.path.join(args.checkpoint_dir, "distance_regressor.pt")

    print(f"{'='*60}")
    print(f"[*] Unified Inference Pipeline: Combined Mesh → Upper + Lower Retainers")
    print(f"[*] Phase 1: PointNet segmentation (upper/lower separation)")
    print(f"[*] Stage 1: Angle prediction for rotation alignment")
    print(f"[*] Phase 2: Distance regression for scalar cutting distances")
    print(f"{'='*60}\n")

    os.makedirs(args.output, exist_ok=True)

    inference = UnifiedInference(ANGLE_CHECKPOINT, POINTNET_CHECKPOINT, DISTANCE_REGRESSOR_CHECKPOINT,
                                  num_back_cuts=args.back_cuts, device=args.device)

    print(f"[*] Input: {os.path.basename(args.input)}")
    upper_result, lower_result = inference.generate_retainer_from_combined(args.input, args.output)

    if upper_result is not None or lower_result is not None:
        print(f"\n[+] Pipeline completed successfully!")
        if upper_result:
            print(f"[+] Upper retainer: {upper_result}")
        if lower_result:
            print(f"[+] Lower retainer: {lower_result}")
    else:
        print(f"\n[!] Pipeline failed")


if __name__ == "__main__":
    main()
