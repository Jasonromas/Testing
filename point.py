import random
from typing import BinaryIO, Dict, List, Optional, Union

import numpy as np
import torch

from  .camera import extract_camera_params


def get_point_clouds(cameras, depths, alphas, rgbs=None):
    """
    Generates a 3D point cloud from camera parameters, depth maps, and optional RGB colors.

    Args:
        cameras: Camera intrinsics and extrinsics.
        depths: Depth maps of shape (N, H, W), where N is the number of images.
        alphas: Binary mask indicating valid depth points.
        rgbs: Optional RGB color values corresponding to depth points.

    Returns:
        PointCloud: A structured point cloud representation with 3D coordinates and color information.
    """
    Hs, Ws, intrinsics, c2ws = extract_camera_params(cameras)
    W, H = int(Ws[0].item()), int(Hs[0].item())
    assert (depths.shape == alphas.shape)
    coords = []
    rgbas = []

    # TODO: Compute ray origins and directions for each pixel
    # Hint: You need to use the camera intrinsics (intrinsics) and extrinsics (c2ws)
    # to convert pixel coordinates into world-space rays.
    # rays_o, rays_d = ......

    # TODO: Compute 3D world coordinates using depth values
    # Hint: Use the ray equation: P = O + D * depth
    # P: 3D point, O: ray origin, D: ray direction, depth: depth value
    # pts = ......

    # TODO: Apply the alpha mask to filter valid points
    # Hint: Mask should be applied to both coordinates and RGB values (if provided)
    # mask = ......
    # coords = pts[mask].cpu().numpy()

    for i in range(len(cameras)):
        intrinsic = intrinsics[i]
        c2w = c2ws[i]
        depth = depths[i]  # (H, W)
        alpha = alphas[i]  # (H, W)

        # Step 1: Get pixel grid in image space
        ys, xs = torch.meshgrid(
            torch.arange(H, dtype=torch.float32, device=depth.device),
            torch.arange(W, dtype=torch.float32, device=depth.device),
            indexing="ij"
        )
        ones = torch.ones_like(xs)
        pixels_h = torch.stack([xs, ys, ones, ones], dim=-1).reshape(-1, 4)  # (H*W, 4)

        # Step 2: Invert intrinsics to get directions in camera space
        inv_K = torch.linalg.inv(intrinsic)
        rays_d_cam = (inv_K @ pixels_h.T).T[:, :3]  # (H*W, 3)

        # Step 3: Transform directions and origins to world space
        rays_d_world = (c2w[:3, :3] @ rays_d_cam.T).T  # Rotate direction
        rays_o_world = c2w[:3, 3].expand_as(rays_d_world)  # Origin is same for all rays

        # Step 4: Compute 3D points in world space
        depth_flat = depth.reshape(-1, 1)
        points = rays_o_world + rays_d_world * depth_flat  # (H*W, 3)

        # Step 5: Apply alpha mask
        mask = alpha.reshape(-1) > 0
        valid_points = points[mask]
        coords.append(valid_points)

        # Step 6: Handle RGB (if provided)
        if rgbs is not None:
            rgb = rgbs[i].reshape(-1, 3)
            rgba = torch.cat([rgb, alpha.reshape(-1, 1)], dim=-1)  # (H*W, 4)
            rgbas.append(rgba[mask])

    coords = torch.cat(coords, dim=0).cpu().numpy()

    if rgbs is not None:
        rgbas = torch.cat(rgbas, dim=0).cpu().numpy()
        channels = dict(
            R=rgbas[..., 0],
            G=rgbas[..., 1],
            B=rgbas[..., 2],
            A=rgbas[..., 3],
        )
    else:
        channels = {}

    point_cloud = PointCloud(coords, channels)
    return point_cloud


def preprocess(data, channel):
    if channel in ["R", "G", "B", "A"]:
        return np.round(data * 255.0)
    return data


class PointCloud:
    def __init__(self, coords: np.ndarray, channels: Dict[str, np.ndarray]) -> None:
        self.coords = coords
        self.channels = channels

    def __repr__(self) -> str:
        str = f"coords:{len(self.coords)} \t channels:{list(self.channels.keys())}"
        return str

    def random_sample(self, num_points: int, **subsample_kwargs) -> "PointCloud":
        """
        Sample a random subset of this PointCloud.

        :param num_points: maximum number of points to sample.
        :param subsample_kwargs: arguments to self.subsample().
        :return: a reduced PointCloud, or self if num_points is not less than
                 the current number of points.
        """
        if len(self.coords) <= num_points:
            return self
        indices = np.random.choice(len(self.coords), size=(num_points,), replace=False)
        return self.subsample(indices, **subsample_kwargs)

    @classmethod
    def load(cls, f: Union[str, BinaryIO]) -> "PointCloud":
        """
        Load the point cloud from a .npz file.
        """
        if isinstance(f, str):
            with open(f, "rb") as reader:
                return cls.load(reader)
        else:
            obj = np.load(f)
            keys = list(obj.keys())
            return PointCloud(
                coords=obj["coords"],
                channels={k: obj[k] for k in keys if k != "coords"},
            )

    def save(self, f: Union[str, BinaryIO]):
        """
        Save the point cloud to a .npz file.
        """
        if isinstance(f, str):
            with open(f, "wb") as writer:
                self.save(writer)
        else:
            np.savez(f, coords=self.coords, **self.channels)

    def farthest_point_sample(
        self, num_points: int, init_idx: Optional[int] = None, **subsample_kwargs
    ) -> "PointCloud":
        """
        Sample a subset of the point cloud that is evenly distributed in space.

        First, a random point is selected. Then each successive point is chosen
        such that it is furthest from the currently selected points.

        The time complexity of this operation is O(NM), where N is the original
        number of points and M is the reduced number. Therefore, performance
        can be improved by randomly subsampling points with random_sample()
        before running farthest_point_sample().

        :param num_points: maximum number of points to sample.
        :param init_idx: if specified, the first point to sample.
        :param subsample_kwargs: arguments to self.subsample().
        :return: a reduced PointCloud, or self if num_points is not less than
                 the current number of points.
        """
        if len(self.coords) <= num_points:
            return self
        init_idx = random.randrange(len(self.coords)) if init_idx is None else init_idx
        indices = np.zeros([num_points], dtype=np.int64)
        indices[0] = init_idx
        sq_norms = np.sum(self.coords**2, axis=-1)

        def compute_dists(idx: int):
            # Utilize equality: ||A-B||^2 = ||A||^2 + ||B||^2 - 2*(A @ B).
            return sq_norms + sq_norms[idx] - 2 * (self.coords @ self.coords[idx])

        cur_dists = compute_dists(init_idx)
        for i in range(1, num_points):
            idx = np.argmax(cur_dists)
            indices[i] = idx
            cur_dists = np.minimum(cur_dists, compute_dists(idx))
        return self.subsample(indices, **subsample_kwargs)

    def subsample(self, indices: np.ndarray, average_neighbors: bool = False) -> "PointCloud":
        if not average_neighbors:
            return PointCloud(
                coords=self.coords[indices],
                channels={k: v[indices] for k, v in self.channels.items()},
            )

        new_coords = self.coords[indices]
        neighbor_indices = PointCloud(coords=new_coords, channels={}).nearest_points(self.coords)

        # Make sure every point points to itself, which might not
        # be the case if points are duplicated or there is rounding
        # error.
        neighbor_indices[indices] = np.arange(len(indices))

        new_channels = {}
        for k, v in self.channels.items():
            v_sum = np.zeros_like(v[: len(indices)])
            v_count = np.zeros_like(v[: len(indices)])
            np.add.at(v_sum, neighbor_indices, v)
            np.add.at(v_count, neighbor_indices, 1)
            new_channels[k] = v_sum / v_count
        return PointCloud(coords=new_coords, channels=new_channels)

    def select_channels(self, channel_names: List[str]) -> np.ndarray:
        data = np.stack([preprocess(self.channels[name], name) for name in channel_names], axis=-1)
        return data

    def nearest_points(self, points: np.ndarray, batch_size: int = 16384) -> np.ndarray:
        """
        For each point in another set of points, compute the point in this
        pointcloud which is closest.

        :param points: an [N x 3] array of points.
        :param batch_size: the number of neighbor distances to compute at once.
                           Smaller values save memory, while larger values may
                           make the computation faster.
        :return: an [N] array of indices into self.coords.
        """
        norms = np.sum(self.coords**2, axis=-1)
        all_indices = []
        for i in range(0, len(points), batch_size):
            batch = points[i : i + batch_size]
            dists = norms + np.sum(batch**2, axis=-1)[:, None] - 2 * (batch @ self.coords.T)
            all_indices.append(np.argmin(dists, axis=-1))
        return np.concatenate(all_indices, axis=0)

    def combine(self, other: "PointCloud") -> "PointCloud":
        assert self.channels.keys() == other.channels.keys()
        return PointCloud(
            coords=np.concatenate([self.coords, other.coords], axis=0),
            channels={
                k: np.concatenate([v, other.channels[k]], axis=0) for k, v in self.channels.items()
            },
        )
