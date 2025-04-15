import math
import torch
from torch import nn


def extract_camera_params(params):
    Hs = params[:, 0]
    Ws = params[:, 1]
    intrinsics = params[:, 2:18].reshape((-1, 4, 4))
    c2ws = params[:, 18:34].reshape((-1, 4, 4))
    return Hs, Ws, intrinsics, c2ws


def to_viewpoint_camera(camera):
    Hs, Ws, intrinsics, c2ws = extract_camera_params(camera.unsqueeze(0))
    return Camera(
        image_width=int(Ws[0]),
        image_height=int(Hs[0]),
        intrinsic_matrix=intrinsics[0],
        camera_to_world=c2ws[0]
    )


class Camera(nn.Module):
    def __init__(
        self,
        image_width: int,
        image_height: int,
        intrinsic_matrix: torch.Tensor,
        camera_to_world: torch.Tensor,
        near_plane: float = 0.1,
        far_plane: float = 100.0,
    ):
        super().__init__()

        device = camera_to_world.device

        self.image_width = int(image_width)
        self.image_height = int(image_height)
        self.near = near_plane
        self.far = far_plane

        self.focal_x = intrinsic_matrix[0, 0]
        self.focal_y = intrinsic_matrix[1, 1]

        self.fov_x = self.focal_to_fov(self.focal_x, self.image_width)
        self.fov_y = self.focal_to_fov(self.focal_y, self.image_height)

        self.intrinsic = intrinsic_matrix
        self.c2w = camera_to_world
        self.world_to_view = torch.linalg.inv(camera_to_world).T  # 4x4 matrix

        self.projection = self._build_projection_matrix(
            self.near, self.far, self.fov_x, self.fov_y
        ).T.to(device)
        self.world_view_projection = self.world_to_view @ self.projection

        self.camera_center = torch.linalg.inv(self.world_to_view)[3, :3]

    @staticmethod
    def focal_to_fov(focal_length: float, sensor_size_px: int) -> float:
        return 2 * math.atan(sensor_size_px / (2 * focal_length))

    @staticmethod
    def fov_to_focal(fov: float, sensor_size_px: int) -> float:
        return sensor_size_px / (2 * math.tan(fov / 2))

    @staticmethod
    def _build_projection_matrix(near: float, far: float, fov_x: float, fov_y: float) -> torch.Tensor:
        tan_half_fov_y = math.tan(fov_y / 2)
        tan_half_fov_x = math.tan(fov_x / 2)

        top = tan_half_fov_y * near
        bottom = -top
        right = tan_half_fov_x * near
        left = -right

        P = torch.zeros(4, 4)
        z_sign = 1.0  # Used in OpenGL-style projection

        P[0, 0] = 2 * near / (right - left)
        P[1, 1] = 2 * near / (top - bottom)
        P[0, 2] = (right + left) / (right - left)
        P[1, 2] = (top + bottom) / (top - bottom)
        P[2, 2] = z_sign * far / (far - near)
        P[2, 3] = -(far * near) / (far - near)
        P[3, 2] = z_sign

        return P
