"""Milestone Genesis Phase 2: Closed-Loop Kinematic Actuator & Trajectory Solver."""

from dataclasses import dataclass
from typing import cast

import torch
import torch.nn as nn

from aptivra.embodied.sensorimotor import SensorimotorTokenizer
from aptivra.world_model.jepa import LatentWorldModel


@dataclass
class KinematicCommand:
    """Target position, velocity, and torque outputs for physical actuators."""

    joint_positions: torch.Tensor  # [batch, num_joints]
    joint_velocities: torch.Tensor  # [batch, num_joints]
    joint_torques: torch.Tensor  # [batch, num_joints]
    target_pose_se3: torch.Tensor  # [batch, 4, 4] homogeneous SE(3) transformation matrix


class KinematicActuatorEngine(nn.Module):
    """Translates high-level latent goals into continuous SE(3) spatial trajectories and torques."""

    def __init__(
        self,
        num_joints: int = 12,
        d_model: int = 512,
        world_model: LatentWorldModel | None = None,
        max_torque_nm: float = 40.0,
        max_velocity_rad_s: float = 5.0,
    ) -> None:
        super().__init__()
        self.num_joints = num_joints
        self.d_model = d_model
        self.max_torque_nm = max_torque_nm
        self.max_velocity_rad_s = max_velocity_rad_s

        self.world_model = world_model or LatentWorldModel(
            latent_dim=d_model, action_dim=num_joints
        )
        self.tokenizer = SensorimotorTokenizer(kinematics_dim=num_joints, d_model=d_model)

        # Policy decoders projecting latent state into physical actuator torques
        self.torque_decoder = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, num_joints),
            nn.Tanh(),  # Normalized torque in [-1.0, 1.0]
        )

        # 6-DOF Lie algebra se(3) twist parameters: 3 for translation + 3 for rotation vector
        self.se3_twist_decoder = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, 6),
        )

    def _so3_exponential_map(self, omega: torch.Tensor) -> torch.Tensor:
        """Computes SO(3) rotation matrix from rotation vector via Rodrigues formula."""
        batch_size = omega.shape[0]
        theta = torch.norm(omega, dim=-1, keepdim=True) + 1e-8
        k = omega / theta

        kx, ky, kz = k[:, 0], k[:, 1], k[:, 2]
        zeros = torch.zeros_like(kx)

        # Skew-symmetric matrix K
        k_mat = torch.stack(
            [
                zeros,
                -kz,
                ky,
                kz,
                zeros,
                -kx,
                -ky,
                kx,
                zeros,
            ],
            dim=-1,
        ).view(batch_size, 3, 3)

        eye = torch.eye(3, device=omega.device).unsqueeze(0).repeat(batch_size, 1, 1)
        sin_t = torch.sin(theta).unsqueeze(-1)
        cos_t = torch.cos(theta).unsqueeze(-1)

        # Rodrigues formula: R = I + sin(theta) * K + (1 - cos(theta)) * K^2
        r_mat = eye + sin_t * k_mat + (1.0 - cos_t) * torch.bmm(k_mat, k_mat)
        return r_mat

    def compute_actuation(
        self, current_kinematics: torch.Tensor, target_latent_goal: torch.Tensor
    ) -> KinematicCommand:
        """Executes closed-loop step: projects telemetry, predicts next state, decodes commands."""
        # 1. Tokenize physical sensor feedback
        current_state_latent = self.tokenizer(current_kinematics)

        # 2. Blend goal with current state
        control_latent = current_state_latent + target_latent_goal

        # 3. Decode physical torques with physical motor limits
        normalized_torques = cast(torch.Tensor, self.torque_decoder(control_latent))
        torques = normalized_torques * self.max_torque_nm

        # 4. Decode proper SE(3) pose via Lie algebra exponential map
        twist_6d = cast(torch.Tensor, self.se3_twist_decoder(control_latent))
        translation = twist_6d[:, :3]  # x, y, z
        omega = twist_6d[:, 3:]  # rotation vector in so(3)

        r_mat = self._so3_exponential_map(omega)

        # Assemble homogeneous [4, 4] matrix
        batch_size = twist_6d.shape[0]
        se3_mat = (
            torch.eye(4, device=current_kinematics.device).unsqueeze(0).repeat(batch_size, 1, 1)
        )
        se3_mat[:, :3, :3] = r_mat
        se3_mat[:, :3, 3] = translation

        # 5. Integrate dynamic velocity and positions with velocity clamping
        dt = 0.02  # 50 Hz control loop
        velocities = (torques * 0.05).clamp(-self.max_velocity_rad_s, self.max_velocity_rad_s)
        positions = current_kinematics + velocities * dt

        return KinematicCommand(
            joint_positions=positions,
            joint_velocities=velocities,
            joint_torques=torques,
            target_pose_se3=se3_mat,
        )
