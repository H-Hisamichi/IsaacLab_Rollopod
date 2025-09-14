# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
from collections.abc import Sequence
from typing import TYPE_CHECKING

import omni.log

import isaaclab.utils.math as math_utils
from isaaclab.assets import Articulation
from isaaclab.managers import CommandTerm
from isaaclab.markers import VisualizationMarkers
from isaaclab.utils.math import quat_apply, quat_conjugate

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv, ManagerBasedRLEnv

    from .commands_cfg import NormalVelocityCommandCfg, UniformVelocityCommandCfg, CamberAngleANDRollingVelocityCommandCfg, CamberAngleANDRollingAngularVelocityCommandCfg, UniformWorldVelocityCommandCfg


class UniformWorldVelocityCommand(CommandTerm):
    r"""

    """

    cfg: UniformWorldVelocityCommandCfg
    """The configuration of the command generator."""

    def __init__(self, cfg: UniformWorldVelocityCommandCfg, env: ManagerBasedEnv):
        """Initialize the command generator.

        Args:
            cfg: The configuration of the command generator.
            env: The environment.

        Raises:
            ValueError: If the heading command is active but the heading range is not provided.
        """
        # initialize the base class
        super().__init__(cfg, env)

        # obtain the robot asset
        # -- robot
        self.robot: Articulation = env.scene[cfg.asset_name]

        # crete buffers to store the command
        # -- command: x vel, y vel, yaw vel
        self.vel_command_w = torch.zeros(self.num_envs, 3, device=self.device)
        self.vel_command_b = torch.zeros(self.num_envs, 4, device=self.device)
        #self.heading_target = torch.zeros(self.num_envs, device=self.device)
        self.is_heading_env = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.is_standing_env = torch.zeros_like(self.is_heading_env)
        # -- metrics
        self.metrics["error_vel_xy"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["error_ang_z"] = torch.zeros(self.num_envs, device=self.device)

    def __str__(self) -> str:
        """Return a string representation of the command generator."""
        msg = "UniformVelocityCommand:\n"
        msg += f"\tCommand dimension: {tuple(self.command.shape[1:])}\n"
        msg += f"\tResampling time range: {self.cfg.resampling_time_range}\n"
        #msg += f"\tHeading command: {self.cfg.heading_command}\n"
        #if self.cfg.heading_command:
        #    msg += f"\tHeading probability: {self.cfg.rel_heading_envs}\n"
        msg += f"\tStanding probability: {self.cfg.rel_standing_envs}"
        return msg

    """
    Properties
    """

    @property
    def command(self) -> torch.Tensor:
        """The desired base velocity command in the base frame. Shape is (num_envs, 3)."""
        return self.vel_command_b

    """
    Implementation specific functions.
    """

    def _update_metrics(self):
        # time for which the command was executed
        max_command_time = self.cfg.resampling_time_range[1]
        max_command_step = max_command_time / self._env.step_dt
        vel_xy = self.robot.data.root_com_lin_vel_w[:, :2]
        vel_norm = torch.norm(vel_xy, dim=1, keepdim=True).clamp(min=1e-6)
        vel_dir = vel_xy / vel_norm
        # logs data
        self.metrics["error_vel_xy"] += (
            torch.sum(torch.abs(self.vel_command_w[:, :2] - vel_dir), dim=-1) / max_command_step
        )
        self.metrics["error_ang_z"] += (
            (self.vel_command_w[:, -1] - self.robot.data.root_ang_vel_b[:, -1]) / max_command_step
        )

    def _resample_command(self, env_ids: Sequence[int]):
        # sample velocity commands
        r = torch.empty(len(env_ids), device=self.device)
        theta = r.uniform_(*self.cfg.ranges.heading)
        # -- linear velocity - x direction
        self.vel_command_w[env_ids, 0] = torch.cos(theta)
        # -- linear velocity - y direction
        self.vel_command_w[env_ids, 1] = torch.sin(theta)
        # -- angular velocity - z direction
        self.vel_command_w[env_ids, 2] = r.uniform_(*self.cfg.ranges.rolling_speed)
        #self.vel_command_b[env_ids, 2] = (
        #    torch.norm(self.vel_command_b[env_ids, :2], dim=1) / self.cfg.robot_radius
        #    ) *(torch.randint(0, 2, (len(env_ids),), device=self.vel_command_b.device) * 2 - 1
        #)
        
        # update standing envs
        self.is_standing_env[env_ids] = r.uniform_(0.0, 1.0) <= self.cfg.rel_standing_envs

    def _update_command(self):
        root_quat_w = self.robot.data.root_link_quat_w
        vel_command_w_3d = self.vel_command_w.clone()
        vel_command_w_3d[:, 2] = 0.0
        root_quat_inv = quat_conjugate(root_quat_w)
        v_l_3d = quat_apply(root_quat_inv, vel_command_w_3d)
        self.vel_command_b[:, 0:3] = v_l_3d
        self.vel_command_b[:, -1] = self.vel_command_w[:, -1]
        # Enforce standing (i.e., zero velocity command) for standing envs
        # TODO: check if conversion is needed
        standing_env_ids = self.is_standing_env.nonzero(as_tuple=False).flatten()
        self.vel_command_w[standing_env_ids, :] = 0.0

    def _set_debug_vis_impl(self, debug_vis: bool):
        # set visibility of markers
        # note: parent only deals with callbacks. not their visibility
        if debug_vis:
            # create markers if necessary for the first time
            if not hasattr(self, "goal_vel_visualizer"):
                # -- goal
                self.goal_vel_visualizer = VisualizationMarkers(self.cfg.goal_vel_visualizer_cfg)
                # -- current
                self.current_vel_visualizer = VisualizationMarkers(self.cfg.current_vel_visualizer_cfg)
            # set their visibility to true
            self.goal_vel_visualizer.set_visibility(True)
            self.current_vel_visualizer.set_visibility(True)
        else:
            if hasattr(self, "goal_vel_visualizer"):
                self.goal_vel_visualizer.set_visibility(False)
                self.current_vel_visualizer.set_visibility(False)

    def _debug_vis_callback(self, event):
        # check if robot is initialized
        # note: this is needed in-case the robot is de-initialized. we can't access the data
        if not self.robot.is_initialized:
            return
        # get marker location
        # -- base state
        base_pos_w = self.robot.data.root_pos_w.clone()
        base_pos_w[:, 2] += 0.5
        # compute goal xy velocity from unit vector and wheel angular speed
        direction = self.command[:, :2]
        wheel_radius = 0.33
        linear_speed = torch.abs(self.command[:, -1]) * wheel_radius
        goal_xy_velocity = direction * linear_speed.unsqueeze(1)
        # -- resolve the scales and quaternions
        vel_des_arrow_scale, vel_des_arrow_quat = self._resolve_xy_velocity_to_arrow(goal_xy_velocity)
        vel_arrow_scale, vel_arrow_quat = self._resolve_xy_velocity_to_arrow(self.robot.data.root_lin_vel_b[:, :2])
        # display markers
        self.goal_vel_visualizer.visualize(base_pos_w, vel_des_arrow_quat, vel_des_arrow_scale)
        self.current_vel_visualizer.visualize(base_pos_w, vel_arrow_quat, vel_arrow_scale)

    """
    Internal helpers.
    """

    def _resolve_xy_velocity_to_arrow(self, xy_velocity: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Converts the XY base velocity command to arrow direction rotation."""
        # obtain default scale of the marker
        default_scale = self.goal_vel_visualizer.cfg.markers["arrow"].scale
        # arrow-scale
        arrow_scale = torch.tensor(default_scale, device=self.device).repeat(xy_velocity.shape[0], 1)
        arrow_scale[:, 0] *= torch.linalg.norm(xy_velocity, dim=1) * 3.0
        # arrow-direction
        heading_angle = torch.atan2(xy_velocity[:, 1], xy_velocity[:, 0])
        zeros = torch.zeros_like(heading_angle)
        arrow_quat = math_utils.quat_from_euler_xyz(zeros, zeros, heading_angle)
        # convert everything back from base to world frame
        base_quat_w = self.robot.data.root_quat_w
        arrow_quat = math_utils.quat_mul(base_quat_w, arrow_quat)

        return arrow_scale, arrow_quat

class CamberAngleANDRollingAngularVelocityCommand(CommandTerm):
    r"""A command generator that generates a rolling speed command from a uniform distribution.

    This command is decomposed into an xy vector, where a vector is calculated that indicates
    the rolling direction and speed. This is given in the robot's base frame.

    """

    #cfg: UniformVelocityCommandCfg
    cfg: CamberAngleANDRollingAngularVelocityCommandCfg
    """The configuration of the command generator."""

    #def __init__(self, cfg: UniformVelocityCommandCfg, env: ManagerBasedEnv):
    def __init__(self, cfg: CamberAngleANDRollingAngularVelocityCommandCfg, env: ManagerBasedEnv):
        """Initialize the command generator.

        Args:
            cfg: The configuration of the command generator.
            env: The environment.

        Raises:
            ValueError: If the heading command is active but the heading range is not provided. <- This function is disabled.
        """
        # initialize the base class
        super().__init__(cfg, env)
        """
        # check configuration
        if self.cfg.heading_command and self.cfg.ranges.heading is None:
            raise ValueError(
                "The velocity command has heading commands active (heading_command=True) but the `ranges.heading`"
                " parameter is set to None."
            )
        if self.cfg.ranges.heading and not self.cfg.heading_command:
            omni.log.warn(
                f"The velocity command has the 'ranges.heading' attribute set to '{self.cfg.ranges.heading}'"
                " but the heading command is not active. Consider setting the flag for the heading command to True."
            )
        """

        # obtain the robot asset
        # -- robot
        self.robot: Articulation = env.scene[cfg.asset_name]

        # crete buffers to store the command
        # -- command: x vel, y vel, yaw vel, heading
        self.vel_command_b = torch.zeros(self.num_envs, 2, device=self.device)
        #self.heading_target = torch.zeros(self.num_envs, device=self.device)
        self.is_heading_env = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.is_standing_env = torch.zeros_like(self.is_heading_env)
        # -- metrics
        #self.metrics["error_camber_angle"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["error_ang_vel"] = torch.zeros(self.num_envs, device=self.device)

    def __str__(self) -> str:
        """Return a string representation of the command generator."""
        msg = "UniformVelocityCommand:\n"
        msg += f"\tCommand dimension: {tuple(self.command.shape[1:])}\n"
        msg += f"\tResampling time range: {self.cfg.resampling_time_range}\n"
        #msg += f"\tHeading command: {self.cfg.heading_command}\n"
        #if self.cfg.heading_command:
        #    msg += f"\tHeading probability: {self.cfg.rel_heading_envs}\n"
        msg += f"\tStanding probability: {self.cfg.rel_standing_envs}"
        return msg

    """
    Properties
    """

    @property
    def command(self) -> torch.Tensor:
        """The desired base velocity command in the base frame. Shape is (num_envs, 3)."""
        return self.vel_command_b

    """
    Implementation specific functions.
    """

    def _update_metrics(self):
        # time for which the command was executed
        max_command_time = self.cfg.resampling_time_range[1]
        max_command_step = max_command_time / self._env.step_dt

        #_projected_gravity_b = self.robot.data.projected_gravity_b
        #_norms = torch.sqrt(torch.sum(_projected_gravity_b ** 2, dim=1))
        #_z = _projected_gravity_b[:, 2]
        #_cos_phi = torch.clamp(torch.abs(_z) / _norms, 0, 1)
        #_phi = torch.acos(_cos_phi)
        #_theta = torch.pi / 2 - _phi
        #_theta_z_sign = torch.sign(_z) * _theta
        #_theta_z_sign[_norms == 0] = 0

        # speed error
        #dot_product = torch.sum(_root_lin_vel_b * _projected_gravity_b, dim=1)
        #g_norm_squared = torch.sum(_projected_gravity_b * _projected_gravity_b, dim=1)
        #mask = g_norm_squared > 0
        #projection = torch.zeros_like(_projected_gravity_b)
        #projection[mask] = (dot_product[mask] / g_norm_squared[mask]).unsqueeze(1) * _projected_gravity_b[mask]
        #vel_perp = _root_lin_vel_b - projection
        #vel_perp_magnitude = torch.norm(vel_perp, dim=1)
        
        # logs data
        #self.metrics["error_camber_angle"] += (
        #    torch.norm(self.vel_command_b[:, 1] - _theta_z_sign) / max_command_step
        #)
        self.metrics["error_ang_vel"] += (
            torch.abs(torch.abs(self.vel_command_b[:, 0]) - self.robot.data.root_com_ang_vel_b[:, 2]) / max_command_step
        )

    def _resample_command(self, env_ids: Sequence[int]):
        # sample velocity commands
        r = torch.empty(len(env_ids), device=self.device)
        _r_scalar = torch.empty(len(env_ids), device=self.device)
        _r_camber_angle = torch.empty(len(env_ids), device=self.device)
        #
        _scalar = _r_scalar.uniform_(*self.cfg.ranges.rolling_ang_vel)
        _camber_angle = _r_camber_angle.uniform_(*self.cfg.ranges.steer_ang_vel)
        # -- linear velocity - x direction
        #self.vel_command_b[env_ids, 0] = r.uniform_(*self.cfg.ranges.lin_vel_x)
        #self.vel_command_b[env_ids, 0] = _scalar * torch.sin(_command_range)
        self.vel_command_b[env_ids, 0] = _scalar
        # -- linear velocity - y direction
        #self.vel_command_b[env_ids, 1] = r.uniform_(*self.cfg.ranges.lin_vel_y)
        #self.vel_command_b[env_ids, 1] = _scalar * torch.cos(_command_range)
        self.vel_command_b[env_ids, 1] = _camber_angle
        # -- ang vel yaw - rotation around z
        #self.vel_command_b[env_ids, 2] = r.uniform_(*self.cfg.ranges.ang_vel_z)
        #self.vel_command_b[env_ids, 2] = _scalar / self.cfg.ranges.rolling_radius
        """
        # heading target
        if self.cfg.heading_command:
            self.heading_target[env_ids] = r.uniform_(*self.cfg.ranges.heading)
            # update heading envs
            self.is_heading_env[env_ids] = r.uniform_(0.0, 1.0) <= self.cfg.rel_heading_envs
        """
        # update standing envs
        self.is_standing_env[env_ids] = r.uniform_(0.0, 1.0) <= self.cfg.rel_standing_envs

    def _update_command(self):
        """Post-processes the velocity command.

        This function sets the velocity command to zero in a stationary environment.
        """
        """
        # Compute angular velocity from heading direction
        if self.cfg.heading_command:
            # resolve indices of heading envs
            env_ids = self.is_heading_env.nonzero(as_tuple=False).flatten()
            # compute angular velocity
            heading_error = math_utils.wrap_to_pi(self.heading_target[env_ids] - self.robot.data.heading_w[env_ids])
            self.vel_command_b[env_ids, 2] = torch.clip(
                self.cfg.heading_control_stiffness * heading_error,
                min=self.cfg.ranges.ang_vel_z[0],
                max=self.cfg.ranges.ang_vel_z[1],
            )
        """
        # Enforce standing (i.e., zero velocity command) for standing envs
        # TODO: check if conversion is needed
        standing_env_ids = self.is_standing_env.nonzero(as_tuple=False).flatten()
        self.vel_command_b[standing_env_ids, :] = 0.0

    def _set_debug_vis_impl(self, debug_vis: bool):
        # set visibility of markers
        # note: parent only deals with callbacks. not their visibility
        if debug_vis:
            # create markers if necessary for the first tome
            if not hasattr(self, "goal_vel_visualizer"):
                # -- goal
                self.goal_vel_visualizer = VisualizationMarkers(self.cfg.goal_vel_visualizer_cfg)
                # -- current
                self.current_vel_visualizer = VisualizationMarkers(self.cfg.current_vel_visualizer_cfg)
            # set their visibility to true
            self.goal_vel_visualizer.set_visibility(True)
            self.current_vel_visualizer.set_visibility(True)
        else:
            if hasattr(self, "goal_vel_visualizer"):
                self.goal_vel_visualizer.set_visibility(False)
                self.current_vel_visualizer.set_visibility(False)

    def _debug_vis_callback(self, event):
        # check if robot is initialized
        # note: this is needed in-case the robot is de-initialized. we can't access the data
        if not self.robot.is_initialized:
            return
        # get marker location
        # -- base state
        base_pos_w = self.robot.data.root_pos_w.clone()
        base_pos_w[:, 2] += 0.5
        # -- resolve the scales and quaternions
        vel_des_arrow_scale, vel_des_arrow_quat = self._resolve_xy_velocity_to_arrow(self.command[:, :2])
        vel_arrow_scale, vel_arrow_quat = self._resolve_xy_velocity_to_arrow(self.robot.data.root_lin_vel_b[:, :2])
        # display markers
        self.goal_vel_visualizer.visualize(base_pos_w, vel_des_arrow_quat, vel_des_arrow_scale)
        self.current_vel_visualizer.visualize(base_pos_w, vel_arrow_quat, vel_arrow_scale)

class CamberAngleANDRollingVelocityCommand(CommandTerm):
    r"""A command generator that generates a rolling speed command from a uniform distribution.

    This command is decomposed into an xy vector, where a vector is calculated that indicates
    the rolling direction and speed. This is given in the robot's base frame.

    """

    #cfg: UniformVelocityCommandCfg
    cfg: CamberAngleANDRollingVelocityCommandCfg
    """The configuration of the command generator."""

    #def __init__(self, cfg: UniformVelocityCommandCfg, env: ManagerBasedEnv):
    def __init__(self, cfg: CamberAngleANDRollingVelocityCommandCfg, env: ManagerBasedEnv):
        """Initialize the command generator.

        Args:
            cfg: The configuration of the command generator.
            env: The environment.

        Raises:
            ValueError: If the heading command is active but the heading range is not provided. <- This function is disabled.
        """
        # initialize the base class
        super().__init__(cfg, env)
        """
        # check configuration
        if self.cfg.heading_command and self.cfg.ranges.heading is None:
            raise ValueError(
                "The velocity command has heading commands active (heading_command=True) but the `ranges.heading`"
                " parameter is set to None."
            )
        if self.cfg.ranges.heading and not self.cfg.heading_command:
            omni.log.warn(
                f"The velocity command has the 'ranges.heading' attribute set to '{self.cfg.ranges.heading}'"
                " but the heading command is not active. Consider setting the flag for the heading command to True."
            )
        """

        # obtain the robot asset
        # -- robot
        self.robot: Articulation = env.scene[cfg.asset_name]

        # crete buffers to store the command
        # -- command: x vel, y vel, yaw vel, heading
        self.vel_command_b = torch.zeros(self.num_envs, 2, device=self.device)
        #self.heading_target = torch.zeros(self.num_envs, device=self.device)
        self.is_heading_env = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.is_standing_env = torch.zeros_like(self.is_heading_env)
        # -- metrics
        self.metrics["error_lin_vel"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["error_steer_ang_vel"] = torch.zeros(self.num_envs, device=self.device)

    def __str__(self) -> str:
        """Return a string representation of the command generator."""
        msg = "UniformVelocityCommand:\n"
        msg += f"\tCommand dimension: {tuple(self.command.shape[1:])}\n"
        msg += f"\tResampling time range: {self.cfg.resampling_time_range}\n"
        #msg += f"\tHeading command: {self.cfg.heading_command}\n"
        #if self.cfg.heading_command:
        #    msg += f"\tHeading probability: {self.cfg.rel_heading_envs}\n"
        msg += f"\tStanding probability: {self.cfg.rel_standing_envs}"
        return msg

    """
    Properties
    """

    @property
    def command(self) -> torch.Tensor:
        """The desired base velocity command in the base frame. Shape is (num_envs, 3)."""
        return self.vel_command_b

    """
    Implementation specific functions.
    """

    def _update_metrics(self):
        # time for which the command was executed
        max_command_time = self.cfg.resampling_time_range[1]
        max_command_step = max_command_time / self._env.step_dt

        projected_gravity_b = self.robot.data.projected_gravity_b
        root_com_ang_vel_b = self.robot.data.root_com_ang_vel_b
        xy_axis = torch.zeros_like(projected_gravity_b)
        xy_axis[:, :2] = projected_gravity_b[:, :2]
        norm = torch.norm(xy_axis, dim=1, keepdim=True)
        projected_vel = torch.zeros(root_com_ang_vel_b.shape[0], device=root_com_ang_vel_b.device)
        valid_mask = norm.squeeze() > 1e-8
        unit_axis = torch.zeros_like(xy_axis)
        unit_axis[valid_mask] = xy_axis[valid_mask] / norm[valid_mask]
        projected_vel[valid_mask] = torch.sum(root_com_ang_vel_b[valid_mask] * unit_axis[valid_mask], dim=1)
        
        # logs data
        self.metrics["error_lin_vel"] += (
            torch.abs(torch.abs(self.vel_command_b[:, 0]) - self.robot.data.root_com_ang_vel_b[:, 2]) / max_command_step
        )
        self.metrics["error_steer_ang_vel"] += (
            torch.norm(self.vel_command_b[:, 1] - projected_vel) / max_command_step
        )

    def _resample_command(self, env_ids: Sequence[int]):
        # sample velocity commands
        r = torch.empty(len(env_ids), device=self.device)
        _r_scalar = torch.empty(len(env_ids), device=self.device)
        _r_steer_ang_vel = torch.empty(len(env_ids), device=self.device)
        #
        _rolling_lin_vel = _r_scalar.uniform_(*self.cfg.ranges.rolling_lin_vel)
        _steer_ang_vel = _r_steer_ang_vel.uniform_(*self.cfg.ranges.steer_ang_vel)
        # -- linear velocity - x direction
        #self.vel_command_b[env_ids, 0] = r.uniform_(*self.cfg.ranges.lin_vel_x)
        #self.vel_command_b[env_ids, 0] = _scalar * torch.sin(_command_range)
        self.vel_command_b[env_ids, 0] = _rolling_lin_vel
        # -- linear velocity - y direction
        #self.vel_command_b[env_ids, 1] = r.uniform_(*self.cfg.ranges.lin_vel_y)
        #self.vel_command_b[env_ids, 1] = _scalar * torch.cos(_command_range)
        self.vel_command_b[env_ids, 1] = _steer_ang_vel
        # -- ang vel yaw - rotation around z
        #self.vel_command_b[env_ids, 2] = r.uniform_(*self.cfg.ranges.ang_vel_z)
        #self.vel_command_b[env_ids, 2] = _scalar / self.cfg.ranges.rolling_radius
        """
        # heading target
        if self.cfg.heading_command:
            self.heading_target[env_ids] = r.uniform_(*self.cfg.ranges.heading)
            # update heading envs
            self.is_heading_env[env_ids] = r.uniform_(0.0, 1.0) <= self.cfg.rel_heading_envs
        """
        # update standing envs
        self.is_standing_env[env_ids] = r.uniform_(0.0, 1.0) <= self.cfg.rel_standing_envs

    def _update_command(self):
        """Post-processes the velocity command.

        This function sets the velocity command to zero in a stationary environment.
        """
        """
        # Compute angular velocity from heading direction
        if self.cfg.heading_command:
            # resolve indices of heading envs
            env_ids = self.is_heading_env.nonzero(as_tuple=False).flatten()
            # compute angular velocity
            heading_error = math_utils.wrap_to_pi(self.heading_target[env_ids] - self.robot.data.heading_w[env_ids])
            self.vel_command_b[env_ids, 2] = torch.clip(
                self.cfg.heading_control_stiffness * heading_error,
                min=self.cfg.ranges.ang_vel_z[0],
                max=self.cfg.ranges.ang_vel_z[1],
            )
        """
        # Enforce standing (i.e., zero velocity command) for standing envs
        # TODO: check if conversion is needed
        standing_env_ids = self.is_standing_env.nonzero(as_tuple=False).flatten()
        self.vel_command_b[standing_env_ids, :] = 0.0

    def _set_debug_vis_impl(self, debug_vis: bool):
        # set visibility of markers
        # note: parent only deals with callbacks. not their visibility
        if debug_vis:
            # create markers if necessary for the first tome
            if not hasattr(self, "goal_vel_visualizer"):
                # -- goal
                self.goal_vel_visualizer = VisualizationMarkers(self.cfg.goal_vel_visualizer_cfg)
                # -- current
                self.current_vel_visualizer = VisualizationMarkers(self.cfg.current_vel_visualizer_cfg)
            # set their visibility to true
            self.goal_vel_visualizer.set_visibility(True)
            self.current_vel_visualizer.set_visibility(True)
        else:
            if hasattr(self, "goal_vel_visualizer"):
                self.goal_vel_visualizer.set_visibility(False)
                self.current_vel_visualizer.set_visibility(False)

    def _debug_vis_callback(self, event):
        # check if robot is initialized
        # note: this is needed in-case the robot is de-initialized. we can't access the data
        if not self.robot.is_initialized:
            return
        # get marker location
        # -- base state
        base_pos_w = self.robot.data.root_pos_w.clone()
        base_pos_w[:, 2] += 0.5
        # -- resolve the scales and quaternions
        vel_des_arrow_scale, vel_des_arrow_quat = self._resolve_xy_velocity_to_arrow(self.command[:, :2])
        vel_arrow_scale, vel_arrow_quat = self._resolve_xy_velocity_to_arrow(self.robot.data.root_lin_vel_b[:, :2])
        # display markers
        self.goal_vel_visualizer.visualize(base_pos_w, vel_des_arrow_quat, vel_des_arrow_scale)
        self.current_vel_visualizer.visualize(base_pos_w, vel_arrow_quat, vel_arrow_scale)

