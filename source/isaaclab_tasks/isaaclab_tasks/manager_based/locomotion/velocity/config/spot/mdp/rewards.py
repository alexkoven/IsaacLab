# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""This sub-module contains the reward functions that can be used for Spot's locomotion task.

The functions can be passed to the :class:`isaaclab.managers.RewardTermCfg` object to
specify the reward function and its parameters.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import ManagerTermBase, SceneEntityCfg
from isaaclab.sensors import ContactSensor

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv
    from isaaclab.managers import RewardTermCfg


##
# Task Rewards
##


def air_time_reward(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
    sensor_cfg: SceneEntityCfg,
    mode_time: float,
    velocity_threshold: float,
) -> torch.Tensor:
    """Reward longer feet air and contact time."""
    # extract the used quantities (to enable type-hinting)
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    asset: Articulation = env.scene[asset_cfg.name]
    if contact_sensor.cfg.track_air_time is False:
        raise RuntimeError("Activate ContactSensor's track_air_time!")
    # compute the reward
    current_air_time = contact_sensor.data.current_air_time[:, sensor_cfg.body_ids]
    current_contact_time = contact_sensor.data.current_contact_time[:, sensor_cfg.body_ids]

    t_max = torch.max(current_air_time, current_contact_time)
    t_min = torch.clip(t_max, max=mode_time)
    stance_cmd_reward = torch.clip(current_contact_time - current_air_time, -mode_time, mode_time)
    # Broadcast command magnitude and realized base speed to match selected feet count.
    # current_air_time/current_contact_time: (num_envs, num_selected_feet)
    cmd_mag = torch.norm(env.command_manager.get_command("base_velocity"), dim=1)  # (num_envs,)
    body_vel_mag = torch.linalg.norm(asset.data.root_lin_vel_b[:, :2], dim=1)  # (num_envs,)
    cmd = cmd_mag.unsqueeze(dim=1).expand_as(current_air_time)  # (num_envs, num_selected_feet)
    body_vel = body_vel_mag.unsqueeze(dim=1).expand_as(current_air_time)  # (num_envs, num_selected_feet)
    reward = torch.where(
        torch.logical_or(cmd > 0.0, body_vel > velocity_threshold),
        torch.where(t_max < mode_time, t_min, 0),
        stance_cmd_reward,
    )
    return torch.sum(reward, dim=1)


def base_angular_velocity_reward(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg, std: float) -> torch.Tensor:
    """Reward tracking of angular velocity commands (yaw) using abs exponential kernel."""
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    # compute the error
    target = env.command_manager.get_command("base_velocity")[:, 2]
    ang_vel_error = torch.linalg.norm((target - asset.data.root_ang_vel_b[:, 2]).unsqueeze(1), dim=1)
    return torch.exp(-ang_vel_error / std)


def base_linear_velocity_reward(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg, std: float, ramp_at_vel: float = 1.0, ramp_rate: float = 0.5
) -> torch.Tensor:
    """Reward tracking of linear velocity commands (xy axes) using abs exponential kernel."""
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    # compute the error
    target = env.command_manager.get_command("base_velocity")[:, :2]
    lin_vel_error = torch.linalg.norm((target - asset.data.root_lin_vel_b[:, :2]), dim=1)
    # fixed 1.0 multiple for tracking below the ramp_at_vel value, then scale by the rate above
    vel_cmd_magnitude = torch.linalg.norm(target, dim=1)
    velocity_scaling_multiple = torch.clamp(1.0 + ramp_rate * (vel_cmd_magnitude - ramp_at_vel), min=1.0)
    return torch.exp(-lin_vel_error / std) * velocity_scaling_multiple


class GaitReward(ManagerTermBase):
    """Gait enforcing reward term for quadrupeds.

    This reward penalizes contact timing differences between selected foot pairs defined in
    :attr:`synced_feet_pair_names` to bias the policy towards a desired gait, i.e trotting,
    bounding, or pacing. Note that this reward is only for quadrupedal gaits with two pairs
    of synchronized feet.
    """

    def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRLEnv):
        """Initialize the term.

        Args:
            cfg: The configuration of the reward.
            env: The RL environment instance.
        """
        super().__init__(cfg, env)
        self.std: float = cfg.params["std"]
        self.max_err: float = cfg.params["max_err"]
        self.velocity_threshold: float = cfg.params["velocity_threshold"]
        self.contact_sensor: ContactSensor = env.scene.sensors[cfg.params["sensor_cfg"].name]
        self.asset: Articulation = env.scene[cfg.params["asset_cfg"].name]
        # match foot body names with corresponding foot body ids
        synced_feet_pair_names = cfg.params["synced_feet_pair_names"]
        if (
            len(synced_feet_pair_names) != 2
            or len(synced_feet_pair_names[0]) != 2
            or len(synced_feet_pair_names[1]) != 2
        ):
            raise ValueError("This reward only supports gaits with two pairs of synchronized feet, like trotting.")
        synced_feet_pair_0 = self.contact_sensor.find_bodies(synced_feet_pair_names[0])[0]
        synced_feet_pair_1 = self.contact_sensor.find_bodies(synced_feet_pair_names[1])[0]
        self.synced_feet_pairs = [synced_feet_pair_0, synced_feet_pair_1]

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        std: float,
        max_err: float,
        velocity_threshold: float,
        synced_feet_pair_names,
        asset_cfg: SceneEntityCfg,
        sensor_cfg: SceneEntityCfg,
    ) -> torch.Tensor:
        """Compute the reward.

        This reward is defined as a multiplication between six terms where two of them enforce pair feet
        being in sync and the other four rewards if all the other remaining pairs are out of sync

        Args:
            env: The RL environment instance.
        Returns:
            The reward value.
        """
        # for synchronous feet, the contact (air) times of two feet should match
        sync_reward_0 = self._sync_reward_func(self.synced_feet_pairs[0][0], self.synced_feet_pairs[0][1])
        sync_reward_1 = self._sync_reward_func(self.synced_feet_pairs[1][0], self.synced_feet_pairs[1][1])
        sync_reward = sync_reward_0 * sync_reward_1
        # for asynchronous feet, the contact time of one foot should match the air time of the other one
        async_reward_0 = self._async_reward_func(self.synced_feet_pairs[0][0], self.synced_feet_pairs[1][0])
        async_reward_1 = self._async_reward_func(self.synced_feet_pairs[0][1], self.synced_feet_pairs[1][1])
        async_reward_2 = self._async_reward_func(self.synced_feet_pairs[0][0], self.synced_feet_pairs[1][1])
        async_reward_3 = self._async_reward_func(self.synced_feet_pairs[1][0], self.synced_feet_pairs[0][1])
        async_reward = async_reward_0 * async_reward_1 * async_reward_2 * async_reward_3
        # only enforce gait if cmd > 0
        cmd = torch.norm(env.command_manager.get_command("base_velocity"), dim=1)
        body_vel = torch.linalg.norm(self.asset.data.root_lin_vel_b[:, :2], dim=1)
        return torch.where(
            torch.logical_or(cmd > 0.0, body_vel > self.velocity_threshold), sync_reward * async_reward, 0.0
        )

    """
    Helper functions.
    """

    def _sync_reward_func(self, foot_0: int, foot_1: int) -> torch.Tensor:
        """Reward synchronization of two feet."""
        air_time = self.contact_sensor.data.current_air_time
        contact_time = self.contact_sensor.data.current_contact_time
        # penalize the difference between the most recent air time and contact time of synced feet pairs.
        se_air = torch.clip(torch.square(air_time[:, foot_0] - air_time[:, foot_1]), max=self.max_err**2)
        se_contact = torch.clip(torch.square(contact_time[:, foot_0] - contact_time[:, foot_1]), max=self.max_err**2)
        return torch.exp(-(se_air + se_contact) / self.std)

    def _async_reward_func(self, foot_0: int, foot_1: int) -> torch.Tensor:
        """Reward anti-synchronization of two feet."""
        air_time = self.contact_sensor.data.current_air_time
        contact_time = self.contact_sensor.data.current_contact_time
        # penalize the difference between opposing contact modes air time of feet 1 to contact time of feet 2
        # and contact time of feet 1 to air time of feet 2) of feet pairs that are not in sync with each other.
        se_act_0 = torch.clip(torch.square(air_time[:, foot_0] - contact_time[:, foot_1]), max=self.max_err**2)
        se_act_1 = torch.clip(torch.square(contact_time[:, foot_0] - air_time[:, foot_1]), max=self.max_err**2)
        return torch.exp(-(se_act_0 + se_act_1) / self.std)


def foot_clearance_reward(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg, target_height: float, std: float, tanh_mult: float
) -> torch.Tensor:
    """Reward the swinging feet for clearing a specified height off the ground"""
    asset: RigidObject = env.scene[asset_cfg.name]
    foot_z_target_error = torch.square(asset.data.body_pos_w[:, asset_cfg.body_ids, 2] - target_height)
    foot_velocity_tanh = torch.tanh(tanh_mult * torch.norm(asset.data.body_lin_vel_w[:, asset_cfg.body_ids, :2], dim=2))
    reward = foot_z_target_error * foot_velocity_tanh
    return torch.exp(-torch.sum(reward, dim=1) / std)


##
# Regularization Penalties
##


def action_smoothness_penalty(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Penalize large instantaneous changes in the network action output"""
    return torch.linalg.norm((env.action_manager.action - env.action_manager.prev_action), dim=1)


def air_time_variance_penalty(env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    """Penalize variance in the amount of time each foot spends in the air/on the ground relative to each other"""
    # extract the used quantities (to enable type-hinting)
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    if contact_sensor.cfg.track_air_time is False:
        raise RuntimeError("Activate ContactSensor's track_air_time!")
    # compute the reward
    last_air_time = contact_sensor.data.last_air_time[:, sensor_cfg.body_ids]
    last_contact_time = contact_sensor.data.last_contact_time[:, sensor_cfg.body_ids]
    return torch.var(torch.clip(last_air_time, max=0.5), dim=1) + torch.var(
        torch.clip(last_contact_time, max=0.5), dim=1
    )


# ! look into simplifying the kernel here; it's a little oddly complex
def base_motion_penalty(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Penalize base vertical and roll/pitch velocity"""
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    return 0.0 * torch.square(asset.data.root_lin_vel_b[:, 2]) + 0.2 * torch.sum(
        torch.abs(asset.data.root_ang_vel_b[:, :2]), dim=1
    )


def base_orientation_penalty(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Penalize non-flat base orientation (roll-only).

    This is computed by penalizing only the *roll* component from the projected gravity vector.

    Notes:
        - ``asset.data.projected_gravity_b`` is gravity direction expressed in the base frame, ordered as ``[x, y, z]``.
        - With the IsaacLab base-frame convention (x forward, y left, z up), roll shows up primarily in the ``y``
          component.
    """
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    # projected_gravity_b: (num_envs, 3)
    projected_gravity_b = asset.data.projected_gravity_b
    g_y = projected_gravity_b[:, 1]  # (num_envs,)
    return torch.abs(g_y)


def base_pitch_upright_reward(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
    min_pitch_deg: float,
    max_pitch_deg: float,
    pitch_std_deg: float = 5.0,
    min_height_m: float = 0.6,
    max_height_m: float = 0.8,
    height_std_m: float = 0.02,
) -> torch.Tensor:
    """Smooth band-pass reward for being *upright* at the right *height*.

    This term is intentionally simple: it is the product of two smooth band-pass filters:

    - **Pitch band-pass**: 1.0 when pitch is within ``[min_pitch_deg, max_pitch_deg]`` and decays smoothly outside.
    - **Height band-pass**: 1.0 when base height (world z) is within ``[min_height_m, max_height_m]`` and decays
      smoothly outside.

    The final reward is:

    ``r = exp(-(d_theta/sigma_theta)^2) * exp(-(d_z/sigma_z)^2)``

    where ``d_theta`` and ``d_z`` are the distances to the pitch/height bands
    (0 inside the band, positive outside).

    Notes:
        - Pitch is inferred from ``projected_gravity_b`` (gravity direction expressed in base frame).
        - With base-frame convention (x forward, y left, z up), for pure pitch about +y:
          gx = sin(theta), gz = -cos(theta), and thus theta = atan2(gx, -gz).

    Args:
        env: The RL environment instance.
        asset_cfg: The robot rigid-body configuration.
        min_pitch_deg: Lower bound of desired upright pitch band [deg].
        max_pitch_deg: Upper bound of desired upright pitch band [deg].
        pitch_std_deg: Smoothness (decay length-scale) outside pitch band [deg].
        min_height_m: Lower bound of desired base height band [m] using ``root_pos_w[:, 2]``.
        max_height_m: Upper bound of desired base height band [m].
        height_std_m: Smoothness (decay length-scale) outside height band [m].

    Returns:
        Per-environment reward in (0, 1]. Shape is (num_envs,).
    """
    asset: RigidObject = env.scene[asset_cfg.name]

    # -----------------------
    # Pitch band-pass shaping
    # -----------------------
    # projected_gravity_b: (num_envs, 3)
    projected_gravity_b = asset.data.projected_gravity_b
    g_x = projected_gravity_b[:, 0]  # (num_envs,)
    g_z = projected_gravity_b[:, 2]  # (num_envs,)

    # pitch_rad: (num_envs,)
    pitch_rad = torch.atan2(g_x, -g_z)
    min_pitch_rad = pitch_rad.new_tensor(min_pitch_deg) * torch.pi / 180.0
    max_pitch_rad = pitch_rad.new_tensor(max_pitch_deg) * torch.pi / 180.0
    min_pitch_rad, max_pitch_rad = torch.min(min_pitch_rad, max_pitch_rad), torch.max(min_pitch_rad, max_pitch_rad)

    # Distance outside pitch band (0 inside): (num_envs,) [rad]
    d_pitch_below = torch.relu(min_pitch_rad - pitch_rad)
    d_pitch_above = torch.relu(pitch_rad - max_pitch_rad)
    d_pitch_out = d_pitch_below + d_pitch_above

    pitch_std_rad = torch.clamp(pitch_rad.new_tensor(pitch_std_deg) * torch.pi / 180.0, min=1.0e-6)
    r_pitch = torch.exp(-torch.square(d_pitch_out / pitch_std_rad))  # (num_envs,)

    # ------------------------
    # Height band-pass shaping
    # ------------------------
    # root_pos_w: (num_envs, 3) [m]
    root_pos_w = asset.data.root_pos_w
    z_w = root_pos_w[:, 2]  # (num_envs,) [m]

    min_h = z_w.new_tensor(min_height_m)
    max_h = z_w.new_tensor(max_height_m)
    min_h, max_h = torch.min(min_h, max_h), torch.max(min_h, max_h)

    # Distance outside height band (0 inside): (num_envs,) [m]
    d_h_below = torch.relu(min_h - z_w)
    d_h_above = torch.relu(z_w - max_h)
    d_h_out = d_h_below + d_h_above

    h_std = torch.clamp(z_w.new_tensor(height_std_m), min=1.0e-6)
    r_height = torch.exp(-torch.square(d_h_out / h_std))  # (num_envs,)

    return r_pitch * r_height


def base_height_in_range_reward(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
    min_height_m: float,
    max_height_m: float,
    std_m: float = 0.02,
) -> torch.Tensor:
    """Reward keeping the robot base height within a desired band.

    The base height is defined as the robot root/base position in world frame along +z:
    ``z_w = asset.data.root_pos_w[:, 2]``.

    The reward is a smooth band-pass:

    - If ``z_w`` is inside ``[min_height_m, max_height_m]`` then reward is 1.0.
    - If ``z_w`` is outside the band, reward decays as ``exp(-(d/std_m)^2)``, where ``d`` is the distance
      to the closest bound.

    This is useful as a separate shaping term (Option C) so that other orientation/velocity rewards remain
    active regardless of height, while still biasing the policy toward the target standing range.

    Args:
        env: The RL environment instance.
        asset_cfg: The robot rigid-body configuration.
        min_height_m: Lower bound on base height, in meters.
        max_height_m: Upper bound on base height, in meters.
        std_m: Decay length-scale (meters) for being outside the band.

    Returns:
        Per-environment reward. Shape is (num_envs,).
    """
    asset: RigidObject = env.scene[asset_cfg.name]

    # root_pos_w: (num_envs, 3) [m]
    root_pos_w = asset.data.root_pos_w
    z_w = root_pos_w[:, 2]  # (num_envs,) [m]

    # Put scalars on the correct device/dtype and enforce ordering.
    min_h = z_w.new_tensor(min_height_m)
    max_h = z_w.new_tensor(max_height_m)
    min_h, max_h = torch.min(min_h, max_h), torch.max(min_h, max_h)

    # Distance outside the band (0 inside, positive outside): (num_envs,) [m]
    d_below = torch.relu(min_h - z_w)
    d_above = torch.relu(z_w - max_h)
    d_out = d_below + d_above

    std = torch.clamp(z_w.new_tensor(std_m), min=1.0e-6)
    return torch.exp(-torch.square(d_out / std))


# Backwards-compatible alias (was used briefly during experimentation).
def base_orientation_pitch_relaxed_penalty(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
    pitch_relax_up_to_deg: float,
    pitch_relax_down_to_deg: float = 0.0,
) -> torch.Tensor:
    """Alias for older experimental term name.

    This previously meant "roll penalty + pitch relaxation". It is kept only to avoid breaking configs if referenced.
    """
    # Map to the new band-pass reward using the provided relaxed pitch band.
    # Keep height band effectively "always on" for backwards compatibility.
    # % TO-DO: Remove this alias once configs are migrated.
    return base_pitch_upright_reward(
        env=env,
        asset_cfg=asset_cfg,
        min_pitch_deg=pitch_relax_down_to_deg,
        max_pitch_deg=pitch_relax_up_to_deg,
        pitch_std_deg=5.0,
        min_height_m=-1.0e6,
        max_height_m=1.0e6,
        height_std_m=1.0,
    )


def front_feet_contact_penalty(env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg, threshold: float) -> torch.Tensor:
    """Penalize front-feet ground contact.

    This term is intended as a simple shaping signal for learning hind-legs-only behaviors.
    It counts how many selected feet have contact forces above a threshold.

    Args:
        env: The RL environment instance.
        sensor_cfg: The contact sensor entity configuration. Its ``body_ids`` should refer to the *front feet*.
        threshold: Contact-force magnitude threshold to classify contact.

    Returns:
        Per-environment penalty. Shape is (num_envs,).
    """
    contact_sensor = env.scene.sensors[sensor_cfg.name]
    if not isinstance(contact_sensor, ContactSensor):
        raise TypeError(
            f"Sensor '{sensor_cfg.name}' is of type '{type(contact_sensor)}' but this term requires ContactSensor."
        )
    body_ids = sensor_cfg.body_ids
    if body_ids is None:
        raise ValueError(
            "SceneEntityCfg.body_ids is None. Provide front-feet selection via sensor_cfg.body_names/body_ids."
        )

    # net_forces_w_history: (num_envs, history, num_bodies, 3)
    net_contact_forces = contact_sensor.data.net_forces_w_history
    if net_contact_forces is None:
        raise RuntimeError("ContactSensor.net_forces_w_history is None. Increase history length or enable force tracking.")
    # forces_mag: (num_envs, history, num_front_feet)
    forces_mag = torch.norm(net_contact_forces[:, :, body_ids], dim=-1)
    # max_mag: (num_envs, num_front_feet)
    max_mag = torch.max(forces_mag, dim=1)[0]
    # is_contact: (num_envs, num_front_feet)
    is_contact = max_mag > threshold
    # contact_count: (num_envs,)
    contact_count = torch.sum(is_contact.to(dtype=torch.float32), dim=1)
    return contact_count


def foot_slip_penalty(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg, sensor_cfg: SceneEntityCfg, threshold: float
) -> torch.Tensor:
    """Penalize foot planar (xy) slip when in contact with the ground"""
    asset: RigidObject = env.scene[asset_cfg.name]
    # extract the used quantities (to enable type-hinting)
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]

    # check if contact force is above threshold
    net_contact_forces = contact_sensor.data.net_forces_w_history
    is_contact = torch.max(torch.norm(net_contact_forces[:, :, sensor_cfg.body_ids], dim=-1), dim=1)[0] > threshold
    foot_planar_velocity = torch.linalg.norm(asset.data.body_lin_vel_w[:, asset_cfg.body_ids, :2], dim=2)

    reward = is_contact * foot_planar_velocity
    return torch.sum(reward, dim=1)


def joint_acceleration_penalty(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Penalize joint accelerations on the articulation."""
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    return torch.linalg.norm((asset.data.joint_acc), dim=1)


def joint_position_penalty(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg, stand_still_scale: float, velocity_threshold: float
) -> torch.Tensor:
    """Penalize joint position error from default on the articulation."""
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    cmd = torch.linalg.norm(env.command_manager.get_command("base_velocity"), dim=1)
    body_vel = torch.linalg.norm(asset.data.root_lin_vel_b[:, :2], dim=1)
    reward = torch.linalg.norm((asset.data.joint_pos - asset.data.default_joint_pos), dim=1)
    return torch.where(torch.logical_or(cmd > 0.0, body_vel > velocity_threshold), reward, stand_still_scale * reward)


def joint_torques_penalty(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Penalize joint torques on the articulation."""
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    return torch.linalg.norm((asset.data.applied_torque), dim=1)


def joint_velocity_penalty(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Penalize joint velocities on the articulation."""
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    return torch.linalg.norm((asset.data.joint_vel), dim=1)
