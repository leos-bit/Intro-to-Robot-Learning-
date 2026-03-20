"""Drone Skydio env """
import argparse
import json
import time
from typing import Any, Dict, Optional, Union
import warnings
from pathlib import Path
from etils import epath
import jax
import jax.numpy as jp
from lxml import etree
from ml_collections import config_dict
import mujoco
import mujoco.viewer
from mujoco import mjx
import numpy as np

from mujoco_playground._src import mjx_env
from mujoco_playground._src import reward
from mujoco_playground._src.dm_control_suite import common

_XML_PATH = "mujoco_drone_imp/Drone_MJCFs/skydio_x2/scene.xml"
NUM_PID_GAINS = 12


def _safe_l2_norm(x: jax.Array, axis=None) -> jax.Array:
    """Differentiable L2 norm that avoids NaN gradients at zero."""
    x = jp.asarray(x, dtype=jp.float32)
    return jp.sqrt(jp.sum(jp.square(x), axis=axis) + 1e-8)


def _softmin(x: jax.Array, temperature: float) -> jax.Array:
    """Smooth minimum for stable clearance shaping."""
    x = jp.asarray(x, dtype=jp.float32)
    tau = jp.asarray(max(float(temperature), 1e-3), dtype=jp.float32)
    return -tau * jax.nn.logsumexp(-x / tau)

def default_config() -> config_dict.ConfigDict:
      cfg = config_dict.create(
        ctrl_dt=0.01,
        sim_dt=0.001,
        solver_iterations=None,
        episode_length=15000,
        action_repeat=1,
        impl="jax",
        nconmax=64,
        njmax=256,
        model_path=_XML_PATH,
        xylim=10,
        zlim=8,
        vellim=2,
        yawrate_lim=2,
        action_scale=0.5,
        spawn_z_min=0.3,
        target_dist_min=2,
        target_dist_max=7,
        collision_terminate_steps=50,
        w_progress=30.0,
        w_energy=0.01,
        w_smooth=0.02,
        w_speed=0.005,
        r_collision=10.0,
        r_goal=500.0,
        termination_penalty=8.0,
        terminal_distance_penalty=20.0,
        eps_goal=0.85,
        terminate_on_collision=True,
        # k_xy=0.45,
        # k_z=1.2,
        # k_yaw=0.5,
        # ki_xy=0.4,
        # ki_z=0.4,
        # kd_xy=0.3,
        # kd_z=0.1,
        # kp_att=4.5,
        # kd_att=0.45,
        # ki_att=0.4,
        # ki_yaw=0.4,
        # kd_yaw=0.3,
        k_xy=0.25,
        k_z=1.2,
        k_yaw=0.2,
        ki_xy=0.0,
        ki_z=0.00,
        kd_xy=2,
        kd_z=2,
        kp_att=3.0,
        kd_att=2,
        ki_att=0.0,
        ki_yaw=0.0,
        kd_yaw=2.0,
        max_tilt=0.25,
        collective_limit=2.0,
        attitude_limit=1.2,
        yaw_limit=0.7,
        outer_i_limit=2.0,
        inner_i_limit=1.0,
        policy_decim=1,
        outer_decim=2,
        position_hold_epsilon=0.05,
        yaw_hold_epsilon=0.1,
        hover_speed_epsilon=0.15,
        hover_success_steps=25,
        landing_radius=1.5,
        landing_xy_speed=0.35,
        landing_z_speed=0.25,
        landing_xy_damping=1.0,
        landing_z_damping=0.8,
        safety_xy_scale=1.5,
        safety_z_low=-0.1,
        safety_z_high_scale=1.5,

        safety_speed_scale=5.0,
        max_steps=15000,
        max_active_obstacles=15,
        obstacle_center_z=3.2,
        obstacle_spawn_clearance=1.0,
        obstacle_target_clearance=0.8,
        obstacle_min_separation=0.8,
        obstacle_sample_margin=0.6,
        w_obs=0.5,
        lidar_warn_dist=2.0,
        obstacle_safe_dist=0.8,
        lidar_softmin_tau=0.5,
        lidar_risk_weight=0.5,
        true_obstacle_risk_weight=1.0,
        drone_clearance_radius=0.25,
        obstacle_radius=0.2,
      )
      cfg.gain_arr = jp.array([
          cfg.k_xy,
          cfg.ki_xy,
          cfg.kd_xy,
          cfg.k_z,
          cfg.ki_z,
          cfg.kd_z,
          cfg.k_yaw,
          cfg.ki_yaw,
          cfg.kd_yaw,
          cfg.kp_att,
          cfg.kd_att,
          cfg.ki_att,
      ])
      return cfg




class newDrone(mjx_env.MjxEnv):
    def __init__(self, config: config_dict.ConfigDict = default_config(),
      config_overrides: Optional[Dict[str, Union[str, int, list[Any]]]] = None,):
            super().__init__(config, config_overrides)

            self._xml_path = str(Path(self._config.get("model_path", _XML_PATH)).resolve())
            self._model_assets = self._collect_model_assets(Path(self._xml_path))
            self._mj_model = mujoco.MjModel.from_xml_path(self._xml_path)
            self._mj_model.opt.timestep = self.sim_dt
            solver_iterations = self._config.get("solver_iterations", None)
            if solver_iterations is not None:
                self._mj_model.opt.iterations = int(solver_iterations)

            self._mjx_model = mjx.put_model(self._mj_model, impl=self._config.impl)
            self._qpos0 = jp.asarray(self._mj_model.qpos0, dtype=jp.float32)
            self._track_camera_id = mujoco.mj_name2id(
                self._mj_model, mujoco.mjtObj.mjOBJ_CAMERA, "track"
            )
            self._last_info: Optional[dict[str, jax.Array]] = None
            self.max_steps = int(self._config.get("max_steps", self._config.episode_length))
            self.xylim = float(self._config.xylim)
            self.zlim = float(self._config.zlim)
            self.spawn_z_min = float(min(max(float(self._config.spawn_z_min), 0.0), self.zlim))
            self.target_dist_min = float(max(0.0, float(self._config.target_dist_min)))
            tdmax = self._config.get("target_dist_max", None)
            self.target_dist_max = (
                None if tdmax is None else float(max(self.target_dist_min, float(tdmax)))
            )
            self.terminate_on_collision = bool(self._config.terminate_on_collision)
            self.collision_terminate_steps = max(1, int(self._config.collision_terminate_steps))
            self.w_progress = float(self._config.w_progress)
            self.w_energy = float(max(0.0, float(self._config.w_energy)))
            self.w_smooth = float(max(0.0, float(self._config.w_smooth)))
            self.w_speed = float(max(0.0, float(self._config.w_speed)))
            self.r_collision = float(max(0.0, float(self._config.r_collision)))
            self.r_goal = float(max(0.0, float(self._config.r_goal)))
            self.termination_penalty = float(max(0.0, float(self._config.termination_penalty)))
            self.terminal_distance_penalty = float(
                max(0.0, float(self._config.get("terminal_distance_penalty", 0.0)))
            )
            self.eps_goal = float(max(0.1, float(self._config.eps_goal)))
            # self.k_xy = float(self._config.k_xy)
            # self.k_z = float(self._config.k_z)
            # self.k_yaw = float(self._config.k_yaw)
            # self.ki_xy = float(max(0.0, float(self._config.get("ki_xy", 0.0))))
            # self.ki_z = float(max(0.0, float(self._config.get("ki_z", 0.0))))
            # self.kd_xy = float(max(0.0, float(self._config.get("kd_xy", 0.0))))
            # self.kd_z = float(max(0.0, float(self._config.get("kd_z", 0.0))))
            # self.kp_att = float(max(0.0, float(self._config.get("kp_att", 4.0))))
            # self.kd_att = float(max(0.0, float(self._config.get("kd_att", 0.35))))
            # self.ki_att = float(max(0.0, float(self._config.get("ki_att", 0.0))))
            # self.ki_yaw = float(max(0.0, float(self._config.get("ki_yaw", 0.0))))
            # self.kd_yaw = float(max(0.0, float(self._config.get("kd_yaw", 0.0))))
            self.max_tilt = float(
                min(
                    max(float(self._config.get("max_tilt", 0.35)), 0.0),
                    float(np.pi / 3.0),
                )
            )
            self.collective_limit = float(
                max(0.0, float(self._config.get("collective_limit", 2.0)))
            )
            self.attitude_limit = float(
                max(0.0, float(self._config.get("attitude_limit", 1.0)))
            )
            self.yaw_limit = float(max(0.0, float(self._config.get("yaw_limit", 0.3))))
            self.outer_i_limit = float(max(0.0, float(self._config.get("outer_i_limit", 2.0))))
            self.inner_i_limit = float(max(0.0, float(self._config.get("inner_i_limit", 1.0))))
            self.policy_decim = max(1, int(self._config.get("policy_decim", 1)))
            self.outer_decim = max(1, int(self._config.get("outer_decim", 2)))
            # self.kp_pos_xy = float(max(0.0, float(self._config.kp_pos_xy)))
            # self.kp_pos_z = float(max(0.0, float(self._config.kp_pos_z)))
            # self.kp_pos_yaw = float(max(0.0, float(self._config.kp_pos_yaw)))
            self.gain_arr = jp.asarray(self._config.get("gain_arr"))
            self.position_hold_epsilon = float(max(0.0, float(self._config.position_hold_epsilon)))
            self.yaw_hold_epsilon = float(max(0.0, float(self._config.yaw_hold_epsilon)))
            self.hover_speed_epsilon = float(
                max(0.0, float(self._config.get("hover_speed_epsilon", 0.3)))
            )
            self.hover_success_steps = max(
                1, int(self._config.get("hover_success_steps", 15))
            )
            self.landing_radius = float(max(0.1, float(self._config.get("landing_radius", 1.5))))
            self.landing_xy_speed = float(max(0.05, float(self._config.get("landing_xy_speed", 0.35))))
            self.landing_z_speed = float(max(0.05, float(self._config.get("landing_z_speed", 0.25))))
            self.safety_xy_scale = float(max(1.0, float(self._config.safety_xy_scale)))
            self.safety_z_low = float(self._config.safety_z_low)
            self.safety_z_high_scale = float(max(1.0, float(self._config.safety_z_high_scale)))
            self.safety_speed_scale = float(max(1.0, float(self._config.safety_speed_scale)))
            self.vellim = float(self._config.vellim)
            self.yawrate_lim = float(self._config.yawrate_lim)
            self.action_scale = float(min(max(float(self._config.action_scale), 0.05), 1.0))
            self.inner_dt = float(self.sim_dt)
            self.outer_dt = float(self.sim_dt * self.outer_decim)
            self._outer_pid_dim = 3
            self._inner_pid_dim = 3

            self.obs_xy_lim = self.safety_xy_scale * self.xylim
            self.obs_z_low = self.safety_z_low
            self.obs_z_high = self.safety_z_high_scale * self.zlim
            self.obs_vel_lim = self.safety_speed_scale * self.vellim
            self.obs_yawrate_lim = self.safety_speed_scale * self.yawrate_lim
            self.goal_vec_xy_lim = self.xylim + self.obs_xy_lim
            self.goal_vec_z_low = -self.obs_z_high
            self.goal_vec_z_high = self.zlim - self.obs_z_low

            self.body_id = mujoco.mj_name2id(self._mj_model, mujoco.mjtObj.mjOBJ_BODY, "x2")
            self.motor_low = jp.array(self._mj_model.actuator_ctrlrange[:, 0], dtype=jp.float32)
            self.motor_high = jp.array(self._mj_model.actuator_ctrlrange[:, 1], dtype=jp.float32)
            total_mass = float(self._mj_model.body_mass.sum())
            gravity = float(-self._mj_model.opt.gravity[2])
            self.hover_thrust = total_mass * gravity / self._mj_model.nu
            self.hover_ctrl = jp.full((self._mj_model.nu,), self.hover_thrust, dtype=jp.float32)
            if self._mj_model.nkey > 0:
                hover_key_id = mujoco.mj_name2id(
                    self._mj_model,
                    mujoco.mjtObj.mjOBJ_KEY,
                    "hover",
                )
                if hover_key_id != -1:
                    self.hover_ctrl = jp.array(
                        self._mj_model.key_ctrl[hover_key_id],
                        dtype=jp.float32,
                    )

            self._obs_target_low = jp.array([-self.xylim, -self.xylim, 0.0], dtype=jp.float32)
            self._obs_target_high = jp.array([self.xylim, self.xylim, self.zlim], dtype=jp.float32)
            self._obs_xy_low = jp.array([-self.obs_xy_lim, -self.obs_xy_lim], dtype=jp.float32)
            self._obs_xy_high = jp.array([self.obs_xy_lim, self.obs_xy_lim], dtype=jp.float32)
            self._obs_z_low = jp.array([self.obs_z_low], dtype=jp.float32)
            self._obs_z_high = jp.array([self.obs_z_high], dtype=jp.float32)
            self._obs_vel_low = jp.array(
                [-self.obs_vel_lim, -self.obs_vel_lim, -self.obs_vel_lim], dtype=jp.float32
            )
            self._obs_vel_high = jp.array(
                [self.obs_vel_lim, self.obs_vel_lim, self.obs_vel_lim], dtype=jp.float32
            )
            self._obs_yaw_low = jp.array([-self.obs_yawrate_lim], dtype=jp.float32)
            self._obs_yaw_high = jp.array([self.obs_yawrate_lim], dtype=jp.float32)
            self._obs_goal_low = jp.array(
                [-self.goal_vec_xy_lim, -self.goal_vec_xy_lim, self.goal_vec_z_low], dtype=jp.float32
            )
            self._obs_goal_high = jp.array(
                [self.goal_vec_xy_lim, self.goal_vec_xy_lim, self.goal_vec_z_high], dtype=jp.float32
            )
            self._action_low = jp.full((4,), -1.0, dtype=jp.float32)
            self._action_high = jp.full((4,), 1.0, dtype=jp.float32)
            self.obstacle_center_z = float(self._config.get("obstacle_center_z", 0.75))
            self.obstacle_spawn_clearance = float(
                max(0.0, float(self._config.get("obstacle_spawn_clearance", 1.0)))
            )
            self.obstacle_target_clearance = float(
                max(0.0, float(self._config.get("obstacle_target_clearance", 0.8)))
            )
            self.obstacle_min_separation = float(
                max(0.0, float(self._config.get("obstacle_min_separation", 0.8)))
            )
            self.obstacle_sample_margin = float(
                max(0.0, float(self._config.get("obstacle_sample_margin", 0.6)))
            )
            self.w_obs = float(max(0.0, float(self._config.get("w_obs", 0.1))))
            self.lidar_warn_dist = float(
                max(0.0, float(self._config.get("lidar_warn_dist", 2.0)))
            )
            self.obstacle_safe_dist = float(
                max(0.0, float(self._config.get("obstacle_safe_dist", 0.8)))
            )
            self.lidar_softmin_tau = float(
                max(1e-3, float(self._config.get("lidar_softmin_tau", 0.5)))
            )
            self.lidar_risk_weight = float(
                max(0.0, float(self._config.get("lidar_risk_weight", 0.5)))
            )
            self.true_obstacle_risk_weight = float(
                max(0.0, float(self._config.get("true_obstacle_risk_weight", 1.0)))
            )
            self.drone_clearance_radius = float(
                max(0.0, float(self._config.get("drone_clearance_radius", 0.25)))
            )
            self.obstacle_radius = float(
                max(0.0, float(self._config.get("obstacle_radius", 0.2)))
            )
            self._obstacle_quat = jp.array([1.0, 0.0, 0.0, 0.0], dtype=jp.float32)
            self._obstacle_body_names = self._discover_obstacle_body_names()
            self.max_obstacles = len(self._obstacle_body_names)
            requested_max_active = int(
                self._config.get("max_active_obstacles", self.max_obstacles)
            )
            self.max_active_obstacles = min(
                max(requested_max_active, 0), self.max_obstacles
            )
            self._obstacle_mocap_ids = self._build_obstacle_mocap_ids(self._obstacle_body_names)
            self._drone_geom_mask, self._obstacle_geom_mask = self._build_contact_geom_masks(
                self._obstacle_body_names
            )
            template_data = mujoco.MjData(self._mj_model)
            mujoco.mj_forward(self._mj_model, template_data)
            self._data0 = mjx.put_data(
                self._mj_model,
                template_data,
                impl=self._config.impl,
                nconmax=self._config.nconmax,
                njmax=self._config.njmax,
            )
            self._default_mocap_pos = jp.asarray(template_data.mocap_pos, dtype=jp.float32)
            self._default_mocap_quat = jp.asarray(template_data.mocap_quat, dtype=jp.float32)
            if self.max_obstacles > 0:
                self._obstacle_park_positions = self._default_mocap_pos[
                    jp.asarray(self._obstacle_mocap_ids, dtype=jp.int32)
                ]
                self._obstacle_park_quats = self._default_mocap_quat[
                    jp.asarray(self._obstacle_mocap_ids, dtype=jp.int32)
                ]
            else:
                self._obstacle_park_positions = jp.zeros((0, 3), dtype=jp.float32)
                self._obstacle_park_quats = jp.zeros((0, 4), dtype=jp.float32)
            self._obstacle_rel_xy_lim = self.obs_xy_lim + self.xylim
            self._obstacle_rel_z_lim = self.obs_z_high + abs(self.obstacle_center_z)
            self._obstacle_rel_low = jp.tile(
                jp.array(
                    [
                        -self._obstacle_rel_xy_lim,
                        -self._obstacle_rel_xy_lim,
                        -self._obstacle_rel_z_lim,
                    ],
                    dtype=jp.float32,
                )[None, :],
                (self.max_obstacles, 1),
            )
            self._obstacle_rel_high = jp.tile(
                jp.array(
                    [
                        self._obstacle_rel_xy_lim,
                        self._obstacle_rel_xy_lim,
                        self._obstacle_rel_z_lim,
                    ],
                    dtype=jp.float32,
                )[None, :],
                (self.max_obstacles, 1),
            )

            self.lidar_max_dist = 6.0
            # Cache sensor slices once so sensordata reads do not depend on XML order.
            self._gyro_sensor_slice = self._sensor_slice("body_gyro")
            self._accel_sensor_slice = self._sensor_slice("body_linacc")
            self._quat_sensor_slice = self._sensor_slice("body_quat")
            self._lidar_sensor_names = (
                "lidar_front",
                "lidar_left",
                "lidar_back",
                "lidar_right",
                "lidar_up",
                "lidar_down",
                "lidar_front_left",
                "lidar_front_right",
                "lidar_back_left",
                "lidar_back_right",
                "lidar_up_left",
                "lidar_up_right",
                "lidar_up_front",
                "lidar_up_back",
                "lidar_down_left",
                "lidar_down_right",
                "lidar_down_front",
                "lidar_down_back",
            )
            self._lidar_sensor_slices = tuple(
                self._sensor_slice(name) for name in self._lidar_sensor_names
            )
            self.num_lidar = len(self._lidar_sensor_slices)
            self._horizontal_lidar_indices = jp.asarray(
                [0, 1, 2, 3, 6, 7, 8, 9],
                dtype=jp.int32,
            )
            
            self.obs_spec = {
                "agent_pos_xy": {
                    "low": self._obs_xy_low,
                    "high": self._obs_xy_high,
                    "shape": (2,),
                    "dtype": jp.float32,
                },
                "agent_pos_z": {
                    "low": self._obs_z_low,
                    "high": self._obs_z_high,
                    "shape": (1,),
                    "dtype": jp.float32,
                },
                "agent_orientation": {
                    "low": jp.full((4,), -1.0, dtype=jp.float32),
                    "high": jp.full((4,), 1.0, dtype=jp.float32),
                    "shape": (4,),
                    "dtype": jp.float32,
                },
                "agent_vel": {
                    "low": self._obs_vel_low,
                    "high": self._obs_vel_high,
                    "shape": (3,),
                    "dtype": jp.float32,
                },
                "target": {
                    "low": self._obs_target_low,
                    "high": self._obs_target_high,
                    "shape": (3,),
                    "dtype": jp.float32,
                },
                "agent_yawrate": {
                    "low": self._obs_yaw_low,
                    "high": self._obs_yaw_high,
                    "shape": (1,),
                    "dtype": jp.float32,
                },
                "goal_vec": {
                    "low": self._obs_goal_low,
                    "high": self._obs_goal_high,
                    "shape": (3,),
                    "dtype": jp.float32,
                },
                "lidar": {
                    "low": jp.zeros((self.num_lidar,), dtype=jp.float32),
                    "high": jp.full((self.num_lidar,), self.lidar_max_dist, dtype=jp.float32),
                    "shape": (self.num_lidar,),
                    "dtype": jp.float32,
                },
                "obstacle_rel": {
                    "low": self._obstacle_rel_low,
                    "high": self._obstacle_rel_high,
                    "shape": (self.max_obstacles, 3),
                    "dtype": jp.float32,
                },
                "obstacle_mask": {
                    "low": jp.zeros((self.max_obstacles,), dtype=jp.float32),
                    "high": jp.ones((self.max_obstacles,), dtype=jp.float32),
                    "shape": (self.max_obstacles,),
                    "dtype": jp.float32,
                },
                "num_active": {
                    "low": jp.zeros((1,), dtype=jp.float32),
                    "high": jp.full((1,), float(self.max_active_obstacles), dtype=jp.float32),
                    "shape": (1,),
                    "dtype": jp.float32,
                },

            }
            self.action_spec = {
                "low": self._action_low,
                "high": self._action_high,
                "shape": (4,),
                "dtype": jp.float32,
            }

    @staticmethod
    def _collect_model_assets(xml_path: Path) -> dict[str, bytes]:
        """Collect local XML assets for downstream tools (e.g., rscope)."""
        assets: dict[str, bytes] = {}
        root = xml_path.parent
        for file_path in root.rglob("*"):
            if not file_path.is_file():
                continue
            content = file_path.read_bytes()
            rel_key = file_path.relative_to(root).as_posix()
            # Use canonical relative paths only. MuJoCo/rscope rejects duplicated
            # assets that differ only by directory prefix.
            assets[rel_key] = content
        return assets

    def _sensor_slice(self, name: str) -> slice:
        sensor_id = mujoco.mj_name2id(self._mj_model, mujoco.mjtObj.mjOBJ_SENSOR, name)
        if sensor_id == -1:
            raise ValueError(f"Unknown sensor: {name}")
        start = int(self._mj_model.sensor_adr[sensor_id])
        dim = int(self._mj_model.sensor_dim[sensor_id])
        return slice(start, start + dim)

    def _discover_obstacle_body_names(self) -> tuple[str, ...]:
        obstacle_names: list[str] = []
        for body_id in range(self._mj_model.nbody):
            body_name = mujoco.mj_id2name(self._mj_model, mujoco.mjtObj.mjOBJ_BODY, body_id)
            if body_name and body_name.startswith("obstacle_"):
                obstacle_names.append(body_name)
        return tuple(sorted(obstacle_names, key=lambda name: int(name.rsplit("_", 1)[1])))

    def _build_obstacle_mocap_ids(self, obstacle_body_names: tuple[str, ...]) -> tuple[int, ...]:
        mocap_ids = []
        for body_name in obstacle_body_names:
            body_id = mujoco.mj_name2id(self._mj_model, mujoco.mjtObj.mjOBJ_BODY, body_name)
            mocap_id = int(self._mj_model.body_mocapid[body_id])
            if mocap_id < 0:
                raise ValueError(f"Obstacle body {body_name} must be a mocap body.")
            mocap_ids.append(mocap_id)
        return tuple(mocap_ids)

    def _build_contact_geom_masks(
        self, obstacle_body_names: tuple[str, ...]
    ) -> tuple[jax.Array, jax.Array]:
        geom_bodyid = np.asarray(self._mj_model.geom_bodyid, dtype=np.int32)
        drone_geom_mask = geom_bodyid == self.body_id
        if obstacle_body_names:
            obstacle_body_ids = np.asarray(
                [
                    mujoco.mj_name2id(self._mj_model, mujoco.mjtObj.mjOBJ_BODY, body_name)
                    for body_name in obstacle_body_names
                ],
                dtype=np.int32,
            )
            obstacle_geom_mask = np.isin(geom_bodyid, obstacle_body_ids)
        else:
            obstacle_geom_mask = np.zeros_like(drone_geom_mask, dtype=bool)
        return (
            jp.asarray(drone_geom_mask, dtype=bool),
            jp.asarray(obstacle_geom_mask, dtype=bool),
        )

    def _detect_drone_contacts(
        self, data: mjx.Data
    ) -> tuple[jax.Array, jax.Array, jax.Array]:
        impl = data._impl if hasattr(data, "_impl") else data
        contact_geom = impl.contact.geom
        contact_dist = impl.contact.dist
        valid_contact = contact_dist <= 0.0
        geom1 = jp.where(valid_contact, contact_geom[:, 0], 0)
        geom2 = jp.where(valid_contact, contact_geom[:, 1], 0)

        drone1 = self._drone_geom_mask[geom1]
        drone2 = self._drone_geom_mask[geom2]
        involves_drone = valid_contact & jp.logical_xor(drone1, drone2)

        obstacle1 = self._obstacle_geom_mask[geom1]
        obstacle2 = self._obstacle_geom_mask[geom2]
        obstacle_contact = involves_drone & (obstacle1 | obstacle2)
        environment_contact = involves_drone & (~(obstacle1 | obstacle2))

        return (
            jp.any(involves_drone),
            jp.any(obstacle_contact),
            jp.any(environment_contact),
        )

    def _extract_obstacle_positions(self, data: mjx.Data) -> jax.Array:
        if self.max_obstacles == 0:
            return jp.zeros((0, 3), dtype=jp.float32)
        return jp.stack([data.mocap_pos[mocap_id] for mocap_id in self._obstacle_mocap_ids])

    def _place_obstacles_in_mocap(
        self,
        base_mocap_pos: jax.Array,
        base_mocap_quat: jax.Array,
        obstacle_positions: jax.Array,
        obstacle_mask: jax.Array,
    ) -> tuple[jax.Array, jax.Array]:
        mocap_pos = jp.asarray(base_mocap_pos, dtype=jp.float32)
        mocap_quat = jp.asarray(base_mocap_quat, dtype=jp.float32)
        obstacle_mask = jp.asarray(obstacle_mask, dtype=jp.bool_)
        for idx, mocap_id in enumerate(self._obstacle_mocap_ids):
            mocap_pos = mocap_pos.at[mocap_id].set(
                jp.where(obstacle_mask[idx], obstacle_positions[idx], self._obstacle_park_positions[idx])
            )
            mocap_quat = mocap_quat.at[mocap_id].set(
                jp.where(obstacle_mask[idx], self._obstacle_quat, self._obstacle_park_quats[idx])
            )
        return mocap_pos, mocap_quat

    def _sample_obstacles(
        self,
        rng: jax.Array,
        agent_location: jax.Array,
    ) -> tuple[jax.Array, jax.Array, jax.Array]:
        if self.max_obstacles == 0 or self.max_active_obstacles == 0:
            empty_mask = jp.zeros((self.max_obstacles,), dtype=jp.float32)
            return self._obstacle_park_positions, empty_mask, jp.array(0, dtype=jp.int32)

        rng, count_rng = jax.random.split(rng)
        num_active = jax.random.randint(
            count_rng,
            (),
            minval=7,
            maxval=self.max_active_obstacles + 1,
            dtype=jp.int32,
        )
        candidate_rngs = jax.random.split(rng, self.max_obstacles)
        sample_limit = max(self.xylim - self.obstacle_sample_margin, 1e-3)
        obstacle_positions = self._obstacle_park_positions
        obstacle_mask = jp.zeros((self.max_obstacles,), dtype=jp.bool_)
        num_candidates = 256

        for idx in range(self.max_obstacles):
            rx, ry = jax.random.split(candidate_rngs[idx])
            candidates_xy = jp.stack(
                [
                    jax.random.uniform(
                        rx,
                        shape=(num_candidates,),
                        minval=-sample_limit,
                        maxval=sample_limit,
                    ),
                    jax.random.uniform(
                        ry,
                        shape=(num_candidates,),
                        minval=-sample_limit,
                        maxval=sample_limit,
                    ),
                ],
                axis=1,
            ).astype(jp.float32)
            candidates = jp.concatenate(
                [
                    candidates_xy,
                    jp.full((num_candidates, 1), self.obstacle_center_z, dtype=jp.float32),
                ],
                axis=1,
            )

            agent_dists = jp.linalg.norm(candidates[:, :2] - agent_location[None, :2], axis=1)
            valid = agent_dists >= self.obstacle_spawn_clearance
            if idx > 0:
                pairwise_dists = jp.linalg.norm(
                    candidates[:, None, :2] - obstacle_positions[None, :idx, :2],
                    axis=-1,
                )
                prev_mask = obstacle_mask[:idx][None, :]
                valid = valid & jp.all(
                    (~prev_mask) | (pairwise_dists >= self.obstacle_min_separation),
                    axis=1,
                )

            first_valid = jp.argmax(valid.astype(jp.int32))
            chosen_idx = jp.where(jp.any(valid), first_valid, num_candidates - 1)
            place_obstacle = (idx < num_active) & jp.any(valid)
            chosen_position = jp.where(
                place_obstacle,
                candidates[chosen_idx],
                self._obstacle_park_positions[idx],
            )
            obstacle_positions = obstacle_positions.at[idx].set(chosen_position)
            obstacle_mask = obstacle_mask.at[idx].set(place_obstacle)

        actual_num_active = jp.sum(obstacle_mask.astype(jp.int32))
        return obstacle_positions, obstacle_mask.astype(jp.float32), actual_num_active
    
    def _sample_target(
        self,
        rng: jax.Array,
        agent_location: jax.Array,
        obstacle_positions: Optional[jax.Array] = None,
        obstacle_mask: Optional[jax.Array] = None,
    ) -> jax.Array:

        num_samples = 200
        rx, ry, rz = jax.random.split(rng, 3)
        z_span = max(self.zlim - self.spawn_z_min, 0.0)
        target_z_samples = jax.random.uniform(
            rz,
            shape=(num_samples,),
            minval=self.spawn_z_min,
            maxval=self.spawn_z_min + max(z_span, 1e-6),
        )
        if z_span <= 1e-6:
            target_z_samples = jp.full((num_samples,), self.spawn_z_min, dtype=jp.float32)
        candidates = jp.stack(
            [
                jax.random.uniform(
                    rx, shape=(num_samples,), minval=-self.xylim, maxval=self.xylim
                ),
                jax.random.uniform(
                    ry, shape=(num_samples,), minval=-self.xylim, maxval=self.xylim
                ),
                target_z_samples,
            ],
            axis=1,
        ).astype(jp.float32)

        dists = jp.linalg.norm(candidates - agent_location[None, :], axis=1)
        valid = dists >= self.target_dist_min
        if self.target_dist_max is not None:
            valid = jp.logical_and(valid, dists <= self.target_dist_max)
        if obstacle_positions is not None and obstacle_mask is not None and self.max_obstacles > 0:
            obstacle_mask = jp.asarray(obstacle_mask, dtype=jp.bool_)
            obstacle_dists = jp.linalg.norm(
                candidates[:, None, :2] - obstacle_positions[None, :, :2],
                axis=-1,
            )
            valid = valid & jp.all(
                (~obstacle_mask[None, :]) | (obstacle_dists >= self.obstacle_target_clearance),
                axis=1,
            )

        first_valid = jp.argmax(valid.astype(jp.int32))
        chosen_idx = jp.where(jp.any(valid), first_valid, num_samples - 1)
        return candidates[chosen_idx]

    def _obstacle_reward_terms(
        self,
        agent_location: jax.Array,
        lidar: jax.Array,
        obstacle_positions: jax.Array,
        obstacle_mask: jax.Array,
    ) -> tuple[jax.Array, jax.Array, jax.Array]:
        horizontal_lidar = jp.asarray(lidar, dtype=jp.float32)[self._horizontal_lidar_indices]
        lidar_clearance = _softmin(horizontal_lidar, self.lidar_softmin_tau)

        obstacle_mask = jp.asarray(obstacle_mask, dtype=jp.bool_)
        if self.max_obstacles > 0:
            obstacle_xy_dist = jp.linalg.norm(
                obstacle_positions[:, :2] - agent_location[None, :2],
                axis=-1,
            )
            true_clearance_all = obstacle_xy_dist - (
                self.drone_clearance_radius + self.obstacle_radius
            )
            true_clearance_all = jp.where(
                obstacle_mask,
                true_clearance_all,
                jp.full_like(true_clearance_all, jp.inf),
            )
            true_clearance = jp.where(
                jp.any(obstacle_mask),
                jp.min(true_clearance_all),
                jp.asarray(self.lidar_max_dist, dtype=jp.float32),
            )
        else:
            true_clearance = jp.asarray(self.lidar_max_dist, dtype=jp.float32)

        lidar_risk = self.lidar_risk_weight * jp.square(
            jp.maximum(0.0, self.lidar_warn_dist - lidar_clearance)
        )
        true_risk = self.true_obstacle_risk_weight * jp.square(
            jp.maximum(0.0, self.obstacle_safe_dist - true_clearance)
        )
        obstacle_risk = lidar_risk + true_risk
        return (
            jp.asarray(lidar_clearance, dtype=jp.float32),
            jp.asarray(true_clearance, dtype=jp.float32),
            jp.asarray(obstacle_risk, dtype=jp.float32),
        )

    def _get_info(self, agent_location: jax.Array, target: jax.Array, initial_distance: jax.Array):
        distance = jp.linalg.norm(agent_location - target)
        return {
            "distance": distance,
            "initial_distance": initial_distance,
        }
    def reset(self, rng: jax.Array) -> mjx_env.State:
        rng, spawn_rng, obstacle_rng, target_rng = jax.random.split(rng, 4)
        sx, sy, sz = jax.random.split(spawn_rng, 3)
        z_span = max(self.zlim - self.spawn_z_min, 0.0)
        spawn_z = jax.random.uniform(
            sz,
            minval=self.spawn_z_min,
            maxval=self.spawn_z_min + max(z_span, 1e-6),
        )
        if z_span <= 1e-6:
            spawn_z = jp.array(self.spawn_z_min, dtype=jp.float32)
        agent_location = jp.array(
            [
                jax.random.uniform(sx, minval=-self.xylim, maxval=self.xylim),
                jax.random.uniform(sy, minval=-self.xylim, maxval=self.xylim),
                spawn_z,
            ],
            dtype=jp.float32,
        )

        qpos = self._qpos0
        qpos = qpos.at[:3].set(agent_location)
        qpos = qpos.at[3:7].set(jp.array([1.0, 0.0, 0.0, 0.0], dtype=jp.float32))
        obstacle_positions, obstacle_mask, num_active = self._sample_obstacles(
            obstacle_rng, agent_location
        )
        mocap_pos, mocap_quat = self._place_obstacles_in_mocap(
            self._default_mocap_pos,
            self._default_mocap_quat,
            obstacle_positions,
            obstacle_mask,
        )
        qvel = jp.zeros((self.mjx_model.nv,), dtype=jp.float32)
        data = self._data0.replace(
            qpos=qpos,
            qvel=qvel,
            act=jp.zeros_like(self._data0.act),
            qacc_warmstart=jp.zeros_like(self._data0.qacc_warmstart),
            mocap_pos=mocap_pos,
            mocap_quat=mocap_quat,
        )
        data = data.replace(ctrl=self.hover_ctrl.astype(data.ctrl.dtype))
        data = mjx.forward(self.mjx_model, data)

        (
            agent_location,
            agent_vel,
            agent_angvel,
            agent_orientation,
            agent_yawrate,
            lidar,
        ) = self._extract_body_state(data)
        obstacle_positions = self._extract_obstacle_positions(data)
        target = self._sample_target(
            target_rng,
            agent_location,
            obstacle_positions=obstacle_positions,
            obstacle_mask=obstacle_mask,
        )
        initial_target_distance = jp.linalg.norm(target - agent_location)
        lidar_clearance, true_obstacle_clearance, obstacle_risk = self._obstacle_reward_terms(
            agent_location,
            lidar,
            obstacle_positions,
            obstacle_mask,
        )

        info = {
            "rng": rng,
            "agent_location": agent_location,
            "agent_vel": agent_vel,
            "agent_angvel": agent_angvel,
            "agent_orientation": agent_orientation,
            "agent_yawrate": agent_yawrate,
            "target": target,
            "prev_action": jp.zeros((4,), dtype=jp.float32),
            "held_action": jp.zeros((4,), dtype=jp.float32),
            "outer_cmd": jp.zeros((4,), dtype=jp.float32),
            "last_motor_cmd": self.hover_ctrl.astype(jp.float32),
            "outer_pid_state": self._zero_pid_state(self._outer_pid_dim),
            "inner_pid_state": self._zero_pid_state(self._inner_pid_dim),
            "sim_step": jp.array(0, dtype=jp.int32),
            "prev_distance": initial_target_distance,
            "initial_target_distance": initial_target_distance,
            "min_distance_to_goal": initial_target_distance,
            "step": jp.array(0, dtype=jp.int32),
            "collision_streak": jp.array(0, dtype=jp.int32),
            "goal_hold_streak": jp.array(0, dtype=jp.int32),
            "distance": initial_target_distance,
            "initial_distance": initial_target_distance,
            "lidar": lidar,
            "obstacle_positions": obstacle_positions,
            "obstacle_mask": obstacle_mask,
            "num_active": num_active.astype(jp.int32),
            "prev_obstacle_risk": obstacle_risk,
            "obstacle_risk": obstacle_risk,
            "lidar_clearance": lidar_clearance,
            "true_obstacle_clearance": true_obstacle_clearance,
            "r_prog": jp.array(0.0, dtype=jp.float32),
            "r_obs": jp.array(0.0, dtype=jp.float32),
            "r_coll": jp.array(0.0, dtype=jp.float32),
            "r_energy": jp.array(0.0, dtype=jp.float32),
            "r_smooth": jp.array(0.0, dtype=jp.float32),
            "r_safety": jp.array(0.0, dtype=jp.float32),
            "r_speed": jp.array(0.0, dtype=jp.float32),
            "r_terminal": jp.array(0.0, dtype=jp.float32),
            "raw_action_l2": jp.array(0.0, dtype=jp.float32),
            "scaled_action_l2": jp.array(0.0, dtype=jp.float32),
            "success": jp.array(False),
            "hovering_at_goal": jp.array(False),
            "collision": jp.array(False),
            "obstacle_collision": jp.array(False),
            "environment_collision": jp.array(False),
            "collision_terminated": jp.array(False),
            "out_of_bounds": jp.array(False),
            "excessive_speed": jp.array(False),
            "numerical_issue": jp.array(False),
            "terminated": jp.array(False),
            "truncated": jp.array(False),
            "reward": jp.array(0.0, dtype=jp.float32),
            
        }
        metrics = self._init_step_metrics(initial_target_distance, initial_target_distance)
        obs = self._get_obs(info)
        self._last_info = info
        return mjx_env.State(
            data=data,
            obs=obs,
            reward=jp.array(0.0, dtype=jp.float32),
            # Brax wrappers expect numeric done for stable truncation/episode_done dtypes.
            done=jp.array(0.0, dtype=jp.float32),
            metrics=metrics,
            info=info,
        )
    def _init_step_metrics(
        self,
        distance: jax.Array,
        initial_distance: jax.Array,
    ) -> dict[str, jax.Array]:
        zero = jp.array(0.0, dtype=jp.float32)
        dist_f32 = jp.asarray(distance, dtype=jp.float32)
        return {
            "distance": dist_f32,
            "distance_to_goal_per_step": dist_f32,
            "final_distance_to_goal": zero,
            "best_distance_to_goal": zero,
            "initial_distance": jp.asarray(initial_distance, dtype=jp.float32),
            "r_prog": zero,
            "r_obs": zero,
            "r_coll": zero,
            "r_energy": zero,
            "r_smooth": zero,
            "r_safety": zero,
            "r_speed": zero,
            "r_terminal": zero,
            "lidar_clearance": jp.asarray(self.lidar_max_dist, dtype=jp.float32),
            "true_obstacle_clearance": jp.asarray(self.lidar_max_dist, dtype=jp.float32),
            "raw_action_l2": zero,
            "scaled_action_l2": zero,
            # Keep all metrics float32 so Brax EvalWrapper aggregation is type-stable.
            "success": zero,
            "hovering_at_goal": zero,
            "collision": zero,
            "obstacle_collision": zero,
            "environment_collision": zero,
            "collision_streak": zero,
            "goal_hold_streak": zero,
            "collision_terminated": zero,
            "out_of_bounds": zero,
            "excessive_speed": zero,
            "numerical_issue": zero,
            "terminated": zero,
            "truncated": zero,
            "reward": zero,
        }

    def _get_obs(self, info: dict[str, jax.Array]):
        agent_location = jp.nan_to_num(
            info["agent_location"],
            nan=0.0,
            posinf=self.obs_xy_lim,
            neginf=-self.obs_xy_lim,
        ).astype(jp.float32)
        agent_vel = jp.nan_to_num(
            info["agent_vel"],
            nan=0.0,
            posinf=self.obs_vel_lim,
            neginf=-self.obs_vel_lim,
        ).astype(jp.float32)
        agent_orientation = jp.nan_to_num(
            info["agent_orientation"],
            nan=0.0,
            posinf=1.0,
            neginf=-1.0,
        ).astype(jp.float32)
        agent_orientation = jp.clip(agent_orientation, -1.0, 1.0)
        agent_yawrate = jp.nan_to_num(
            info["agent_yawrate"],
            nan=0.0,
            posinf=self.obs_yawrate_lim,
            neginf=-self.obs_yawrate_lim,
        ).astype(jp.float32)
        target = jp.nan_to_num(
            info["target"],
            nan=0.0,
            posinf=self.xylim,
            neginf=-self.xylim,
        ).astype(jp.float32)
        target = jp.clip(target, self._obs_target_low, self._obs_target_high)
        goal_vec = (target - agent_location).astype(jp.float32)
        lidar = jp.nan_to_num(
            info["lidar"],
            nan=self.lidar_max_dist,
            posinf=self.lidar_max_dist,
            neginf=0.0,
        ).astype(jp.float32)
        lidar = jp.clip(lidar, 0.0, self.lidar_max_dist)
        obstacle_positions = jp.nan_to_num(
            info["obstacle_positions"],
            nan=0.0,
            posinf=self._obstacle_rel_xy_lim,
            neginf=-self._obstacle_rel_xy_lim,
        ).astype(jp.float32)
        obstacle_mask = jp.clip(
            jp.asarray(info["obstacle_mask"], dtype=jp.float32),
            0.0,
            1.0,
        )
        obstacle_rel = obstacle_positions - agent_location[None, :]
        obstacle_rel = jp.where(obstacle_mask[:, None] > 0.0, obstacle_rel, 0.0)
        obstacle_rel = jp.clip(obstacle_rel, self._obstacle_rel_low, self._obstacle_rel_high)
        num_active = jp.clip(
            jp.asarray(info["num_active"], dtype=jp.float32).reshape((1,)),
            0.0,
            float(self.max_active_obstacles),
        )
        return {
            "agent_pos_xy": jp.clip(agent_location[0:2], self._obs_xy_low, self._obs_xy_high),
            "agent_pos_z": jp.clip(agent_location[2:3], self._obs_z_low, self._obs_z_high),
            "agent_orientation": agent_orientation,
            "agent_vel": jp.clip(agent_vel, self._obs_vel_low, self._obs_vel_high),
            "target": target,
            "agent_yawrate": jp.clip(agent_yawrate, self._obs_yaw_low, self._obs_yaw_high),
            "goal_vec": jp.clip(goal_vec, self._obs_goal_low, self._obs_goal_high),
            "lidar": lidar,
            "obstacle_rel": obstacle_rel,
            "obstacle_mask": obstacle_mask,
            "num_active": num_active,
        }

    def _pack_pid_state(self, integral: jax.Array, prev_error: jax.Array) -> jax.Array:
        return jp.concatenate(
            [
                jp.asarray(integral, dtype=jp.float32),
                jp.asarray(prev_error, dtype=jp.float32),
            ],
            axis=0,
        )

    def _unpack_pid_state(self, pid_state: jax.Array, dim: int) -> tuple[jax.Array, jax.Array]:
        pid_state = jp.asarray(pid_state, dtype=jp.float32).reshape((2 * dim,))
        return pid_state[:dim], pid_state[dim:]

    def _zero_pid_state(self, dim: int) -> jax.Array:
        return jp.zeros((2 * dim,), dtype=jp.float32)

    def _goal_hover_blend(
        self,
        agent_location: jax.Array,
        agent_vel: jax.Array,
        target: jax.Array,
    ) -> jax.Array:
        dist = _safe_l2_norm(target - agent_location)
        del agent_vel
        hover_radius = jp.asarray(self.landing_radius, dtype=jp.float32)
        return jp.clip(1.0 - (dist / jp.maximum(hover_radius, 1e-6)), 0.0, 1.0)

    def _goal_landing_command(
        self,
        agent_location: jax.Array,
        target: jax.Array,
    ) -> jax.Array:
        goal_vec = jp.asarray(target - agent_location, dtype=jp.float32)
        return jp.array(
            [
                jp.clip(goal_vec[0], -self.landing_xy_speed, self.landing_xy_speed),
                jp.clip(goal_vec[1], -self.landing_xy_speed, self.landing_xy_speed),
                jp.clip(goal_vec[2], -self.landing_z_speed, self.landing_z_speed),
                0.0,
            ],
            dtype=jp.float32,
        )

    def _quat_to_roll_pitch_yaw(self, quat: jax.Array) -> tuple[jax.Array, jax.Array, jax.Array]:
        quat = jp.nan_to_num(quat, nan=0.0, posinf=1.0, neginf=-1.0)
        quat = quat / jp.maximum(jp.linalg.norm(quat), 1e-6)
        w, x, y, z = quat

        sin_roll = 2.0 * ((w * x) + (y * z))
        cos_roll = 1.0 - (2.0 * ((x * x) + (y * y)))
        roll = jp.arctan2(sin_roll, cos_roll)

        sin_pitch = 2.0 * ((w * y) - (z * x))
        pitch = jp.arcsin(jp.clip(sin_pitch, -1.0 + 1e-6, 1.0 - 1e-6))

        sin_yaw = 2.0 * ((w * z) + (x * y))
        cos_yaw = 1.0 - (2.0 * ((y * y) + (z * z)))
        yaw = jp.arctan2(sin_yaw, cos_yaw)
        return roll, pitch, yaw

    def action_to_command(self, action: jax.Array) -> jax.Array:
        cmd = jp.asarray(action, dtype=jp.float32).reshape((4,))
        cmd = jp.nan_to_num(cmd, nan=0.0, posinf=1.0, neginf=-1.0)
        cmd = jp.clip(cmd, -1.0, 1.0)
        return jp.array(
            [
                cmd[0] * self.vellim,
                cmd[1] * self.vellim,
                cmd[2] * self.vellim,
                cmd[3] * self.yawrate_lim,
            ],
            dtype=jp.float32,
        )

    def velocity_controller(
        self,
        agent_vel: jax.Array,
        agent_orientation: jax.Array,
        cmd: jax.Array,
        pid_state: jax.Array,
        dt: float,
        gain_arr: jax.Array
    ) -> tuple[jax.Array, jax.Array]:
        agent_vel = jp.nan_to_num(agent_vel, nan=0.0, posinf=0.0, neginf=0.0)
        cmd = jp.nan_to_num(cmd, nan=0.0, posinf=0.0, neginf=0.0)
        roll, pitch, yaw = self._quat_to_roll_pitch_yaw(agent_orientation)
        del roll, pitch
        k_xy = gain_arr[0]
        ki_xy = gain_arr[1] 
        kd_xy = gain_arr[2]
        k_z = gain_arr[3]
        ki_z = gain_arr[4]
        kd_z = gain_arr[5]
        k_yaw = gain_arr[6]
        ki_yaw = gain_arr[7]
        kd_yaw = gain_arr[8]
        kp_att = gain_arr[9]
        kd_att = gain_arr[10]
        ki_att = gain_arr[11]
        vel_err_world = cmd[:3] - agent_vel
        cos_yaw = jp.cos(yaw)
        sin_yaw = jp.sin(yaw)
        vel_err_body = jp.array(
            [
                (cos_yaw * vel_err_world[0]) + (sin_yaw * vel_err_world[1]),
                (-sin_yaw * vel_err_world[0]) + (cos_yaw * vel_err_world[1]),
                vel_err_world[2],
            ],
            dtype=jp.float32,
        )

        integral, prev_error = self._unpack_pid_state(pid_state, self._outer_pid_dim)
        dt = jp.asarray(max(dt, 1e-6), dtype=jp.float32)
        derivative = (vel_err_body - prev_error) / dt
        prev_integral = integral
        candidate_integral = jp.clip(
            integral + (vel_err_body * dt),
            -self.outer_i_limit,
            self.outer_i_limit,
        )

        roll_unsat = -(
            (k_xy * vel_err_body[1])
            + (ki_xy * candidate_integral[1])
            + (kd_xy * derivative[1])
        )
        roll_des = jp.clip(
            roll_unsat,
            -self.max_tilt,
            self.max_tilt,
        )
        pitch_unsat = (
            (k_xy * vel_err_body[0])
            + (ki_xy * candidate_integral[0])
            + (kd_xy * derivative[0])
        )
        pitch_des = jp.clip(
            pitch_unsat,
            -self.max_tilt,
            self.max_tilt,
        )
        collective_unsat = (
            (k_z * vel_err_body[2])
            + (ki_z * candidate_integral[2])
            + (kd_z * derivative[2])
        )
        collective = jp.clip(
            collective_unsat,
            -self.collective_limit,
            self.collective_limit,
        )
        sat_eps = jp.asarray(1e-6, dtype=jp.float32)
        integral = candidate_integral
        integral = integral.at[0].set(
            jp.where(jp.abs(pitch_unsat - pitch_des) > sat_eps, prev_integral[0], candidate_integral[0])
        )
        integral = integral.at[1].set(
            jp.where(jp.abs(roll_unsat - roll_des) > sat_eps, prev_integral[1], candidate_integral[1])
        )
        integral = integral.at[2].set(
            jp.where(
                jp.abs(collective_unsat - collective) > sat_eps,
                prev_integral[2],
                candidate_integral[2],
            )
        )

        outer_cmd = jp.array([collective, roll_des, pitch_des, cmd[3]], dtype=jp.float32)
        return outer_cmd, self._pack_pid_state(integral, vel_err_body)

    def attitude_rate_controller(
        self,
        agent_orientation: jax.Array,
        agent_angvel: jax.Array,
        cmd: jax.Array,
        pid_state: jax.Array,
        dt: float,
        gain_arr: jax.Array, 
    ) -> tuple[jax.Array, jax.Array]:
        agent_angvel = jp.nan_to_num(agent_angvel, nan=0.0, posinf=0.0, neginf=0.0)
        cmd = jp.nan_to_num(cmd, nan=0.0, posinf=0.0, neginf=0.0)
        roll, pitch, _ = self._quat_to_roll_pitch_yaw(agent_orientation)
        k_xy = gain_arr[0]
        ki_xy = gain_arr[1] 
        kd_xy = gain_arr[2]
        k_z = gain_arr[3]
        ki_z = gain_arr[4]
        kd_z = gain_arr[5]
        k_yaw = gain_arr[6]
        ki_yaw = gain_arr[7]
        kd_yaw = gain_arr[8]
        kp_att = gain_arr[9]
        kd_att = gain_arr[10]
        ki_att = gain_arr[11]
        err = jp.array(
            [
                cmd[1] - roll,
                cmd[2] - pitch,
                cmd[3] - agent_angvel[2],
            ],
            dtype=jp.float32,
        )
        integral, prev_error = self._unpack_pid_state(pid_state, self._inner_pid_dim)
        dt = jp.asarray(max(dt, 1e-6), dtype=jp.float32)
        derivative = (err - prev_error) / dt
        prev_integral = integral
        candidate_integral = jp.clip(
            integral + (err * dt),
            -self.inner_i_limit,
            self.inner_i_limit,
        )

        u_collective = jp.clip(cmd[0], -self.collective_limit, self.collective_limit)
        u_roll_unsat = (
            (kp_att * err[0])
            + (ki_att * candidate_integral[0])
            - (kd_att * agent_angvel[0])
        )
        u_roll = jp.clip(
            u_roll_unsat,
            -self.attitude_limit,
            self.attitude_limit,
        )
        u_pitch_unsat = (
            (kp_att * err[1])
            + (ki_att * candidate_integral[1])
            - (kd_att * agent_angvel[1])
        )
        u_pitch = jp.clip(
            u_pitch_unsat,
            -self.attitude_limit,
            self.attitude_limit,
        )
        u_yaw_unsat = (
            (k_yaw * err[2])
            + (ki_yaw * candidate_integral[2])
            + (kd_yaw * derivative[2])
        )
        u_yaw = jp.clip(
            u_yaw_unsat,
            -self.yaw_limit,
            self.yaw_limit,
        )
        sat_eps = jp.asarray(1e-6, dtype=jp.float32)
        integral = candidate_integral
        integral = integral.at[0].set(
            jp.where(jp.abs(u_roll_unsat - u_roll) > sat_eps, prev_integral[0], candidate_integral[0])
        )
        integral = integral.at[1].set(
            jp.where(jp.abs(u_pitch_unsat - u_pitch) > sat_eps, prev_integral[1], candidate_integral[1])
        )
        integral = integral.at[2].set(
            jp.where(jp.abs(u_yaw_unsat - u_yaw) > sat_eps, prev_integral[2], candidate_integral[2])
        )

        tilt_comp = jp.clip(jp.cos(roll) * jp.cos(pitch), 0.5, 1.0)
        hover_ctrl = self.hover_ctrl / tilt_comp
        thrust = jp.array(
            [
                hover_ctrl[0] + u_collective - u_roll + u_pitch - u_yaw,
                hover_ctrl[1] + u_collective + u_roll + u_pitch + u_yaw,
                hover_ctrl[2] + u_collective + u_roll - u_pitch - u_yaw,
                hover_ctrl[3] + u_collective - u_roll - u_pitch + u_yaw,
            ],
            dtype=jp.float32,
        )
        thrust = jp.nan_to_num(
            thrust,
            nan=self.hover_thrust,
            posinf=jp.max(self.motor_high),
            neginf=jp.min(self.motor_low),
        )
        thrust = jp.clip(thrust, self.motor_low, self.motor_high)
        return thrust, self._pack_pid_state(integral, err)

    def _physics_step(self, data: mjx.Data, motor_cmd: jax.Array) -> mjx.Data:
        ctrl = jp.asarray(motor_cmd, dtype=data.ctrl.dtype)
        return mjx_env.step(self.mjx_model, data, ctrl, 1)

    def _extract_body_state(
        self,
        data: mjx.Data,
    ) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:
        agent_location = jp.nan_to_num(
            data.subtree_com[self.body_id],
            nan=0.0,
            posinf=self.obs_xy_lim,
            neginf=-self.obs_xy_lim,
        ).astype(jp.float32)
        agent_vel = jp.nan_to_num(
            data.cvel[self.body_id, 3:6],
            nan=0.0,
            posinf=self.obs_vel_lim,
            neginf=-self.obs_vel_lim,
        ).astype(jp.float32)
        sensor_data = jp.nan_to_num(data.sensordata, nan=0.0, posinf=0.0, neginf=0.0).astype(
            jp.float32
        )
        agent_angvel = sensor_data[self._gyro_sensor_slice]
        agent_yawrate = jp.clip(agent_angvel[2:3], self._obs_yaw_low, self._obs_yaw_high)
        agent_orientation = sensor_data[self._quat_sensor_slice]
        agent_orientation = agent_orientation / jp.maximum(jp.linalg.norm(agent_orientation), 1e-6)
        lidar = jp.concatenate(
            [sensor_data[sensor_slice] for sensor_slice in self._lidar_sensor_slices],
            axis=0,
        )
        lidar = jp.where(lidar < 0.0, self.lidar_max_dist, lidar)
        lidar = jp.clip(lidar, 0.0, self.lidar_max_dist).astype(jp.float32)
        return agent_location, agent_vel, agent_angvel, agent_orientation, agent_yawrate, lidar

    def _run_cascaded_controller(
        self,
        data: mjx.Data,
        info: dict[str, jax.Array],
        policy_action: jax.Array,
        gain_arr: jax.Array,
    ) -> tuple[mjx.Data, dict[str, jax.Array]]:
        carry = (
            data,
            info["agent_location"],
            info["agent_vel"],
            info["agent_angvel"],
            info["agent_orientation"],
            info["agent_yawrate"],
            info["held_action"],
            info["outer_cmd"],
            info["outer_pid_state"],
            info["inner_pid_state"],
            info["sim_step"],
            info["last_motor_cmd"],
        )

        def _controller_step(carry, _):
            (
                step_data,
                agent_location,
                agent_vel,
                agent_angvel,
                agent_orientation,
                agent_yawrate,
                held_action,
                outer_cmd,
                outer_pid_state,
                inner_pid_state,
                sim_step,
                last_motor_cmd,
            ) = carry

            update_policy = (sim_step % self.policy_decim) == 0
            held_action = jp.where(update_policy, policy_action, held_action)
            hover_blend = self._goal_hover_blend(agent_location, agent_vel, info["target"])
            landing_cmd = self._goal_landing_command(agent_location, info["target"])
            hover_speed = _safe_l2_norm(agent_vel)
            hover_thrust_blend = hover_blend * jp.clip(
                1.0 - (hover_speed / jp.asarray(max(3.0 * self.hover_speed_epsilon, 0.4), dtype=jp.float32)),
                0.0,
                1.0,
            )
            outer_pid_state = outer_pid_state * (1.0 - hover_blend)
            inner_pid_state = inner_pid_state * (1.0 - hover_blend)
            outer_cmd = outer_cmd * (1.0 - hover_blend)
            vel_cmd = self.action_to_command(held_action)
            vel_cmd = ((1.0 - hover_blend) * vel_cmd) + (hover_blend * landing_cmd)

            candidate_outer_cmd, candidate_outer_pid_state = self.velocity_controller(
                agent_vel=agent_vel,
                agent_orientation=agent_orientation,
                cmd=vel_cmd,
                pid_state=outer_pid_state,
                dt=self.outer_dt,
                gain_arr = gain_arr
            )
            update_outer = (sim_step % self.outer_decim) == 0
            outer_cmd = jp.where(update_outer, candidate_outer_cmd, outer_cmd)
            outer_pid_state = jp.where(update_outer, candidate_outer_pid_state, outer_pid_state)

            motor_cmd, inner_pid_state = self.attitude_rate_controller(
                agent_orientation=agent_orientation,
                agent_angvel=agent_angvel,
                cmd=outer_cmd,
                pid_state=inner_pid_state,
                dt=self.inner_dt,
                gain_arr = gain_arr
            )
            motor_cmd = ((1.0 - hover_thrust_blend) * motor_cmd) + (hover_thrust_blend * self.hover_ctrl)
            motor_cmd = jp.clip(motor_cmd, self.motor_low, self.motor_high)
            step_data = self._physics_step(step_data, motor_cmd)
            (
                agent_location,
                agent_vel,
                agent_angvel,
                agent_orientation,
                agent_yawrate,
                _,
            ) = self._extract_body_state(step_data)

            return (
                step_data,
                agent_location,
                agent_vel,
                agent_angvel,
                agent_orientation,
                agent_yawrate,
                held_action,
                outer_cmd,
                outer_pid_state,
                inner_pid_state,
                sim_step + jp.array(1, dtype=jp.int32),
                motor_cmd,
            ), None

        final_carry, _ = jax.lax.scan(_controller_step, carry, xs=None, length=self.n_substeps)
        (
            data,
            _,
            _,
            _,
            _,
            _,
            held_action,
            outer_cmd,
            outer_pid_state,
            inner_pid_state,
            sim_step,
            last_motor_cmd,
        ) = final_carry
        controller_state = {
            "held_action": held_action,
            "outer_cmd": outer_cmd,
            "outer_pid_state": outer_pid_state,
            "inner_pid_state": inner_pid_state,
            "sim_step": sim_step,
            "last_motor_cmd": last_motor_cmd,
        }
        return data, controller_state

    def step(
        self,
        state: mjx_env.State,
        action: jax.Array,
        gain_arr: Optional[jax.Array] = None,
    ) -> mjx_env.State:
        raw_action = jp.asarray(action, dtype=jp.float32).reshape((4,))
        raw_action = jp.nan_to_num(raw_action, nan=0.0, posinf=1.0, neginf=-1.0)
        raw_action = jp.clip(raw_action, self._action_low, self._action_high)
        scaled_action = raw_action * self.action_scale
        gain_ar = self.gain_arr if gain_arr is None else jp.asarray(gain_arr, dtype=jp.float32)
        gain_ar = gain_ar.reshape((NUM_PID_GAINS,))
        data, controller_state = self._run_cascaded_controller(state.data, state.info, scaled_action, gain_ar)
        applied_action = controller_state["held_action"]
        (
            agent_location,
            agent_vel,
            agent_angvel,
            agent_orientation,
            agent_yawrate,
            lidar,
        ) = self._extract_body_state(data)
        obstacle_positions = self._extract_obstacle_positions(data)
        info = {
            **state.info,
            **controller_state,
            "agent_location": agent_location,
            "agent_vel": agent_vel,
            "agent_angvel": agent_angvel,
            "agent_orientation": agent_orientation,
            "agent_yawrate": agent_yawrate,
            "lidar": lidar,
            "obstacle_positions": obstacle_positions,
        }
        obs = self._get_obs(info)

        invalid_state = (
            (~jp.all(jp.isfinite(data.qpos)))
            | (~jp.all(jp.isfinite(data.qvel)))
            | (~jp.all(jp.isfinite(data.cvel)))
            | (~jp.all(jp.isfinite(data.ctrl)))
            | (~jp.all(jp.isfinite(data.sensordata)))
        )

        has_contact, has_obstacle_contact, has_environment_contact = self._detect_drone_contacts(
            data
        )
        reward, done, metrics, info = self._reward_and_done(
            info,
            raw_action,
            applied_action,
            has_contact,
            has_obstacle_contact,
            has_environment_contact,
            invalid_state,
        )
        self._last_info = info
        return mjx_env.State(
            data=data,
            obs=obs,
            reward=reward.astype(jp.float32),
            done=done,
            metrics=metrics,
            info=info,
        )

    def _reward_and_done(
        self,
        info: dict[str, jax.Array],
        raw_action: jax.Array,
        scaled_action: jax.Array,
        has_contact: jax.Array,
        has_obstacle_contact: jax.Array,
        has_environment_contact: jax.Array,
        invalid_state: jax.Array,
    ):
        goal_delta = info["target"] - info["agent_location"]
        dist = _safe_l2_norm(goal_delta)
        delta_action = scaled_action - info["prev_action"]

        collision_streak = jp.where(
            has_contact,
            info["collision_streak"] + 1,
            jp.array(0, dtype=jp.int32),
        )
        collision = collision_streak >= self.collision_terminate_steps

        r_prog = self.w_progress * (info["prev_distance"] - dist)
        lidar_clearance, true_obstacle_clearance, obstacle_risk = self._obstacle_reward_terms(
            info["agent_location"],
            info["lidar"],
            info["obstacle_positions"],
            info["obstacle_mask"],
        )
        r_obs = self.w_obs * (info["prev_obstacle_risk"] - obstacle_risk)
        r_coll = jp.where(collision, -self.r_collision, 0.0)
        r_energy = -self.w_energy * jp.dot(scaled_action, scaled_action)
        r_smooth = -self.w_smooth * jp.dot(delta_action, delta_action)

        out_of_bounds = (
            (jp.abs(info["agent_location"][0]) > (self.safety_xy_scale * self.xylim))
            | (jp.abs(info["agent_location"][1]) > (self.safety_xy_scale * self.xylim))
            | (info["agent_location"][2] < self.safety_z_low)
            | (info["agent_location"][2] > (self.safety_z_high_scale * self.zlim))
        )
        speed_sq = jp.dot(info["agent_vel"], info["agent_vel"])
        speed = jp.sqrt(speed_sq + 1e-8)
        excessive_speed = speed_sq > jp.square(self.safety_speed_scale * self.vellim)
        safety_terminated = out_of_bounds | excessive_speed
        r_safety = jp.where(safety_terminated, -self.r_collision, 0.0)
        r_speed = -self.w_speed * speed_sq

        reward = r_prog + r_obs + r_coll + r_energy + r_smooth + r_safety + r_speed
        hovering_at_goal = (dist <= self.eps_goal) & (
            speed_sq <= jp.square(self.hover_speed_epsilon)
        )
        goal_hold_streak = jp.where(
            hovering_at_goal,
            info["goal_hold_streak"] + 1,
            jp.array(0, dtype=jp.int32),
        )
        success = goal_hold_streak >= self.hover_success_steps
        reward = reward + jp.where(success, self.r_goal, 0.0)

        step_count = info["step"] + 1
        terminated = success | (self.terminate_on_collision & collision) | safety_terminated
        truncated = step_count >= self.max_steps
        failure_episode_end = (terminated | truncated) & (~success)
        r_terminal = jp.where(
            failure_episode_end,
            -(self.termination_penalty + self.terminal_distance_penalty * dist),
            0.0,
        )
        reward = reward + r_terminal
        episode_end = terminated | truncated
        min_distance_to_goal = jp.minimum(info["min_distance_to_goal"], dist)

        invalid_reward = (-2.0 * self.r_collision) - self.termination_penalty
        reward = jp.where(invalid_state, invalid_reward, reward)
        terminated = jp.where(invalid_state, True, terminated)
        truncated = jp.where(invalid_state, False, truncated)

        success_i = jp.where(invalid_state, False, success)
        hovering_at_goal_i = jp.where(invalid_state, False, hovering_at_goal)
        collision_i = jp.where(invalid_state, False, has_contact)
        obstacle_collision_i = jp.where(invalid_state, False, has_obstacle_contact)
        environment_collision_i = jp.where(invalid_state, False, has_environment_contact)
        collision_terminated_i = jp.where(
            invalid_state, False, self.terminate_on_collision & collision
        )
        out_of_bounds_i = jp.where(invalid_state, False, out_of_bounds)
        excessive_speed_i = jp.where(invalid_state, False, excessive_speed)
        numerical_issue_i = invalid_state
        terminated_i = terminated
        truncated_i = truncated

        to_f32 = lambda x: jp.asarray(x, dtype=jp.float32)
        metrics = {
            "distance": to_f32(dist),
            "distance_to_goal_per_step": to_f32(dist),
            "final_distance_to_goal": to_f32(
                jp.where(invalid_state | (~episode_end), 0.0, dist)
            ),
            "best_distance_to_goal": to_f32(
                jp.where(invalid_state | (~episode_end), 0.0, min_distance_to_goal)
            ),
            "initial_distance": to_f32(info["initial_target_distance"]),
            "r_prog": to_f32(jp.where(invalid_state, 0.0, r_prog)),
            "r_obs": to_f32(jp.where(invalid_state, 0.0, r_obs)),
            "r_coll": to_f32(jp.where(invalid_state, -2.0 * self.r_collision, r_coll)),
            "r_energy": to_f32(jp.where(invalid_state, 0.0, r_energy)),
            "r_smooth": to_f32(jp.where(invalid_state, 0.0, r_smooth)),
            "r_safety": to_f32(jp.where(invalid_state, 0.0, r_safety)),
            "r_speed": to_f32(jp.where(invalid_state, 0.0, r_speed)),
            "r_terminal": to_f32(jp.where(invalid_state, -self.termination_penalty, r_terminal)),
            "lidar_clearance": to_f32(
                jp.where(invalid_state, info["lidar_clearance"], lidar_clearance)
            ),
            "true_obstacle_clearance": to_f32(
                jp.where(invalid_state, info["true_obstacle_clearance"], true_obstacle_clearance)
            ),
            "raw_action_l2": to_f32(_safe_l2_norm(raw_action)),
            "scaled_action_l2": to_f32(_safe_l2_norm(scaled_action)),
            "success": to_f32(success_i),
            "hovering_at_goal": to_f32(hovering_at_goal_i),
            "collision": to_f32(collision_i),
            "obstacle_collision": to_f32(obstacle_collision_i),
            "environment_collision": to_f32(environment_collision_i),
            "collision_streak": to_f32(
                jp.where(invalid_state, info["collision_streak"], collision_streak)
            ),
            "goal_hold_streak": to_f32(
                jp.where(invalid_state, info["goal_hold_streak"], goal_hold_streak)
            ),
            "collision_terminated": to_f32(collision_terminated_i),
            "out_of_bounds": to_f32(out_of_bounds_i),
            "excessive_speed": to_f32(excessive_speed_i),
            "numerical_issue": to_f32(numerical_issue_i),
            "terminated": to_f32(terminated_i),
            "truncated": to_f32(truncated_i),
            "reward": to_f32(reward),
        }

        info = {
            **info,
            "prev_action": jp.where(
                invalid_state,
                jp.zeros((4,), dtype=jp.float32),
                scaled_action,
            ),
            "prev_distance": jp.where(invalid_state, info["prev_distance"], dist),
            "min_distance_to_goal": jp.where(
                invalid_state, info["min_distance_to_goal"], min_distance_to_goal
            ),
            "step": step_count,
            "collision_streak": jp.where(invalid_state, info["collision_streak"], collision_streak),
            "goal_hold_streak": jp.where(
                invalid_state, info["goal_hold_streak"], goal_hold_streak
            ),
            "distance": dist,
            "initial_distance": info["initial_target_distance"],
            "prev_obstacle_risk": jp.where(
                invalid_state,
                info["prev_obstacle_risk"],
                obstacle_risk,
            ),
            "obstacle_risk": jp.where(invalid_state, info["obstacle_risk"], obstacle_risk),
            "lidar_clearance": metrics["lidar_clearance"],
            "true_obstacle_clearance": metrics["true_obstacle_clearance"],
            "r_prog": metrics["r_prog"],
            "r_obs": metrics["r_obs"],
            "r_coll": metrics["r_coll"],
            "r_energy": metrics["r_energy"],
            "r_smooth": metrics["r_smooth"],
            "r_safety": metrics["r_safety"],
            "r_speed": metrics["r_speed"],
            "r_terminal": metrics["r_terminal"],
            "raw_action_l2": metrics["raw_action_l2"],
            "scaled_action_l2": metrics["scaled_action_l2"],
            # Keep diagnostics as booleans in info for readability.
            "success": success_i,
            "hovering_at_goal": hovering_at_goal_i,
            "collision": collision_i,
            "obstacle_collision": obstacle_collision_i,
            "environment_collision": environment_collision_i,
            "collision_terminated": collision_terminated_i,
            "out_of_bounds": out_of_bounds_i,
            "excessive_speed": excessive_speed_i,
            "numerical_issue": numerical_issue_i,
            "terminated": terminated_i,
            "truncated": truncated_i,
            "reward": metrics["reward"],
        }
        done = to_f32(terminated | truncated)
        return metrics["reward"], done, metrics, info

    def _resolve_info(
        self,
        state: Optional[mjx_env.State] = None,
        info: Optional[dict[str, jax.Array]] = None,
    ) -> dict[str, jax.Array]:
        if info is not None:
            return info
        if state is not None:
            return state.info
        if self._last_info is not None:
            return self._last_info
        raise ValueError("Pass `state` or `info`, or call `reset`/`step` first.")

    def _drone_to_target(
        self,
        state: Optional[mjx_env.State] = None,
        info: Optional[dict[str, jax.Array]] = None,
    ) -> jax.Array:
        env_info = self._resolve_info(state=state, info=info)
        return jp.asarray(
            env_info["target"] - env_info["agent_location"],
            dtype=jp.float32,
        )

    def _dist_drone_to_target(self, dist: jax.Array) -> jax.Array:
        return jp.asarray(jp.linalg.norm(jp.atleast_1d(dist)), dtype=jp.float32)

    def _drone_vels_yawrate(
        self,
        state: Optional[mjx_env.State] = None,
        info: Optional[dict[str, jax.Array]] = None,
    ) -> jax.Array:
        env_info = self._resolve_info(state=state, info=info)
        agent_vel = jp.asarray(env_info["agent_vel"], dtype=jp.float32)
        agent_yawrate = jp.asarray(jp.atleast_1d(env_info["agent_yawrate"]), dtype=jp.float32)
        return jp.concatenate([agent_vel, agent_yawrate[:1]], axis=0)

   
    @property
    def xml_path(self):
        return self._xml_path

    @property
    def action_size(self):
        return 4

    @property
    def action_low(self):
        return self._action_low

    @property
    def action_high(self):
        return self._action_high

    @property
    def mj_model(self):
        return self._mj_model

    @property
    def mjx_model(self):
        return self._mjx_model





    def _apply_config_overrides(self, 
        cfg: config_dict.ConfigDict,
        overrides: Optional[dict[str, Any]],
    ) -> config_dict.ConfigDict:
        if not overrides:
            return cfg
        for key, value in overrides.items():
            cfg[key] = value
        return cfg


def _pid_demo_action(env: newDrone, state: mjx_env.State) -> jax.Array:
    goal_vec = jp.asarray(state.info["target"] - state.info["agent_location"], dtype=jp.float32)
    vel_cmd = jp.clip(
        jp.array(
            [
                0.1 * goal_vec[0],
                0.1 * goal_vec[1],
                0.1 * goal_vec[2],
            ],
            dtype=jp.float32,
        ),
        -env.vellim,
        env.vellim,
    )
    yawrate_cmd = jp.array([0.5], dtype=jp.float32)
    action = jp.concatenate([vel_cmd / env.vellim, yawrate_cmd], axis=0)
    return jp.clip(action, env.action_low, env.action_high).astype(jp.float32)


def _sync_viewer_data(env: newDrone, viewer_data: mujoco.MjData, state: mjx_env.State) -> None:
    viewer_data.qpos[:] = np.asarray(state.data.qpos)
    viewer_data.qvel[:] = np.asarray(state.data.qvel)
    viewer_data.ctrl[:] = np.asarray(state.data.ctrl)
    if env.mj_model.nmocap > 0:
        viewer_data.mocap_pos[:] = np.asarray(state.data.mocap_pos)
        viewer_data.mocap_quat[:] = np.asarray(state.data.mocap_quat)
    mujoco.mj_forward(env.mj_model, viewer_data)


def run_pid_demo(
    num_steps: int = 2000,
    seed: int = 0,
    render: bool = True,
    jit_step: bool = True,
    real_time: bool = True,
    env_overrides: Optional[dict[str, Any]] = None,
) -> None:
    cfg = default_config()
    if env_overrides:
        for key, value in env_overrides.items():
            cfg[key] = value

    env = newDrone(config=cfg)
    rng = jax.random.PRNGKey(seed)
    rng, reset_rng = jax.random.split(rng)
    state = env.reset(reset_rng)

    step_fn = jax.jit(env.step) if jit_step else env.step
    if jit_step:
        print("Compiling first step...")
        compile_start = time.perf_counter()
        warm_action = _pid_demo_action(env, state)
        warm_state = step_fn(state, warm_action)
        jax.block_until_ready(warm_state.reward)
        print(f"Compile done in {time.perf_counter() - compile_start:.2f}s")

    viewer = None
    viewer_data = None
    if render:
        viewer_data = mujoco.MjData(env.mj_model)
        _sync_viewer_data(env, viewer_data, state)
        viewer = mujoco.viewer.launch_passive(env.mj_model, viewer_data)
        if env._track_camera_id >= 0:
            viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FIXED
            viewer.cam.fixedcamid = env._track_camera_id
            viewer.sync()

    print(
        "Running PID demo with render. "

    )

    try:
        for step in range(num_steps):
            action = _pid_demo_action(env, state)
            step_start = time.perf_counter()
            state = step_fn(state, action)

            if render and viewer is not None:
                _sync_viewer_data(env, viewer_data, state)
                viewer.sync()
                if hasattr(viewer, "is_running") and not viewer.is_running():
                    print("Viewer closed, stopping demo.")
                    break

            if step % 50 == 0:
                dist = float(state.info["distance"])
                pos = np.asarray(state.info["agent_location"])
                vel = np.asarray(state.info["agent_vel"])
                hold_streak = int(state.info["goal_hold_streak"])
                print(
                    f"step={step:04d} reward={float(state.reward): .3f} "
                    f"dist={dist: .3f} hold={hold_streak:03d} "
                    f"pos={np.round(pos, 3)} vel={np.round(vel, 3)}"
                )

            if bool(state.done):
                print(
                    "episode ended",
                    f"step={step}",
                    f"success={bool(state.info['success'])}",
                    f"collision={bool(state.info['collision'])}",
                    f"oob={bool(state.info['out_of_bounds'])}",
                    f"numerical_issue={bool(state.info['numerical_issue'])}",
                )
                rng, reset_rng = jax.random.split(rng)
                state = env.reset(reset_rng)

            if real_time:
                elapsed = time.perf_counter() - step_start
                time.sleep(max(env._ctrl_dt - elapsed, 0.0))
    except KeyboardInterrupt:
        print("\nStopped by user.")
    finally:
        if viewer is not None:
            viewer.close()

def _rollout_loss_impl(
    gain_arr: jax.Array,
    env: newDrone,
    n_steps: int = 500,
    seed: int = 0,
    n_episodes: int = 3,
) -> float:
    """Run headless rollouts and return mean episode reward (higher = better).

    gain_vec layout (15 floats):
        [0]  kp_pos_xy   [1]  kp_pos_z    [2]  kp_pos_yaw
        [3]  k_xy        [4]  ki_xy       [5]  kd_xy
        [6]  k_z         [7]  ki_z        [8]  kd_z
        [9]  k_yaw       [10] ki_yaw      [11] kd_yaw
        [12] kp_att      [13] kd_att      [14] ki_att
    """
    if int(env.mj_model.opt.iterations) != 1:
        raise ValueError(
            "jax.grad through MJX requires solver_iterations=1. "
            "MJX uses a lax.while_loop solver for larger iteration counts, "
            "and JAX reverse-mode autodiff does not support that path."
        )
    gain_arr = jp.asarray(gain_arr, dtype=jp.float32).reshape((NUM_PID_GAINS,))
    episode_rngs = jax.random.split(jax.random.PRNGKey(seed), n_episodes)

    def _run_episode(reset_rng: jax.Array) -> jax.Array:
        init_state = env.reset(reset_rng)
        init_carry = (
            init_state,
            jp.array(1.0, dtype=jp.float32),
            jp.array(0.0, dtype=jp.float32),
        )

        def _episode_step(carry, _):
            state, active, total_reward = carry
            action = _pid_demo_action(env, state)
            next_state = jax.lax.cond(
                active > 0.0,
                lambda s: env.step(s, action, gain_arr),
                lambda s: jax.tree_util.tree_map(jax.lax.stop_gradient, s),
                state,
            )
            total_reward = jax.lax.cond(
                active > 0.0,
                lambda tr: tr + next_state.reward.astype(jp.float32),
                lambda tr: tr,
                total_reward,
            )
            next_active = active * (1.0 - next_state.done.astype(jp.float32))
            # Keep `jax.grad` on the rollout objective, but stop gradients through
            # the recurrent state carry. Full reverse-mode through long MJX
            # rollouts is numerically unstable here and produces NaN cotangents.
            next_state = jax.tree_util.tree_map(jax.lax.stop_gradient, next_state)
            return (next_state, next_active, total_reward), None

        (_, _, total_reward), _ = jax.lax.scan(_episode_step, init_carry, xs=None, length=n_steps)
        return total_reward

    def _episode_scan(total_reward, reset_rng):
        episode_reward = _run_episode(reset_rng)
        return total_reward + episode_reward, episode_reward

    total_reward, _ = jax.lax.scan(
        _episode_scan,
        jp.array(0.0, dtype=jp.float32),
        episode_rngs,
    )
    return total_reward / jp.asarray(n_episodes, dtype=jp.float32)


def _make_rollout_objective(
    env: newDrone,
    n_steps: int = 500,
    seed: int = 0,
    n_episodes: int = 3,
):
    def rollout_impl(gain_arr: jax.Array) -> jax.Array:
        return _rollout_loss_impl(
            gain_arr,
            env,
            n_steps=n_steps,
            seed=seed,
            n_episodes=n_episodes,
        )

    @jax.custom_vjp
    def rollout_objective(gain_arr: jax.Array) -> jax.Array:
        return rollout_impl(gain_arr)

    def rollout_fwd(gain_arr: jax.Array):
        value = rollout_impl(gain_arr)
        return value, gain_arr

    def rollout_bwd(gain_arr: jax.Array, cotangent: jax.Array):
        jac = jax.jacfwd(rollout_impl)(gain_arr)
        return (cotangent * jac,)

    rollout_objective.defvjp(rollout_fwd, rollout_bwd)
    return rollout_objective


def _clip_gradient_norm(
    grad: jax.Array,
    max_norm: float,
) -> tuple[jax.Array, jax.Array]:
    """Clip a gradient vector by global norm."""
    grad = jp.nan_to_num(
        jp.asarray(grad, dtype=jp.float32),
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )
    grad_norm = jp.linalg.norm(grad)
    max_norm = jp.asarray(max_norm, dtype=jp.float32)
    scale = jp.minimum(1.0, max_norm / jp.maximum(grad_norm, 1e-8))
    return grad * scale, grad_norm





def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Quick PID render demo for the MJX drone env.")
    parser.add_argument("--steps", type=int, default=2000, help="Number of env steps to run.")
    parser.add_argument("--seed", type=int, default=0, help="PRNG seed.")
    parser.add_argument(
        "--no-render",
        action="store_true",
        help="Run the PID demo without opening the MuJoCo viewer.",
    )
    parser.add_argument(
        "--no-jit",
        action="store_true",
        help="Disable jitting the env step for easier debugging.",
    )
    parser.add_argument(
        "--no-real-time",
        action="store_true",
        help="Run as fast as possible instead of sleeping to roughly match ctrl_dt.",
    )
    parser.add_argument(
        "--env-overrides",
        type=str,
        default=None,
        help='JSON dict of config overrides, e.g. \'{"outer_decim": 4, "ki_z": 0.1}\'',
    )
    return parser.parse_args()


if __name__ == "__main__":
    # args = _parse_args()
    # overrides = json.loads(args.env_overrides) if args.env_overrides else None
    # run_pid_demo(
    #     num_steps=args.steps,
    #     seed=args.seed,
    #     render=not args.no_render,
    #     jit_step=not args.no_jit,
    #     real_time=not args.no_real_time,
    #     env_overrides=overrides,
    # )

    # n_iter = 100
    # cfg = default_config()
    # cfg.solver_iterations = 1

    # env = newDrone(config=cfg)
    # train_n_steps = 50
    # eval_n_steps = 500
    # train_n_episodes = 1
    # eval_n_episodes = 1
    # rollout_objective = _make_rollout_objective(
    #     env,
    #     n_steps=train_n_steps,
    #     n_episodes=train_n_episodes,
    # )
    # eval_rollout = jax.jit(
    #     lambda gain_arr: _rollout_loss_impl(
    #         gain_arr,
    #         env,
    #         n_steps=eval_n_steps,
    #         n_episodes=eval_n_episodes,
    #     )
    # )
    # value_and_grad = jax.jit(jax.value_and_grad(rollout_objective))
    # eta = 1
    # grad_clip_norm = 1.0
    # gain_arr = env.gain_arr
    # for n in range(n_iter):
    #     low_rews, grad = value_and_grad(gain_arr)
    #     grad_finite = bool(jp.all(jp.isfinite(grad)))
    #     clipped_grad, grad_norm = _clip_gradient_norm(grad, grad_clip_norm)
    #     clipped_grad_norm = jp.linalg.norm(clipped_grad)
    #     gain_arr = gain_arr + (eta * clipped_grad)
    #     env.gain_arr = gain_arr
    #     if n % 10 == 0 or n == n_iter - 1:
    #         full_rew = eval_rollout(gain_arr)
    #         print(
    #             f"iter={n:03d} train_reward={float(low_rews): .5f} "
    #             f"eval_reward={float(full_rew): .5f} "
    #         )
    #         print(
    #             "gain_arr=",
    #             np.array2string(np.asarray(gain_arr), precision=5, floatmode="fixed"),
    #         )
            
    # print("final gains, ", env.gain_arr)
        
        
    args = _parse_args()
    overrides = json.loads(args.env_overrides) if args.env_overrides else None
    run_pid_demo(
        num_steps=args.steps,
        seed=args.seed,
        render=not args.no_render,
        jit_step=not args.no_jit,
        real_time=not args.no_real_time,
        env_overrides=overrides,
    )
