from typing import Optional
import numpy as np
import gymnasium as gym
import mujoco as mj

class Drone_Env(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 100}

    def __init__(
        self,
        model_path,
        dt,
        max_steps,
        render_mode,
        xylim,
        zlim,
        vellim,
        yawrate_lim,
        action_scale=1.0,
        spawn_z_min=0.5,
        target_dist_min=0.0,
        target_dist_max=None,
        terminate_on_collision=True,
        collision_terminate_steps=1,
        w_progress=1.0,
        w_energy=0.01,
        w_smooth=0.05,
        w_speed=0.01,
        r_collision=5.0,
        r_goal=10.0,
        termination_penalty=0.0,
        eps_goal=0.15,
        k_xy=0.6,
        k_z=1.2,
        k_yaw=0.2,
        safety_xy_scale=1.5,
        safety_z_low=-0.1,
        safety_z_high_scale=1.5,
        safety_speed_scale=4.0,
    ):
        super().__init__()
        self.model_path = model_path
        self.dt = dt
        self.max_steps = max_steps
        self.render_mode = render_mode
        self.xylim = xylim
        self.zlim = zlim
        self.spawn_z_min = float(np.clip(spawn_z_min, 0.0, self.zlim))
        self.target_dist_min = float(max(0.0, target_dist_min))
        self.target_dist_max = (
            None if target_dist_max is None else float(max(self.target_dist_min, target_dist_max))
        )
        self.terminate_on_collision = bool(terminate_on_collision)
        self.collision_terminate_steps = max(1, int(collision_terminate_steps))
        self.w_progress = float(w_progress)
        self.w_energy = float(max(0.0, w_energy))
        self.w_smooth = float(max(0.0, w_smooth))
        self.w_speed = float(max(0.0, w_speed))
        self.r_collision = float(max(0.0, r_collision))
        self.r_goal = float(max(0.0, r_goal))
        self.termination_penalty = float(max(0.0, termination_penalty))
        self.eps_goal = float(max(1e-3, eps_goal))
        self.k_xy = float(k_xy)
        self.k_z = float(k_z)
        self.k_yaw = float(k_yaw)
        self.safety_xy_scale = float(max(1.0, safety_xy_scale))
        self.safety_z_low = float(safety_z_low)
        self.safety_z_high_scale = float(max(1.0, safety_z_high_scale))
        self.safety_speed_scale = float(max(1.0, safety_speed_scale))
        self.vellim = vellim
        self.yawrate_lim = yawrate_lim
        self.obs_xy_lim = self.safety_xy_scale * float(self.xylim)
        self.obs_z_low = float(self.safety_z_low)
        self.obs_z_high = self.safety_z_high_scale * float(self.zlim)
        self.obs_vel_lim = self.safety_speed_scale * float(self.vellim)
        self.obs_yawrate_lim = self.safety_speed_scale * float(self.yawrate_lim)
        self.goal_vec_xy_lim = float(self.xylim) + self.obs_xy_lim
        self.goal_vec_z_low = -self.obs_z_high
        self.goal_vec_z_high = float(self.zlim) - self.obs_z_low
        self._agent_location = np.array([-1.0, -1.0, -1.0], dtype = np.float32)
        self._agent_vel = np.array([0, 0, 0], dtype = np.float32)
        self._agent_orientation = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
        self._agent_yawrate = np.array([0], dtype = np.float32)
        self._target_location = np.array([-1.0, -1.0, -1.0], dtype=np.float32)
        self._prev_action = np.zeros(4, dtype=np.float32)
        self._prev_distance = 0.0
        self._initial_target_distance = 0.0
        self.action_scale = float(np.clip(action_scale, 0.05, 1.0))
        self.observation_space = gym.spaces.Dict({
                "agent_pos_xy": gym.spaces.Box(
                    low=np.array([-self.obs_xy_lim, -self.obs_xy_lim], dtype=np.float32),
                    high=np.array([self.obs_xy_lim, self.obs_xy_lim], dtype=np.float32),
                    dtype=np.float32,
                ),
                "agent_pos_z": gym.spaces.Box(
                    low=np.array([self.obs_z_low], dtype=np.float32),
                    high=np.array([self.obs_z_high], dtype=np.float32),
                    dtype=np.float32,
                ),
                "agent_orientation": gym.spaces.Box(
                    low=np.array([-1.0, -1.0, -1.0, -1.0], dtype=np.float32),
                    high=np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float32),
                    dtype=np.float32,
                ),
                "agent_vel": gym.spaces.Box(
                    low=np.array([-self.obs_vel_lim, -self.obs_vel_lim, -self.obs_vel_lim], dtype=np.float32),
                    high=np.array([self.obs_vel_lim, self.obs_vel_lim, self.obs_vel_lim], dtype=np.float32),
                    dtype=np.float32,
                ),
                "target": gym.spaces.Box(
                    low=np.array([-self.xylim, -self.xylim, 0.0], dtype=np.float32),
                    high=np.array([self.xylim, self.xylim, self.zlim], dtype=np.float32),
                    dtype=np.float32,
                ),
                "agent_yawrate": gym.spaces.Box(
                    low=np.array([-self.obs_yawrate_lim], dtype=np.float32),
                    high=np.array([self.obs_yawrate_lim], dtype=np.float32),
                    dtype=np.float32,
                ),
                "goal_vec": gym.spaces.Box(
                    low=np.array(
                        [-self.goal_vec_xy_lim, -self.goal_vec_xy_lim, self.goal_vec_z_low],
                        dtype=np.float32,
                    ),
                    high=np.array(
                        [self.goal_vec_xy_lim, self.goal_vec_xy_lim, self.goal_vec_z_high],
                        dtype=np.float32,
                    ),
                    dtype=np.float32,
                ),
            })
        # Normalized command action: [vx_cmd, vy_cmd, vz_cmd, yawrate_cmd] in [-1, 1].
        self.action_space = gym.spaces.Box(-1.0, 1.0, (4, ), dtype=np.float32)
        
        self.model = mj.MjModel.from_xml_path(model_path)
        self.data = mj.MjData(self.model)
        mj.mj_forward(self.model, self.data)   # or after mj_step
        self.body_id = mj.mj_name2id(self.model, mj.mjtObj.mjOBJ_BODY, "x2")
        self.motor_low = self.model.actuator_ctrlrange[:, 0].astype(np.float32)
        self.motor_high = self.model.actuator_ctrlrange[:, 1].astype(np.float32)
        total_mass = float(self.model.body_mass.sum())
        gravity = float(-self.model.opt.gravity[2])
        self.hover_thrust = total_mass * gravity / self.model.nu
        self.hover_ctrl = np.full(self.model.nu, self.hover_thrust, dtype=np.float32)
        if self.model.nkey > 0:
            hover_key_id = mj.mj_name2id(self.model, mj.mjtObj.mjOBJ_KEY, "hover")
            if hover_key_id != -1:
                self.hover_ctrl = self.model.key_ctrl[hover_key_id].astype(np.float32).copy()
        self._step= 0
        self._collision_streak = 0
        self.render_mode = render_mode
        self._renderer = None
        self._viewer = None
    def _get_obs(self):
        agent_location = np.nan_to_num(
            self._agent_location, nan=0.0, posinf=self.obs_xy_lim, neginf=-self.obs_xy_lim
        ).astype(np.float32)
        agent_vel = np.nan_to_num(
            self._agent_vel, nan=0.0, posinf=self.obs_vel_lim, neginf=-self.obs_vel_lim
        ).astype(np.float32)
        agent_orientation = np.nan_to_num(
            self._agent_orientation, nan=0.0, posinf=1.0, neginf=-1.0
        ).astype(np.float32)
        agent_orientation = np.clip(
            agent_orientation,
            self.observation_space["agent_orientation"].low,
            self.observation_space["agent_orientation"].high,
        )
        agent_yawrate = np.nan_to_num(
            self._agent_yawrate, nan=0.0, posinf=self.obs_yawrate_lim, neginf=-self.obs_yawrate_lim
        ).astype(np.float32)
        target = np.nan_to_num(
            self._target_location, nan=0.0, posinf=self.xylim, neginf=-self.xylim
        ).astype(np.float32)
        target = np.clip(
            target,
            self.observation_space["target"].low,
            self.observation_space["target"].high,
        )
        goal_vec = (target - agent_location).astype(np.float32)
        return {
            "agent_pos_xy": np.clip(
                agent_location[0:2],
                self.observation_space["agent_pos_xy"].low,
                self.observation_space["agent_pos_xy"].high,
            ),
            "agent_pos_z": np.clip(
                agent_location[2:3],
                self.observation_space["agent_pos_z"].low,
                self.observation_space["agent_pos_z"].high,
            ),
            "agent_orientation": agent_orientation,
            "agent_vel": np.clip(
                agent_vel,
                self.observation_space["agent_vel"].low,
                self.observation_space["agent_vel"].high,
            ),
            "target": target,
            "agent_yawrate": np.clip(
                agent_yawrate,
                self.observation_space["agent_yawrate"].low,
                self.observation_space["agent_yawrate"].high,
            ),
            "goal_vec": np.clip(
                goal_vec,
                self.observation_space["goal_vec"].low,
                self.observation_space["goal_vec"].high,
            ),
        }
        
    def _get_info(self):
        distance = np.linalg.norm(self._agent_location - self._target_location)
        return {
            "distance": distance,
            "initial_distance": self._initial_target_distance,
        }
        
    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        # IMPORTANT: Must call this first to seed the random number generator
        super().reset(seed=seed)

        # Randomly place the agent anywhere on the grid
        # Random spawn in a box: x,y in [-xylim, xylim], z in [spawn_z_min, zlim]
        self._agent_location = np.array(
            [
                self.np_random.uniform(-self.xylim, self.xylim),
                self.np_random.uniform(-self.xylim, self.xylim),
                self.np_random.uniform(self.spawn_z_min, self.zlim),
            ],
            dtype=np.float32,
        )

        # Keep simulator state consistent with observation
        self.data.qpos[:3] = self._agent_location
        self.data.qpos[3:7] = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
        self.data.qvel[:] = 0.0
        self.data.ctrl[:] = self.hover_ctrl
        mj.mj_forward(self.model, self.data)


        # Randomly place target with optional distance constraints from spawn.
        candidate = self._agent_location.copy()
        for _ in range(200):
            candidate = np.array(
                [
                    self.np_random.uniform(-self.xylim, self.xylim),
                    self.np_random.uniform(-self.xylim, self.xylim),
                    self.np_random.uniform(self.spawn_z_min, self.zlim),
                ],
                dtype=np.float32,
            )
            dist = float(np.linalg.norm(candidate - self._agent_location))
            max_ok = self.target_dist_max is None or dist <= self.target_dist_max
            if dist >= self.target_dist_min and max_ok:
                break
        self._target_location = candidate
        self._initial_target_distance = float(
            np.linalg.norm(self._target_location - self._agent_location)
        )
        self._agent_yawrate = np.zeros_like(self._agent_yawrate)
        self._agent_vel = np.zeros_like(self._agent_vel)
        self._agent_orientation = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
        self._prev_action = np.zeros(4, dtype=np.float32)
        self._prev_distance = self._initial_target_distance
        observation = self._get_obs()
        info = self._get_info()
        self._step = 0
        self._collision_streak = 0
        return observation, info
    
    
    def step(self, action):
        """Execute one timestep within the environment.

        Args:
            action: The action to take (0-3 for directions)

        Returns:
            tuple: (observation, reward, terminated, truncated, info)
        """
        raw_action = np.asarray(action, dtype=np.float32).reshape(4,)
        raw_action = np.nan_to_num(raw_action, nan=0.0, posinf=1.0, neginf=-1.0)
        raw_action = np.clip(raw_action, self.action_space.low, self.action_space.high)
        action = raw_action * self.action_scale

        thrust = self.vel_to_thrust(action)
        self.data.ctrl[:] = thrust
        mj.mj_step(self.model, self.data)

        invalid_state = (
            not np.isfinite(self.data.qpos).all()
            or not np.isfinite(self.data.qvel).all()
            or not np.isfinite(self.data.cvel).all()
            or not np.isfinite(self.data.ctrl).all()
        )
        if invalid_state:
            self._step += 1
            info = self._get_info()
            info.update(
                {
                    "r_prog": 0.0,
                    "r_coll": -2.0 * self.r_collision,
                    "r_energy": 0.0,
                    "r_smooth": 0.0,
                    "r_speed": 0.0,
                    "r_terminal": -self.termination_penalty,
                    "raw_action_l2": float(np.linalg.norm(raw_action)),
                    "scaled_action_l2": float(np.linalg.norm(action)),
                    "success": False,
                    "collision": False,
                    "collision_streak": self._collision_streak,
                    "collision_terminated": False,
                    "numerical_issue": True,
                }
            )
            self._prev_action = np.zeros(4, dtype=np.float32)
            return self._get_obs(), (-2.0 * self.r_collision) - self.termination_penalty, True, False, info
        
        self._agent_location = self.data.subtree_com[self.body_id].astype(np.float32).copy()
        self._agent_vel = self.data.cvel[self.body_id, 3:6].astype(np.float32).copy()
        self._agent_yawrate = self.data.cvel[self.body_id, 2:3].astype(np.float32).copy()
        self._agent_orientation = self.data.qpos[3:7].astype(np.float32).copy()

        dist = np.linalg.norm(self._target_location - self._agent_location)
        delta_action = action - self._prev_action
        has_contact = self.data.ncon > 0
        if has_contact:
            self._collision_streak += 1
        else:
            self._collision_streak = 0
        collision = self._collision_streak >= self.collision_terminate_steps

        r_prog = self.w_progress * (self._prev_distance - dist)
        # Only penalise when the collision streak reaches the termination threshold,
        # not on every fleeting contact — avoids the gradient collapsing to "don't move".
        r_coll = -self.r_collision if collision else 0.0
        r_energy = -self.w_energy * float(np.dot(action, action))
        r_smooth = -self.w_smooth * float(np.dot(delta_action, delta_action))
        out_of_bounds = (
            abs(float(self._agent_location[0])) > (self.safety_xy_scale * self.xylim)
            or abs(float(self._agent_location[1])) > (self.safety_xy_scale * self.xylim)
            or float(self._agent_location[2]) < self.safety_z_low
            or float(self._agent_location[2]) > (self.safety_z_high_scale * self.zlim)
        )
        speed = float(np.linalg.norm(self._agent_vel))
        excessive_speed = speed > (self.safety_speed_scale * self.vellim)
        safety_terminated = out_of_bounds or excessive_speed
        r_safety = -self.r_collision if safety_terminated else 0.0
        r_speed = -self.w_speed * (speed * speed)
        reward = r_prog + r_coll + r_energy + r_smooth + r_safety + r_speed

        success = dist <= self.eps_goal
        if success:
            reward += self.r_goal

        obs = self._get_obs()
        self._step += 1
        terminated = success or (self.terminate_on_collision and collision) or safety_terminated
        truncated = self._step >= self.max_steps
        r_terminal = 0.0
        if terminated and not success:
            r_terminal = -self.termination_penalty
            reward += r_terminal
        info = self._get_info()
        info.update(
            {
                "r_prog": r_prog,
                "r_coll": r_coll,
                "r_energy": r_energy,
                "r_smooth": r_smooth,
                "r_safety": r_safety,
                "r_speed": r_speed,
                "r_terminal": r_terminal,
                "raw_action_l2": float(np.linalg.norm(raw_action)),
                "scaled_action_l2": float(np.linalg.norm(action)),
                "success": success,
                "collision": has_contact,
                "collision_streak": self._collision_streak,
                "collision_terminated": bool(self.terminate_on_collision and collision),
                "out_of_bounds": out_of_bounds,
                "excessive_speed": excessive_speed,
                "numerical_issue": False,
            }
        )

        self._prev_action = action.copy()
        self._prev_distance = dist
        return obs, reward, terminated, truncated, info

    def vel_to_thrust(self, action):
        # Convert normalized commands to physical setpoints.
        cmd = np.clip(action, self.action_space.low, self.action_space.high)
        cmd = np.nan_to_num(cmd, nan=0.0, posinf=1.0, neginf=-1.0)
        agent_vel = np.nan_to_num(self._agent_vel, nan=0.0, posinf=0.0, neginf=0.0)
        agent_yawrate = np.nan_to_num(self._agent_yawrate, nan=0.0, posinf=0.0, neginf=0.0)
        v_des = np.array(
            [
                cmd[0] * self.vellim,
                cmd[1] * self.vellim,
                cmd[2] * self.vellim,
            ],
            dtype=np.float32,
        )
        yawrate_des = float(cmd[3] * self.yawrate_lim)

        v_err = np.nan_to_num(v_des - agent_vel, nan=0.0, posinf=0.0, neginf=0.0)
        yaw_err = float(np.nan_to_num(yawrate_des - float(agent_yawrate[0]), nan=0.0, posinf=0.0, neginf=0.0))

        u_collective = self.k_z * float(v_err[2])
        u_roll = -self.k_xy * float(v_err[1])
        u_pitch = self.k_xy * float(v_err[0])
        u_yaw = self.k_yaw * yaw_err

        # Mixer in XML actuator order:
        # thrust1 (-x, -y), thrust2 (-x, +y), thrust3 (+x, +y), thrust4 (+x, -y)
        # roll contribution follows y-sign, pitch follows -x-sign, yaw follows rotor spin sign.
        f1 = self.hover_ctrl[0] + u_collective - u_roll + u_pitch - u_yaw
        f2 = self.hover_ctrl[1] + u_collective + u_roll + u_pitch + u_yaw
        f3 = self.hover_ctrl[2] + u_collective + u_roll - u_pitch - u_yaw
        f4 = self.hover_ctrl[3] + u_collective - u_roll - u_pitch + u_yaw
        thrust = np.array([f1, f2, f3, f4], dtype=np.float32)
        thrust = np.nan_to_num(
            thrust,
            nan=float(self.hover_thrust),
            posinf=float(self.motor_high.max()),
            neginf=float(self.motor_low.min()),
        )
        return np.clip(thrust, self.motor_low, self.motor_high)
    
    def render(self):
        if self.render_mode is None:
            return None

        if self.render_mode == "rgb_array":
            if self._renderer is None:
                self._renderer = mj.Renderer(self.model, height=480, width=640)
            self._renderer.update_scene(self.data)
            frame = self._renderer.render()  # np.ndarray HxWx3 uint8
            return frame

        if self.render_mode == "human":
            # simplest: no-op for now (or implement mj.viewer path)
            return None

        raise ValueError(f"Unsupported render_mode: {self.render_mode}")

    def close(self):
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None


if __name__ == "__main__":
    import drone_training


    env = gym.make(
    "DroneEnv-v0",
    model_path="Drone_MJCFs/skydio_x2/scene.xml",
    dt=0.01,
    max_steps=200,
    render_mode=None,
    xylim=2.0,
    zlim=2.0,
    vellim=1.0,
    yawrate_lim=1.0,
)

    obs, info = env.reset(seed=0)
    print("reset ok, keys:", obs.keys(), "distance:", info["distance"])

    for t in range(100):
        action = env.action_space.sample()  # in [-1, 1]
        
        obs, reward, terminated, truncated, info = env.step(action)

        assert env.observation_space.contains(obs), f"bad obs at t={t}"
        if t % 10 == 0:
            print(
                f"t={t:03d} reward={reward:.3f} d={info['distance']:.3f} "
                f"term={terminated} trunc={truncated}"
            )

        if terminated or truncated:
            print("episode ended at step", t)
            obs, info = env.reset()

    print("smoke test passed")
    model = mj.MjModel.from_xml_path("Drone_MJCFs/skydio_x2/scene.xml")
    data = mj.MjData(model)
    
    print("dt ", model.opt.timestep)
