from typing import Optional
import numpy as np
import gymnasium as gym
import mujoco as mj

class Drone_Env(gym.Env):
    
    def __init__(self, model_path, dt, max_steps, render_mode, xylim, zlim, vellim, yawrate_lim):
        super().__init__()
        self.model_path = model_path
        self.dt = dt
        self.max_steps = max_steps
        self.render_mode = render_mode
        self.xylim = xylim
        self.zlim = zlim
        self._agent_location = np.array([-1.0, -1.0, -1.0], dtype = np.float32)
        self._agent_vel = np.array([0, 0, 0], dtype = np.float32)
        self._agent_orientation = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
        self._agent_yawrate = np.array([0], dtype = np.float32)
        self._target_location = np.array([-1.0, -1.0, -1.0], dtype=np.float32)
        self._prev_action = np.zeros(4, dtype=np.float32)
        self._prev_distance = 0.0
        self.vellim = vellim
        self.yawrate_lim = yawrate_lim
        self.w_progress = 1.0
        self.w_energy = 0.01
        self.w_smooth = 0.05
        self.r_collision = 5.0
        self.r_goal = 10.0
        self.eps_goal = 0.15
        self.observation_space = gym.spaces.Dict({
                "agent_pos_xy": gym.spaces.Box(
                    low=np.array([-self.xylim, -self.xylim], dtype=np.float32),
                    high=np.array([self.xylim, self.xylim], dtype=np.float32),
                    dtype=np.float32,
                ),
                "agent_pos_z": gym.spaces.Box(
                    low=np.array([0.0], dtype=np.float32),
                    high=np.array([self.zlim], dtype=np.float32),
                    dtype=np.float32,
                ),
                "agent_orientation": gym.spaces.Box(
                    low=np.array([-1.0, -1.0, -1.0, -1.0], dtype=np.float32),
                    high=np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float32),
                    dtype=np.float32,
                ),
                "agent_vel": gym.spaces.Box(
                    low=np.array([-self.vellim, -self.vellim, -self.vellim], dtype=np.float32),
                    high=np.array([self.vellim, self.vellim, self.vellim], dtype=np.float32),
                    dtype=np.float32,
                ),
                "target": gym.spaces.Box(
                    low=np.array([-self.xylim, -self.xylim, 0.0], dtype=np.float32),
                    high=np.array([self.xylim, self.xylim, self.zlim], dtype=np.float32),
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
        self.k_xy = 0.6
        self.k_z = 1.2
        self.k_yaw = 0.2

        self._step= 0
    def _get_obs(self):
        return {
            "agent_pos_xy": self._agent_location[0:2].astype(np.float32),
            "agent_pos_z": self._agent_location[2:3].astype(np.float32),
            "agent_orientation": self._agent_orientation.astype(np.float32),
            "agent_vel": self._agent_vel.astype(np.float32),
            "target": self._target_location.astype(np.float32),
        }
        
    def _get_info(self):
        distance = np.linalg.norm(self._agent_location - self._target_location)
        return {
            "distance": distance
        }
        
    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        # IMPORTANT: Must call this first to seed the random number generator
        super().reset(seed=seed)

        # Randomly place the agent anywhere on the grid
        # Random spawn in a box: x,y in [-xylim, xylim], z in [0.2, zlim]
        self._agent_location = np.array(
            [
                self.np_random.uniform(-self.xylim, self.xylim),
                self.np_random.uniform(-self.xylim, self.xylim),
                self.np_random.uniform(0.2, self.zlim),
            ],
            dtype=np.float32,
        )

        # Keep simulator state consistent with observation
        self.data.qpos[:3] = self._agent_location
        self.data.qvel[:] = 0.0
        mj.mj_forward(self.model, self.data)


        # Randomly place target, ensuring it's different from agent position
        self._target_location = self._agent_location.copy()
        while np.array_equal(self._target_location, self._agent_location):
            self._target_location = np.array(
            [
                self.np_random.uniform(-self.xylim, self.xylim),
                self.np_random.uniform(-self.xylim, self.xylim),
                self.np_random.uniform(0.2, self.zlim),
            ],
            dtype=np.float32,
        )
        self._agent_yawrate = np.zeros_like(self._agent_yawrate)
        self._agent_vel = np.zeros_like(self._agent_vel)
        self._agent_orientation = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
        self._prev_action = np.zeros(4, dtype=np.float32)
        self._prev_distance = np.linalg.norm(self._target_location - self._agent_location)
        observation = self._get_obs()
        info = self._get_info()
        self._step = 0
        return observation, info
    
    
    def step(self, action):
        """Execute one timestep within the environment.

        Args:
            action: The action to take (0-3 for directions)

        Returns:
            tuple: (observation, reward, terminated, truncated, info)
        """
        action = np.asarray(action, dtype=np.float32).reshape(4,)
        action = np.clip(action, self.action_space.low, self.action_space.high)

        thrust = self.vel_to_thrust(action)
        self.data.ctrl[:] = thrust
        mj.mj_step(self.model, self.data)
        
        self._agent_location = self.data.subtree_com[self.body_id].astype(np.float32).copy()
        self._agent_vel = self.data.cvel[self.body_id, 3:6].astype(np.float32).copy()
        self._agent_yawrate = self.data.cvel[self.body_id, 2:3].astype(np.float32).copy()
        self._agent_orientation = self.data.qpos[3:7].astype(np.float32).copy()

        dist = np.linalg.norm(self._target_location - self._agent_location)
        delta_action = action - self._prev_action
        collision = self.data.ncon > 0

        r_prog = self.w_progress * (self._prev_distance - dist)
        r_coll = -self.r_collision if collision else 0.0
        r_energy = -self.w_energy * float(np.dot(action, action))
        r_smooth = -self.w_smooth * float(np.dot(delta_action, delta_action))
        reward = r_prog + r_coll + r_energy + r_smooth

        success = dist <= self.eps_goal
        if success:
            reward += self.r_goal

        obs = self._get_obs()
        terminated = success or collision
        truncated = (self._step >= self.max_steps)
        info = self._get_info()
        info.update(
            {
                "r_prog": r_prog,
                "r_coll": r_coll,
                "r_energy": r_energy,
                "r_smooth": r_smooth,
                "success": success,
                "collision": collision,
            }
        )

        self._prev_action = action.copy()
        self._prev_distance = dist
        self._step+=1
        return obs, reward, terminated, truncated, info

    def vel_to_thrust(self, action):
        # Convert normalized commands to physical setpoints.
        cmd = np.clip(action, self.action_space.low, self.action_space.high)
        v_des = np.array(
            [
                cmd[0] * self.vellim,
                cmd[1] * self.vellim,
                cmd[2] * self.vellim,
            ],
            dtype=np.float32,
        )
        yawrate_des = float(cmd[3] * self.yawrate_lim)

        v_err = v_des - self._agent_vel
        yaw_err = yawrate_des - float(self._agent_yawrate[0])

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
        return np.clip(thrust, self.motor_low, self.motor_high)


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
