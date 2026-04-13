import jax
import jax.numpy as jp
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


