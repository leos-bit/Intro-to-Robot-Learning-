import argparse
import time
from functools import partial
from pathlib import Path

import jax
import jax.numpy as jp
import mujoco
import mujoco.viewer
import numpy as np
from mujoco import mjx

outer_i_limit = 2.0
_outer_pid_dim = 3
max_tilt = 0.45
collective_limit = 2.0
_inner_pid_dim = 3
inner_i_limit = 1.0
attitude_limit = 1.2
yaw_limit = 0.7

sim_dt = 0.01
ctrl_dt = 0.01
outer_decim = 2

_xml_path = "mujoco_drone_imp/Drone_MJCFs/skydio_x2/scene.xml"
_mj_model = mujoco.MjModel.from_xml_path(_xml_path)
_mj_model.opt.timestep = sim_dt
_mjx_model = mjx.put_model(_mj_model)
_mjx_template_data = mujoco.MjData(_mj_model)
mujoco.mj_forward(_mj_model, _mjx_template_data)
_mjx_data0 = mjx.put_data(_mj_model, _mjx_template_data)
_mjx_qpos0 = jp.asarray(_mj_model.qpos0, dtype=jp.float32)

total_mass = float(_mj_model.body_mass.sum())
gravity = float(-_mj_model.opt.gravity[2])
hover_thrust = total_mass * gravity / _mj_model.nu
motor_low = jp.array(_mj_model.actuator_ctrlrange[:, 0], dtype=jp.float32)
motor_high = jp.array(_mj_model.actuator_ctrlrange[:, 1], dtype=jp.float32)
hover_ctrl = jp.full((_mj_model.nu,), hover_thrust, dtype=jp.float32)
if _mj_model.nkey > 0:
    hover_key_id = mujoco.mj_name2id(_mj_model, mujoco.mjtObj.mjOBJ_KEY, "hover")
    if hover_key_id != -1:
        hover_ctrl = jp.array(_mj_model.key_ctrl[hover_key_id], dtype=jp.float32)

vellim = 2.5
yawrate_lim = 2.0
gain_arr = jp.array(
    [0.55, 0.0, 0.35, 1.2, 0.0, 0.2, 0.35, 0.0, 0.35, 4.5, 0.45, 0.0],
    dtype=jp.float32,
)
NUM_PID_GAINS = 12

N_SUBSTEPS = max(1, int(round(ctrl_dt / sim_dt)))
INNER_DT = sim_dt
OUTER_DT = sim_dt * outer_decim


def action_to_command(action: jax.Array) -> jax.Array:
    cmd = jp.asarray(action, dtype=jp.float32).reshape((4,))
    cmd = jp.nan_to_num(cmd, nan=0.0, posinf=1.0, neginf=-1.0)
    cmd = jp.clip(cmd, -1.0, 1.0)
    return jp.array(
        [
            cmd[0] * vellim,
            cmd[1] * vellim,
            cmd[2] * vellim,
            cmd[3] * yawrate_lim,
        ],
        dtype=jp.float32,
    )


def _zero_pid_state(dim: int) -> jax.Array:
    return jp.zeros((2 * dim,), dtype=jp.float32)


def _pack_pid_state(integral: jax.Array, prev_error: jax.Array) -> jax.Array:
    return jp.concatenate(
        [
            jp.asarray(integral, dtype=jp.float32),
            jp.asarray(prev_error, dtype=jp.float32),
        ],
        axis=0,
    )


def _unpack_pid_state(pid_state: jax.Array, dim: int) -> tuple[jax.Array, jax.Array]:
    pid_state = jp.asarray(pid_state, dtype=jp.float32).reshape((2 * dim,))
    return pid_state[:dim], pid_state[dim:]


def _quat_to_roll_pitch_yaw(quat: jax.Array) -> tuple[jax.Array, jax.Array, jax.Array]:
    quat = jp.nan_to_num(quat, nan=0.0, posinf=1.0, neginf=-1.0)
    quat = quat / jp.maximum(jp.linalg.norm(quat), 1e-6)
    w, x, y, z = quat

    sin_roll = 2.0 * ((w * x) + (y * z))
    cos_roll = 1.0 - (2.0 * ((x * x) + (y * y)))
    roll = jp.arctan2(sin_roll, cos_roll)

    sin_pitch = 2.0 * ((w * y) - (z * x))
    pitch = jp.arcsin(jp.clip(sin_pitch, -0.999999, 0.999999))

    sin_yaw = 2.0 * ((w * z) + (x * y))
    cos_yaw = 1.0 - (2.0 * ((y * y) + (z * z)))
    yaw = jp.arctan2(sin_yaw, cos_yaw)
    return roll, pitch, yaw


def velocity_controller(
    agent_vel: jax.Array,
    agent_orientation: jax.Array,
    cmd: jax.Array,
    pid_state: jax.Array,
    dt: float,
    gain_arr: jax.Array,
) -> tuple[jax.Array, jax.Array]:
    agent_vel = jp.nan_to_num(agent_vel, nan=0.0, posinf=0.0, neginf=0.0)
    cmd = jp.nan_to_num(cmd, nan=0.0, posinf=0.0, neginf=0.0)
    roll, pitch, yaw = _quat_to_roll_pitch_yaw(agent_orientation)
    del roll, pitch

    k_xy = gain_arr[0]
    ki_xy = gain_arr[1]
    kd_xy = gain_arr[2]
    k_z = gain_arr[3]
    ki_z = gain_arr[4]
    kd_z = gain_arr[5]

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

    integral, prev_error = _unpack_pid_state(pid_state, _outer_pid_dim)
    dt = jp.asarray(max(dt, 1e-6), dtype=jp.float32)
    derivative = (vel_err_body - prev_error) / dt
    prev_integral = integral
    candidate_integral = jp.clip(
        integral + (vel_err_body * dt),
        -outer_i_limit,
        outer_i_limit,
    )

    roll_unsat = -(
        (k_xy * vel_err_body[1])
        + (ki_xy * candidate_integral[1])
        + (kd_xy * derivative[1])
    )
    roll_des = jp.clip(roll_unsat, -max_tilt, max_tilt)

    pitch_unsat = (
        (k_xy * vel_err_body[0])
        + (ki_xy * candidate_integral[0])
        + (kd_xy * derivative[0])
    )
    pitch_des = jp.clip(pitch_unsat, -max_tilt, max_tilt)

    collective_unsat = (
        (k_z * vel_err_body[2])
        + (ki_z * candidate_integral[2])
        + (kd_z * derivative[2])
    )
    collective = jp.clip(collective_unsat, -collective_limit, collective_limit)

    sat_eps = jp.asarray(1e-6, dtype=jp.float32)
    integral = candidate_integral
    integral = integral.at[0].set(
        jp.where(
            jp.abs(pitch_unsat - pitch_des) > sat_eps,
            prev_integral[0],
            candidate_integral[0],
        )
    )
    integral = integral.at[1].set(
        jp.where(
            jp.abs(roll_unsat - roll_des) > sat_eps,
            prev_integral[1],
            candidate_integral[1],
        )
    )
    integral = integral.at[2].set(
        jp.where(
            jp.abs(collective_unsat - collective) > sat_eps,
            prev_integral[2],
            candidate_integral[2],
        )
    )

    outer_cmd = jp.array([collective, roll_des, pitch_des, cmd[3]], dtype=jp.float32)
    return outer_cmd, _pack_pid_state(integral, vel_err_body)


def _attitude_rate_controller_core(
    agent_orientation: jax.Array,
    agent_angvel: jax.Array,
    cmd: jax.Array,
    pid_state: jax.Array,
    dt: float,
    gain_arr: jax.Array,
    hover_ctrl_arr: jax.Array,
) -> tuple[jax.Array, jax.Array]:
    agent_angvel = jp.nan_to_num(agent_angvel, nan=0.0, posinf=0.0, neginf=0.0)
    cmd = jp.nan_to_num(cmd, nan=0.0, posinf=0.0, neginf=0.0)
    roll, pitch, _ = _quat_to_roll_pitch_yaw(agent_orientation)

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
    integral, prev_error = _unpack_pid_state(pid_state, _inner_pid_dim)
    dt = jp.asarray(max(dt, 1e-6), dtype=jp.float32)
    derivative = (err - prev_error) / dt
    prev_integral = integral
    candidate_integral = jp.clip(
        integral + (err * dt),
        -inner_i_limit,
        inner_i_limit,
    )

    u_collective = jp.clip(cmd[0], -collective_limit, collective_limit)
    u_roll_unsat = (
        (kp_att * err[0])
        + (ki_att * candidate_integral[0])
        - (kd_att * agent_angvel[0])
    )
    u_roll = jp.clip(u_roll_unsat, -attitude_limit, attitude_limit)

    u_pitch_unsat = (
        (kp_att * err[1])
        + (ki_att * candidate_integral[1])
        - (kd_att * agent_angvel[1])
    )
    u_pitch = jp.clip(u_pitch_unsat, -attitude_limit, attitude_limit)

    u_yaw_unsat = (
        (k_yaw * err[2])
        + (ki_yaw * candidate_integral[2])
        + (kd_yaw * derivative[2])
    )
    u_yaw = jp.clip(u_yaw_unsat, -yaw_limit, yaw_limit)

    sat_eps = jp.asarray(1e-6, dtype=jp.float32)
    integral = candidate_integral
    integral = integral.at[0].set(
        jp.where(
            jp.abs(u_roll_unsat - u_roll) > sat_eps,
            prev_integral[0],
            candidate_integral[0],
        )
    )
    integral = integral.at[1].set(
        jp.where(
            jp.abs(u_pitch_unsat - u_pitch) > sat_eps,
            prev_integral[1],
            candidate_integral[1],
        )
    )
    integral = integral.at[2].set(
        jp.where(
            jp.abs(u_yaw_unsat - u_yaw) > sat_eps,
            prev_integral[2],
            candidate_integral[2],
        )
    )

    tilt_comp = jp.clip(jp.cos(roll) * jp.cos(pitch), 0.5, 1.0)
    hover_cmd = jp.asarray(hover_ctrl_arr, dtype=jp.float32) / tilt_comp
    thrust = jp.array(
        [
            hover_cmd[0] + u_collective - u_roll + u_pitch - u_yaw,
            hover_cmd[1] + u_collective + u_roll + u_pitch + u_yaw,
            hover_cmd[2] + u_collective + u_roll - u_pitch - u_yaw,
            hover_cmd[3] + u_collective - u_roll - u_pitch + u_yaw,
        ],
        dtype=jp.float32,
    )
    thrust = jp.nan_to_num(
        thrust,
        nan=hover_thrust,
        posinf=jp.max(motor_high),
        neginf=jp.min(motor_low),
    )
    thrust = jp.clip(thrust, motor_low, motor_high)
    return thrust, _pack_pid_state(integral, err)


def attitude_rate_controller(
    agent_orientation: jax.Array,
    agent_angvel: jax.Array,
    cmd: jax.Array,
    pid_state: jax.Array,
    dt: float,
    gain_arr: jax.Array,
) -> tuple[jax.Array, jax.Array]:
    return _attitude_rate_controller_core(
        agent_orientation=agent_orientation,
        agent_angvel=agent_angvel,
        cmd=cmd,
        pid_state=pid_state,
        dt=dt,
        gain_arr=gain_arr,
        hover_ctrl_arr=hover_ctrl,
    )


def _attitude_rate_controller_step(
    agent_orientation: jax.Array,
    agent_angvel: jax.Array,
    cmd: jax.Array,
    pid_state: jax.Array,
    dt: float,
    gain_arr: jax.Array,
    hover_ctrl_arr: jax.Array,
) -> tuple[jax.Array, jax.Array]:
    return _attitude_rate_controller_core(
        agent_orientation=agent_orientation,
        agent_angvel=agent_angvel,
        cmd=cmd,
        pid_state=pid_state,
        dt=dt,
        gain_arr=gain_arr,
        hover_ctrl_arr=hover_ctrl_arr,
    )


def _sensor_slice(mj_model: mujoco.MjModel, name: str) -> slice:
    sensor_id = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_SENSOR, name)
    if sensor_id == -1:
        raise ValueError(f"Unknown sensor: {name}")
    start = int(mj_model.sensor_adr[sensor_id])
    dim = int(mj_model.sensor_dim[sensor_id])
    return slice(start, start + dim)


_BODY_ID = mujoco.mj_name2id(_mj_model, mujoco.mjtObj.mjOBJ_BODY, "x2")
if _BODY_ID == -1:
    raise ValueError("Could not find body 'x2' in the MuJoCo model.")
_GYRO_SENSOR_SLICE = _sensor_slice(_mj_model, "body_gyro")
_QUAT_SENSOR_SLICE = _sensor_slice(_mj_model, "body_quat")


def _extract_body_state(
    mj_data: mujoco.MjData,
    body_id: int,
    gyro_sensor_slice: slice,
    quat_sensor_slice: slice,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    agent_pos = jp.asarray(mj_data.subtree_com[body_id], dtype=jp.float32)
    agent_vel = jp.asarray(mj_data.cvel[body_id, 3:6], dtype=jp.float32)
    sensor_data = jp.asarray(mj_data.sensordata, dtype=jp.float32)
    agent_angvel = sensor_data[gyro_sensor_slice]
    agent_orientation = sensor_data[quat_sensor_slice]
    agent_orientation = agent_orientation / jp.maximum(jp.linalg.norm(agent_orientation), 1e-6)
    return agent_pos, agent_vel, agent_angvel, agent_orientation


def run_cascaded_controller(
    action: jax.Array,
    agent_vel: jax.Array,
    agent_angvel: jax.Array,
    agent_orientation: jax.Array,
    outer_cmd: jax.Array,
    outer_pid_state: jax.Array,
    inner_pid_state: jax.Array,
    gain_arr: jax.Array,
    vellim: float,
    yawrate_lim: float,
    hover_ctrl_arr: jax.Array,
    sim_step: int = 0,
    outer_dt: float | None = None,
    inner_dt: float | None = None,
) -> tuple[jax.Array, dict[str, jax.Array]]:
    del vellim, yawrate_lim

    gain_arr = jp.asarray(gain_arr, dtype=jp.float32).reshape((NUM_PID_GAINS,))
    outer_dt = OUTER_DT if outer_dt is None else float(outer_dt)
    inner_dt = INNER_DT if inner_dt is None else float(inner_dt)

    vel_cmd = action_to_command(action)
    next_outer_cmd = jp.asarray(outer_cmd, dtype=jp.float32)
    next_outer_pid_state = jp.asarray(outer_pid_state, dtype=jp.float32)

    if sim_step % outer_decim == 0:
        next_outer_cmd, next_outer_pid_state = velocity_controller(
            agent_vel=agent_vel,
            agent_orientation=agent_orientation,
            cmd=vel_cmd,
            pid_state=outer_pid_state,
            dt=outer_dt,
            gain_arr=gain_arr,
        )

    motor_cmd, next_inner_pid_state = _attitude_rate_controller_step(
        agent_orientation=agent_orientation,
        agent_angvel=agent_angvel,
        cmd=next_outer_cmd,
        pid_state=inner_pid_state,
        dt=inner_dt,
        gain_arr=gain_arr,
        hover_ctrl_arr=hover_ctrl_arr,
    )

    return motor_cmd, {
        "vel_cmd": vel_cmd,
        "outer_cmd": next_outer_cmd,
        "outer_pid_state": next_outer_pid_state,
        "inner_pid_state": next_inner_pid_state,
        "sim_step": sim_step + 1,
    }


def step_dynamics(
    mj_model: mujoco.MjModel,
    mj_data: mujoco.MjData,
    action: jax.Array,
    outer_cmd: jax.Array | None = None,
    outer_pid_state: jax.Array | None = None,
    inner_pid_state: jax.Array | None = None,
    gain_arr: jax.Array | None = None,
    vellim: float = vellim,
    yawrate_lim: float = yawrate_lim,
    hover_ctrl_arr: jax.Array | None = None,
    sim_step: int = 0,
    body_id: int | None = None,
    gyro_sensor_slice: slice | None = None,
    quat_sensor_slice: slice | None = None,
    outer_dt: float | None = None,
    inner_dt: float | None = None,
) -> tuple[mujoco.MjData, dict[str, jax.Array]]:
    if body_id is None:
        body_id = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_BODY, "x2")
    if body_id == -1:
        raise ValueError("Could not find body 'x2' in the MuJoCo model.")
    if gyro_sensor_slice is None:
        gyro_sensor_slice = _sensor_slice(mj_model, "body_gyro")
    if quat_sensor_slice is None:
        quat_sensor_slice = _sensor_slice(mj_model, "body_quat")
    if hover_ctrl_arr is None:
        hover_ctrl_arr = jp.full((mj_model.nu,), hover_thrust, dtype=jp.float32)
    if outer_cmd is None:
        outer_cmd = jp.zeros((4,), dtype=jp.float32)
    if outer_pid_state is None:
        outer_pid_state = _zero_pid_state(_outer_pid_dim)
    if inner_pid_state is None:
        inner_pid_state = _zero_pid_state(_inner_pid_dim)
    if gain_arr is None:
        gain_arr = globals()["gain_arr"]
    outer_dt = OUTER_DT if outer_dt is None else float(outer_dt)
    inner_dt = INNER_DT if inner_dt is None else float(inner_dt)

    pos, vel, angvel, ori = _extract_body_state(
        mj_data,
        body_id=body_id,
        gyro_sensor_slice=gyro_sensor_slice,
        quat_sensor_slice=quat_sensor_slice,
    )
    state_pos = pos
    state_ori = ori
    state_vel = vel
    state_angvel = angvel

    current_outer_cmd = jp.asarray(outer_cmd, dtype=jp.float32)
    current_outer_pid_state = jp.asarray(outer_pid_state, dtype=jp.float32)
    current_inner_pid_state = jp.asarray(inner_pid_state, dtype=jp.float32)
    current_sim_step = int(sim_step)
    current_vel_cmd = action_to_command(action)

    for _ in range(N_SUBSTEPS):
        motor_cmd, controller_state = run_cascaded_controller(
            action=action,
            agent_vel=vel,
            agent_angvel=angvel,
            agent_orientation=ori,
            outer_cmd=current_outer_cmd,
            outer_pid_state=current_outer_pid_state,
            inner_pid_state=current_inner_pid_state,
            gain_arr=gain_arr,
            vellim=vellim,
            yawrate_lim=yawrate_lim,
            hover_ctrl_arr=hover_ctrl_arr,
            sim_step=current_sim_step,
            outer_dt=outer_dt,
            inner_dt=inner_dt,
        )
        current_vel_cmd = controller_state["vel_cmd"]
        current_outer_cmd = controller_state["outer_cmd"]
        current_outer_pid_state = controller_state["outer_pid_state"]
        current_inner_pid_state = controller_state["inner_pid_state"]
        current_sim_step = int(controller_state["sim_step"])

        mj_data.ctrl[:] = np.asarray(jax.device_get(motor_cmd), dtype=np.float64)
        mujoco.mj_step(mj_model, mj_data)
        pos, vel, angvel, ori = _extract_body_state(
            mj_data,
            body_id=body_id,
            gyro_sensor_slice=gyro_sensor_slice,
            quat_sensor_slice=quat_sensor_slice,
        )

    pos2, vel2, angvel2, ori2 = pos, vel, angvel, ori
    raw_action = jp.asarray(action, dtype=jp.float32).reshape((4,))

    transition = {
        "state": jp.concatenate([state_pos, state_ori], axis=0),
        "action": current_vel_cmd,
        "raw_action": raw_action,
        "next_state": jp.concatenate([pos2, ori2], axis=0),
        "vel": state_vel,
        "angvel": state_angvel,
        "vel2": vel2,
        "angvel2": angvel2,
        "motor_cmd": motor_cmd,
        "outer_cmd": current_outer_cmd,
        "outer_pid_state": current_outer_pid_state,
        "inner_pid_state": current_inner_pid_state,
        "sim_step": current_sim_step,
    }
    return mj_data, transition


def _physics_step_mjx(mjx_data: mjx.Data, motor_cmd: jax.Array) -> mjx.Data:
    ctrl = jp.asarray(motor_cmd, dtype=mjx_data.ctrl.dtype)
    mjx_data = mjx_data.replace(ctrl=ctrl)
    return mjx.step(_mjx_model, mjx_data)


def _run_cascaded_controller_scan(
    action: jax.Array,
    agent_vel: jax.Array,
    agent_angvel: jax.Array,
    agent_orientation: jax.Array,
    outer_cmd: jax.Array,
    outer_pid_state: jax.Array,
    inner_pid_state: jax.Array,
    gain_arr: jax.Array,
    hover_ctrl_arr: jax.Array,
    sim_step: jax.Array,
    outer_dt: float | None = None,
    inner_dt: float | None = None,
) -> tuple[jax.Array, dict[str, jax.Array]]:
    gain_arr = jp.asarray(gain_arr, dtype=jp.float32).reshape((NUM_PID_GAINS,))
    outer_dt = OUTER_DT if outer_dt is None else float(outer_dt)
    inner_dt = INNER_DT if inner_dt is None else float(inner_dt)

    vel_cmd = action_to_command(action)
    current_outer_cmd = jp.asarray(outer_cmd, dtype=jp.float32)
    current_outer_pid_state = jp.asarray(outer_pid_state, dtype=jp.float32)
    current_inner_pid_state = jp.asarray(inner_pid_state, dtype=jp.float32)
    sim_step = jp.asarray(sim_step, dtype=jp.int32)

    def _update_outer(_: None) -> tuple[jax.Array, jax.Array]:
        return velocity_controller(
            agent_vel=agent_vel,
            agent_orientation=agent_orientation,
            cmd=vel_cmd,
            pid_state=current_outer_pid_state,
            dt=outer_dt,
            gain_arr=gain_arr,
        )

    next_outer_cmd, next_outer_pid_state = jax.lax.cond(
        (sim_step % outer_decim) == 0,
        _update_outer,
        lambda _: (current_outer_cmd, current_outer_pid_state),
        operand=None,
    )

    motor_cmd, next_inner_pid_state = _attitude_rate_controller_step(
        agent_orientation=agent_orientation,
        agent_angvel=agent_angvel,
        cmd=next_outer_cmd,
        pid_state=current_inner_pid_state,
        dt=inner_dt,
        gain_arr=gain_arr,
        hover_ctrl_arr=hover_ctrl_arr,
    )

    return motor_cmd, {
        "vel_cmd": vel_cmd,
        "outer_cmd": next_outer_cmd,
        "outer_pid_state": next_outer_pid_state,
        "inner_pid_state": next_inner_pid_state,
        "sim_step": sim_step + jp.array(1, dtype=jp.int32),
    }


def _step_dynamics_mjx(
    mjx_data: mjx.Data,
    action: jax.Array,
    outer_cmd: jax.Array,
    outer_pid_state: jax.Array,
    inner_pid_state: jax.Array,
    gain_arr: jax.Array,
    hover_ctrl_arr: jax.Array,
    sim_step: jax.Array,
    outer_dt: float | None = None,
    inner_dt: float | None = None,
) -> tuple[mjx.Data, dict[str, jax.Array]]:
    pos, vel, angvel, ori = _extract_body_state(
        mjx_data,
        body_id=_BODY_ID,
        gyro_sensor_slice=_GYRO_SENSOR_SLICE,
        quat_sensor_slice=_QUAT_SENSOR_SLICE,
    )
    state_pos = pos
    state_ori = ori
    state_vel = vel
    state_angvel = angvel
    current_vel_cmd = action_to_command(action)
    initial_motor_cmd = jp.zeros((int(_mj_model.nu),), dtype=jp.float32)

    carry = (
        mjx_data,
        pos,
        vel,
        angvel,
        ori,
        jp.asarray(outer_cmd, dtype=jp.float32),
        jp.asarray(outer_pid_state, dtype=jp.float32),
        jp.asarray(inner_pid_state, dtype=jp.float32),
        jp.asarray(sim_step, dtype=jp.int32),
        initial_motor_cmd,
        current_vel_cmd,
    )

    def _substep(carry, _):
        (
            step_data,
            _,
            step_vel,
            step_angvel,
            step_ori,
            step_outer_cmd,
            step_outer_pid_state,
            step_inner_pid_state,
            step_sim_step,
            _,
            _,
        ) = carry

        motor_cmd, controller_state = _run_cascaded_controller_scan(
            action=action,
            agent_vel=step_vel,
            agent_angvel=step_angvel,
            agent_orientation=step_ori,
            outer_cmd=step_outer_cmd,
            outer_pid_state=step_outer_pid_state,
            inner_pid_state=step_inner_pid_state,
            gain_arr=gain_arr,
            hover_ctrl_arr=hover_ctrl_arr,
            sim_step=step_sim_step,
            outer_dt=outer_dt,
            inner_dt=inner_dt,
        )
        step_data = _physics_step_mjx(step_data, motor_cmd)
        pos2, vel2, angvel2, ori2 = _extract_body_state(
            step_data,
            body_id=_BODY_ID,
            gyro_sensor_slice=_GYRO_SENSOR_SLICE,
            quat_sensor_slice=_QUAT_SENSOR_SLICE,
        )
        return (
            step_data,
            pos2,
            vel2,
            angvel2,
            ori2,
            controller_state["outer_cmd"],
            controller_state["outer_pid_state"],
            controller_state["inner_pid_state"],
            controller_state["sim_step"],
            motor_cmd,
            controller_state["vel_cmd"],
        ), None

    final_carry, _ = jax.lax.scan(_substep, carry, xs=None, length=N_SUBSTEPS)
    (
        mjx_data,
        pos2,
        vel2,
        angvel2,
        ori2,
        current_outer_cmd,
        current_outer_pid_state,
        current_inner_pid_state,
        current_sim_step,
        motor_cmd,
        current_vel_cmd,
    ) = final_carry
    raw_action = jp.asarray(action, dtype=jp.float32).reshape((4,))

    transition = {
        "state": jp.concatenate([state_pos, state_ori], axis=0),
        "action": current_vel_cmd,
        "raw_action": raw_action,
        "next_state": jp.concatenate([pos2, ori2], axis=0),
        "vel": state_vel,
        "angvel": state_angvel,
        "vel2": vel2,
        "angvel2": angvel2,
        "motor_cmd": motor_cmd,
        "outer_cmd": current_outer_cmd,
        "outer_pid_state": current_outer_pid_state,
        "inner_pid_state": current_inner_pid_state,
        "sim_step": current_sim_step,
    }
    return mjx_data, transition


def _init_rollout_data(mj_model: mujoco.MjModel, init_pos: list[float]) -> mujoco.MjData:
    mj_data = mujoco.MjData(mj_model)
    qpos = np.asarray(mj_model.qpos0, dtype=np.float64).copy()
    qpos[: len(init_pos)] = np.asarray(init_pos, dtype=np.float64)
    mj_data.qpos[:] = qpos
    mj_data.qvel[:] = 0.0
    mj_data.ctrl[:] = np.asarray(jax.device_get(hover_ctrl), dtype=np.float64)
    mujoco.mj_forward(mj_model, mj_data)
    return mj_data


def _init_rollout_data_mjx(init_pos: jax.Array) -> mjx.Data:
    qpos = _mjx_qpos0.at[: init_pos.shape[0]].set(jp.asarray(init_pos, dtype=jp.float32))
    mjx_data = _mjx_data0.replace(
        qpos=qpos,
        qvel=jp.zeros_like(_mjx_data0.qvel),
        act=jp.zeros_like(_mjx_data0.act),
        qacc_warmstart=jp.zeros_like(_mjx_data0.qacc_warmstart),
        ctrl=hover_ctrl.astype(_mjx_data0.ctrl.dtype),
    )
    return mjx.forward(_mjx_model, mjx_data)


def collect_random_rollouts(
    init_pos: list[float],
    actions: jax.Array,
    out_dir: str = "jax_implementation/MBRL/dyn_data",
    render: bool = False,
    seed: int | None = None,
) -> Path:
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    actions_np = np.asarray(jax.device_get(actions), dtype=np.float32)
    B, N, _ = actions_np.shape

    states = np.zeros((B, N, 7), dtype=np.float32)
    next_states = np.zeros((B, N, 7), dtype=np.float32)
    planner_actions = np.zeros((B, N, 4), dtype=np.float32)
    raw_actions = np.zeros((B, N, 4), dtype=np.float32)
    vels = np.zeros((B, N, 3), dtype=np.float32)
    angvels = np.zeros((B, N, 3), dtype=np.float32)
    vels2 = np.zeros((B, N, 3), dtype=np.float32)
    angvels2 = np.zeros((B, N, 3), dtype=np.float32)
    motor_cmds = np.zeros((B, N, int(_mj_model.nu)), dtype=np.float32)
    outer_cmds = np.zeros((B, N, 4), dtype=np.float32)
    outer_pid_states = np.zeros((B, N, 2 * _outer_pid_dim), dtype=np.float32)
    inner_pid_states = np.zeros((B, N, 2 * _inner_pid_dim), dtype=np.float32)

    body_id = mujoco.mj_name2id(_mj_model, mujoco.mjtObj.mjOBJ_BODY, "x2")
    gyro_sensor_slice = _sensor_slice(_mj_model, "body_gyro")
    quat_sensor_slice = _sensor_slice(_mj_model, "body_quat")

    for b in range(B):
        mj_data = _init_rollout_data(_mj_model, init_pos)
        outer_cmd = jp.zeros((4,), dtype=jp.float32)
        outer_pid_state = _zero_pid_state(_outer_pid_dim)
        inner_pid_state = _zero_pid_state(_inner_pid_dim)
        sim_step = 0

        if render and b == 0:
            with mujoco.viewer.launch_passive(_mj_model, mj_data) as viewer:
                for t in range(N):
                    mj_data, transition = step_dynamics(
                        mj_model=_mj_model,
                        mj_data=mj_data,
                        action=jp.asarray(actions_np[b, t], dtype=jp.float32),
                        outer_cmd=outer_cmd,
                        outer_pid_state=outer_pid_state,
                        inner_pid_state=inner_pid_state,
                        gain_arr=gain_arr,
                        vellim=vellim,
                        yawrate_lim=yawrate_lim,
                        hover_ctrl_arr=hover_ctrl,
                        sim_step=sim_step,
                        body_id=body_id,
                        gyro_sensor_slice=gyro_sensor_slice,
                        quat_sensor_slice=quat_sensor_slice,
                    )
                    outer_cmd = transition["outer_cmd"]
                    outer_pid_state = transition["outer_pid_state"]
                    inner_pid_state = transition["inner_pid_state"]
                    sim_step = transition["sim_step"]

                    states[b, t] = np.asarray(jax.device_get(transition["state"]), dtype=np.float32)
                    next_states[b, t] = np.asarray(jax.device_get(transition["next_state"]), dtype=np.float32)
                    planner_actions[b, t] = np.asarray(jax.device_get(transition["action"]), dtype=np.float32)
                    raw_actions[b, t] = np.asarray(jax.device_get(transition["raw_action"]), dtype=np.float32)
                    vels[b, t] = np.asarray(jax.device_get(transition["vel"]), dtype=np.float32)
                    angvels[b, t] = np.asarray(jax.device_get(transition["angvel"]), dtype=np.float32)
                    vels2[b, t] = np.asarray(jax.device_get(transition["vel2"]), dtype=np.float32)
                    angvels2[b, t] = np.asarray(jax.device_get(transition["angvel2"]), dtype=np.float32)
                    motor_cmds[b, t] = np.asarray(jax.device_get(transition["motor_cmd"]), dtype=np.float32)
                    outer_cmds[b, t] = np.asarray(jax.device_get(transition["outer_cmd"]), dtype=np.float32)
                    outer_pid_states[b, t] = np.asarray(
                        jax.device_get(transition["outer_pid_state"]),
                        dtype=np.float32,
                    )
                    inner_pid_states[b, t] = np.asarray(
                        jax.device_get(transition["inner_pid_state"]),
                        dtype=np.float32,
                    )

                    viewer.sync()
                    time.sleep(ctrl_dt)
        else:
            for t in range(N):
                mj_data, transition = step_dynamics(
                    mj_model=_mj_model,
                    mj_data=mj_data,
                    action=jp.asarray(actions_np[b, t], dtype=jp.float32),
                    outer_cmd=outer_cmd,
                    outer_pid_state=outer_pid_state,
                    inner_pid_state=inner_pid_state,
                    gain_arr=gain_arr,
                    vellim=vellim,
                    yawrate_lim=yawrate_lim,
                    hover_ctrl_arr=hover_ctrl,
                    sim_step=sim_step,
                    body_id=body_id,
                    gyro_sensor_slice=gyro_sensor_slice,
                    quat_sensor_slice=quat_sensor_slice,
                )
                outer_cmd = transition["outer_cmd"]
                outer_pid_state = transition["outer_pid_state"]
                inner_pid_state = transition["inner_pid_state"]
                sim_step = transition["sim_step"]

                states[b, t] = np.asarray(jax.device_get(transition["state"]), dtype=np.float32)
                next_states[b, t] = np.asarray(jax.device_get(transition["next_state"]), dtype=np.float32)
                planner_actions[b, t] = np.asarray(jax.device_get(transition["action"]), dtype=np.float32)
                raw_actions[b, t] = np.asarray(jax.device_get(transition["raw_action"]), dtype=np.float32)
                vels[b, t] = np.asarray(jax.device_get(transition["vel"]), dtype=np.float32)
                angvels[b, t] = np.asarray(jax.device_get(transition["angvel"]), dtype=np.float32)
                vels2[b, t] = np.asarray(jax.device_get(transition["vel2"]), dtype=np.float32)
                angvels2[b, t] = np.asarray(jax.device_get(transition["angvel2"]), dtype=np.float32)
                motor_cmds[b, t] = np.asarray(jax.device_get(transition["motor_cmd"]), dtype=np.float32)
                outer_cmds[b, t] = np.asarray(jax.device_get(transition["outer_cmd"]), dtype=np.float32)
                outer_pid_states[b, t] = np.asarray(
                    jax.device_get(transition["outer_pid_state"]),
                    dtype=np.float32,
                )
                inner_pid_states[b, t] = np.asarray(
                    jax.device_get(transition["inner_pid_state"]),
                    dtype=np.float32,
                )

    seed_suffix = "none" if seed is None else str(seed)
    save_path = out_path / f"random_rollouts_B{B}_N{N}_seed{seed_suffix}.npz"
    np.savez(
        save_path,
        seed=np.array(-1 if seed is None else seed, dtype=np.int32),
        init_pos=np.asarray(init_pos, dtype=np.float32),
        state=states,
        action=planner_actions,
        raw_action=raw_actions,
        next_state=next_states,
        vel=vels,
        angvel=angvels,
        vel2=vels2,
        angvel2=angvels2,
        motor_cmd=motor_cmds,
        outer_cmd=outer_cmds,
        outer_pid_state=outer_pid_states,
        inner_pid_state=inner_pid_states,
    )
    return save_path


@partial(jax.jit, static_argnames=("horizon",))
def _collect_random_rollouts_parallel_impl(
    init_pos: jax.Array,
    batch_keys: jax.Array,
    horizon: int,
) -> dict[str, jax.Array]:
    batch_size = batch_keys.shape[0]
    init_pos = jp.asarray(init_pos, dtype=jp.float32)
    batched_init_pos = jp.broadcast_to(init_pos, (batch_size, init_pos.shape[0]))
    batch_data = jax.vmap(_init_rollout_data_mjx)(batched_init_pos)
    outer_cmd = jp.zeros((batch_size, 4), dtype=jp.float32)
    outer_pid_state = jp.zeros((batch_size, 2 * _outer_pid_dim), dtype=jp.float32)
    inner_pid_state = jp.zeros((batch_size, 2 * _inner_pid_dim), dtype=jp.float32)
    sim_step = jp.zeros((batch_size,), dtype=jp.int32)

    def _scan_step(carry, _):
        batch_data, outer_cmd, outer_pid_state, inner_pid_state, sim_step, rollout_keys = carry

        next_keys, action_keys = jax.vmap(lambda key: tuple(jax.random.split(key, 2)))(rollout_keys)
        raw_actions = jax.vmap(
            lambda key: jax.random.uniform(
                key,
                (4,),
                minval=-1.0,
                maxval=1.0,
                dtype=jp.float32,
            )
        )(action_keys)

        batch_data, transition = jax.vmap(
            _step_dynamics_mjx,
            in_axes=(0, 0, 0, 0, 0, None, None, 0, None, None),
        )(
            batch_data,
            raw_actions,
            outer_cmd,
            outer_pid_state,
            inner_pid_state,
            gain_arr,
            hover_ctrl,
            sim_step,
            OUTER_DT,
            INNER_DT,
        )

        next_carry = (
            batch_data,
            transition["outer_cmd"],
            transition["outer_pid_state"],
            transition["inner_pid_state"],
            transition["sim_step"],
            next_keys,
        )
        return next_carry, transition

    _, transitions = jax.lax.scan(
        _scan_step,
        (batch_data, outer_cmd, outer_pid_state, inner_pid_state, sim_step, batch_keys),
        xs=None,
        length=horizon,
    )
    return transitions


def collect_random_rollouts_parallel(
    init_pos: list[float],
    num_rollouts: int,
    horizon: int,
    out_dir: str = "jax_implementation/MBRL/dyn_data",
    seed: int | None = None,
) -> Path:
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    if num_rollouts <= 0:
        raise ValueError("num_rollouts must be positive.")
    if horizon <= 0:
        raise ValueError("horizon must be positive.")

    seed_value = int(seed if seed is not None else np.random.randint(1, 1000))
    batch_keys = jax.random.split(jax.random.PRNGKey(seed_value), num_rollouts)
    transitions = _collect_random_rollouts_parallel_impl(
        jp.asarray(init_pos, dtype=jp.float32),
        batch_keys,
        horizon,
    )

    def _to_numpy(name: str) -> np.ndarray:
        return np.swapaxes(
            np.asarray(jax.device_get(transitions[name]), dtype=np.float32),
            0,
            1,
        )

    states = _to_numpy("state")
    planner_actions = _to_numpy("action")
    raw_actions = _to_numpy("raw_action")
    next_states = _to_numpy("next_state")
    vels = _to_numpy("vel")
    angvels = _to_numpy("angvel")
    vels2 = _to_numpy("vel2")
    angvels2 = _to_numpy("angvel2")
    motor_cmds = _to_numpy("motor_cmd")
    outer_cmds = _to_numpy("outer_cmd")
    outer_pid_states = _to_numpy("outer_pid_state")
    inner_pid_states = _to_numpy("inner_pid_state")

    save_path = out_path / f"random_rollouts_parallel_B{num_rollouts}_N{horizon}_seed{seed_value}.npz"
    np.savez(
        save_path,
        seed=np.array(seed_value, dtype=np.int32),
        init_pos=np.asarray(init_pos, dtype=np.float32),
        state=states,
        action=planner_actions,
        raw_action=raw_actions,
        next_state=next_states,
        vel=vels,
        angvel=angvels,
        vel2=vels2,
        angvel2=angvels2,
        motor_cmd=motor_cmds,
        outer_cmd=outer_cmds,
        outer_pid_state=outer_pid_states,
        inner_pid_state=inner_pid_states,
    )
    return save_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Collect random drone dynamics transitions.")
    parser.add_argument("--parallel", action="store_true", help="Use MJX vmap/scan rollout collection.")
    parser.add_argument("--render", action="store_true", help="Render the first rollout live.")
    parser.add_argument("--num_rollouts", type=int, default=1)
    parser.add_argument("--horizon", type=int, default=1000)
    parser.add_argument("--out_dir", type=str, default="jax_implementation/MBRL/dyn_data")
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    init_pos = [0, 0, 0.1, 1, 0, 0, 0]
    seed = args.seed if args.seed is not None else np.random.randint(1, 1000)
    B = args.num_rollouts
    N = args.horizon
    if args.parallel:
        if args.render:
            raise ValueError("Rendering is not supported when --parallel is enabled.")
        save_path = collect_random_rollouts_parallel(
            init_pos=init_pos,
            num_rollouts=B,
            horizon=N,
            out_dir=args.out_dir,
            seed=seed,
        )
    else:
        key = jax.random.PRNGKey(seed)
        actions = jax.random.uniform(key, (B, N, 4), minval=-1.0, maxval=1.0, dtype=jp.float32)
        save_path = collect_random_rollouts(
            init_pos=init_pos,
            actions=actions,
            out_dir=args.out_dir,
            render=args.render,
            seed=seed,
        )
    print(f"Saved random rollout dataset to {save_path}")
