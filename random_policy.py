#!/usr/bin/env python3
import time, random

from gz.transport13 import Node
from gz.msgs10.twist_pb2 import Twist
from gz.msgs10.boolean_pb2 import Boolean
import matplotlib.pyplot as plt
import numpy as np
DT = 0.02

MAX_ACC = 1.5
MAX_YAW_RATE = 1.0

MAX_VEL_XY = 1.5
MAX_VEL_Z  = 1.5
HOVER_VZ = 0.0
HOVER_Z_TARGET = 1.0
KP_Z = 2.0
KD_Z = 0.8

CMD_TOPIC = "/quad/gazebo/command/twist"
ENABLE_TOPIC = "/quad/enable"

GOAL_X, GOAL_Y, GOAL_Z = 2.0, 0.0, 1.5
WP = 1.0
WU = 0.05
WS = 0.1
R_COLL = 10.0
R_GOAL = 20.0
EPS_GOAL = 0.25
GAMMA = 0.99

def clamp(x, lo, hi): return max(lo, min(hi, x))

def reward_and_return(
    d_prev,
    d_curr,
    a_t,
    a_prev,
    step_idx,
    running_return,
    running_discounted_return,
    collision=False,
    goal_reached=False,
    w_p=WP,
    w_u=WU,
    w_s=WS,
    r_coll=R_COLL,
    r_goal=R_GOAL,
    gamma=GAMMA,
):
    r_prog = w_p * (d_prev - d_curr)
    r_coll_term = -r_coll if collision else 0.0

    ax, ay, az = a_t
    dax, day, daz = (a_t[0] - a_prev[0], a_t[1] - a_prev[1], a_t[2] - a_prev[2])

    r_energy = -w_u * (ax * ax + ay * ay + az * az)
    r_smooth = -w_s * (dax * dax + day * day + daz * daz)

    reward = r_prog + r_coll_term + r_energy + r_smooth
    if goal_reached:
        reward += r_goal

    running_return += reward
    running_discounted_return += (gamma ** step_idx) * reward

    return (
        reward,
        running_return,
        running_discounted_return,
        {
            "r_prog": r_prog,
            "r_coll": r_coll_term,
            "r_energy": r_energy,
            "r_smooth": r_smooth,
            "goal_bonus": r_goal if goal_reached else 0.0,
        },
    )

def main():
    node = Node()

    pub_cmd = node.advertise(CMD_TOPIC, Twist)
    pub_en  = node.advertise(ENABLE_TOPIC, Boolean)

    # Give subscribers time to connect
    time.sleep(0.2)

    en = Boolean()
    en.data = True
    last_en = 0.0

    vx = vy = vz = 0.0
    x, y, z = 0.0, 0.0, 1.0
    prev_action = (0.0, 0.0, 0.0)
    d_prev = ((x - GOAL_X) ** 2 + (y - GOAL_Y) ** 2 + (z - GOAL_Z) ** 2) ** 0.5

    T = 1000
    i = 0
    episode_return = 0.0
    discounted_episode_return = 0.0
    returns = []
    rewards = []
    while i < T:
       
        now = time.time()
        if now - last_en > 1.0:
            pub_en.publish(en)
            last_en = now

        ax = random.uniform(-MAX_ACC, MAX_ACC)
        ay = random.uniform(-MAX_ACC, MAX_ACC)
        # Keep random exploration, but bias z-accel toward hover altitude.
        az_rand = random.uniform(-MAX_ACC, MAX_ACC)
        z_err = HOVER_Z_TARGET - z
        az_hover = KP_Z * z_err - KD_Z * vz
        az = clamp(0.25 * az_rand + az_hover, -MAX_ACC, MAX_ACC)
        yaw_rate = random.uniform(-MAX_YAW_RATE, MAX_YAW_RATE)

        vx = clamp(vx + ax * DT, -MAX_VEL_XY, MAX_VEL_XY)
        vy = clamp(vy + ay * DT, -MAX_VEL_XY, MAX_VEL_XY)
        vz = clamp(vz + az * DT, -MAX_VEL_Z,  MAX_VEL_Z)

        msg = Twist()
        msg.linear.x = vx
        msg.linear.y = vy
        msg.linear.z = vz + HOVER_VZ
        msg.angular.z = yaw_rate

        # Very rough state estimate from commanded velocities.
        x += msg.linear.x * DT
        y += msg.linear.y * DT
        z += msg.linear.z * DT
        d_curr = ((x - GOAL_X) ** 2 + (y - GOAL_Y) ** 2 + (z - GOAL_Z) ** 2) ** 0.5

        reward, episode_return, discounted_episode_return, reward_terms = reward_and_return(
            d_prev=d_prev,
            d_curr=d_curr,
            a_t=(ax, ay, az),
            a_prev=prev_action,
            step_idx=i,
            running_return=episode_return,
            running_discounted_return=discounted_episode_return,
            collision=False,  
            goal_reached=(d_curr <= EPS_GOAL),
        )
        rewards.append((reward, reward_terms))
        returns.append((episode_return, discounted_episode_return))
        prev_action = (ax, ay, az)
        d_prev = d_curr

        pub_cmd.publish(msg)
        time.sleep(DT)
        i+= 1

    print(f"steps={T}")
    print(f"total_return={episode_return:.3f}")
    print(f"discounted_return_gamma_{GAMMA}={discounted_episode_return:.3f}")
    print(f"last_reward={rewards[-1][0]:.3f}")
    times = np.linspace(0, T * DT, num=T)
    episodic_returns = [r[0] for r in returns]
    discounted_returns = [r[1] for r in returns]

    plt.figure(figsize=(10, 6))
    plt.plot(times, episodic_returns, label="return")
    plt.plot(times, discounted_returns, label=f"discounted return (gamma={GAMMA})")
    plt.title("Returns vs Timesteps")
    plt.xlabel("time")
    plt.ylabel("return")
    plt.legend()
    plt.tight_layout()
    plt.show()
    
if __name__ == "__main__":
    main()
