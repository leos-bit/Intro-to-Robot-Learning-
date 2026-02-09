#!/usr/bin/env python3
import time, random

from gz.transport13 import Node
from gz.msgs10.twist_pb2 import Twist
from gz.msgs10.boolean_pb2 import Boolean

DT = 0.02

MAX_ACC = 1.5
MAX_YAW_RATE = 1.0

MAX_VEL_XY = 1.5
MAX_VEL_Z  = 1.5
HOVER_VZ = 1.0

CMD_TOPIC = "/quad/gazebo/command/twist"
ENABLE_TOPIC = "/quad/enable"

def clamp(x, lo, hi): return max(lo, min(hi, x))

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

    while True:
       
        now = time.time()
        if now - last_en > 1.0:
            pub_en.publish(en)
            last_en = now

        ax = random.uniform(-MAX_ACC, MAX_ACC)
        ay = random.uniform(-MAX_ACC, MAX_ACC)
        az = random.uniform(-MAX_ACC, MAX_ACC)
        yaw_rate = random.uniform(-MAX_YAW_RATE, MAX_YAW_RATE)

        vx = clamp(vx + ax * DT, -MAX_VEL_XY, MAX_VEL_XY)
        vy = clamp(vy + ay * DT, -MAX_VEL_XY, MAX_VEL_XY)
        vz = clamp(vz + az * DT, -MAX_VEL_Z,  MAX_VEL_Z)

        msg = Twist()
        msg.linear.x = vx
        msg.linear.y = vy
        msg.linear.z = vz + HOVER_VZ
        msg.angular.z = yaw_rate


        pub_cmd.publish(msg)
        time.sleep(DT)

if __name__ == "__main__":
    main()
