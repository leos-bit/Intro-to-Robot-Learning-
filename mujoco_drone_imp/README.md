# drone-training

Local package for MuJoCo drone environments.

## Install (editable)

```bash
pip install -e .
```

## Quick check

```python
import gymnasium as gym
import drone_training  # triggers env registration

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
```
