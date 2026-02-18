import mujoco
from mujoco.glfw import glfw


MODEL_PATH = "Drone_MJCFs/skydio_x2/scene.xml"


def main():
    model = mujoco.MjModel.from_xml_path(MODEL_PATH)
    data = mujoco.MjData(model)

    mujoco.mj_resetDataKeyframe(model, data, 0)
    data.qpos[2] = 0.35
    data.qvel[:] = 0.0

    total_mass = float(model.body_mass.sum())
    g = float(-model.opt.gravity[2])
    hover_thrust = total_mass * g / model.nu
    thrust_scale = 0.9

    if not glfw.init():
        raise RuntimeError("Failed to initialize GLFW")

    # Request a broadly compatible desktop OpenGL context.
    glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_ANY_PROFILE)
    glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 2)
    glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 1)

    window = glfw.create_window(1400, 900, "Skydio X2 OpenGL Viz", None, None)
    if not window:
        glfw.terminate()
        raise RuntimeError("Failed to create GLFW window")
    glfw.make_context_current(window)
    glfw.swap_interval(1)

    cam = mujoco.MjvCamera()
    opt = mujoco.MjvOption()
    scene = mujoco.MjvScene(model, maxgeom=10000)
    try:
        context = mujoco.MjrContext(model, mujoco.mjtFontScale.mjFONTSCALE_150)
    except mujoco.FatalError as e:
        glfw.destroy_window(window)
        glfw.terminate()
        raise RuntimeError(
            "OpenGL context missing framebuffer support. "
            "Try: MESA_GL_VERSION_OVERRIDE=3.3 python viz/viz.py"
        ) from e

    cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
    cam.trackbodyid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "x2")
    cam.distance = 2.0
    cam.azimuth = 120.0
    cam.elevation = -20.0
    left_down = False
    right_down = False
    last_x = 0.0
    last_y = 0.0

    def on_key(_window, key, _scancode, action, _mods):
        nonlocal thrust_scale
        if action not in (glfw.PRESS, glfw.REPEAT):
            return
        if key == glfw.KEY_UP:
            thrust_scale = min(1.3, thrust_scale + 0.02)
            print(f"thrust_scale={thrust_scale:.2f}")
        elif key == glfw.KEY_DOWN:
            thrust_scale = max(0.0, thrust_scale - 0.02)
            print(f"thrust_scale={thrust_scale:.2f}")
        elif key == glfw.KEY_R:
            mujoco.mj_resetDataKeyframe(model, data, 0)
            data.qpos[2] = 0.35
            data.qvel[:] = 0.0
            print("reset")
        elif key == glfw.KEY_ESCAPE:
            glfw.set_window_should_close(window, True)

    def on_mouse_button(_window, button, action, _mods):
        nonlocal left_down, right_down
        if button == glfw.MOUSE_BUTTON_LEFT:
            left_down = action == glfw.PRESS
        elif button == glfw.MOUSE_BUTTON_RIGHT:
            right_down = action == glfw.PRESS

    def on_cursor_pos(_window, xpos, ypos):
        nonlocal last_x, last_y
        dx = xpos - last_x
        dy = ypos - last_y
        last_x = xpos
        last_y = ypos

        if left_down:
            cam.azimuth -= 0.25 * dx
            cam.elevation = max(-89.0, min(89.0, cam.elevation - 0.25 * dy))
        elif right_down:
            cam.lookat[0] -= 0.002 * dx
            cam.lookat[2] += 0.002 * dy

    def on_scroll(_window, _xoff, yoff):
        cam.distance = max(0.2, cam.distance * (0.9 ** yoff))

    glfw.set_key_callback(window, on_key)
    glfw.set_mouse_button_callback(window, on_mouse_button)
    glfw.set_cursor_pos_callback(window, on_cursor_pos)
    glfw.set_scroll_callback(window, on_scroll)
    last_x, last_y = glfw.get_cursor_pos(window)

    while not glfw.window_should_close(window):
        data.ctrl[:] = hover_thrust * thrust_scale
        mujoco.mj_step(model, data)

        width, height = glfw.get_framebuffer_size(window)
        viewport = mujoco.MjrRect(0, 0, width, height)
        mujoco.mjv_updateScene(
            model,
            data,
            opt,
            None,
            cam,
            mujoco.mjtCatBit.mjCAT_ALL,
            scene,
        )
        mujoco.mjr_render(viewport, scene, context)

        glfw.swap_buffers(window)
        glfw.poll_events()

    glfw.destroy_window(window)
    glfw.terminate()


if __name__ == "__main__":
    main()
