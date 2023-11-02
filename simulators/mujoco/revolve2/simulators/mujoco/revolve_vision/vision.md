# Vision in mujoco runner

> #### Disclaimer:
> Adding vision is both *easy* and *hard*:
> - easy because all "complex" code is already available as APIs (mostly
mujoco & glfw)
> - hard because documentation is patchy, choices will be made and evolution
is a pain (more on that latter)
> The following assumes that you intend to use __small__ cameras (e.g. <<
200x200), assumptions may break for larger framebuffers 

> :warning: This "documentation" does not claim to present the _best_ method.
> Only one that works (for the author)

> All paths are relative to the main repository folder

## 1. Declaring a camera

1. In `runners/mujoco/revolve2/runners/mujoco/_local_runner.py` locate the function `_make_mjcf`
2. This is the primary location to change anything related to the physics engine (ground color,
objects, gravity ...)
3. Locate the line `attachment_frame = env_mjcf.attach(robot)` and __before__ that one create a camera:
```
aabb = posed_actor.actor.calc_aabb()
fps_cam_pos = [
    aabb.offset.x + aabb.size.x / 2,
    aabb.offset.y,
    aabb.offset.z
]
robot.worldbody.add("camera", name="vision", mode="fixed", dclass=robot.full_identifier,
                    pos=fps_cam_pos, xyaxes="0 -1 0 0 0 1")
robot.worldbody.add('site',
                    name=robot.full_identifier[:-1] + "_camera",
                    pos=fps_cam_pos, rgba=[0, 0, 1, 1],
                    type="ellipsoid", size=[0.0001, 0.025, 0.025])
```
with `posed_actor` the revolve object modelling a robot and `robot` the mujoco equivalent.
In this example, the camera is placed at the front of the robot's bounding box.

## 2. Encapsulating OpenGL context

Sample code for working with different types of OpenGL contexts. Restrictions apply:
- GLFW cannot work in a headless/multithreaded environment
- EGL requires a GPU which may not be available in HPC contexts
- OSMESA in CPU only but does not scale to even moderate resolutions
```
class OpenGLVision:
    max_width, max_height = 200, 200
    global_context = None

    def __init__(self, model: MjModel, shape: Tuple[int, int], headless: bool):
        # if OpenGLVision.global_context is None:
        if OpenGLVision.global_context is None and headless:
            match Config.opengl_lib.upper():
                case Config.OpenGLLib.GLFW.name:  # Does not work in multithread
                    from mujoco.glfw import GLContext

                case Config.OpenGLLib.EGL.name:
                    from mujoco.egl import GLContext
                    os.environ['MUJOCO_GL'] = 'egl'
                case Config.OpenGLLib.OSMESA.name:
                    from mujoco.osmesa import GLContext
                    os.environ['MUJOCO_GL'] = 'osmesa'
                case _:
                    raise ValueError(f"Unknown OpenGL backend {Config.opengl_lib}")
            OpenGLVision.global_context = GLContext(self.max_width, self.max_height)
            OpenGLVision.global_context.make_current()
            logging.debug(f"Initialized {OpenGLVision.global_context=}")

        w, h = shape
        assert 0 < w <= self.max_width
        assert 0 < h <= self.max_height

        self.width, self.height = w, h
        self.context = mujoco.MjrContext(model, mujoco.mjtFontScale.mjFONTSCALE_150.value)
        self.viewport = mujoco.MjrRect(0, 0, w, h)

        self.cam = mujoco.MjvCamera()
        self.cam.type = mujoco.mjtCamera.mjCAMERA_FIXED
        self.cam.fixedcamid = 1

        self.vopt = mujoco.MjvOption()
        self.scene = mujoco.MjvScene(model, maxgeom=10000)
        self.pert = mujoco.MjvPerturb()

        self.img = np.zeros((self.height, self.width, 3), dtype=np.uint8)

    def process(self, model, data):
        mujoco.mjv_updateScene(
            model, data,
            self.vopt, self.pert, self.cam, mujoco.mjtCatBit.mjCAT_ALL.value,
            self.scene)
        mujoco.mjr_setBuffer(mujoco.mjtFramebuffer.mjFB_OFFSCREEN, self.context)
        mujoco.mjr_render(self.viewport, self.scene, self.context)
        mujoco.mjr_readPixels(self.img, None, self.viewport, self.context)

        return self.img
```

This piece of code has 3 objectives:
 - Generating a global context object initialized by mujoco
 - Generating the individual "vision" object (i.e. a context, viewport, framebuffer...)
 - Produce a np.array containing the rendered data

> :warning: the return img is provided as-is. Mujoco uses a BGR format with a flipped
> up-down coordinate system. Either use `cv2.cvtColor(np.flipud(img), cv2.COLOR_RGBA2BGR)`
> to transform it into a more natural array or take care when indexing

The variable `Config.opengl_lib` is, in my case, a static configuration variable.
Other than that, the variable `model` and `data` in function `process` refer to the
`MjModel` and `MjData` objects encapsulating Mujoco's static and dynamic data, respectively.

## 3. Getting the picture

Depending on your version of Revolve2 this may or may not be trivial. In a backward
fashion you need to:
- invoke `process(model, data)` from an `OpenGLVision` object (see section 2). This
provides the actual data you can use e.g. for controller input
- for that you need expose Mujoco's `model` and `data` to the `ActorController`
(sub-)class used in your robot. This can be done, for instance, through the following
skeleton class:
```
class SensorControlData(DefaultActorControl):
    def __init__(self, mj_model: MjModel, mj_data: MjData):
        DefaultActorControl.__init__(self, mj_model, mj_data)
        self.sensors = mj_data.sensordata
        self.model, self.data = mj_model, mj_data
```
- Unfortunately, to have such a class provided to your controller, you need to actually
create it in `_local_runner.py`. My personal solution already includes a wrapper around
that class but other methods should be possible (including directly changing Revolve's
source code). The corresponding lines are:
```
    control_user = SensorControlData(self.model, self.data)
    self.controller.control(control_step, control_user)
```

## 4. Salvaging the viewer

If using a viewer for OpenGL rendering, you will probably be faced with incompatible
backends. The default dependency for Revolve2 (`mujoco-python-viewer`) uses glfw out of
the box. 

It is theoretically possible to delegate rendering to another back-end:
```
glfw_window_hint = {
    Config.OpenGLLib.OSMESA.name: glfw.OSMESA_CONTEXT_API,
    Config.OpenGLLib.EGL.name: glfw.EGL_CONTEXT_API
}
if (ogl := Config.opengl_lib.upper()) in glfw_window_hint:
    glfw.window_hint(glfw.CONTEXT_CREATION_API, glfw_window_hint[ogl])
```

However, this does not guarantee byte-wise identical results. In fact, there
does not seem to be much difference between using a different context creator
and the regular glfw (besides the fact that osmesa fails spectacularly to work).

## #. Miscellaneous

Random notes:
- osmesa-delegated glfw is useless (full cpu use without rendering)
- egl-delegated egl seems functional but non-deterministic?
- different results when mixing osmesa/egl (non bitwise-identical vision)
- Deterministic vision in headless and offscreen (movie) cases
- Not always true for the interactive viewer. Unsure yet if this can be fixed

Current rule of thumb:
Use egl for headless situations and glfw (delegating to egl) for movie/viewer.
However, this requires access to a GPU (acceptable in this lab
because of the rippers configuration but might not hold outside or for students)

Seems I have done almost the same as https://github.com/deepmind/dm_control/blob/main/dm_control/mujoco/engine.py