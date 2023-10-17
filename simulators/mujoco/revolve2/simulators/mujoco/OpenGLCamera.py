import mujoco
from typing import Tuple
import logging
import numpy as np
import os

import cv2
import numpy as np

class Config:
    opengl_lib = "glfw"   # Changed from 'egl' to 'glfw'

class OpenGLVision:
    max_width, max_height = 200, 200
    global_context = None

    def __init__(self, model: mujoco.MjModel, shape: Tuple[int, int], headless: bool):
        if OpenGLVision.global_context is None and headless:
            config = Config.opengl_lib.upper()
            if config == "GLFW":
                from mujoco.glfw import GLContext
                os.environ['MUJOCO_GL'] = 'glfw'
            # Uncommenting below just in case you might need other backends in future
            # elif config == "EGL":
            #     from mujoco.egl import GLContext
            #     os.environ['MUJOCO_GL'] = 'egl'
            # elif config == "OSMESA":
            #     from mujoco.osmesa import GLContext
            #     os.environ['MUJOCO_GL'] = 'osmesa'
            else:
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
        self.cam.fixedcamid = 0

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

