from setuptools import setup, find_packages

setup(name='RevolveVision',
      version='0.0.0',
      author='Kevin Godin-Dubois',
      author_email='k.j.m.godin-dubois@vu.nl',
      packages=find_packages(where='src'),
      requires=[
            "abrain",

            "numpy",
            "revolve2_core", "revolve2_actor_controller", "revolve2_runners_mujoco",
            "mujoco", "glfw", "pyrr",

            "colorama",
            "humanize",
            "matplotlib",

            "qdpy"
      ],
      scripts=[

      ])



