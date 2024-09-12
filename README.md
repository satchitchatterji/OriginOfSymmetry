# Testing Emergent Bilateral Symmetry in Evolvable Robots with Vision
***Michele Vanucci*** $^1$, ***Satchit Chatterji*** $^2$, and *Babak H. Kargar* $^1$

$^1$ Vrije Universiteit Amsterdam, $^2$ University of Amsterdam

Within this repository is the code that is needed to reproduce and build on the results of the paper ***Testing Emergent Bilateral Symmetry in Evolvable Robots with Vision***, accepted to ECTA 2024. The code is heavily build upon *Revolve2*, (repository at [github.com/ci-group/revolve2](https://github.com/ci-group/revolve2), documentation at [ci-group.github.io/revolve2](https://ci-group.github.io/revolve2/)), specifically pre-release version v0.4.0-beta2. Due to this, it is not as readable as we'd have hoped, and is a working repository. The database used for analysis in the paper is available on request by emailing the authors.

## Installation

This repo is based on pre-release v0.4.0-beta2 of *Revolve2*. Thus, it follows the [general guidelines](https://ci-group.github.io/revolve2/installation/) for installing *Revolve2*.

Python 3.10 was used during experimentation. First create a virtual environment with the following commands:

```
python3.10 -m pip install virtualenv
python3.10 -m virtualenv .venv
source .venv/bin/activate
```

Officially, conda is not supported by Revolve, but has shown some initial success during development (though we recommend venv). Finally, the code can be installed in editable mode by running the bash file

    sh student_install.sh

Also install W&B using ```pip install wandb``` and login as normal. As per testing this should install necessary libraries. If not, please start a pull request.

## Execution

The main experimental body can be found in the folder [examples/robot_bodybrain_ea](https://github.com/satchitchatterji/OriginOfSymmetry/tree/main/examples/robot_bodybrain_ea). Several additions have been made elsewhere in the code for stability, data saving and adding in the camera -- the last one is the most critical. The camera class can be found in ```OriginOfSymmetry/simulators/mujoco/revolve2/simulators/mujoco/OpenGLCamera.py```, and has been implemented in ```OriginOfSymmetry/simulators/mujoco/revolve2/simulators/mujoco/_local_runner.py```

The body of ```OriginOfSymmetry/examples/robot_bodybrain_ea/main.py``` contains the list of parameters that are tested. Specifically, you may want to change the following body of code around line ~450 (*csv/json config support yet to be added*):

    parameters_to_test = {
        "brain_multineat_parameters": {
            "OverallMutationRate": 0.15,
        },
        "body_multineat_parameters": {
            "OverallMutationRate": 0.15,
        },
        "evolution_parameters": {
            "steer" : [True, False],
            "population_size": 100,
            "num_generations": 200,
            "num_repetitions": 5,
            "offspring_size": 100,
            "tournament_size": 6,
            "database_file" : "./exp.sqlite",
            "target_list": [[(5,5)], [(5,5)], [(0,math.sqrt(50))], [(0,math.sqrt(50)), (5,5),(-5,5)]]
        }
    }

The code can then be run by calling:

    python3 examples/robot_bodybrain_ea/main.py

Finally, W&B has been integrated into the code for ease of monitoring, so please make sure this is set up before running the code.

Plots ready for analysis may be attained by running the notebooks in ```OriginOfSymmetry/examples/robot_bodybrain_ea/database_processes```:
- ```tests.ipynb```: A collection of example use cases for analysis and plotting functions within the ```database_processes``` directory. It is recommended to start here to see if an analysis method you wish is already here, and contains most graphs presented in the paper.
- ```draw_bodies.ipynb```: Plot bodies of existing robot genotypes (from database) in 2D and 3D.
- ```path_plotting.ipynb```: Plot the path of robots from the database.
- ```plot_fitness_histograms.ipynb```: Plot various fitness/symmetry results.


## Citation

*To be updated post publication*...