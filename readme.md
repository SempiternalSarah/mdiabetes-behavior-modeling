# mDiabetes Behavior Modeling source code
This repo contains the code used to model the behavior of human participants
in the mDiabetes-AI study

## Getting Started
- clone this repo
- symlink the `arogya_content` and `local_storage` folders to your working directory
    - `ln -s /home/path/to/arogya_content .` (same for `local_storage`)
- load the `utils` submodule
	- `git submodule init && git submodule update`
- create your own branch: `git checkout -b mycoolbranch`
- run some experiments using the `Experiment` class from `experiment.py`

## Pulling Changes to the `utils/` module
- `git submodule foreach --recursive git pull`

## Files
- `experiment.py`: defines the logic to run an experiment. This involves building the dataset and models with
specified hyperparams, iterating over the data in particular ways, training the model, and calculating performance metrics
- `models/base.py`: the base class for our models, contains helper code. not a real model
- `models/basic.py`: the current model and basic lstm similar to what was used in DKT
- `utils/behavior_data.py`: the logic for building and encoding the behavior dataset

