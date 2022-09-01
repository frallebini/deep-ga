# Deep GA

## Installing dependencies
1. Install [poetry](https://python-poetry.org/) by running
    ```shell
   curl -sSL https://install.python-poetry.org | python3 -
   ```
2. `cd` into the project directory and run
    ```shell
   poetry install
   ```
   
## Evaluation
`cd` into the project directory and run
```shell
poetry shell
```
followed by
```shell
python evaluate.py
```
The model will be evaluated by playing 30 episodes of *Frostbite*. Scores and videos will be stored in a newly 
created `results` directory.

To play *Space Invaders* instead, replace
```
"environment": "ALE/Frostbite-v5"
```
with
```
"environment": "ALE/SpaceInvaders-v5"
```
in [`config.json`](config.json).

## Training
`cd` into the project directory and run
```shell
poetry shell
```
followed by
```shell
python train.py
```
The model will be trained on *Frostbite* until the total number of frames it has seen is `>=1e9` (`"max_train_frames"` 
in [`config.json`](config.json)). You can stop the training at any moment, as checkpoints are saved in the 
[`models`](models) directory at each generation.

To train on *Space Invaders* instead, replace
```
"environment": "ALE/Frostbite-v5"
```
with
```
"environment": "ALE/SpaceInvaders-v5"
```
in [`config.json`](config.json).