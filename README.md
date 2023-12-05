# Simu Heat

Explosive heat distribution simulation

This repository is the source code of this project, but does not include the datasets.

## 1. directory structure

Heat_simu/

* data/
* docs/
* eval_record/
* training_record/
* source/ (clone this repo here)

## 2. requirements

python 3.9

pytorch 1.13.1

torchvision 0.14.1

numpy 1.23.0

tqdm 4.65.0

matplotlib 3.5.2

tensorboard 2.10.0

torchinfo

## 3. training command

pwd: ./source

`python ./train/train.py -bf test --ep 200` 

access to tensorboard

`localhost:6006`

## 4. evaluation command

pwd: ./source



`python ./train/eval.py`

set the checkpoint path in the `eval.py` source code



`python ./train/k_step_eval.py -k 10 -c $CHECKPOINT_WEIGHT_PATH`
