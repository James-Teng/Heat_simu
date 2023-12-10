# Heat Simu

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

## 3. data preparation

`cd ./source`

`bash ./preprocess/batch_process_files.sh`

## 4. training command

pwd: ./source

`python ./train/train.py -bf test --ep 200` 

`python ./train/train.py -bf k_5 -k 5 -bs 8 -ep 100`

access to tensorboard

`localhost:6006`

## 5. evaluation command

`cd ./source`

`python ./train/k_step_eval.py -k 10 -c $CHECKPOINT_WEIGHT_PATH`
