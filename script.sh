#!/bin/bash
python train.py -fedproto True --data mnist --isiid True
python train.py -fedproto True --data cifar --isiid True

python train.py -fedproto False --data mnist --isiid True
python train.py -fedproto False --data cifar --isiid True

python train.py -fedproto True --data mnist --isiid False
python train.py -fedproto True --data cifar --isiid False

python train.py -fedproto False --data mnist --isiid False
python train.py -fedproto False --data cifar --isiid False
