#!/bin/bash

python train.py filter median 7
python attack.py runs/median-7x7 filter median 7

python train.py filter median 5
python attack.py runs/median-5x5 filter median 5

