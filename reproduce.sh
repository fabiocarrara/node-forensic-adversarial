#!/bin/bash

# filtering
for OP in median mean; do
for K in 3 5 7; do
        python train.py filter $OP $K
        python attack.py -n 400 runs/${OP}-${K}x${K} filter $OP $K
done
done

# hist-eq
python train.py hist-eq
python attack.py -n 400 runs/hist-eq hist-eq

# jpeg
for Q in 60; do
        python train.py -e 500 jpeg $Q
        python attack.py -n 400 runs/jpeg-${Q} jpeg $Q
done
