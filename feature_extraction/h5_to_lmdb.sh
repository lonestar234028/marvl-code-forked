#!/bin/bash

LANG="$1"
H5="/science/image/nlp-datasets/emanuele/data/marvl/features/marvl-${LANG}_boxes36.h5"
LMDB="/science/image/nlp-datasets/emanuele/data/marvl/features/marvl-${LANG}_boxes36.lmdb"


python h5_to_lmdb.py --h5 $H5 --lmdb $LMDB

