#!/bin/bash
# Remote (Brev/Ubuntu) env + dataset setup for Stage 2 scoring. Run after the box is up:
#   brev copy scripts/remote_setup.sh <instance>:~/remote_setup.sh
#   brev exec <instance> 'nohup bash ~/remote_setup.sh > ~/setup.log 2>&1 & echo LAUNCHED'
set -e
cd ~
python3 -m venv vg
. vg/bin/activate
pip install -U pip
pip install torch torchaudio transformers huggingface_hub datasets torchcodec librosa soundfile scikit-learn numpy
mkdir -p data && cd data
if [ ! -f release_in_the_wild/meta.csv ]; then
  wget -q -O itw.zip "https://owncloud.fraunhofer.de/index.php/s/JZgXh0JEAF0elxa/download"
  unzip -q itw.zip
fi
echo SETUP_DONE
