#!/bin/bash

mkdir funnels/data/downloads
wget -O funnels/data/downloads/data.tar.gz "https://zenodo.org/record/1161203/files/data.tar.gz?download=1"
tar -xvf funnels/data/downloads/data.tar.gz
mv data/* funnels/data/downloads/
rm -r data

pip install -e .