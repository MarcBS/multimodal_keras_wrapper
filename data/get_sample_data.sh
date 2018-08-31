#!/usr/bin/env bash

cd data 2>/dev/null;
echo "Downloading data...";
wget https://github.com/lvapeab/lvapeab.github.io/raw/master/multimodal_keras_wrapper_data/multimodal_keras_wrapper_data.tar.gz;
echo "Uncompressing data...";
tar xzf multimodal_keras_wrapper_data.tar.gz;
rm multimodal_keras_wrapper_data.tar.gz;
echo "Done.";