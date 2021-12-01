#! /usr/bin/bash -xe

. path

cache=$parent_dir/cache
exp=$parent_dir/exp
data=$parent_dir/data
pretrained_models=$parent_dir/pretrained_models

mkdir -p $cache
mkdir -p $exp
mkdir -p $data
mkdir -p $pretrained_models
