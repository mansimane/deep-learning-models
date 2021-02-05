#!/usr/bin/env bash

image=$1
tag=$2
account=$3

if [[ -z $image || -z $tag || -z $account ]]; then
  echo 'one or more variables are undefined'
  exit 1
fi

aws ecr get-login-password --region us-east-1 | docker login --username AWS --password-stdin ${account}.dkr.ecr.us-east-1.amazonaws.com
docker build --no-cache -t "${image}:${tag}" -f ./Dockerfile.user .

