#!/usr/bin/env bash

cp ../conda.yaml conda.yaml
docker build -t rpl-workshop .
rm conda.yaml
