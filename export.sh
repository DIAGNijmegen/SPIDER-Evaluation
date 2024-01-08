#!/usr/bin/env bash

./build.sh

docker save spider | gzip -c > SPIDER.tar.gz
