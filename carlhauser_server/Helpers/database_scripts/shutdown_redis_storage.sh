#!/bin/bash

# set -e
set -x

# ../../redis/src/redis-cli
redis-cli -s ./cache.sock shutdown
