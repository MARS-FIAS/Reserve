#!/bin/bash

a=10

while [[ $a -gt 0 ]]; do
  echo "a = $a is greater than zero"
  a=$((a - 1))
done
