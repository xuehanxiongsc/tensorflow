#!/bin/bash

for file in *.ipynb; do
  echo "Convert python notebool file $file to python file"
  ipython nbconvert --to python $file
done
