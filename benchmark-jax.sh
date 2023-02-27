#!/bin/bash

echo "Running with all cores..."
jupyter nbconvert --execute benchmark-jax.ipynb --to html
rm benchmark-jax.html
mv timing.pickle timing_all.pickle

for i in $(seq 0 12); do
  arg="0-$i"
  echo -e "\nRunning $arg..."
  taskset -c $arg jupyter nbconvert --execute benchmark-jax.ipynb --to html
  rm benchmark-jax.html
  mv timing.pickle timing_c$arg.pickle
done

jupyter nbconvert --execute visualize-benchmark.ipynb --to html
