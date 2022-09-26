#!/bin/bash
k=0
for i in 0 1 11 111 1111
do
    k=$((k+1))
    echo "Repetition $k"
    echo "Seed $i"
    sh run.sh $k $i
done