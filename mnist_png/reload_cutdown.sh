#!/bin/bash


# delete the 0-9 directories
for i in {0..9}
do
   rm -rf $i/
done


# create the 0-9 directories
for i in {0..9}
do
   mkdir $i
done


# copy the files
for i in {0..9}
do
    find ../training/$i/ -maxdepth 1 -type f | head -500 | xargs cp -t $i/
done


echo "done"
