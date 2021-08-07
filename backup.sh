#!/bin/sh

timestamp=`date "+%Y%m%d_%H%M%S"`

version="v4"

echo $timestamp

mkdir bak 2>/dev/zero
sync
cp thesis_main_$version.ipynb bak/thesis_main_$version""_""$timestamp.ipynb 2>/dev/zero
ls -l bak/*.ipynb
