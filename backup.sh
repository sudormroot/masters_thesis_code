#!/bin/sh

timestamp=`date "+%Y%m%d_%H%M%S"`

echo $timestamp

mkdir bak 2>/dev/zero
sync
cp thesis_main.ipynb bak/thesis_main_$timestamp.ipynb 2>/dev/zero
ls -l bak/*.ipynb
