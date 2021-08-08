#!/bin/sh

version="v6"

jupyter nbconvert \
        --to pdf thesis_main_$version.ipynb \
        --output pdfs/thesis_main_$version.ipynb

