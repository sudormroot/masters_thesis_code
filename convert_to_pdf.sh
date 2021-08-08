#!/bin/sh

version="v5"

jupyter nbconvert --to pdf thesis_main_$version.ipynb --output pdfs/thesis_main_$version.ipynb

