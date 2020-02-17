# EnrichedBarcode

Annotate persistence barcodes with data attributes on simplices.

This branch is for usage under Python 3. The persistent homology dependencies have been updated and are much easier to install.

## Dependencies and requirements

This code depends on a persistent (co)homology software [Dionysus 2](https://github.com/mrzv/dionysus) and its utility for alpha complex filtration [Diode](https://github.com/mrzv/diode) and some common Python packages: `numpy`, `scipy`, `matplotlib`.

## Getting started

1. Install CGAL and cmake.
E.g. in Ubuntu:
```
sudo apt install cmake libcgal-dev
```
2. Install the python dependencies.

3. Install this package by running ``pip install $ThisDir``.

A jupyter notebook is included in the examples folder showing various examples.
