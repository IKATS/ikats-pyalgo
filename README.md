# ![IKATS Logo](https://ikats.github.io/img/Logo-ikats-icon.png) IKATS pyalgo

![Docker Automated build](https://img.shields.io/docker/automated/ikats/pyalgo.svg)
![Docker Build Status](https://img.shields.io/docker/build/ikats/pyalgo.svg)
![MicroBadger Size](https://img.shields.io/microbadger/image-size/ikats/pyalgo.svg)

**An overview of IKATS global architecture is available [here](https://github.com/IKATS/IKATS)**

This repository contains all python algorithms developed by and for IKATS.  
This part of IKATS is the heart of data processing that can implement Big Data treatments.  
In this system, the algorithms from the user point of view are "operators".  

These operators, through inputs provided through workflow design and input parameters, execute scripts of "calculation algorithms" 
that manipulate the data ingested into the IKATS database using the IkatsAPI (see [pybase project](https://github.com/IKATS/ikats-pybase)). 
The outputs of the algorithms can be visualized by means of "VizTools" and used in the rest of the workflow.  

This modularization is planned in the form of plugins so as to allow external contributions to IKATS. 
Thus any IKATS operator is structured in the same way as a contribution and uses the same API to obtain and deliver data to 
the IKATS database or to contribute to the current workflow.
In addition to the VizTools that can be implemented at each operator level, IKATS provides VizTools for the basic functional types.

Each directory, associated to one or several algorithms, provide its own unit tests and a json describing algorithm(s) interface (catalog_def.json):
* family  
* inputs  
* outputs  
* parameters
and associated types, descriptions.

List of python algorithms provided (as operators) at the moment :
(see <a href="https://github.com/IKATS/ikats-datamodel">java operators</a> ) for other operators provided in IKATS.


## Data Exploration


- [Random Projections](https://ikats.github.io/doc/operators/randomProjections.html)
- [TS pattern Matching](https://ikats.github.io/doc/operators/tsPatternMatching.html)


## Pre-Processing on Ts


### Cleaning

- [Rollmean](https://ikats.github.io/doc/operators/rollmean.html)


### Reduction

 - [Cut DS](https://ikats.github.io/doc/operators/cutTs.html)
 - [Cut DS by Metric](https://ikats.github.io/doc/operators/cutByMetric.html)
 - [Cut-Y](https://ikats.github.io/doc/operators/cutY.html)


### Transforming

- [Resample](https://ikats.github.io/doc/operators/resample.html)
- [SAX](https://ikats.github.io/doc/operators/sax.html)
- [Slope](https://ikats.github.io/doc/operators/slope.html)
- [Unwrap](https://ikats.github.io/doc/operators/unwrap.html)


## Stats


### Statiststics on Ts

- [Quality indicators](https://ikats.github.io/doc/operators/qualityIndicators.html)


### Ts Correaltion Computation

- [Correlate ts loop](https://ikats.github.io/doc/operators/correlateTsLoop.html)


## Data Modeling


### Supervised Learning

- [Decision Tree-Fit](https://ikats.github.io/doc/operators/decisionTreeFit.html)
- [Decision Tree-Predict](https://ikats.github.io/doc/operators/decisionTreePredict.html)
- [Decision Tree-Fit CV](https://ikats.github.io/doc/operators/decisionTreeFitCV.html)


### Unsupervised Learning

- [K-Means](https://ikats.github.io/doc/operators/kmeans.html)
- [K-Means on patterns](https://ikats.github.io/doc/operators/kmeansOnPatterns.html)
