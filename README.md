# Selected projects - Lam Ngo

## Table of Contents

- [Introduction](#Introduction)
- [DockerDatabase](#DockerDatabase)
- [MatrixFactorization](#MatrixFactorization)
- [TimeSeriesForecasting](#TimeSeriesForecasting)
- [References](#references)


## Introduction
Dear potential employer,

this repository contains a few selected projects that demonstrate my programming ability and interest in data science.

## DockerDatabase
In this project, I used docker, python (including flask), SQL and MongoDB. The frontend is built with Javascript and HTML/CSS (focus was on the backend components though, website looks unpolished). Random data is generated and inserted into a SQL database. One can modify this data by adding new entries through the website. The data can then be transferred to a MongoDB database. The programme is containerized with Docker.  The goal of this project was to combine all these different tools.


## MatrixFactorization
This project was part of my thesis on "Interpretable representation of latent variables in matrix factorizations". It uses matrix factorization <sup>[1](#myfootnote1)</sup> to predict students' answers to mathematical questions <sup>[2](#myfootnote2)</sup>. Pytorch was used for the implementation. 


## TimeSeriesForecasting
I wanted to find out if neural network state-of-the-art algorithms outperform baseline algorithms such as XGBoost in a multivariate time series forecasting task. Car traffic is predicted using variables such as weather and holidays. The results can be seen here:
| Method  | RMSE | MAE | 
| ------------- | ------------- | -------------- |
| Linear Regression | 0.3735 | 0.2478 |
| XGBoost | 0.2952 | 0.1684 |
| Vanilla LSTM |  0.2838 | 0.2415 |
|  DA-RNN<sup>[3](#myfootnote3)</sup> | 0.1962 | 0.1372 |
| Informer<sup>[4](#myfootnote4)</sup> | 0.1926 | 0.1302 |


## References
<a name="myfootnote1">1</a>. https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=5197422

<a name="myfootnote2">2</a>. https://competitions.codalab.org/competitions/25449

<a name="myfootnote3">3</a>. https://arxiv.org/abs/1704.02971

<a name="myfootnote4">4</a>. https://arxiv.org/abs/2012.07436









