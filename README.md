# MBO-with-constraint-handling
This Python code is for multi/single-objective Bayesian optimization (MBO/SBO) with/without constraint handling.  
This code is released under the MIT License, see LICENSE.txt.  

## Usage
You can start with "tutorial_multi-objective_Bayesian_optimization.ipynb", "tutorial_single-objective_Bayesian_optimization.ipynb", or "main.py".  
Detailed usage is written in "tutorial_*.ipynb".

## Citation
MBO part is based on MBO-EPBII-SRVA and MBO-EPBII published in the following articles:  
* [N. Namura, "Surrogate-Assisted Reference Vector Adaptation to Various Pareto Front Shapes for Many-Objective Bayesian Optimization," IEEE Congress on Evolutionary Computation, Krakow, Poland, pp.901-908, 2021.](https://doi.org/10.1109/CEC45853.2021.9504917)
* [N. Namura, K. Shimoyama, and S. Obayashi, "Expected Improvement of Penalty-based Boundary Intersection for Expensive Multiobjective Optimization," IEEE Transactions on Evolutionary Computation, vol. 21, no. 6, pp. 898-913, 2017.](https://doi.org/10.1109/TEVC.2017.2693320)

Please cite the article(s) if you use the MBO code.  

## Requirement
* numpy
* pandas
* scipy
* matplotlib
* scikit-learn
* pymoo
* pyDOE2
* optproblems
* diversipy

## Contact
Nobuo Namura (nobuo.namura.gp@gmail.com)