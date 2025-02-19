# ExplainReduce Experiments

This directory contains the code necessary to recreate the results and figures in the ExplainReduce publication.
The experiments are for the most part meant to be run in a compute array managed with the `slurm` framework.
As the `slurm` batch scripts are dependent on your particular setup (e.g., in how modules are managed), they will likely need additional modifications.

## Simple example

To reproduce the illustrative example (Figure 1), simply run
```
python3 experiments/example.py
```

## Synthetic end-to-end example

To reproduce the synthetic case example, call
```
python3 experiments/synthetic_case_example.py
```

## Jets case example


To reproduce the synthetic case example, call
```
python3 experiments/jets_example.py
```

## ExplainReduce performance as a function of subset size

This experiment produces figures depicting fidelity, coverage and instability as a function of subset size $k$.
To run a single instance of these experiments, call
```
python3 experiments/k_sensitivity.py $job_id
```
where `$job_id` is a positive integer in the root of this repository.
Alternatively, launch a `slurm` array job by running
```
sbatch experiments/scripts/exp_k_sens.sh
```

After producing a batch of results, create the fidelity figures by calling
```
python3 experiments/k_sensitivity.py
```
i.e., without any arguments.
The generated plots will be stored in the `ms/` directory under the root of the repository.

For the coverage and instability figures, call
```
python3 experiments/coverage_instability.py
```

## ExplainReduce performance as a function of subsample size

This experiment can be run similarly as above, but just replace the commands with
```
python3 experiments/subsample_sensitivity.py ($job_id)?
```
and 
```
sbatch experiments/scripts/exp_n_sens.sh
```

## Hyperparameter sensitivity analysis

Again, repeat the instructions above but with

```
python3 experiments/coverage_epsilon_sensitivity.py ($job_id)?
```
and 
```
sbatch experiments/scripts/exp_sensitivity.sh
```

## Greedy algorithm comparison

Finally, to produce the approximation analysis table, run

```
python3 experiments/greedy_comparison.py
```
and 
```
sbatch experiments/scripts/exp_greedy_comparison.sh
```