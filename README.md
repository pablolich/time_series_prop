# Infering absolute abundances from compositional data

## Project Structure
```
code/
│── __init__.py
│── main.py
│── fit.py
│── data.py
│── models.py
│── cost_functions.py
│── aux_integration.py
│── aux_optimization.py
│── opt_protocols.py
│── logs.md

data/ #folder for dataset files and data.py class to read them
├── davis (exp)
├── jo (exp)
├── exponential_errors_gamma (synth)
├── glv_3spp (synth)
├── glv_4spp (synth)

results/ #folder for results
```

##  Overview
This project implements a pipeline for fitting a relative species abundance time-series data to different models of population dynamics, using different cost functions to evaluate the goodness of fit. 

1. **`main.py`** - Wrapper to run the full pipeline
2. **`fit.py`** - Class combining data, model, and cost functions into a single object
3. **`data.py`** - Class to create a data object from one of the datasets
4. **`models.py`** - Library of models writen as classes
5. **`cost_functions.py`** - Library of cost functions as classes
6. **`aux_integration.py`** - Auxiliary functions assisting with integration routines (processing integration outputs & status)  
7. **`aux_optimization.py`** - Auxiliary functions assisting with optimization routines (different optimization techniques/strategies)
8. **`opt_protocols.py`** - Functions executing different optimization protocols

---

