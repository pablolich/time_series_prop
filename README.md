# Infering absolute abundances from compositional data

## 📌 Project Structure
```
code/
│── __init__.py
│── main.py
│── fit.py
│── aux_integration.py
│── aux_optimization.py
│── data.py
│── models.py
│── cost_functions.py
│── data/ #folder for dataset files and data.py class to read them
│   ├── Davis
│   ├── Hiltunen
│   ├── Jo
│   ├── glv_3spp
│   ├── glv_4spp
```

## 🚀 Overview
This project implements a pipeline for fitting a relative species abundance time-series data to different models of population dynamics, using different cost functions to evaluate the goodness of fit. 

### **🔹 Components**
1. **`data/`** - collection of datasets
2. **`models.py`** - Library of models writen as classes
3. **`cost_functions.py`** - Library of cost functions as classes
5. **`data.py`** - Class to create a data object from one of the datasets
6. **`fit.py`** - Class combining data, model, and cost functions into a single object
7. **`aux_integration.py`** - Auxiliary functions assisting with integration routines (processing integration outputs & status)  
8. **`aux_optimization.py`** - Auxiliary functions assisting with optimization routines (different optimization techniques/strategies)
8. **`main.py`** - Wrapper to run the full pipeline

---

