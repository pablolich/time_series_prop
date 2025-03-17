# GLV Model Fitting Pipeline

## 📌 Project Structure
```
code/
│── main.py
│── fit.py
│── integration_funcs.py
│── optimization_funcs.py
│── models/ #folder with models
│   ├── __init__.py
│   ├── glv.py
│── cost/ #folder for cost functions
│   ├── __init__.py
│   ├── ssq_prop.py
│   ├── log_prop.py
│── data/ #folder for dataset files and data.py class to read them
│   ├── __init__.py
│   ├── data.py
│   ├── Davis
│   ├── Hiltunen
│   ├── Jo
│   ├── glv_3spp
│   ├── glv_4spp
│── opt_protocols/ #folder for dataset files and data.py class to read them
│   ├── __init__.py
```

## 🚀 Overview
This project implements a pipeline for fitting a relative species abundance time-series data to different models of population dynamics, using different cost functions to evaluate the goodness of fit. 

### **🔹 Components**
1. **`data/`** - collection of datasets
2. **`models/`** - Library of models 
3. **`cost/`** - Library of cost functions
5. **`data.py`** - Builds a data object given a dataset to be fed to the fit object
6. **`fit.py`** - Combines data, model, and cost functions into a single object
7. **`utils.py`** - Contains auxiliary functions such those for scoring results, integrating, and optimizing. 
8. **`main.py`** - Runs the full pipeline

---

## 📌 Future Improvements
- Implement additional **cost functions** (e.g., log-likelihood based).
- Implement additional **models**
- Support for multiple datasets in a single run.

---

## 📄 License
This project is open-source and available under the **MIT License**.


