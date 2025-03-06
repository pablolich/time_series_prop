# GLV Model Fitting Pipeline

## 📌 Project Structure
```
code/
│── main.py
│── utils.py
│── fit.py
│── data.py
│── models/ #folder with models
│   ├── __init__.py
│   ├── glv.py
│── cost/ #folder for cost functions
│   ├── __init__.py
│   ├── ssq_prop.py
│── data/ #folder for dataset files
│   ├── glv_chaos_4spp.csv

```

## 🚀 Overview
This project implements a pipeline for fitting a relative species abundance time-series data to different models of population dynamics, using different cost functions to evaluate the goodness of fit. 

### **🔹 Components**
1. **`data/`** - collection of datasets
2. **`models/`** - Library of models 
3. **`cost/`** - Library of cost functions
5. **`data.py`** - Builds a data object given a dataset to be fed to the fit object
6. **`fit.py`** - Combines data, model, and cost functions into a single object
7. **`utils.py`** - Contains auxiliary functions such those for integrating, and optimizing. 
8. **`main.py`** - Runs the full pipeline

---

## 📌 Future Improvements
- Implement additional **cost functions** (e.g., log-likelihood based).
- Implement additional **models**
- Support for multiple datasets in a single run.

---

## 📄 License
This project is open-source and available under the **MIT License**.


