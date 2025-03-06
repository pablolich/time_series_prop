# GLV Model Fitting Pipeline

## 📌 Project Structure
```
project_root/
│── main.py
│── fit.py
├── data.py
│── utils.py
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
4. **`data.py`** - Builds a data object given a dataset to be fed to the fit object
5. **`fit.py`** - Combines data, model, and cost functions into a single object
6. **`utils.py`** - Contains auxiliary functions such those for integrating, and optimizing. 
7. **`main.py`** - Runs the full pipeline

---

## 📂 File Descriptions

### **1️⃣ `data/data.py`**
Loads and normalizes the dataset. Stores:
- **Raw Observed Abundances**
- **Normalized Abundances**
- **Raw and Normalized Time Series**

### **2️⃣ `models/glv.py`**
Implements the **Generalized Lotka-Volterra Model**, including:
- ODE System Solver (`dxdt`)
- Parameter Parsing (`parse_parameters`)
- System Integration (`integrate`)

### **3️⃣ `cost/ssq_prop.py`**
Defines the **Sum of Squared Differences (SSQ) Cost Function**.
- Computes model error based on species abundance proportions.

### **4️⃣ `fit.py`**
A wrapper that combines **Data, Model, and Cost Function**.
- Inherits from `Data`, `GLVModel`, and `SSQCostFunction`

### **5️⃣ `main.py`**
Runs the complete pipeline:
- Loads data
- Initializes model and cost function
- Generates predictions
- Computes the cost function

---

## ▶️ Usage
### **1️⃣ Install Dependencies**
Make sure you have the required libraries installed:
```bash
pip install numpy pandas scipy
```

### **2️⃣ Run the Pipeline**
Execute the `main.py` script:
```bash
python main.py
```
This will:
1. Load the dataset
2. Fit the GLV model
3. Compute the cost function
4. Print the final cost value

---

## 📌 Future Improvements
- Implement additional **cost functions** (e.g., log-likelihood based).
- Implement additional **models**
- Support for multiple datasets in a single run.

---

## 📄 License
This project is open-source and available under the **MIT License**.


