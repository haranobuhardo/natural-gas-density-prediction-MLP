# **Exploring the Capabilities of a Simple MLP for Natural Gas Density Estimation**
Made by Hardo for **PACMANN Deep Learning Final Project**

This repository contains information about experiment of predicting Natural Gas' density using simple MLP ANN

---


## Dataset Information
---
- Source paper: `Isothermal PqT measurements on Qatar’s North Field type synthetic natural gas mixtures using a vibrating-tube densimeter` (Atilhan, et. al 2012))
- DOI: 10.1016/j.jct.2012.04.008
- Shape: 1240 rows × 16 columns ([check this file](/dataset/ng_density_all.csv))
- Input features: Pressure, Temperature, Methane, Ethane, Propane, 2-methylpropane (isobutane), Butane, 2-metylbutane (isopentane), Pentane, Octane, Toluene, Methylcyclopentane, Nitrogen, Carbon Dioxide
- Predictor feature: Density
Note: the data was manually extracted from the paper (you can check the data [on this file](/dataset/Natural%20gas%20density%20data.xlsx)) and then transformed

## MLP Architecture
---
![MLP Architecture](/images/MLP_arch.png)
- number of input layer neuron: 14
- number of hidden layer neuron: 250
- number of output layer neuron: 1 (regression task)
- max epoch: 100
- learning rate: 0.1
- stopping criterion: max_epoch
- optimizer: Adam (beta 0.9 & 0.99)

## Results and Conclusion
---
| Model | rmse | aard | r2 |
|---|---|---|---|
| Our MLP Model | 1.759634 | 0.726616 | 0.999589 |
| Linear Regression | 20.897857 | 9.717687 | 0.941997 |
| Peng-Robinson EoS (with DWSIM) | 2.228784 | 0.717293 | 0.999322 |
In conclusion, we have developed a simple MLP model that can predict the density of synthetic natural gas with high accuracy and speed. While the model's performance is still below the results reported in the referenced paper, it is a promising tool for density prediction. However, the model is sensitive to gas composition changes outside of the training set, which may limit its applicability. Future work includes conducting further training with a more diverse dataset and experimenting with different MLP architectures and optimization techniques to improve the model's performance and generalizability. Overall, our simple MLP model provides a fast and accurate alternative to traditional EoS calculation methods for density prediction.


## How to run this project
---
### via Python Notebook
1. Install Python Poetry (dependency management)
`pip install poetry`
2. Change working dir to current dir
3. Install all current project's Poetry dependencies with this command
`poetry install`
4. Run and do training from the `Training.ipynb` file 

### via Python Notebook
_Needed to run Streamlit_
1. Install Python Poetry (dependency management)
`pip install poetry`
2. Change working dir to current dir
3. Run all the Python script in the /src/ folder (from the main parent folder), starting from `1a_data_prep.py` to `4_hyperparameter_tuning.py`

## How to use the trained model (Streamlit)
---
1. Run the streamlit with this code:
`poetry run streamlit run src/streamlit_predict_density.py`
2. Fill all the input data, and press the predict button
3. Predicted density will shown in the bottom

## Full report
For full analysis report, you can check out (this Medium's article)[https://medium.com]