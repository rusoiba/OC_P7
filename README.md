# OC_P7

OC_P7 is an open-source creditworthiness assessment project. The project is based around a Dashboard and meant to be used by banking sector employees.

This dashboard contains three main parts : Explorer, Prediction and Global Statistics.

## Explorer
The explorer is a Dashboard module made to explore a specific loan ID (unequivocally related to clients).
It enables the user to explore multiple aspects of loaner's data coming for various sources : 
- Personnal data (gender, age, ...)
- Family data (household composition, environment, ...)
- Education data (level of education)
- Profesionnal data (income level, work stability, working years, ...)
- Previous loans history (behaviour and balance)

## Predict
The predict module uses a ML Model to predict the probability of payment default. It is based on the agregation of all data sources available [here](https://www.kaggle.com/c/home-credit-default-risk/data)
