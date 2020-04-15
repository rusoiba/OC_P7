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

The whole modeling process is described is this [technical note](https://github.com/rusoiba/OC_P7/blob/master/Projet7.pdf).

The interpretability of the prediction is given by a **Force Plot** showing the local importance of each variable of the model. Force plot is a method of [SHAP module](https://github.com/slundberg/shap) and it basically defines what makes the prediction what is it with respect to this particular client. As you probably know, decisionnal paths are quite singular when using Tree based model.

## Statistics
The Statistics module displays global and anonymized client's data. Thanks to the data volume we are able to draw conclusions about the clients attribute such that empirical mean of repayment by gender and so on.

Not only is it a regulatory obligation but it is also an excellent way to interpret and explain predictions (in complement to the SHAP explanation).

