# Alphabet Soup 

## Neural Network Model Analysis Report

1) __Overview__ of the analysis: 

As a nonprofit charitable foundation, Alphabet Soup is in need of a tool that can help it select the applicants for funding with the best chance of success in their ventures. Based on the data provided by Alphabet Soup’s business team, we have been able to create a binary classifier that can predict whether applicants will be successful if funded by Alphabet Soup.

2) __Results__: 
* Data Preprocessing
    * What variable(s) are the target(s) for your model?
        * The target variable for the model is `IS_SUCCESSFUL`. 
    * What variable(s) are the features for your model?
        * The features for the model include all the other columns in the dataset, _except_ `IS_SUCCESSFUL`.
    * What variable(s) should be removed from the input data because they are neither targets nor features?
        * The `EIN` column was removed as it was deemed non-beneficial and not used as either a target or feature in the model.
* Compiling, Training, and Evaluating the Model
    * How many neurons, layers, and activation functions did you select for your neural network model, and why?
        * This neural network model, __nn_model3__, consists of three hidden layers with 222 neurons in the _first_ hidden layer and 37 neurons each in the _second_ and _third_ hidden layers.
        * The activation function used for all hidden layers is `relu`, which is a common choice for deep learning models.
        * The output layer consists of a single neuron with a `sigmoid` activation function.
    * Were you able to achieve the target model performance?
        * This version of the was able to achieve a target predictive accuracy higher than 75% with the final evaluation of the model test data yielding an accuracy of approximately __75.11%__.
    * What steps did you take in your attempts to increase model performance?
        * This model represents the third attempt to achieve a target predictive accuracy, the learnings from the previous attempts influenced changes made.
        * In the first two attempts, the `NAME` column was removed along with the `EIN` column for the same reasons. It was put back for this model to provide additional features.
        * In the the first model, __nn_model__, there were only two  hidden layers: 1st hidden layer with 222 neurons and 2nd hidden layer with 37 neurons.
        * With the second model, __nn_model2__, the third hidden layer was added: 1st hidden layer with 222 neurons, 2nd hidden layer with 37 neurons, and 3rd hidden layer with 22.2 neurons.

3) __Summary__: 

In summary, the model provided to you, `AlphabetSoupCharity_Optimization_SDT.ipynb` or __nn_model3__, does achieve an accuracy of approximately 75.11% on the test data. While this accuracy meets a basic level of performance and was able to achieve a target predictive accuracy higher than 75%, there could be room for improvement. As a recommendation we could conduct a more in-depth analysis of the dataset to identify any data patterns or correlations that could lead to better feature selection and engineering. Ultimately, the choice of the best model and approach may require further experimentation.

