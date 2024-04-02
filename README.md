# deep-learning-neural-network-optimization
UNC_data_bootcamp_module_21

## Challenge Description
### Background
> The nonprofit foundation Alphabet Soup wants a tool that can help it select the applicants for funding with the best chance of success in their ventures. With your knowledge of machine learning and neural networks, you’ll use the features in the provided dataset to create a binary classifier that can predict whether applicants will be successful if funded by Alphabet Soup. From Alphabet Soup’s business team, you have received a CSV containing more than 34,000 organizations that have received funding from Alphabet Soup over the years. Within this dataset are a number of columns that capture metadata about each organization, such as:
* __EIN__ and __NAME__—Identification columns
* __APPLICATION_TYPE__—Alphabet Soup application type
* __AFFILIATION__—Affiliated sector of industry
* __CLASSIFICATION__—Government organization classification
* __USE_CASE__—Use case for funding
* __ORGANIZATION__—Organization type
* __STATUS__—Active status
* __INCOME_AMT__—Income classification
* __SPECIAL_CONSIDERATIONS__—Special considerations for application
* __ASK_AMT__—Funding amount requested
* __IS_SUCCESSFUL__—Was the money used effectively

***from the UNC Bootcamp instructions for this challenge***


## Deliverables
This challenge will be executed using Google Colab and completed by performing the following 5 steps per challenge instructions:
* Preprocess the Data
* Compile, Train, and Evaluate the Model
* Optimize the Model
* Write a Report on the Neural Network Model
* Copy Files Into This Repository

### Step-1: Preprocess the Data
In this step I'll use my knowledge of Pandas and scikit-learn’s `StandardScaler()`, to preprocess the dataset, following the instructions outlined below for the initial model called `deep_learning_nn_SDT.ipynb`. These steps prepare the data for __Step 2__, where I'll compile, train, and evaluate the neural network model.

Start by uploading the starter file to Google Colab, then using the information we provided in the Challenge files, follow the instructions to complete the preprocessing steps.
1) Read in the `charity_data.csv` to a Pandas DataFrame, and be sure to identify the following in your dataset:
  * What variable(s) are the target(s) for your model?
  * What variable(s) are the feature(s) for your model?
2) Drop the `EIN` and `NAME` columns.
3) Determine the number of unique values for each column.
4) For columns that have more than 10 unique values, determine the number of data points for each unique value.
5) Use the number of data points for each unique value to pick a cutoff point to bin "rare" categorical variables together in a new value, `Other`, and then check if the binning was successful.
6) Use `pd.get_dummies()` to encode categorical variables.
7) Split the preprocessed data into a features array, `X`, and a target array, `y`. Use these arrays and the `train_test_split` function to split the data into training and testing datasets.
8) Scale the training and testing features datasets by creating a `StandardScaler` instance, fitting it to the training data, then using the `transform` function.


### Step-2: Compile, Train, and Evaluate the Model
For the next step in the challenge, I’ll design a neural network, or deep learning model, to create a binary classification model that can predict if an Alphabet Soup-funded organization will be successful based on the features in the dataset. I will need to think about how many _inputs_ there are before determining the number of _neurons_ and layers in your model. Once that step is completed, then I can compile, train, and evaluate the binary classification model to calculate the model’s loss and accuracy. Complete the following from the challenge instructions:
1) Continue using the file in Google Colab in which you performed the preprocessing steps from __Step 1__.
2) Create a neural network model by assigning the number of input features and nodes for each layer using TensorFlow and Keras.
3) Create the first hidden layer and choose an appropriate activation function.
4) If necessary, add a second hidden layer with an appropriate activation function.
5) Create an output layer with an appropriate activation function.
6) Check the structure of the model.
7) Compile and train the model.
8) ~~Create a callback that saves the model's weights every five epochs.~~  _Not required for this challenge._
9) Evaluate the model using the test data to determine the loss and accuracy.
10) Save and export your results to an HDF5 file. Name the file `AlphabetSoupCharity.h5`.


### Step-3: Optimize the Model
After running the model the for the first time I will need to optimize the model to achieve a target predictive accuracy higher than 75%. This could take multiple attempts, however per the instructions we should not exceed three. Below are listed steps and/or hints to complete the next models, and perhaps achieve the target.

Use any or all of the following methods to optimize your model:
* Adjust the input data to ensure that no variables or outliers are causing confusion in the model, such as:
    * Dropping more or fewer columns.
    * Creating more bins for rare occurrences in columns.
    * Increasing or decreasing the number of values for each bin.
    * Add more neurons to a hidden layer.
    * Add more hidden layers.
    * Use different activation functions for the hidden layers.
    * Add or reduce the number of epochs to the training regimen.

1) Create a new Google Colab file and name it `AlphabetSoupCharity_Optimization.ipynb`.
2) Import your dependencies and read in the `charity_data.csv` to a Pandas DataFrame.
3) Preprocess the dataset as you did in __Step 1__. Be sure to adjust for any modifications that came out of optimizing the model.
4) Design a neural network model, and be sure to adjust for modifications that will optimize the model to achieve higher than 75% accuracy.
5) Save and export your results to an HDF5 file. Name the file `AlphabetSoupCharity_Optimization.h5`.


### Step-4: Write a Report on the Neural Network Model
For this part of the challenge, I’ll write a report in a markdown file on the performance of the deep learning model I created for Alphabet Soup. The report will be called `Report_AlphabetSoup_SDT.md` and it will contain the following format and answer the questions per challenge instructions:
1) __Overview__ of the analysis: Explain the purpose of this analysis.
2) __Results__: Using bulleted lists and images to support your answers, address the following questions:
* Data Preprocessing
    * What variable(s) are the target(s) for your model?
    * What variable(s) are the features for your model?
    * What variable(s) should be removed from the input data because they are neither targets nor features?
* Compiling, Training, and Evaluating the Model
    * How many neurons, layers, and activation functions did you select for your neural network model, and why?
    * Were you able to achieve the target model performance?
    * What steps did you take in your attempts to increase model performance?
3) __Summary__: Summarize the overall results of the deep learning model. Include a recommendation for how a different model could solve this classification problem, and then explain your recommendation.


### Step-5: Copy Files Into This Repository
When finished with the analysis in Google Colab, I'll move the files into my repository for final submission performing the following steps.
1) Download your Colab notebooks to your computer.
2) Move them into your Deep Learning Challenge directory in your local repository.
3) Push the added files to GitHub.


## Resources
### Bootcamp References
[Module 21 Instructions](https://bootcampspot.instructure.com/courses/3285/assignments/52251?module_item_id=937581)

Starter_Code
* Stater_Code.ipynb


***Special Thanks:***
* Jamie Miller
* Mounika Mamindla
* Lisa Shemanciik

### External References
_(where possible will provide link to website)_
* [pandas documentation](https://pandas.pydata.org/docs/reference/general_functions.html)
* [scikit-learn documentation](https://scikit-learn.org/stable/user_guide.html)
* [Google](https://www.google.com)
* [Google Colab](https://colab.research.google.com/notebooks/welcome.ipynb)
* [TensorFlow: Keras model documentation](https://www.tensorflow.org/api_docs/python/tf/keras/Model)
* [YouTube](https://www.youtube.com)
* [TensorFlow playground](https://playground.tensorflow.org/#activation=sigmoid&batchSize=10&dataset=gauss&regDataset=reg-plane&learningRate=0.03&regularizationRate=0&noise=0&networkShape=1&seed=0.10587&showTestData=false&discretize=true&percTrainData=50&x=true&y=true&xTimesY=false&xSquared=false&ySquared=false&cosX=false&sinX=false&cosY=false&sinY=false&collectStats=false&problem=classification&initZero=false&hideText=false&discretize_hide=true&regularization_hide=true&learningRate_hide=true&regularizationRate_hide=true&percTrainData_hide=true&showTestData_hide=true&noise_hide=true&batchSize_hide=true)

