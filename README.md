# Demi5.0
## Suite for semi supervised tweet classification.

Demi downloads every tweet from a given account.
Cleans the data and apply two transformations to it: tfidf and seq2vec. (data_units)

Takes two or more data-units and trains them in 8 different classifier algorithms:

* [LogReg](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html)
* [RandomForest](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html)
* [StochasticGradiantDescent](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.SGDClassifier.html)
* [SupportVectorMachine](https://scikit-learn.org/stable/modules/svm.html)
* [SupportVectorClassifier](https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html)
* [LongShortTermMemory](https://keras.io/getting-started/sequential-model-guide/)
* [MultiLayerPerceptron](https://github.com/google/eng-edu/tree/master/ml/guides/text_classification)
* [SeparableCnn](https://github.com/google/eng-edu/tree/master/ml/guides/text_classification)

Takes any number of data-units and classify them into any model.

Results of each classification are then tranform to class-proportions and cummulative record.

![DATA TRANFORMATION](https://i.imgur.com/BWVOLEd.png "Logo Title Text 1")

Generates plots from every machine.

![MACHINES PLOT](https://i.imgur.com/ml6uW5X.png "Logo Title Text 1")

Generates plot from prediction results.

![DATA PLOT](https://i.imgur.com/BY7NVL1.png "Logo Title Text 1")

Then groups data-units into 'parties' each one associated with a class from a model.

Runs statistical analisys over the prediction results from every data-unit on every model,

and plots the results for predilection on each party from their units.

Columns are the different models, rows are data-units to be classify.

The example assumes the party is 'a' and the target is 'a'.

From red to yellow means that the unit has more 'votes' for the target than for other classes.

From black to blue, the unit has more 'votes' for other than the target.

White means the 'votes' are not statisticaly significant.

![FINAL PLOT](https://i.imgur.com/HpsC7kK.png "Logo Title Text 1")



