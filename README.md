# ID3-Random-Forest-AdaBoosting

## ID3

In decision tree learning, ID3 (Iterative Dichotomiser 3) is an algorithm invented by Ross Quinlan used to generate a decision tree   from a dataset. ID3 is the precursor to the C4.5 algorithm, and is typically used in the machine learning and natural language processing domains.

## Random Forest

Random forests or random decision forests are an ensemble learning method for classification, regression and other tasks, that operate by constructing a multitude of decision trees at training time and outputting the class that is the mode of the classes (classification) or mean prediction (regression)

## AdaBoosting

AdaBoost, short for "Adaptive Boosting", is a machine learning meta-algorithm formulated by Yoav Freund and Robert Schapire who won the Gödel Prize in 2003 for their work. It can be used in conjunction with many other types of learning algorithms to improve their performance.

## Pre-Processing

* The data file contains the values of the attributes, seperated by a ’,’. As such, we use a
filestream object to read each line and parse it using the ’,’ as a delimiter. This results in a
set of 15 strings (as many as the attributes in consideration, which are 15 themselves) that
make up the values for that row. We feed this into the dataset.

* The data set also contains certain missing values, or rather, ’?’ in place of an actual value.
They have to be replaced before the training data can be used. For attributes with discrete
values, we have chosen to replace their missing values with the most frequent value among
those attributes.

* The data set is now mostly ready. However, to make computation (and the code) simpler, we
have chosen to replace the strings with integer values. For attributes with continuous values,
we converted the strings to integers (since the strings had only numbers). For attributes with
discrete values, we replace them with values starting from 0. That is to say, for example if
an attribute could take one among 15 different values, then their integer counterparts would
correspondingly range from 0 - 14.

* The data set has now been converted entirely to integers. The next step is to discretize the
attributes with continuous values. For the ID3 and the Random Forest, we have chosen to
discretize them by simply calculating the total of their integer values and calculating the average
to take as the threshold. All values observed to be lesser than the threshold are given a
value of 0, while those greater are given a value of 1. This is done for all six columns. Thus,
we have converted continuous attributes, to binary attributes.

* We discretize the attributes differently in case of Adaptive Boosting. The age attribute is divided
into 4 possible values instead of 2, 0 for values in 0-20, 1 for 20-40, 2 for 40-60, 3
for 60-80 and so on. For the rest of the continuous attributes, we take the median from the
list of the values, and all those greater than the median take a value of 1 while those less
than the median take a value of 0.
Comparision

* ID3 Decision Trees
  * Accuracy: 78 ~ 79 %
  * Running Time: 1.8808s
  
* Random Forest Algorithm:
  * Accuracy: 81 ~ 82 %
  * Running Time: 34.383s
  
* Adaptive Boosting:
  * Accuracy: 81 ~ 82 % on average
  * Running Time: 49.409s [for 4500 samples]
  
* It is to be noted that discretizing the age attribute as shown above for Adaptive Boosting
increases its accuracy, as opposed to discretizing it into binary values.
