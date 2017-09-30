# spark-machine-learning

In this post, I demonstrate how you can use Apache Spark's machine learning libraries to perform binary classification using logistic regression. The dataset I am using for this demo is taken from Andrew Ng's machine learning course on [Coursera](https://www.coursera.org/learn/machine-learning).

![image](https://user-images.githubusercontent.com/1760859/31045688-8865095e-a602-11e7-8fba-7a171b9eda71.png)

Let's assume a classroom scenario where students go through three exams to pass the class. The dataset have historical data of students with their scores in first two exams and a label column which shows whether each student was able to pass the 3rd and final exam or not. My goal is to train a binary classifier using historical data and predict, given scores of first two exams of a particular student, whether the student will pass the final exam(1) or not(0).

To get a better sense of data, let's plot the scores and labels in a scatter plot using R.

![pass-fail](https://user-images.githubusercontent.com/1760859/31045701-c4f56968-a602-11e7-913a-952f7cb667a9.png)

In above plot, a red dot shows a passed student and black dot represents a failed one. The plot also shows a clear pattern or separation between scores of passed and failed students. My objective is to train a model/classifier that can capture this pattern and to use this model to make predictions later on. In this demo, I am going to use logistic regression algorithm to create the model.

I start by defining the schema that matches the dataset.

```
static StructType SCHEMA = new StructType (new StructField[] {
  new StructField(COLUMN_SCORE_1, DataTypes.DoubleType, false, Metadata.empty()),
  new StructField(COLUMN_SCORE_2, DataTypes.DoubleType, false, Metadata.empty()),
  4new StructField(COLUMN_PREDICTION, DataTypes.IntegerType, false,
  Metadata.empty())
});
```
Then I load the data from CSV file and convert it into vectorized format while specifying feature and label columns.
```
Dataset dataSet = spark .read().schema(SCHEMA).
	format("com.databricks.spark.csv").
	option ("header", "true").
	load(inputFile);

dataSet = spark.createDataFrame(dataSet.javaRDD(), SCHEMA);
VectorAssembler vectorAssembler = new VectorAssembler().
	setInputCols(new String[]{COLUMN_SCORE_1, COLUMN_SCORE_2}).
	setOutputCol(COLUMN_INPUT_FEATURES);
dataSet = vectorAssembler.transform(dataSet);
```
Next step is to split the data into training and cross validation sets and setup the logistic regression classifier.
```
Dataset[] splittedDataSet = dataSet.randomSplit(new double[]{0.7, 0.3},
SPLIT_SEED);
Dataset trainingDataSet = splittedDataSet[0];
Dataset crossValidationDataSet = splittedDataSet[1];

LogisticRegression logisticRegression = new LogisticRegression().
	setMaxIter(LOGISTIC_REGRESSION_ITERATIONS).
	setRegParam(LOGISTIC_REGRESSION_ITERATIONS).
	setElasticNetParam(LOGISTIC_REGRESSION_STEP_SIZE);

logisticRegression.setLabelCol(COLUMN_PREDICTION);
logisticRegression.setFeaturesCol(COLUMN_INPUT_FEATURES);
```
Next, I train the model and get the training results.
```
LogisticRegressionModel logisticRegressionModel = 
logisticRegression.fit(trainingDataSet);
LogisticRegressionTrainingSummary logisticRegressionTrainingSummary =
logisticRegressionModel.summary();
```
You can also print the error on each iteration of logistic regression.
```
double[] objectiveHistory = logisticRegressionTrainingSummary.objectiveHistory();
for (double errorPerIteration : objectiveHistory)
	System.out.println(errorPerIteration);
```
Next, I find the best threshold value based on FScore and use this threshold to create our final model.
```
BinaryLogisticRegressionSummary binaryLogisticRegressionSummary = 
(BinaryLogisticRegressionSummary) logisticRegressionTrainingSummary;
// Get the threshold corresponding to the maximum F-Measure and return 
// LogisticRegression with this selected threshold.
Dataset fScore = binaryLogisticRegressionSummary.fMeasureByThreshold();
double maximumFScore = fScore.select(functions.max("F-Measure")).head().getDouble(0);
double bestThreshold = fScore.where(fScore.col("F-Measure").equalTo(maximumFScore)).select("threshold").head().getDouble(0);
logisticRegressionModel.setThreshold(bestThreshold);
System.out.println("maximum FScore: " + maximumFScore);
```
I use the initially separated cross validation set to find accuracy of our trained model and print the results.
```
Dataset crossValidationDataSetPredictions = logisticRegressionModel.transform(crossValidationDataSet);
JavaPairRDD<Double, Double> crossValidationPredictionRDD = convertToJavaRDDPair(crossValidationDataSetPredictions);
Utils.printFScoreBinaryClassfication(crossValidationPredictionRDD);
printPredictionResult(crossValidationDataSetPredictions);
```
Results look like this in my case, which are not bad for a start.
```
True positives: 11
False positives: 3
False negatives: 0
Precision: 0.7857142857142857
Recall: 1.0
FScore: 0.88
Correct predictions: 22/25
```
Let's plot the results to get a visual intuition on how algorithm did.
![selection_049-e1488866192141](https://user-images.githubusercontent.com/1760859/31045721-376bcfaa-a603-11e7-88ec-b11f62e80066.png)


As you can see in above plot, the prediction pattern match very closely with the initial dataset we plotted which again illustrates correctness of implementation.

Bonus: You can find Random Forest based solutions of the same problem [here](https://github.com/harishasan/spark-machine-learning/blob/master/src/main/java/binary/classification/BinaryRandomForest.java).
