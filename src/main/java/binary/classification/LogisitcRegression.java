package binary.classification;

import java.util.Random;

import org.apache.spark.SparkConf;
import org.apache.spark.ml.classification.BinaryLogisticRegressionSummary;
import org.apache.spark.ml.classification.LogisticRegression;
import org.apache.spark.ml.classification.LogisticRegressionModel;
import org.apache.spark.ml.classification.LogisticRegressionTrainingSummary;
import org.apache.spark.ml.feature.VectorAssembler;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SparkSession;
import org.apache.spark.sql.functions;
import org.apache.spark.sql.types.DataTypes;
import org.apache.spark.sql.types.Metadata;
import org.apache.spark.sql.types.StructField;
import org.apache.spark.sql.types.StructType;

public class LogisitcRegression {
	static final String TRAINING_DATA_FILE_PATH = "/Users/haris/Desktop/train.csv";
	static final String TRAINED_MODEL_FILE_PATH = "/Users/haris/Desktop/model";
	static final String TEST_DATA_FILE_PATH = "/Users/haris/Desktop/test.csv";
	
	static final long SPLIT_SEED = new Random().nextLong();
	static final String COLUMN_SCORE_1 = "score1";
	static final String COLUMN_SCORE_2 = "score2";
	static final String COLUMN_PREDICTION = "result";
	static final String COLUMN_INPUT_FEATURES = "inputFeatures";
	
	static final double LOGISTIC_REGRESSION_REGULARIZATION = 0.01;
	static final int LOGISTIC_REGRESSION_ITERATIONS = 100;
	static final double LOGISTIC_REGRESSION_STEP_SIZE = 0.001;
	
	static StructType SCHEMA = new StructType(new StructField[]{
      new StructField(COLUMN_SCORE_1, DataTypes.DoubleType, false, Metadata.empty()),
      new StructField(COLUMN_SCORE_2, DataTypes.DoubleType, false, Metadata.empty()),
      new StructField(COLUMN_PREDICTION, DataTypes.IntegerType, false, Metadata.empty())
	});
	
	public static void main(String[] args) throws Exception{
				
		SparkConf sparkConfig = new SparkConf();
		sparkConfig.setMaster("local");
		sparkConfig.setAppName("binary-classfication");
		SparkSession sparkSession = SparkSession.builder().config(sparkConfig).getOrCreate();
		Dataset<Row> dataSet = loadDataSetFromFile(sparkSession, TRAINING_DATA_FILE_PATH);
	    
		Dataset<Row>[] splittedDataSet = dataSet.randomSplit(new double[]{0.7, 0.3}, SPLIT_SEED);
		Dataset<Row> trainingDataSet = splittedDataSet[0]; 
		Dataset<Row> crossValidationDataSet = splittedDataSet[1];
		
		LogisticRegression logisticRegression = new LogisticRegression().
				setMaxIter(LOGISTIC_REGRESSION_ITERATIONS).
				setRegParam(LOGISTIC_REGRESSION_ITERATIONS).
				setElasticNetParam(LOGISTIC_REGRESSION_STEP_SIZE);
		
	    logisticRegression.setLabelCol(COLUMN_PREDICTION);
	    logisticRegression.setFeaturesCol(COLUMN_INPUT_FEATURES);
	    System.out.println("<<<<<<<<<<<<<<<<<<<<<<< TRAINING >>>>>>>>>>>>>>>>>>>>>>>>");
	    LogisticRegressionModel logisticRegressionModel = logisticRegression.fit(trainingDataSet);
	    LogisticRegressionTrainingSummary logisticRegressionTrainingSummary = logisticRegressionModel.summary();
	    
	    //Print error per iteration
	    //double[] objectiveHistory = logisticRegressionTrainingSummary.objectiveHistory();
	    //for (double errorPerIteration : objectiveHistory)
	    //	System.out.println(errorPerIteration);
	
	    // Obtain the metrics useful to judge performance on test data.
	    // We cast the summary to a BinaryLogisticRegressionSummary since the problem is a binary classification problem.
	    BinaryLogisticRegressionSummary binaryLogisticRegressionSummary = (BinaryLogisticRegressionSummary) logisticRegressionTrainingSummary;
	    // Get the threshold corresponding to the maximum F-Measure and return LogisticRegression with
	    // this selected threshold.
	    Dataset<Row> fScore = binaryLogisticRegressionSummary.fMeasureByThreshold();
	    double maximumFScore = fScore.select(functions.max("F-Measure")).head().getDouble(0);
	    double bestThreshold = fScore.where(fScore.col("F-Measure").equalTo(maximumFScore)).select("threshold").head().getDouble(0);
	    logisticRegressionModel.setThreshold(bestThreshold);
	    System.out.println("maximum FScore: " + maximumFScore);
	    
	    System.out.println("<<<<<<<<<<<<<<<<<<<<<<< CROSS VALIDATION >>>>>>>>>>>>>>>>>>>>>>>>");
	    //make the predictions on cross validation set
	    Dataset<Row> crossValidationDataSetPredictions = logisticRegressionModel.transform(crossValidationDataSet);
	    printPredictionResult(crossValidationDataSetPredictions);
	    //saved the trained model on disk for later use
	    logisticRegressionModel.save(TRAINED_MODEL_FILE_PATH);
	    
	    System.out.println("<<<<<<<<<<<<<<<<<<<<<<< TEST >>>>>>>>>>>>>>>>>>>>>>>>");
	    Dataset<Row> testDataSet = loadDataSetFromFile(sparkSession, TEST_DATA_FILE_PATH);
		LogisticRegressionModel trainedLogisticRegressionModel = LogisticRegressionModel.load(TRAINED_MODEL_FILE_PATH);
		Dataset<Row> testDataSetPredictions = trainedLogisticRegressionModel.transform(testDataSet);
		printPredictionResult(testDataSetPredictions);
	    
	    sparkSession.stop();
	}
	
	private static Dataset<Row> loadDataSetFromFile(SparkSession spark, String inputFile) throws Exception {
		Dataset<Row> dataSet = spark.read().schema(SCHEMA).
				format("com.databricks.spark.csv").
				option("header", "true").
				load(inputFile);
		
		dataSet = spark.createDataFrame(dataSet.javaRDD(), SCHEMA);
	    VectorAssembler vectorAssembler = new VectorAssembler().
	    		setInputCols(new String[]{COLUMN_SCORE_1, COLUMN_SCORE_2}).
	    		setOutputCol(COLUMN_INPUT_FEATURES);
	    dataSet = vectorAssembler.transform(dataSet);
		return dataSet;
	}
	
	private static void printPredictionResult(Dataset<Row> data) {
		
		int total = 0;
		int correct = 0;
		
		for(Row row : data.collectAsList()){
			total++;
			String prediction = String.valueOf(row.get(row.fieldIndex("prediction"))).replace(".0", "");
			String actual = String.valueOf(row.get(row.fieldIndex(COLUMN_PREDICTION)));
			
			if(prediction.equals(actual))
				correct ++;
	    }
		
		System.out.println("Correct predictions: " + correct + "/" + total);
	}
}
