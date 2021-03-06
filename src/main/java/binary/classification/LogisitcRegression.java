package binary.classification;

import java.util.Random;

import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.function.PairFunction;
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

import scala.Tuple2;
import utils.Utils;

public class LogisitcRegression {
	
	static String TRAINING_DATA_FILE_NAME = "scores_dataset/train.csv";
	static String TEST_DATA_FILE_NAME = "scores_dataset/test.csv";
	//change this according to your system
	static String TRAINED_MODEL_FILE_PATH = "/Users/haris/Desktop/logistic_regression_model";
	
	static final long SPLIT_SEED = new Random().nextLong();
	static final String COLUMN_SCORE_1 = "score1";
	static final String COLUMN_SCORE_2 = "score2";
	static final String COLUMN_PREDICTION = "result";
	static final String COLUMN_INPUT_FEATURES = "inputFeatures";
	
	static final double LOGISTIC_REGRESSION_REGULARIZATION = 0.3;
	static final int LOGISTIC_REGRESSION_ITERATIONS = 100;
	static final double LOGISTIC_REGRESSION_STEP_SIZE = 0.001;
	
	static StructType SCHEMA = new StructType(new StructField[]{
      new StructField(COLUMN_SCORE_1, DataTypes.DoubleType, false, Metadata.empty()),
      new StructField(COLUMN_SCORE_2, DataTypes.DoubleType, false, Metadata.empty()),
      new StructField(COLUMN_PREDICTION, DataTypes.IntegerType, false, Metadata.empty())
	});
	
	public static void main(String[] args) throws Exception{
				
		String resourceDirecyoryPath = Utils.getResourcesDirectoryPath();
		String trainingDataFilePath = resourceDirecyoryPath + TRAINING_DATA_FILE_NAME;
		String testDataFilePath = resourceDirecyoryPath + TEST_DATA_FILE_NAME;
		
		SparkConf sparkConfig = new SparkConf();
		sparkConfig.setMaster("local");
		sparkConfig.setAppName("binary-classfication");
		SparkSession sparkSession = SparkSession.builder().config(sparkConfig).getOrCreate();
		Dataset<Row> dataSet = loadDataSetFromFile(sparkSession, trainingDataFilePath);
	    
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
	    
	    //maximum 1, higher the better
	    //https://en.wikipedia.org/wiki/Receiver_operating_characteristic
	    System.out.println("Area under ROC: " + binaryLogisticRegressionSummary.areaUnderROC());
	    Dataset<Row> fScore = binaryLogisticRegressionSummary.fMeasureByThreshold();
	    double maximumFScore = fScore.select(functions.max("F-Measure")).head().getDouble(0);
	    double bestThreshold = fScore.where(fScore.col("F-Measure").equalTo(maximumFScore)).select("threshold").head().getDouble(0);
	    logisticRegressionModel.setThreshold(bestThreshold);
	    System.out.println("maximum FScore: " + maximumFScore);
	    
	    System.out.println("<<<<<<<<<<<<<<<<<<<<<<< CROSS VALIDATION >>>>>>>>>>>>>>>>>>>>>>>>");
	    //make the predictions on cross validation set
	    Dataset<Row> crossValidationDataSetPredictions = logisticRegressionModel.transform(crossValidationDataSet);	    
	    JavaPairRDD<Double, Double> crossValidationPredictionRDD = convertToJavaRDDPair(crossValidationDataSetPredictions);
	    Utils.printFScoreBinaryClassfication(crossValidationPredictionRDD);
	    printPredictionResult(crossValidationDataSetPredictions);
	    //saved the trained model on disk for later use
	    logisticRegressionModel.save(TRAINED_MODEL_FILE_PATH);
	    
	    System.out.println("<<<<<<<<<<<<<<<<<<<<<<< TEST >>>>>>>>>>>>>>>>>>>>>>>>");
	    Dataset<Row> testDataSet = loadDataSetFromFile(sparkSession, testDataFilePath);
		LogisticRegressionModel trainedLogisticRegressionModel = LogisticRegressionModel.load(TRAINED_MODEL_FILE_PATH);
		Dataset<Row> testDataSetPredictions = trainedLogisticRegressionModel.transform(testDataSet);
		printPredictionResult(testDataSetPredictions);
	    
	    sparkSession.stop();
	}

	private static JavaPairRDD<Double, Double> convertToJavaRDDPair(Dataset<Row> rowsData) {
		JavaRDD<Row> rowsRdd = rowsData.toJavaRDD();
		JavaPairRDD<Double, Double> pairRDD = rowsRdd.mapToPair(RowToTuplePairer);
		return pairRDD;
	}
	
	@SuppressWarnings("serial")
	public static final PairFunction<Row, Double, Double> RowToTuplePairer =
		 new PairFunction<Row, Double, Double>() {
	 		public Tuple2<Double, Double> call(Row row) throws Exception {
	 			Double prediction = Double.valueOf(String.valueOf(row.get(row.fieldIndex("prediction"))));
				Double actual = Double.valueOf(String.valueOf(row.get(row.fieldIndex(COLUMN_PREDICTION))));
	 			return new Tuple2<Double, Double>(prediction, actual);
 		}
	};
	
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
