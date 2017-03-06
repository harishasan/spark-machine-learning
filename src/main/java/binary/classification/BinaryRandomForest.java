package binary.classification;

import java.util.HashMap;
import java.util.Map;
import java.util.Random;

import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.api.java.function.Function;
import org.apache.spark.mllib.linalg.Vector;
import org.apache.spark.mllib.linalg.Vectors;
import org.apache.spark.mllib.regression.LabeledPoint;
import org.apache.spark.mllib.tree.model.RandomForestModel;

import scala.Tuple2;
import utils.Utils;

public class BinaryRandomForest {
	
	static String TRAINING_DATA_FILE_NAME = "scores_dataset/train.csv";
	static String TEST_DATA_FILE_NAME = "scores_dataset/test.csv";
	
	static final long SPLIT_SEED = new Random().nextLong();
	
	static final int NUMBER_OF_CLASSES = 2;
	//Use more in practice.
	static final int NUMBER_OF_TREES = 2;
	static final String FEATURE_SUBSET_STRATEGY = "auto";
	static final String IMPURITY = "gini";
	static final int MAX_DEPTH = 5; 
	static final int MAX_BINS = 32; 
	
	public static void main(String[] args) throws Exception{
	
		SparkConf sparkConfig = new SparkConf();
		sparkConfig.setMaster("local");
		sparkConfig.setAppName("binary-classfication");
		
		String resourceDirecyoryPath = Utils.getResourcesDirectoryPath();
		String trainingDataFilePath = resourceDirecyoryPath + TRAINING_DATA_FILE_NAME;
		String testDataFilePath = resourceDirecyoryPath + TEST_DATA_FILE_NAME;
		
		
		JavaSparkContext sparkContext = new JavaSparkContext(sparkConfig);
        JavaRDD<String> data = sparkContext.textFile(trainingDataFilePath);
        //remove the header row from data
        final String headerRow = data.first();
		data = data.filter(item -> !item.equals(headerRow));
        JavaRDD<LabeledPoint> formattedData = data.map(getFunctionToConvertLineToLabelledPoint());
        
        JavaRDD<LabeledPoint>[] splits = formattedData.randomSplit(new double[]{0.7, 0.3}, SPLIT_SEED);
        JavaRDD<LabeledPoint> trainingData = splits[0];
        JavaRDD<LabeledPoint> crossValidationData = splits[1];

        // Train a RandomForest model.
        // Empty categoricalFeaturesInfo indicates all features are continuous.
        Map<Integer, Integer> categoricalFeaturesInfo = new HashMap<>();
        Integer seed = new Random().nextInt();
        RandomForestModel randomForestModel = org.apache.spark.mllib.tree.RandomForest.trainClassifier(trainingData, 
    		NUMBER_OF_CLASSES,
    		categoricalFeaturesInfo, 
    		NUMBER_OF_TREES,
    		FEATURE_SUBSET_STRATEGY,
    		IMPURITY,
    		MAX_DEPTH,
    		MAX_BINS,
    		seed);

        //each entry in RDD contains <prediction,actual_label>
        JavaPairRDD<Double, Double> predictionAndLabels = crossValidationData.mapToPair(
    		dataPoint -> new Tuple2<>(randomForestModel.predict(dataPoint.features()), dataPoint.label())
        );
        
        System.out.println("<<<<<<<<<<<<<<<<<<<<<<< Cross validation set stats >>>>>>>>>>>>>>>>>>>>>>>>");
        Utils.printFScoreBinaryClassfication(predictionAndLabels);
        //cross validation error = correct/actual
        double crossValidationError = predictionAndLabels.filter(pAndL -> !pAndL._1().equals(pAndL._2())).count() / (double) crossValidationData.count();
        System.out.println("Cross validation Error: " + crossValidationError);

        System.out.println("<<<<<<<<<<<<<<<<<<<<<<< Test set stats >>>>>>>>>>>>>>>>>>>>>>>>");
        JavaRDD<String> testData = sparkContext.textFile(testDataFilePath);
        //remove the header row from data
        final String testHeaderRow = testData.first();
		testData = testData.filter(item -> !item.equals(testHeaderRow ));
        JavaRDD<LabeledPoint> formattedTestData = testData.map(getFunctionToConvertLineToLabelledPoint());
        JavaPairRDD<Double, Double> testPredictionAndLabels = formattedTestData.mapToPair(
    		dataPoint -> new Tuple2<>(randomForestModel.predict(dataPoint.features()), dataPoint.label())
		);
        double testSetError = testPredictionAndLabels.filter(pAndL -> !pAndL._1().equals(pAndL._2())).count() / (double) crossValidationData.count();
        System.out.println("Test Error: " + testSetError );
        Utils.printFScoreBinaryClassfication(testPredictionAndLabels);
        sparkContext.close();
	}

	@SuppressWarnings("serial")
	private static Function<String, LabeledPoint> getFunctionToConvertLineToLabelledPoint() {
		
		return new Function<String, LabeledPoint>() {
		    
			public LabeledPoint call(String line) throws Exception {
		        
				String[] parts = line.split(",");
				double score1 = Double.parseDouble(parts[0]);
				double score2 = Double.parseDouble(parts[1]);
				double label = Double.parseDouble(parts[2]);
				
				Vector featureVector = Vectors.dense(new double[]{score1, score2});  
		        return new LabeledPoint(label, featureVector);
		    }
		};
	}
}
