package binary.classification;

import java.util.HashMap;
import java.util.LinkedHashMap;
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

public class MultiClassRandomForest {
	
	static String TRAINING_DATA_FILE_NAME = "kdd_dataset/train.csv";
	static String TEST_DATA_FILE_NAME = "kdd_dataset/test.csv";
	
	static final long SPLIT_SEED = new Random().nextLong();
	
	static final int NUMBER_OF_TREES = 100;
	static final String FEATURE_SUBSET_STRATEGY = "auto";
	static final String IMPURITY = "gini";
	static final int MAX_DEPTH = 4; 
	static final int MAX_BINS = 100;
	static final int NUMBER_OF_FEATURES = 41;
	
	/**
	 * Map<class_label, assigned_index>
	 * e.g. <normal, 1>
	 * e.g. <dos, 2>
	 * Linked hash map to get label using index  
	 */
	private static final LinkedHashMap<String, Double> classLabelsMap = new LinkedHashMap<>();
	/**
	 * These maps are required to handle discrete value features
	 * Each map represents one discrete value feature
	 * Map<discrete_value, assigned_index>
	 * E.g <tcp, 1>
	 * E.g <udp, 2>
	 */
	private static final HashMap<String, Double> protocolTypeMap = new HashMap<>();
	private static final HashMap<String, Double> serviceMap = new HashMap<>();
	private static final HashMap<String, Double> flagMap = new HashMap<>();
	
	public static void main(String[] args) throws Exception{
	
		SparkConf sparkConfig = new SparkConf();
		sparkConfig.setMaster("local");
		sparkConfig.setAppName("binary-classfication");
		
		String resourceDirecyoryPath = Utils.getResourcesDirectoryPath();
		String trainingDataFilePath = resourceDirecyoryPath + TRAINING_DATA_FILE_NAME;
		String testDataFilePath = resourceDirecyoryPath + TEST_DATA_FILE_NAME;
		
		
		JavaSparkContext sparkContext = new JavaSparkContext(sparkConfig);
		sparkContext.setLogLevel("ERROR");
        JavaRDD<String> data = sparkContext.textFile(trainingDataFilePath);
        //remove the header row from data
        final String headerRow = data.first();
		data = data.filter(item -> !item.equals(headerRow));
        JavaRDD<LabeledPoint> formattedData = data.map(getFunctionToConvertLineToLabelledPoint());
        
        //to avoid lazy loading because in lazy loading labelMap won't get populated
        formattedData.count();
        
        JavaRDD<LabeledPoint>[] splits = formattedData.randomSplit(new double[]{0.7, 0.3}, SPLIT_SEED);
        JavaRDD<LabeledPoint> trainingData = splits[0];
        JavaRDD<LabeledPoint> crossValidationData = splits[1];
        //categoricalFeaturesInfo contains information about discrete features
        //E.g in case of this example protocol type can have values like TCP/UDP/ICMP
        Map<Integer, Integer> categoricalFeaturesInfo = new HashMap<>();
        categoricalFeaturesInfo.put(1, protocolTypeMap.size());
        categoricalFeaturesInfo.put(2, serviceMap.size());
        categoricalFeaturesInfo.put(3, flagMap.size());
        
        Integer seed = new Random().nextInt();
        RandomForestModel randomForestModel = org.apache.spark.mllib.tree.RandomForest.trainClassifier(trainingData, 
    		classLabelsMap.size(),
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
        
        System.out.println("Total examples: " + predictionAndLabels.count());
        System.out.println("======= Cross validation ==========");
        Utils.printFScoreMultiClassClassfication(predictionAndLabels, classLabelsMap);
        
        //cross validation error = correct/actual
        double crossValidationError = predictionAndLabels.filter(pAndL -> !pAndL._1().equals(pAndL._2())).count() / (double) crossValidationData.count();
        System.out.println("Cross validation Error: " + crossValidationError);

        System.out.println("======= Test ==========");
        //clear the populated maps
        classLabelsMap.clear();
        flagMap.clear();
        protocolTypeMap.clear();
        serviceMap.clear();
        
        JavaRDD<String> testData = sparkContext.textFile(testDataFilePath);
        final String testHeaderRow = testData.first();
        testData = testData.filter(item -> !item.equals(testHeaderRow));
        
        JavaRDD<LabeledPoint> formattedTestData = testData.map(getFunctionToConvertLineToLabelledPoint());
        //to avoid lazy loading because in lazy loading labelMap won't get populated
        formattedTestData.count();
        
        JavaPairRDD<Double, Double> testPredictionAndLabels = formattedTestData.mapToPair(
    		dataPoint -> new Tuple2<>(randomForestModel.predict(dataPoint.features()), dataPoint.label())
		);
        
        System.out.println("Total examples: " + testPredictionAndLabels.count());
        double testError = testPredictionAndLabels.filter(pAndL -> !pAndL._1().equals(pAndL._2())).count() / (double) testData.count();
        System.out.println("Test Error: " + testError);
        System.out.println("For better results download and train on complete dataset given here: https://kdd.ics.uci.edu/databases/kddcup99/kddcup99.html");
        
        sparkContext.close();
	}

	@SuppressWarnings("serial")
	private static Function<String, LabeledPoint> getFunctionToConvertLineToLabelledPoint() {
				
		return new Function<String, LabeledPoint>() {
		    
			public LabeledPoint call(String line) throws Exception {
		        
				double[] featureValues = new double[NUMBER_OF_FEATURES];				
				String[] parts = line.split(",");
				if(parts.length != 42)
					throw new Exception("Expected columns are 42 but were " + parts.length);
				
				for(int index = 0; index < 40; index ++){
					
					double currentValue = 0;
					
					if(index == 1)
						currentValue = getAndSetClassIndexFromMap(protocolTypeMap, parts[index]);	
					else if(index == 2)
						currentValue = getAndSetClassIndexFromMap(serviceMap, parts[index]);
					else if(index == 3)
						currentValue = getAndSetClassIndexFromMap(flagMap, parts[index]);
					else
						currentValue = Double.parseDouble(parts[index]);
					
					featureValues[index] = currentValue;
				}
				
				double labelIndex = getAndSetClassIndexFromMap(classLabelsMap, parts[NUMBER_OF_FEATURES]);
				Vector featureVector = Vectors.dense(featureValues);
		        return new LabeledPoint(labelIndex, featureVector);
		    }
		};
	}

	private static double getAndSetClassIndexFromMap(HashMap<String, Double> labelAndIndexMap, String label) {
		if(!labelAndIndexMap.containsKey(label))
			labelAndIndexMap.put(label, 1.0 * labelAndIndexMap.size());
			
		return labelAndIndexMap.get(label);
	}
}
