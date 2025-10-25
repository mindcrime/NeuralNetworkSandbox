package org.fogbeam.dl4j.spark;

import java.util.Arrays;
import java.util.List;

import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.input.PortableDataStream;
import org.apache.spark.mllib.regression.LabeledPoint;
import org.datavec.api.io.labels.ParentPathLabelGenerator;
import org.datavec.image.recordreader.ImageRecordReader;
import org.deeplearning4j.spark.util.MLLibUtil;
import org.nd4j.linalg.api.buffer.DataBuffer;
import org.nd4j.linalg.api.buffer.DataBuffer.Type;
import org.nd4j.linalg.factory.Nd4j;

public class ExpMain 
{

	public static void main(String[] args) throws Exception
	{
		
		SparkConf sparkConf = new SparkConf();
		sparkConf.setMaster("local");
		sparkConf.setAppName("SparkNeuralNetwork");
		
		// Nd4j.setDataType(Type.
		
		JavaSparkContext sc = new JavaSparkContext( sparkConf );

		// https://github.com/deeplearning4j/DataVec/blob/master/datavec-spark/src/test/java/org/datavec/spark/functions/TestRecordReaderBytesFunction.java
			
		// String recursiveSetting = sc.hadoopConfiguration().get("mapreduce.input.fileinputformat.input.dir.recursive");
		// System.out.println( "recursiveSetting: " + recursiveSetting );
		
		JavaPairRDD<String, PortableDataStream> files = sc.binaryFiles("/home/prhodes/development/experimental/ai_exp/NeuralNetworkSandbox/mnist_png/training/1/*.png");
		
		System.out.println( "binary data RDD created" );
			
		ImageRecordReader reader = new ImageRecordReader(28, 28, 1, new ParentPathLabelGenerator());
		List<String> labelsList = Arrays.asList("0", "1", "2", "3", "4", "5", "6", "7", "8", "9" );   //Need this for Spark: can't infer without init call
		reader.setLabels(labelsList);
		
		JavaRDD<LabeledPoint> labeledPoints = MLLibUtil.fromBinary(files, reader);
		
		System.out.println( "labeledPoints RDD created");
		
		LabeledPoint point = labeledPoints.first();
		
		System.out.println( "point: " + point.toString() );
		
		
		// JavaRDD<DataSet> trainingData = MLLibUtil.fromLabeledPoint( labeledPoints, 10, 50);

		
		System.out.println( "DataSet RDD created");
		
		
		/* 
        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .seed(12345)
                .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT).iterations(1)
                .activation(Activation.LEAKYRELU)
                .weightInit(WeightInit.XAVIER)
                .learningRate(0.02)
                .updater(Updater.NESTEROVS).momentum(0.9)
                .regularization(true).l2(1e-4)
                .list()
                .layer(0, new DenseLayer.Builder().nIn(28 * 28).nOut(500).build())
                .layer(1, new DenseLayer.Builder().nIn(500).nOut(100).build())
                .layer(2, new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
                    .activation(Activation.SOFTMAX).nIn(100).nOut(10).build())
                .pretrain(false).backprop(true)
                .build();

		// Create the TrainingMaster instance
		int examplesPerDataSetObject = 1;
		TrainingMaster trainingMaster = new ParameterAveragingTrainingMaster.Builder(examplesPerDataSetObject)
				.build();
		
		// Create the SparkDl4jMultiLayer instance
		// Create the Spark network
        SparkDl4jMultiLayer sparkNet = new SparkDl4jMultiLayer(sc, conf, trainingMaster);

        
        sparkNet.fit( trainingData );
        */
		
		

	}
}