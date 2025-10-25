package org.fogbeam.dl4j.spark;

import java.util.ArrayList;
import java.util.List;

import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.deeplearning4j.datasets.iterator.impl.MnistDataSetIterator;
import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.Updater;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.spark.api.TrainingMaster;
import org.deeplearning4j.spark.impl.multilayer.SparkDl4jMultiLayer;
import org.deeplearning4j.spark.impl.paramavg.ParameterAveragingTrainingMaster;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import com.beust.jcommander.Parameter;

public class SparkNeuralNetwork 
{

	private static final Logger log = LoggerFactory.getLogger(SparkNeuralNetwork.class);

    @Parameter(names = "-useSparkLocal", description = "Use spark local (helper for testing/running without spark submit)", arity = 1)
    private boolean useSparkLocal = true;

    @Parameter(names = "-batchSizePerWorker", description = "Number of examples to fit each worker with")
    private int batchSizePerWorker = 16;

    @Parameter(names = "-numEpochs", description = "Number of epochs for training")
    private int numEpochs = 15;

	
	
	public static void main(String[] args) throws Exception
	{
		new SparkNeuralNetwork().entryPoint(args);

		System.out.println( "done" );
	}

	
    protected void entryPoint(String[] args) throws Exception 
    {

		
		SparkConf sparkConf = new SparkConf();
		sparkConf.setMaster("local");
		sparkConf.setAppName("SparkNeuralNetwork");
			
		JavaSparkContext sc = new JavaSparkContext( sparkConf );

		
		DataSetIterator iterTrain = new MnistDataSetIterator(batchSizePerWorker, true, 12345);
        DataSetIterator iterTest = new MnistDataSetIterator(batchSizePerWorker, true, 12345);
        
        List<DataSet> trainingDataList = new ArrayList<>();
        List<DataSet> testDataList = new ArrayList<>();
        
        while (iterTrain.hasNext()) {
            trainingDataList.add(iterTrain.next());
        }
        while (iterTest.hasNext()) {
            testDataList.add(iterTest.next());
        }

        JavaRDD<DataSet> trainingData = sc.parallelize(trainingDataList);
        JavaRDD<DataSet> testData = sc.parallelize(testDataList);
		
		
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

		//Create the TrainingMaster instance
		int examplesPerDataSetObject = 1;
		TrainingMaster trainingMaster = new ParameterAveragingTrainingMaster.Builder(examplesPerDataSetObject)
				.build();
		
		//Create the SparkDl4jMultiLayer instance
		//Create the Spark network
        SparkDl4jMultiLayer sparkNet = new SparkDl4jMultiLayer(sc, conf, trainingMaster);

		//Fit the network using the training data:
        //Execute training:
        for (int i = 0; i < numEpochs; i++) 
        {
            sparkNet.fit(trainingData);
            log.info("Completed Epoch {}", i);
        }

		
        // Perform evaluation (distributed)
        Evaluation evaluation = sparkNet.evaluate(testData);
        log.info("***** Evaluation *****");
        log.info(evaluation.stats());

        //Delete the temp training files, now that we are done with them
        trainingMaster.deleteTempFiles(sc);

        
        log.info("***** Example Complete *****");
        
    }	
}