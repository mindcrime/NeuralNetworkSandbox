package org.fogbeam.dl4j.spark.eval;

import java.io.File;
import java.util.Random;

import org.apache.log4j.BasicConfigurator;
import org.datavec.api.io.labels.ParentPathLabelGenerator;
import org.datavec.api.split.FileSplit;
import org.datavec.image.loader.NativeImageLoader;
import org.datavec.image.recordreader.ImageRecordReader;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.Updater;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.deeplearning4j.util.ModelSerializer;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.api.preprocessor.DataNormalization;
import org.nd4j.linalg.dataset.api.preprocessor.ImagePreProcessingScaler;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public class LoadAndEvaluateModel 
{

	private static Logger log = LoggerFactory.getLogger(LoadAndEvaluateModel.class);
	
	public static void main(String[] args) throws Exception
	{
		BasicConfigurator.configure();
		
		// image information
		// 28x28 grayscale (single channel)
		int height = 28;
		int width = 28;
		int channels = 1;
		int rngSeed = 123;
		Random randomGen = new Random( rngSeed );
		int batchSize = 128;
		int outputNum = 10;
				
		
		File modelLocation = new File( "sparkTrainedNetwork.zip" );
		MultiLayerNetwork model = ModelSerializer.restoreMultiLayerNetwork(modelLocation);

		
		File testData = new File( "/home/prhodes/development/experimental/ai_exp/NeuralNetworkSandbox/mnist_png/testing/" );
		FileSplit testSplit = new FileSplit( testData, NativeImageLoader.ALLOWED_FORMATS, randomGen );

		ParentPathLabelGenerator labelGen = new ParentPathLabelGenerator();
		ImageRecordReader imageReader = new ImageRecordReader( height, width, channels, labelGen );		
		imageReader.initialize(testSplit);
		
		DataSetIterator testIterator = new RecordReaderDataSetIterator(imageReader, batchSize, 1, outputNum );
		DataNormalization scaler = new ImagePreProcessingScaler(0,1);
		scaler.fit(testIterator);
		testIterator.setPreProcessor(scaler);
		
		// create eval object with 10 possible classes
		Evaluation eval = new Evaluation(outputNum);
		
		while( testIterator.hasNext())
		{
			DataSet ds = testIterator.next();
			INDArray output = model.output(ds.getFeatureMatrix());
			eval.eval(ds.getLabels(), output);
		}
		
		log.info( eval.stats() );
		
		
		System.out.println( "done" );
		log.info( "done" );
	}
}