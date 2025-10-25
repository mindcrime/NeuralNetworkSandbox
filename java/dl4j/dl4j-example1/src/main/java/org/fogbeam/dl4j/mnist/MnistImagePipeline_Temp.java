package org.fogbeam.dl4j.mnist;

import java.io.File;
import java.util.List;
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
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.api.preprocessor.DataNormalization;
import org.nd4j.linalg.dataset.api.preprocessor.ImagePreProcessingScaler;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public class MnistImagePipeline_Temp 
{

	private static Logger log = LoggerFactory.getLogger(MnistImagePipeline_Temp.class);
	
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
		
		
		int numEpochs = 15;
		double learningRate = 0.006;
		
		
		
		File trainingData = new File( "/home/prhodes/development/experimental/ai_exp/NeuralNetworkSandbox/mnist_png/training/" );
		File testData = new File( "/home/prhodes/development/experimental/ai_exp/NeuralNetworkSandbox/mnist_png/testing/" );
		
		
		FileSplit trainingSplit = new FileSplit( trainingData, NativeImageLoader.ALLOWED_FORMATS, randomGen );
		FileSplit testSplit = new FileSplit( testData, NativeImageLoader.ALLOWED_FORMATS, randomGen );
		

		ParentPathLabelGenerator labelGen = new ParentPathLabelGenerator();
		
		ImageRecordReader imageReader = new ImageRecordReader( height, width, channels, labelGen );
		
		imageReader.initialize(trainingSplit);
		// imageReader.setListeners( new LogRecordListener() );
		
			
		List<String> labels = imageReader.getLabels();
		System.out.println( "labels: " + labels );
		
		
		DataSetIterator dataIterator = new RecordReaderDataSetIterator(imageReader, batchSize, 1, outputNum);

		DataSet ds = dataIterator.next();

		INDArray featureMatrix = ds.getFeatureMatrix();
		
		System.out.println( "***************************************\n");
		System.out.println( "featureMatrix, rank: " + featureMatrix.rank() );
		System.out.println( featureMatrix.toString());
		System.out.println( "***************************************\n");
		
		
		System.out.println( "done" );
		log.info( "done" );
	}

}
