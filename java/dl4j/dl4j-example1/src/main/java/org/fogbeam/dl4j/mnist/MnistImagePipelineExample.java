package org.fogbeam.dl4j.mnist;

import java.io.File;
import java.util.Random;

import org.apache.log4j.BasicConfigurator;
import org.datavec.api.io.labels.ParentPathLabelGenerator;
import org.datavec.api.records.listener.impl.LogRecordListener;
import org.datavec.api.split.FileSplit;
import org.datavec.image.loader.NativeImageLoader;
import org.datavec.image.recordreader.ImageRecordReader;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.nd4j.linalg.dataset.api.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.api.preprocessor.DataNormalization;
import org.nd4j.linalg.dataset.api.preprocessor.ImagePreProcessingScaler;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public class MnistImagePipelineExample 
{

	private static Logger log = LoggerFactory.getLogger(MnistImagePipelineExample.class);
	
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
		int batchSize = 1;
		int outputNum = 10;
		
		File trainingData = new File( "/home/prhodes/development/experimental/ai_exp/NeuralNetworkSandbox/mnist_png/training/" );
		File testData = new File( "/home/prhodes/development/experimental/ai_exp/NeuralNetworkSandbox/mnist_png/testing/" );
		
		
		FileSplit trainingSplit = new FileSplit( trainingData, NativeImageLoader.ALLOWED_FORMATS, randomGen );
		FileSplit testSplit = new FileSplit( testData, NativeImageLoader.ALLOWED_FORMATS, randomGen );
		

		ParentPathLabelGenerator labelGen = new ParentPathLabelGenerator();
		
		ImageRecordReader imageReader = new ImageRecordReader( height, width, channels, labelGen );
		
		imageReader.initialize(trainingSplit);
		imageReader.setListeners( new LogRecordListener() );
		
		DataSetIterator dataIterator = new RecordReaderDataSetIterator(imageReader, batchSize, 1, outputNum);

		DataNormalization scaler = new ImagePreProcessingScaler(0,1);
		dataIterator.setPreProcessor(scaler);
		
		
		for(int i = 1; i <= 3; i++ )
		{
			
			DataSet ds = dataIterator.next();
			System.out.println( ds );
			System.out.println( dataIterator.getLabels());
			
			
		}
		
		
		
		
		
		
		
		System.out.println( "done" );
		log.info( "done" );
	}

}
