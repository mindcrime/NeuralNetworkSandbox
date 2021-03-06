package org.fogbeam.dl4j.mnist;

import java.io.File;
import java.util.Random;

import org.apache.log4j.BasicConfigurator;
import org.datavec.api.io.labels.ParentPathLabelGenerator;
import org.datavec.api.split.FileSplit;
import org.datavec.image.loader.NativeImageLoader;
import org.datavec.image.recordreader.ImageRecordReader;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.Updater;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.util.ModelSerializer;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.api.preprocessor.DataNormalization;
import org.nd4j.linalg.dataset.api.preprocessor.ImagePreProcessingScaler;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public class MnistImagePipelineWithNetworkSave 
{

	private static Logger log = LoggerFactory.getLogger(MnistImagePipelineWithNetworkSave.class);
	
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
		
		FileSplit trainingSplit = new FileSplit( trainingData, NativeImageLoader.ALLOWED_FORMATS, randomGen );
		

		ParentPathLabelGenerator labelGen = new ParentPathLabelGenerator();
		
		ImageRecordReader imageReader = new ImageRecordReader( height, width, channels, labelGen );
		
		imageReader.initialize(trainingSplit);
		// imageReader.setListeners( new LogRecordListener() );
		
		DataSetIterator dataIterator = new RecordReaderDataSetIterator(imageReader, batchSize, 1, outputNum);

		DataNormalization scaler = new ImagePreProcessingScaler(0,1);
		dataIterator.setPreProcessor(scaler);
		
		
		MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
				.seed(rngSeed)
				.optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
				.iterations(1)
				.learningRate(learningRate)
				.updater(Updater.NESTEROVS)
				.momentum(0.9)
				.regularization(true)
				.l2(1e-4)
				.list()
				.layer(0, new DenseLayer.Builder()
						.nIn(height * width)
						.nOut(100)
						.activation("relu")
						.weightInit( WeightInit.XAVIER)
						.build())
				.layer(1, new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
						.nIn(100)
						.nOut(outputNum)
						.activation("softmax")
						.weightInit(WeightInit.XAVIER)
						.build())
				.pretrain(false)
				.backprop(true)
				.setInputType(InputType.convolutional(height, width, channels))
				.build();
		
		
		MultiLayerNetwork model = new MultiLayerNetwork( conf );
		model.init();
			
		
		log.info("***************** TRAINING MODEL ********************");
		
		for( int i = 0; i < numEpochs; i++ )
		{
			model.fit(dataIterator);
		}
		
		
		log.info("***************** SAVING MODEL ********************");
		File modelOutputLocation = new File( "mnist_model_nn.zip" );
		
		ModelSerializer.writeModel(model, modelOutputLocation, false);
		
		System.out.println( "done" );
		log.info( "done" );
	}

}
