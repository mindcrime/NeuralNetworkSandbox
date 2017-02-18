package org.fogbeam.dl4j.mnist;

import java.io.File;
import java.util.Random;
import java.util.concurrent.TimeUnit;

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

import com.google.common.base.Stopwatch;

public class MnistImagePipelineWith6LayerNNExample 
{

	private static Logger log = LoggerFactory.getLogger(MnistImagePipelineWith6LayerNNExample.class);
	
	public static void main(String[] args) throws Exception
	{
		BasicConfigurator.configure();

		boolean saveModel = false;
		if( args.length > 0 )
		{
			saveModel = Boolean.parseBoolean(args[0]);
		}

		log.info( "saveModel: " + saveModel );
		
	
		Stopwatch stopwatch = Stopwatch.createStarted();
		
		// image information
		// 28x28 grayscale (single channel)
		int height = 28;
		int width = 28;
		int channels = 1;
		int rngSeed = 123;
		Random randomGen = new Random( rngSeed );
		int batchSize = 128;
		int outputNum = 10;
		
		
		int numEpochs = 45;
		double learningRate = 0.0006;
		
		
		
		File trainingData = new File( "/home/prhodes/development/experimental/ai_exp/NeuralNetworkSandbox/mnist_png/training/" );
		File testData = new File( "/home/prhodes/development/experimental/ai_exp/NeuralNetworkSandbox/mnist_png/testing/" );
		
		
		FileSplit trainingSplit = new FileSplit( trainingData, NativeImageLoader.ALLOWED_FORMATS, randomGen );
		FileSplit testSplit = new FileSplit( testData, NativeImageLoader.ALLOWED_FORMATS, randomGen );
		

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
				.layer(1, new DenseLayer.Builder()
						.nIn(100)
						.nOut(100)
						.activation("relu")
						.weightInit( WeightInit.XAVIER)
						.build())
				.layer(2, new DenseLayer.Builder()
						.nIn(100)
						.nOut(100)
						.activation("relu")
						.weightInit( WeightInit.XAVIER)
						.build())
				.layer(3, new DenseLayer.Builder()
						.nIn(100)
						.nOut(100)
						.activation("relu")
						.weightInit( WeightInit.XAVIER)
						.build())
				.layer(4, new DenseLayer.Builder()
						.nIn(100)
						.nOut(100)
						.activation("relu")
						.weightInit( WeightInit.XAVIER)
						.build())
				.layer(5, new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
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
		
		
		model.setListeners( new ScoreIterationListener( 10 ) );
		
		
		log.info("***************** TRAINING MODEL ********************");
		
		for( int i = 0; i < numEpochs; i++ )
		{
			model.fit(dataIterator);
		}
		
		stopwatch.stop();
		
		
		imageReader.reset();
		
		imageReader.initialize(testSplit);
		
		DataSetIterator testIterator = new RecordReaderDataSetIterator(imageReader, batchSize, 1, outputNum );
		
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
		log.info( "Training time: " + stopwatch.elapsed(TimeUnit.SECONDS ) + " seconds, or " +  stopwatch.elapsed(TimeUnit.MINUTES ) + " minutes.");
		
		if( saveModel )
		{
			log.info( "**************** SAVING MODEL ****************");
			File modelOutputLocation = new File( "mnist_model_nn.zip" );
			if( modelOutputLocation.exists() )
			{
				modelOutputLocation.delete();
				modelOutputLocation = new File( "mnist_model_nn.zip" );
			}
			ModelSerializer.writeModel(model, modelOutputLocation, false);
		}
		
		
		
		System.out.println( "done" );
		log.info( "done" );
	}

}
