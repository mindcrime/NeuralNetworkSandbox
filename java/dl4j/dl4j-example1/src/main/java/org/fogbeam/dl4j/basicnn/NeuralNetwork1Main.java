package org.fogbeam.dl4j.basicnn;

import java.io.File;

import org.datavec.api.records.reader.RecordReader;
import org.datavec.api.records.reader.impl.csv.CSVRecordReader;
import org.datavec.api.split.FileSplit;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.Updater;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.lossfunctions.LossFunctions;


public class NeuralNetwork1Main 
{

	public static void main(String[] args) throws Exception
	{

		int seed = 123;
		double learningRate = 0.01;
		int batchSize = 50;
		int numEpochs = 30;
		
		int numInputs = 2;
		int numOutputs = 2;
		
		int numHiddenNodes = 20;
		
		// load the training data
		RecordReader trainReader = new CSVRecordReader();
		trainReader.initialize(new FileSplit( new File("linear_data_train.csv")));
		DataSetIterator trainIterator = new RecordReaderDataSetIterator(trainReader, batchSize,0,2);
		
		// load the eval data
		RecordReader evalReader = new CSVRecordReader();
		evalReader.initialize(new FileSplit( new File("linear_data_eval.csv")));
		DataSetIterator evalIterator = new RecordReaderDataSetIterator(evalReader, batchSize,0,2);
				
		
		MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
										.seed(seed)
										.iterations(1)
										.optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
										.learningRate(learningRate)
										.updater(Updater.NESTEROVS)
										.momentum(0.9)
										.list()
										.layer(0, new DenseLayer.Builder()
											.nIn(numInputs)
											.nOut(numHiddenNodes)
											.weightInit(WeightInit.XAVIER)
											.activation("relu")
											.build() )
										.layer(1, new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
												.weightInit(WeightInit.XAVIER)
												.activation("softmax")
												.nIn(numHiddenNodes)
												.nOut(numOutputs)
												.build() )
										.pretrain(false)
										.backprop(true)
										.build();
		
		// System.out.println( conf.toJson() );
		
		MultiLayerNetwork model = new MultiLayerNetwork(conf);
		model.init();
		model.setListeners(new ScoreIterationListener(10));
		
		for( int n = 0; n < numEpochs; n++ )
		{
			model.fit(trainIterator);
			
		}
		
		
		
		System.out.println( "Evaluating model...");
		Evaluation eval = new Evaluation(numOutputs);
		while(evalIterator.hasNext())
		{
			DataSet t = evalIterator.next();
			INDArray features = t.getFeatureMatrix();
			INDArray labels = t.getLabels();
			INDArray predicted = model.output(features,false);
			eval.eval(labels, predicted);
		}
		
		System.out.println( eval.stats());
		
		System.out.println( "done" );

	}

}
