package org.fogbeam.dl4j.basicnn;

import java.io.File;

import org.datavec.api.records.reader.RecordReader;
import org.datavec.api.records.reader.impl.csv.CSVRecordReader;
import org.datavec.api.split.FileSplit;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.util.ModelSerializer;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;


public class LoadNeuralNetworkMain 
{

	public static void main(String[] args) throws Exception
	{
		
		int batchSize = 50;
		int numOutputs = 2;
		

		File locationToSave = new File("MyMultiLayerNetwork.zip");
		
		MultiLayerNetwork model = ModelSerializer.restoreMultiLayerNetwork(locationToSave);
		
		
		// load the eval data
		RecordReader evalReader = new CSVRecordReader();
		evalReader.initialize(new FileSplit( new File("linear_data_eval.csv")));
		DataSetIterator evalIterator = new RecordReaderDataSetIterator(evalReader, batchSize,0,2);
		
		
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
