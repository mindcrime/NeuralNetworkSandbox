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


public class MakePredictionMain 
{

	public static void main(String[] args) throws Exception
	{
		
		int batchSize = 1;
		int numOutputs = 2;
		

		File locationToSave = new File("MyMultiLayerNetwork.zip");
		
		MultiLayerNetwork model = ModelSerializer.restoreMultiLayerNetwork(locationToSave);
		
		
		// load the eval data
		RecordReader predictDataReader = new CSVRecordReader();
		predictDataReader.initialize(new FileSplit( new File("linear_data_eval.csv")));
		DataSetIterator predictDataIterator = new RecordReaderDataSetIterator(predictDataReader, batchSize,0,2);
		
		
		System.out.println( "Predicting with model...");
		
		
		while(predictDataIterator.hasNext())
		{
			DataSet t = predictDataIterator.next();
			INDArray labels = t.getLabels();
			System.out.println( "labels: " + labels );
			
			
			INDArray features = t.getFeatures();
			

			INDArray predicted = model.output(features,false);
			System.out.println( "prediction: " + (  ( predicted.getDouble(0) < predicted.getDouble(1) ) ? 1 : 0 ) );
			
		}

		
		System.out.println( "done" );

	}

}
