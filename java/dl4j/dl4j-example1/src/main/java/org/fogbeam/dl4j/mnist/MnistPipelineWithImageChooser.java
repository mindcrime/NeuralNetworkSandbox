package org.fogbeam.dl4j.mnist;

import java.io.File;
import java.util.Arrays;
import java.util.List;

import javax.swing.JFileChooser;

import org.apache.log4j.BasicConfigurator;
import org.datavec.image.loader.NativeImageLoader;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.util.ModelSerializer;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.api.preprocessor.DataNormalization;
import org.nd4j.linalg.dataset.api.preprocessor.ImagePreProcessingScaler;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;


public class MnistPipelineWithImageChooser 
{

	private static Logger logger = LoggerFactory.getLogger(MnistPipelineWithImageChooser.class);
	
	public static String fileChoose()
	{
		JFileChooser jfc = new JFileChooser();
		int ret = jfc.showOpenDialog(null);
		if( ret == JFileChooser.APPROVE_OPTION) 
		{
			File file = jfc.getSelectedFile();
			String fileName = file.getAbsolutePath();
			return fileName;
		}
		else
		{
			return null;
		}
	}
	
	public static void main(String[] args) throws Exception
	{
		BasicConfigurator.configure();
		
		int height = 28;
		int width = 28;
		int channels = 1;
		
		List<Integer> labelList = Arrays.asList(0, 1, 2, 3, 4, 5, 6, 7, 8, 9);

		String fileName = fileChoose();
		logger.info( "****** fileName: " + fileName + "   **************");
		
		logger.info("*********** LOAD THE MODEL **********");
		File modelLocation = new File( "mnist_model_nn.zip" );
		MultiLayerNetwork model = ModelSerializer.restoreMultiLayerNetwork(modelLocation);

		
		logger.info( "********** TEST YOUR IMAGE AGAINST SAVED MODEL **********");
		
		File imageFile = new File( fileName );
		NativeImageLoader loader = new NativeImageLoader( height, width, channels );
		INDArray image = loader.asMatrix(imageFile);
		
		logger.info( "********** SCALE OUR IMAGE TO MATCH TRAINING DATA **********" );
		DataNormalization scaler = new ImagePreProcessingScaler(0,1);
		scaler.transform(image);
		
		INDArray output = model.output(image);
		
		logger.info( output.toString());
		logger.info( "labels: " + labelList );
		
		System.out.println( "Image was a " + maxProbIndex(output));
		
	}

	public static int maxProbIndex( INDArray data )
	{
		int maxProbIndex = 0;
		double prevMax = 0;
		
		for( int i = 0; i < data.length(); i++ )
		{
			double temp = data.getDouble(i);
			if( temp > prevMax )
			{
				prevMax = temp;
				maxProbIndex = i;
			}
		}
		
		return maxProbIndex;
	}
}
