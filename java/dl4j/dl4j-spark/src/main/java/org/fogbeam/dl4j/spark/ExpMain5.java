package org.fogbeam.dl4j.spark;

import java.io.BufferedWriter;
import java.io.File;
import java.io.FileWriter;
import java.nio.file.Paths;
import java.sql.Connection;
import java.sql.DriverManager;
import java.sql.PreparedStatement;
import java.util.Arrays;
import java.util.Date;
import java.util.List;
import java.util.Properties;
import java.util.UUID;
import java.util.concurrent.TimeUnit;

import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.api.java.function.Function;
import org.apache.spark.input.PortableDataStream;
import org.datavec.api.io.labels.ParentPathLabelGenerator;
import org.datavec.api.writable.Writable;
import org.datavec.image.recordreader.ImageRecordReader;
import org.datavec.spark.functions.RecordReaderFunction;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.Updater;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.spark.api.TrainingMaster;
import org.deeplearning4j.spark.datavec.DataVecDataSetFunction;
import org.deeplearning4j.spark.impl.multilayer.SparkDl4jMultiLayer;
import org.deeplearning4j.spark.impl.paramavg.ParameterAveragingTrainingMaster;
import org.deeplearning4j.util.ModelSerializer;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.preprocessor.DataNormalization;
import org.nd4j.linalg.dataset.api.preprocessor.ImagePreProcessingScaler;
import org.nd4j.linalg.lossfunctions.LossFunctions;

import com.google.common.base.Stopwatch;


public class ExpMain5 
{
	public static void main(String[] args) throws Exception
	{
		if( args != null && ! (args.length==0) ) 
		{
			for( String arg : args ) 
			{
				System.out.println( "arg: " + arg );
			}
		}
		
		
		SparkConf sparkConf = new SparkConf();
		
		String appName = "SparkNeuralNetwork5";
		sparkConf.setAppName(appName);
		
		Stopwatch sw = Stopwatch.createStarted();		
		
		JavaSparkContext sc = new JavaSparkContext( sparkConf );

		String appId = "";
		for( int i = 0; i < 10; i++ )
		{
			try
			{
				appId = sc.getConf().getAppId();
				break;
			}
			catch( Exception e )
			{
				Thread.sleep( 750 );
				continue;
			}
		}

		
		Class.forName("org.postgresql.Driver");
		
		String url = "jdbc:postgresql://" + args[2] + "/neuralobjects";
		Properties props = new Properties();
		props.setProperty("user","postgres");
		props.setProperty("password","");
		// props.setProperty("ssl","true");
		
		Connection dbConn = DriverManager.getConnection(url, props);
		
		PreparedStatement statement = dbConn.prepareStatement("insert into enginejob values(?, ?, ?, ?, ?)");
		// insert into enginejob values( '1', 'abc', 'SparkNeuralNetwork', '2017', '2018');
		
		UUID uuid = UUID.randomUUID();
		statement.setString( 1, uuid.toString() );
		statement.setString( 2, appId );
		statement.setString( 3, appName );
		statement.setString( 4, new Date().toString() );
		statement.setString(5, "not_set");
		
		statement.executeUpdate();
		
		dbConn.close();
		
		sc.hadoopConfiguration().set("mapreduce.input.fileinputformat.input.dir.recursive", "true");
		
		// files...
		JavaPairRDD<String, PortableDataStream> origData = sc.binaryFiles( args[0]);		
		
		ImageRecordReader irr = new ImageRecordReader(28, 28, 1, new ParentPathLabelGenerator() );
        List<String> labelsList = Arrays.asList( "0", "1", "2", "3", "4", "5", "6", "7", "8", "9" );
		irr.setLabels(labelsList);
        RecordReaderFunction rrf = new RecordReaderFunction(irr);
        JavaRDD<List<Writable>> rdd = origData.map(rrf);
		
        
		System.out.println( "DataSet RDD created");
				
		
		JavaRDD<DataSet> unscaledData = rdd.map(new DataVecDataSetFunction(1,10, false, null, null ));
		
		System.out.println( "unscaled data RDD created");
		
		JavaRDD<DataSet> trainingData = unscaledData.map( new Function<DataSet,DataSet>() {

			DataNormalization scaler = new ImagePreProcessingScaler(0,1);
			@Override
			public DataSet call(DataSet v1) throws Exception {
				
				DataSet newDS = new DataSet( v1.getFeatures(), v1.getLabels());
				scaler.preProcess(newDS);
				return newDS;
			}} );
		
		System.out.println( "trainingData RDD created");
						
		System.out.println( "Building network configuration...");
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
                .setInputType(InputType.convolutional(28, 28, 1))
                .build();

		// Create the TrainingMaster instance
		int examplesPerDataSetObject = 1;
		TrainingMaster trainingMaster = new ParameterAveragingTrainingMaster.Builder(examplesPerDataSetObject)
				.batchSizePerWorker(10)
				.workerPrefetchNumBatches(2)
				.build();
		
		// Create the SparkDl4jMultiLayer instance
		// Create the Spark network
        SparkDl4jMultiLayer sparkNet = new SparkDl4jMultiLayer(sc, conf, trainingMaster);

        long elapsedPhase1 = sw.elapsed(TimeUnit.SECONDS);
        System.out.println( "Loading data took " + elapsedPhase1 + " seconds.  Starting to train model now.");

        
        MultiLayerNetwork trainedNetwork  = null;
        for( int i = 0; i < 6; i++ ) 
        {
        	trainedNetwork = sparkNet.fit( trainingData );
        }
		
        long elapsedPhase2 = sw.elapsed(TimeUnit.SECONDS);
        System.out.println( "Training model took " + ( elapsedPhase2 - elapsedPhase1) + " seconds.");
        System.out.println( "Total elapsed time: " + elapsedPhase2 );

        File metricsFile = new File( args[1] + "/run_metrics.log" );      
        Date jobDate = new Date();
        String workingDir = Paths.get(".").toAbsolutePath().normalize().toString();
        
        System.out.println( "Working dir is: " + workingDir );
        
        BufferedWriter metricsWriter = null;
        try
        {
        	metricsWriter = new BufferedWriter( new FileWriter(metricsFile));
        	
        	metricsWriter.write( "Job run at: " + jobDate.toString() + "\n");
        	metricsWriter.write( "Training model took " + ( elapsedPhase2 - elapsedPhase1) + " seconds.\n");
        	metricsWriter.write( "Total elapsed time: " + elapsedPhase2 + "\n" );
        	metricsWriter.write( "Working dir: " + workingDir + "\n" );
        	metricsWriter.write( "\n" );
        	metricsWriter.flush();
        }
        catch( Exception e )
        {
        	e.printStackTrace();
        }
        finally 
        {
        	if( metricsWriter != null )
        	{
        		metricsWriter.close();
        	}
        }
        
        /* delete any existing model if there is one */
        String modelFileName = args[1] + "/sparkTrainedNetwork_" + jobDate.toString() + ".zip";
        File oldModelFile = new File( modelFileName );
        if( oldModelFile.exists())
        {
        	oldModelFile.delete();
        	oldModelFile = null;
        }
        
        ModelSerializer.writeModel(trainedNetwork, new File( modelFileName), false);
        
        dbConn = DriverManager.getConnection(url, props);
		PreparedStatement statement2 = dbConn.prepareStatement("update enginejob set end_time = ? where id = ?");
		statement2.setString(1, new Date().toString());
		statement2.setString(2, uuid.toString());
		statement2.executeUpdate();
		
		dbConn.close();
        
		System.out.println( "done" );

	}
}
