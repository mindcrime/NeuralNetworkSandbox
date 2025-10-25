package org.fogbeam.dl4j.spark.launcher;

import java.io.IOException;
import java.io.InputStream;
import java.util.Arrays;

import org.apache.spark.launcher.SparkLauncher;


public class SparkLauncherMain1 
{

	public static void main(String[] args) throws Exception
	{
		
		SparkLauncher launcher = new SparkLauncher();
		
		Process sparkJob = launcher
		//.setSparkHome("/usr/hadoop/spark-1.6.3-bin-hadoop2.6")
		  .setSparkHome( args[0])
	    //.setAppResource("/home/prhodes/development/experimental/ai_exp/NeuralNetworkSandbox/java/dl4j/dl4j-spark/target/dl4j-spark-0.0.1-SNAPSHOT.jar")
		  .setAppResource( args[1] )
		//.setMainClass("org.fogbeam.dl4j.spark.ExpMain4")
		  .setMainClass( args[2] )
	    //.setMaster("local[*]")
		  .setMaster( args[3] )
	    //.setDeployMode("client")
		  .setDeployMode( args[4] )
	    // .addAppArgs("foo=bar", "path=/usr/local/opt", "haltandcatchfire=false")
		  .addAppArgs( args[5], args[6], args[7])
		  
		  .launch();
		
		
		byte[] isData = new byte[4096];
		byte[] esData = new byte[4096];
		
		
		InputStream is = sparkJob.getInputStream();
		
		Thread isGobbler = new Thread( new Runnable() {
			public void run() {
				try
				{
					while( is.read(isData) != -1 ) {
						String s = new String(isData).trim();
						System.out.println( s );
						Arrays.fill( isData, (byte)0);
						
					}
				}
				catch( IOException e ) {}
			};
			
		} );
		
		
		
		InputStream es = sparkJob.getErrorStream();
		Thread esGobbler = new Thread( new Runnable() {
			public void run() {
				try
				{
					while( es.read(esData) != -1 ) {
						String s = new String(esData).trim();
						System.out.println( s );
						Arrays.fill( esData, (byte)0);
					}
				}
				catch( IOException e ) {}
			};
			
		} );
		
		isGobbler.start();
		esGobbler.start();
		
		sparkJob.waitFor();
		
		System.out.println( "done" );
		
		System.exit(0);
	}

}
