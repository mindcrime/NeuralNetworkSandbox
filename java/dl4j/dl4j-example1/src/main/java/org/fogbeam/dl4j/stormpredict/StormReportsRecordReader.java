package org.fogbeam.dl4j.stormpredict;

import java.util.Date;
import java.util.List;

import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.datavec.api.records.reader.impl.csv.CSVRecordReader;
import org.datavec.api.transform.TransformProcess;
import org.datavec.api.transform.schema.Schema;
import org.datavec.api.writable.Writable;
import org.datavec.spark.transform.SparkTransformExecutor;
import org.datavec.spark.transform.misc.StringToWritablesFunction;
import org.datavec.spark.transform.misc.WritablesToStringFunction;

public class StormReportsRecordReader 
{

	public static void main( String[] args )
	{
		
		int numLinesToSkip = 0;
		String delimiter = ",";
		
		String baseDir = "/home/prhodes/development/experimental/ai_exp/NeuralNetworkSandbox/java/dl4j/dl4j-example1/";
		String fileName = "reports.csv";
		String inputPath = baseDir + fileName;
		
		String timeStamp = String.valueOf( new Date().getTime());
		
		String outputPath = baseDir + "output/reports_processed" + timeStamp;
		
		/* Fields are:
		 	datetime,severity,location,county,state,lat,long,comment,type
		 */
		Schema inputDataSchema = new Schema.Builder()
				.addColumnsString( "datetime", "severity", "location", "county", "state" )
				.addColumnsDouble( "lat", "long" )
				.addColumnString( "comment" )
				.addColumnCategorical("type", "TOR", "WIND", "HAIL")
				.build();
				
		
		/* extract lat and long, and transform type to 0,1,2 */
		TransformProcess transform = new TransformProcess.Builder(inputDataSchema)
				.removeColumns("datetime", "severity", "location", "county", "state", "comment")
				.categoricalToInteger("type")
				.build();
		
		
		/* print the before and after schema */
		int numActions = transform.getActionList().size();
		for( int i = 0; i < numActions; i++ )
		{
			System.out.println( "\n\n=============================================" );
			System.out.println(  "Schema after step " + i + "(" + transform.getActionList().get(i) + ")--" );
			System.out.println( transform.getSchemaAfterStep(i));
			
		}
		
		
		SparkConf sparkConf = new SparkConf();
		
		sparkConf.setMaster("local[*]");
		sparkConf.setAppName( "StormReportsRecordReaderTransform" );
		
		JavaSparkContext sparkContext = new JavaSparkContext( sparkConf );
		
		try
		{
			JavaRDD<String> lines = sparkContext.textFile(inputPath);
			JavaRDD<List<Writable>> stormReports = lines.map( new StringToWritablesFunction(new CSVRecordReader()));
			JavaRDD<List<Writable>> processed = SparkTransformExecutor.execute(stormReports, transform);
			JavaRDD<String> toSave = processed.map( new WritablesToStringFunction(",") );
		
			toSave.saveAsTextFile( outputPath );
		}
		finally
		{
			sparkContext.close();
		}
	}
}
