package lily.lab.mpred_demo;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.filecache.DistributedCache;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;

public class KMeansHadoop	 
{
    public static void main( String[] args ) throws Exception
    {
    	Configuration conf = new Configuration();
        if (args.length != 4) {
        	System.err.println("Usage: KMeansHadoop <input path> "
        			+ "<output path> <centroid path> <numClusters>");
        	System.exit(2);
        }
    	    	
    	Job job = new Job(conf, "K-Means Image Compression");
    	job.setJarByClass(KMeansHadoop.class);
    	
    	job.setMapperClass(KMeansMapper.class);
    	job.setReducerClass(KMeansReducer.class);
    	
    	FileInputFormat.addInputPath(job, new Path(args[0]));
    	FileOutputFormat.setOutputPath(job, new Path(args[1]));
    	DistributedCache.addCacheFile((new Path(args[2])).toUri(), job.getConfiguration());
    	job.getConfiguration().set("numClusters", args[3]);
    	job.setNumReduceTasks(Integer.parseInt(args[3]));
    	
    	job.setMapOutputKeyClass(PixelWritableComparable.class);
    	job.setMapOutputValueClass(PixIndexTuple.class);
    	job.setOutputKeyClass(PixelWritableComparable.class);
    	job.setOutputValueClass(Text.class);
    	 	
    	System.exit(job.waitForCompletion(true)? 0 : 1);   	
    }
}