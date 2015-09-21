package lily.lab.mpred_demo;

import java.io.BufferedReader;
import java.io.InputStreamReader;
import java.util.ArrayList;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.filecache.DistributedCache;
import org.apache.hadoop.fs.FileStatus;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;


public class KMeansHadoop	 
{
	public static String ROOT = "/user/myproject/";
	
    public static void main( String[] args ) throws Exception
    {
    	Configuration conf = new Configuration();
        if (args.length != 4) {
        	System.err.println("Usage: KMeansHadoop <input path> "
        			+ "<output path> <centroid path> <numClusters>");
        	System.exit(2);
        }
        
        String input = ROOT+args[0];
        String output = ROOT+args[1]+"/"+System.nanoTime();
        String centroid = ROOT+args[2];
        
        String numClusters = args[3];
        
        int numIter = 10;
        double tol = 1e-3;
        
        int iter = 0;
        double dist = 1.0;
        int jobStatus = 0;
        while(iter<numIter && dist>tol && jobStatus == 0) {
    	    	
    	    Job job = new Job(conf, "K-Means Image Compression");
    	    job.setJarByClass(KMeansHadoop.class);
    	    
    	    Path centroid_path = new Path(centroid);
    	    DistributedCache.addCacheFile(centroid_path.toUri(), job.getConfiguration());
    	    
    	    job.setMapperClass(KMeansMapper.class);
    	    job.setReducerClass(KMeansReducer.class);
    	
    	    FileInputFormat.addInputPath(job, new Path(input));
    	    FileOutputFormat.setOutputPath(job, new Path(output));
    	    
    	    job.getConfiguration().set("numClusters", numClusters);
    	    job.setNumReduceTasks(Integer.parseInt(numClusters));
    	
    	    job.setMapOutputKeyClass(PixelWritableComparable.class);
    	    job.setMapOutputValueClass(PixIndexTuple.class);
    	    job.setOutputKeyClass(PixelWritableComparable.class);
    	    job.setOutputValueClass(Text.class);
    	 	
    	    jobStatus = job.waitForCompletion(true)? 0 : 1;
    	    
    	    // get next centers
    	    FileSystem fs = FileSystem.get(conf);
    	    FileStatus[] fss = fs.listStatus(new Path(output));
    	    if (fss == null) {
    	    	System.err.println("No output for current job.");
    	    	System.exit(-1);
    	    }
    	    
    	    ArrayList<ArrayList<Double>> next_centers = new ArrayList<ArrayList<Double>>();
    	    
    	    for (FileStatus status : fss) {
    	    	if (status.getPath().getName().startsWith("_"))
    	    		continue;
    	    	
    	    	Path path = status.getPath();
    	    	BufferedReader br=new BufferedReader(new InputStreamReader(fs.open(path)));

    	        String line;
			    try {
				    while ((line = br.readLine()) != null) {
					    String[] center_rgba_str = line.split("\t")[0].split(",");
					    if (center_rgba_str.length != 4) {
					    	System.err.println("The rgba value of centroids is not valid!");
					    	System.exit(-1);
					    }
					    
					    ArrayList<Double> center = new ArrayList<Double>();
					    for (int i=0; i<center_rgba_str.length; i++) {
					    	center.add(Double.parseDouble(center_rgba_str[i]));
					    }
					    
					    next_centers.add(center);
				    }
			    } finally {
				    br.close();
			    }
    	    }
    	    
    	    // get previous centers
    	    fss = fs.listStatus(new Path(centroid));
    	    ArrayList<ArrayList<Double>> cur_centers = new ArrayList<ArrayList<Double>>();
    	    
    	    for (FileStatus status : fss) {
    	    	if (status.getPath().getName().startsWith("_"))
    	    		continue;
    	    		
    	    	Path path = status.getPath();
    	    	BufferedReader br=new BufferedReader(new InputStreamReader(fs.open(path)));

    	        String line;
			    try {
				    while ((line = br.readLine()) != null) {
					    String[] center_rgba_str = line.split("\t")[0].split(",");
					    if (center_rgba_str.length != 4) {
					    	System.err.println("The rgba value of centroids is not valid!");
					    	System.exit(-1);
					    }
					    
					    ArrayList<Double> center = new ArrayList<Double>();
					    for (int i=0; i<center_rgba_str.length; i++) {
					    	center.add(Double.parseDouble(center_rgba_str[i]));
					    }
					    
					    cur_centers.add(center);
				    }
			    } finally {
				    br.close();
			    }
    	    }
    	    
    	    if (cur_centers.size() != next_centers.size()) {
    	    	System.err.println("The size of new centers is not correct.");
    	    	System.exit(-1);
    	    }
    	    
    	    // compute distance of current centers and previous centers
    	    double next_dist = 0.0;
    	    for (int i=0; i<cur_centers.size(); i++) {
    	    	ArrayList<Double> cur_rgba = cur_centers.get(i);
    	    	ArrayList<Double> next_rgba = next_centers.get(i);
    	    	for (int j=0; j<cur_rgba.size(); j++) {
    	    		next_dist += Math.pow(cur_rgba.get(j)-next_rgba.get(j), 2.0);
    	    	}
    	    }
    	    dist = Math.sqrt(next_dist);
    	    
    	    // update status    	    
    	    centroid = output;
    	    output = ROOT+args[1]+"/"+System.nanoTime(); 
    	    iter++;
        }
    }
}