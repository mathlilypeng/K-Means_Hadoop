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

public class KMeansDrive {
    // set root directory on HDFS test test test
    private final static String ROOT = "/user/myproject/";

    public static void main( String[] args ) throws Exception {
        if (args.length != 4) {
        	System.err.println("Usage: KMeansHadoop <input path> <output path> <centroid path> <numClusters>");
        	System.exit(2);
        }
	    // set number of iterations and value of tolerance for kmeans iterative algorithm
        int numIter = 10;
        double tol = 1e-3;
        
	    // start iterations
        int iter = 0;
        double dist = 1.0;
        int jobStatus = 0;
        String input = ROOT+args[0];
        String output = ROOT+args[1]+"/" + iter;
        String centroid = ROOT+args[2];
        String numClusters = args[3];
        Configuration conf = new Configuration();
        while(iter<numIter && dist>tol && jobStatus == 0) {
            // start a new map reduce job for each iteration
            conf.set("iteration.depth", iter + "");
    	    Job job = new Job(conf, "K-Means Pixel Clustering " + iter);

            // distribute center pixels to distributed cache
    	    Path centroidPath = new Path(centroid);
    	    DistributedCache.addCacheFile(centroidPath.toUri(), job.getConfiguration());
            // set driver class, mapper class and reducer class
            job.setJarByClass(KMeansDrive.class);
    	    job.setMapperClass(KMeansMapper.class);
    	    job.setReducerClass(KMeansReducer.class);
            // set input path and output path for the current loop
    	    FileInputFormat.addInputPath(job, new Path(input));
    	    FileOutputFormat.setOutputPath(job, new Path(output));
            job.setOutputKeyClass(PixelWritableComparable.class);
            job.setMapOutputValueClass(PixelWritableComparable.class);
            job.setOutputValueClass(Text.class);
            // set number of reducers to equal to number of clusters of kmeans
    	    job.setNumReduceTasks(Integer.parseInt(numClusters));
            // set number of clusters for kmeans algorithm
            job.getConfiguration().set("numClusters", numClusters);

            // start map-reduce job and wait for it to complete
    	    jobStatus = job.waitForCompletion(true)? 0 : 1;

            // compute centers for next iteration
    	    FileSystem fs = FileSystem.get(conf);
    	    FileStatus[] fss = fs.listStatus(new Path(output));
    	    if (fss == null) {
    	    	System.err.println("No output for iteration " + iter);
    	    	System.exit(2);
    	    }    	    
    	    ArrayList<ArrayList<Double>> nextCenters = new ArrayList<ArrayList<Double>>();
    	    for (FileStatus status : fss) {
    	    	if (status.getPath().getName().startsWith("_")) {
    	    		continue;
                }
                Path path = status.getPath();
                BufferedReader br=new BufferedReader(new InputStreamReader(fs.open(path)));
                String line;
                try {
                    while ((line = br.readLine()) != null) {
                        String[] rgbaValues = line.split("\t")[0].split(",");
                        // each array is a RGBR value which has a length of 4
                        if (rgbaValues.length != 4) {
                            System.err.println("Invalid rgba value!");
                            System.exit(2);
                        }
                        ArrayList<Double> center = new ArrayList<Double>();
                        for (String rgbaValue : rgbaValues) {
                            center.add(Double.parseDouble(rgbaValue));
                        }
                        nextCenters.add(center);
                    }
                } finally {
                    br.close();
                }
            }

            // get current centers
            fss = fs.listStatus(new Path(centroid));
            ArrayList<ArrayList<Double>> curCenters = new ArrayList<ArrayList<Double>>();
            for (FileStatus status : fss) {
                if (status.getPath().getName().startsWith("_")) {
                    continue;
                }
                Path path = status.getPath();
                BufferedReader br=new BufferedReader(new InputStreamReader(fs.open(path)));
                String line;
                try {
                    while ((line = br.readLine()) != null) {
                        String[] rgbaValues = line.split("\t")[0].split(",");
                        if (rgbaValues.length != 4) {
                            System.err.println("The rgba value of centroids is not valid!");
                            System.exit(2);
                        }
                        ArrayList<Double> center = new ArrayList<Double>();
                        for (String rgbaValue : rgbaValues) {
                            center.add(Double.parseDouble(rgbaValue));
                        }
                        curCenters.add(center);
                    }
                } finally {
                    br.close();
                }
            }
            if (curCenters.size() != nextCenters.size()) {
                System.err.println("Incorrect number of centers!");
                System.exit(2);
            }

            // compute distance between current centers and next centers
            double nextDist = 0.0;
            for (int i=0; i<curCenters.size(); i++) {
                ArrayList<Double> curRgba = curCenters.get(i);
                ArrayList<Double> nextRgba = nextCenters.get(i);
                for (int j=0; j<curRgba.size(); j++) {
                    nextDist += Math.pow(curRgba.get(j)-nextRgba.get(j), 2.0);
                }
            }

            // update status
            iter++;
            dist = Math.sqrt(nextDist);
            centroid = output;
            output = ROOT+args[1]+"/" + iter;
        }
    }
}