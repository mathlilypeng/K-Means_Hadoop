package lily.lab.mpred_demo;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;

import org.apache.hadoop.mapreduce.Mapper;
import org.apache.hadoop.filecache.DistributedCache;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.Text;

public class KMeansMapper extends Mapper<LongWritable, Text, PixelWritableComparable, PixIndexTuple> {
    double[][] mCenters;
    private PixelWritableComparable outKey = new PixelWritableComparable();
    private PixIndexTuple outValue = new PixIndexTuple();
		
    public void setup(Context context) throws IOException, InterruptedException {
	    int k = Integer.parseInt(context.getConfiguration().get("numClusters"));
	    mCenters = new double[k][4];		
        try {
            Path[] files = DistributedCache.getLocalCacheFiles(context.getConfiguration());
            int cnt = 0;
	        for (Path p : files) {
	            String line;
                BufferedReader rdr = new BufferedReader(new FileReader(p.toString()));
                try {
                    while ((line = rdr.readLine()) != null) {
                        String[] center_rgba_str = line.split("\t")[0].split(",");
                        if (center_rgba_str.length != 4) {
                            System.err.println("The rgba value of centroids is not valid!");
                            System.exit(2);
                        }					    
                        double[] center_rgba = new double[center_rgba_str.length];
                        for (int i=0; i<center_rgba_str.length; i++) {
		                    center_rgba[i] = Double.parseDouble(center_rgba_str[i]);
	                    }
                        mCenters[cnt] = center_rgba;
                        cnt++;
                    }
                } finally {
				    rdr.close();
			    }
		    }
		} catch (IOException e) {
			System.err.println("Exception Reading DistributedCache: " + e);
		}
	}
	
	@Override 
	public void map(LongWritable key, Text value, Context context) throws IOException, InterruptedException {
        String[] line = value.toString().split(",");
        if (line.length != 5) {
            System.err.println("Invalid input data format!");
            System.exit(2);
        }
        int id = Integer.parseInt(line[0]);
        double[] rgba = new double[line.length-1];
        for(int i=0; i<line.length-1; i++) {
            rgba[i] = Double.parseDouble(line[i+1]);
        }
        outValue.setPixel(rgba);
        outValue.setID(id);
        int centerIndex = 0;
        double dist = Double.POSITIVE_INFINITY;
        for (int i=0; i<mCenters.length; i++) {
            double tmp = computeDist(rgba, mCenters[i]);
            if (tmp < dist) {
                dist = tmp;
                centerIndex = i;
            }
        }
        outKey.setPixel(mCenters[centerIndex]);
        context.write(outKey, outValue);
    }
    
    private double computeDist(double[] point, double[] center) {
        double res = 0.0;
        for(int i=0; i<point.length; i++) {
            res += Math.pow(point[i]-center[i], 2);
        }
        return res;
    }		
}
