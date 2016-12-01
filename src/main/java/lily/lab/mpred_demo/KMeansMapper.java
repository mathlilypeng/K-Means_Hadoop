package lily.lab.mpred_demo;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;

import org.apache.hadoop.mapreduce.Mapper;
import org.apache.hadoop.filecache.DistributedCache;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.Text;

public class KMeansMapper extends Mapper<LongWritable, Text, PixelWritableComparable, PixelWritableComparable> {
    private double[][] centers;
    private PixelWritableComparable outKey = new PixelWritableComparable();
    private PixelWritableComparable outValue = new PixelWritableComparable();
		
    public void setup(Context context) throws IOException, InterruptedException {
	    int k = Integer.parseInt(context.getConfiguration().get("numClusters"));
        // initialize centers for each mapper
	    this.centers = new double[k][4];
        try {
            Path[] files = DistributedCache.getLocalCacheFiles(context.getConfiguration());
            int cnt = 0;
	        for (Path p : files) {
	            String line;
                BufferedReader rdr = new BufferedReader(new FileReader(p.toString()));
                try {
                    while ((line = rdr.readLine()) != null) {
                        String[] pixelStr = line.split("\t")[0].split(",");
                        if (pixelStr.length != 5) {
                            System.err.println("Invalid pixel string!");
                            System.exit(2);
                        }					    
                        double[] centerRGBA = new double[4];
                        for (int i=1; i<5; i++) {
		                    centerRGBA[i] = Double.parseDouble(pixelStr[i]);
	                    }
                        this.centers[cnt] = centerRGBA;
                        cnt++;
                    }
                } finally {
				    rdr.close();
			    }
		    }
		} catch (IOException e) {
			System.err.println("Exception reading distributedCache: " + e);
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
        this.outValue.setPixel(rgba);
        this.outValue.setID(id);

        // compare distances of the pixel to all centers and find its closest center pixel
        int centerIndex = 0;
        double dist = Double.POSITIVE_INFINITY;
        for (int i=0; i<this.centers.length; i++) {
            double tmp = computeDist(rgba, this.centers[i]);
            if (tmp < dist) {
                dist = tmp;
                centerIndex = i;
            }
        }
        // assign the pixel to cluster of its closest center
        this.outKey.setPixel(this.centers[centerIndex]);
        context.write(this.outKey, this.outValue);
    }
    
    private double computeDist(double[] point, double[] center) {
        double res = 0.0;
        for(int i=0; i<point.length; i++) {
            res += Math.pow(point[i]-center[i], 2);
        }
        return res;
    }		
}