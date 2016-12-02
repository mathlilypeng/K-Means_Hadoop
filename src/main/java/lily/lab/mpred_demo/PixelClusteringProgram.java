package lily.lab.mpred_demo;

import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;
import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.filecache.DistributedCache;
import org.apache.hadoop.fs.FileStatus;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.io.WritableComparable;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.hadoop.mapreduce.Reducer;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;

import javax.annotation.Nonnull;
import java.io.BufferedReader;
import java.io.DataInput;
import java.io.DataOutput;
import java.io.FileReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.util.ArrayList;
import java.util.Arrays;

/**
 * The PixelClusteringProgram class implements k-means clustering
 * in mapreduce paradigm on HDFS to cluster pixels.
 * The clustering process consists of two steps in each iteration:
 * 1. map step: compute closest center for each pixel point
 * 2. reduce step: update centers according to new clusters
 * final output would be pixels devided into k clusters and center of each cluster
 */
public class PixelClusteringProgram {

    public static final Log log = LogFactory.getLog(PixelWritableComparable.class);

    // root directory on HDFS
    private final static String ROOT = "/user/myproject/";
    // number of iterations and value of tolerance of kmeans algorithm
    private final static int NUM_ITERS = 20;
    private final static double TOL = 1e-3;

    private Configuration conf = new Configuration();

    /** Construct a new PixelClusteringProgram using a default Configuration */
    public PixelClusteringProgram() {
        this.conf = new Configuration();
    }

    /**
     * Pixel class
     */
    private static class PixelWritableComparable implements WritableComparable<PixelWritableComparable> {
        // id of pixel, default to -1
        private long id = -1;
        // rgb value of pixel
        private double[] rgb = new double[3];

        public void setPixel(@Nonnull double[] pixel) throws RuntimeException {
            if (pixel.length != 3) {
                throw new RuntimeException("Invalid pixel value!");
            }
            System.arraycopy(pixel, 0, rgb, 0, 3);
        }

        public double[] getPixel() {
            return rgb;
        }

        public void setID(int id) {
            this.id = id;
        }

        public long getID() {
            return id;
        }

        public void readFields(DataInput in) throws IOException {
            // Read the data out in the order it is written
            for (int i=0; i<rgb.length; i++) {
                rgb[i] = in.readDouble();
            }
        }

        public void write(DataOutput out) throws IOException {
            // Write the data out in the order it is read
            for (double aRgb : rgb) {
                out.writeDouble(aRgb);
            }
        }

        public String toString() {
            StringBuilder sb = new StringBuilder();
            sb.append(id);
            sb.append(',');
            for (int i=0; i<rgb.length-1; i++) {
                sb.append(String.valueOf(rgb[i]));
                sb.append(',');
            }
            sb.append(rgb[rgb.length-1]);
            return sb.toString();
        }

        public int compareTo(@Nonnull PixelWritableComparable o) {
            double[] thisValue = this.rgb;
            double[] thatValue = o.rgb;
            for(int i=0; i<thisValue.length; i++) {
                if(thisValue[i] != thatValue[i]) {
                    return ((Double)thisValue[i]).compareTo((Double)thatValue[i]);
                }
            }
            return 0;
        }

        public int hashCode() {
            return Arrays.hashCode(rgb);
        }
    }

    /**
     * Mapper that takes in a stream of RGB values, computes for each RGB value its closest center,
     * and outputs each RGB value with its closest center
     */
    private static class KMeansMapper
            extends Mapper<LongWritable, Text, PixelWritableComparable, PixelWritableComparable> {
        private double[][] centers;
        private PixelWritableComparable outKey = new PixelWritableComparable();
        private PixelWritableComparable outValue = new PixelWritableComparable();

        @Override
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

    /**
     * Reducer that computes new center of each cluster
     */
    private static class KMeansReducer
            extends Reducer<PixelWritableComparable, PixelWritableComparable, PixelWritableComparable, Text> {
        private StringBuilder sb = new StringBuilder();
        private long cnt = 0;
        private double[] newCenter = new double[4];
        private PixelWritableComparable outKey = new PixelWritableComparable();
        private Text outValue = new Text();

        @Override
        public void reduce(PixelWritableComparable key, Iterable<PixelWritableComparable> values,
                           Context context) throws IOException, InterruptedException {
            for(PixelWritableComparable value : values) {
                this.cnt++;
                long id = value.getID();
                double[] rgba = value.getPixel();
                this.sb.append(String.valueOf(id));
                this.sb.append(',');
                for (int i=0; i<4; i++) {
                    this.newCenter[i] += rgba[i];
                }
            }
        }

        protected void cleanup(Context context) throws IOException, InterruptedException {
            if (cnt > 0) {
                for (int i=0; i<this.newCenter.length; i++) {
                    this.newCenter[i] /= (double)cnt;
                }
                this.sb.deleteCharAt(sb.length()-1);
            } else {
                for(int i=0; i<newCenter.length; i++) {
                    this.newCenter[i] = Math.random();
                }
            }
            this.outKey.setPixel(this.newCenter);
            this.outValue.set(this.sb.toString());
            context.write(this.outKey, this.outValue);
        }
    }

    private static Job submitJob (Configuration conf, Path inputPath, Path outputPath, Path centersPath, int k, int cnt)
            throws Exception {
        Job job = new Job(conf, "K-Means Pixel Clustering " + cnt);
        // set driver class, mapper class and reducer class
        job.setJarByClass(KMeansDrive.class);
        job.setMapperClass(KMeansMapper.class);
        job.setReducerClass(KMeansReducer.class);

        FileInputFormat.addInputPath(job, inputPath);
        FileOutputFormat.setOutputPath(job, outputPath);
        job.setOutputKeyClass(PixelWritableComparable.class);
        job.setMapOutputValueClass(PixelWritableComparable.class);
        job.setOutputValueClass(Text.class);
        job.setNumReduceTasks(k);
        job.getConfiguration().set("numClusters", String.valueOf(k));

        // distribute centers to distributed cache
        DistributedCache.addCacheFile(centersPath.toUri(), job.getConfiguration());

        //submit job and immediately return, rather than waiting for completion
        log.info("Runing iteration " + cnt);
        job.submit();
        return job;
    }

    /**
     * Given paths to two group of centers, compute their distance in frobenius norm
     * @param nextCentersPath Path containing centers of next iteration
     * @param curCentersPath Path of centers of current iteration
     * @param k number of clusters
     * @return distance of two group of centers
     * @throws IOException
     */
    private double computeDist(Path nextCentersPath, Path curCentersPath, int k) throws IOException {
        double dist = 0;
        ArrayList<double[]> nextCenters = getCenters(nextCentersPath);
        ArrayList<double[]> curCenters = getCenters(curCentersPath);

        if (nextCenters.size()!=k || curCenters.size() != k) {
            throw new RuntimeException("Incorrect number of centers!");
        }

        for (int i=0; i<k; i++) {
            double[] curRgb = curCenters.get(i);
            double[] nextRgb = nextCenters.get(i);
            for (int j=0; j<3; j++) {
                dist += Math.pow(curRgb[j]-nextRgb[j], 2.0);
            }
        }

        return Math.sqrt(dist);
    }

    /**
     * Read centers from HDFS
     * @param centersPath Path to centers on HDFS
     * @return An ArrayList of rgb values
     * @throws IOException
     */
    private ArrayList<double[]> getCenters(Path centersPath) throws IOException {
        FileSystem fs = FileSystem.get(conf);
        FileStatus[] fss = fs.listStatus(centersPath);
        if (fss == null) {
            throw new RuntimeException ("Empty directory!");
        }

        ArrayList<double[]> centers = new ArrayList<double[]>();
        for (FileStatus status : fss) {
            if (status.getPath().getName().startsWith("_")) {
                continue;
            }
            Path path = status.getPath();
            BufferedReader br= null;
            try {
                br = new BufferedReader(new InputStreamReader(fs.open(path)));
            } catch (IOException e) {
                e.printStackTrace();
            }
            String line;

            assert (br != null);
            try {
                while ((line = br.readLine()) != null) {
                    String[] rgbStr = line.split("\t")[0].split(",");

                    assert (rgbStr.length == 4);

                    double[] rgb = new double[3];
                    for (int i=0; i<3; i++) {
                        rgb[i] = Double.parseDouble(rgbStr[i+1]);
                    }
                    centers.add(rgb);
                }
            } finally {
                br.close();
            }
        }

        return centers;
    }

    /**
     * Start the program
     *
     * @param inputDirStr The directory to the input rgb data
     * @param outputDirStr The directory for the results to be output to
     * @param centersDirStr The path for to the centers of kmeans clustering
     * @param numClusters The number of clusters set for kmeans clustering
     * @throws IOException
     */
    public boolean start(String inputDirStr, String outputDirStr, String centersDirStr, int numClusters)
            throws Exception {

        // set initial parameters
        int cnt = 0;
        double dist = 1.0;
        Path inputPath = new Path(inputDirStr);
        Path outputPath = new Path(outputDirStr + "/" + cnt);
        Path centersPath = new Path(centersDirStr);
        while(cnt< NUM_ITERS && dist> TOL ) {
            Job job = submitJob(this.conf, inputPath, outputPath, centersPath, numClusters, cnt);
            // wait for job to complete
            while (!job.isComplete()) {
                try {
                    Thread.sleep(5000);
                } catch (InterruptedException e) {
                    log.info("Job interrupted.");
                    return false;
                }
            }

            if (job.isSuccessful()) {
                dist = computeDist(outputPath, centersPath, numClusters);
                centersPath = outputPath;
                outputPath = new Path(outputDirStr + "/" + (++cnt));
            } else {
                log.info("Job failed in iteration " + cnt);
                return false;
            }
        }

        return true;
    }

    /**
     * Main method for running the PixelCLusteringProgram. This taks four parameters:
     * 1. path to the directory containing RGB values
     * 2. path to the output directory
     * 3. path to the directory containing k initial centers
     * 4. number of clusters set for kmeans clustering
     * @param args Command line arguments
     * @throws Exception
     */
    public static void main( String[] args ) throws Exception {
        if (args.length < 4) {
            throw new IllegalArgumentException("Illegal arguments! " +
                    "Usage: KMeansHadoop <input path> <output path> <centers path> <numClusters>");
        }
        String inputDirStr = ROOT+args[0];
        String outputDirStr = ROOT+args[1];
        String centersDirStr = ROOT+args[2];
        int numClusters = Integer.parseInt(args[3]);

        PixelClusteringProgram listener = new PixelClusteringProgram();

        log.info("Running K-means clustering...");
        int exitCode = listener.start(inputDirStr, outputDirStr, centersDirStr, numClusters)? 0 : 1;

        System.exit(exitCode);
    }
}
