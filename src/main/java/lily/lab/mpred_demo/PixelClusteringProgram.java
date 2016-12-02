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
import org.apache.hadoop.io.Writable;
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
import java.util.List;
import java.util.Random;

/**
 * The PixelClusteringProgram class implements k-means clustering
 * in mapreduce paradigm on HDFS to cluster pixel RGB values.
 * The clustering process consists of two steps in each iteration:
 * 1. map step: find closest center pixel for each input pixel point
 * 2. reduce step: update centers according to new clusters
 * The final output would be pixel IDs in k clusters and center of each cluster
 */
public class PixelClusteringProgram {

    public static final Log log = LogFactory.getLog(PixelClusteringProgram.class);

    // number of iterations and value of tolerance of kmeans algorithm
    private final static int NUM_ITERS = 20;
    private final static double TOL = 1e-3;

    private Configuration conf = new Configuration();

    /**
     * Construct a new PixelClusteringProgram using a default Configuration
     * */
    public PixelClusteringProgram() {
        this.conf = new Configuration();
    }

    /**
     * Data type class for centers
     */
    private static class CenterPixel implements WritableComparable<CenterPixel> {
        private double[] rgb = new double[3]; // pixel rgb value

        public void setRgb(@Nonnull double[] rgb) {
            assert (rgb.length == 3);
            System.arraycopy(rgb, 0, this.rgb, 0, 3);
        }

        public double[] getRgb() {
            return this.rgb;
        }

        public void readFields(DataInput in) throws IOException {
            for (int i=0; i<3; i++) {
                this.rgb[i] = in.readDouble();
            }
        }

        public void write(DataOutput out) throws IOException {
            for (double val : rgb) {
                out.writeDouble(val);
            }
        }

        public String toString() {
            StringBuilder sb = new StringBuilder();
            for (int i=0; i<2; i++) {
                sb.append(rgb[i]);
                sb.append(',');
            }
            sb.append(rgb[2]);
            return sb.toString();
        }

        public int compareTo(@Nonnull CenterPixel o) {
            double[] thisRgb = this.rgb;
            double[] thatRgb = o.getRgb();
            for(int i=0; i<3; i++) {
                if(thisRgb[i] != thatRgb[i]) {
                    return ((Double)thisRgb[i]).compareTo((Double)thatRgb[i]);
                }
            }
            return 0;
        }

        public int hashCode() {
            return Arrays.hashCode(rgb);
        }
    }

    /**
     * Data type class for aggregation of pixel points in a group
     */
    private static class SumCountTuple implements Writable {
        private double[] rgbSum = new double[3]; // sum of rgb values in this group
        private long count = 0; // count of points in this group
        private List<Long> idList = new ArrayList<Long>(); // list of all pixel ids in this group

        public void setRgbSum(@Nonnull double[] rgb) {
            assert (rgb.length == 3);
            System.arraycopy(rgb, 0, this.rgbSum, 0, 3);
        }

        public double[] getRgbSum() {
            return this.rgbSum;
        }

        public void setCount(long count) {
            this.count = count;
        }

        public long getCount() {
            return this.count;
        }

        public void setIdList(List<Long> ids) {
            this.idList = ids;
        }

        public List<Long> getIdList() {
            return this.idList;
        }

        public void concatIds(List<Long> newIds) {
            this.idList.addAll(newIds);
        }

        public void cleanupIdList() {
            this.idList.clear();
        }

        public void readFields(DataInput in) throws IOException {
            for (int i=0; i<3; i++) {
                rgbSum[i] = in.readDouble();
            }
            this.count = in.readLong();

            this.idList.clear();
            while (true) {
                try {
                    this.idList.add(in.readLong());
                } catch (IOException e) {
                    break;
                }
            }
        }

        public void write(DataOutput out) throws IOException {
            for (double val : rgbSum) {
                out.writeDouble(val);
            }
            out.writeLong(this.count);
            for (Long id : this.idList) {
                out.writeLong(id);
            }
        }

        public String toString() {
            StringBuilder sb = new StringBuilder();
            for (int i=0; i<3; i++) {
                sb.append(this.rgbSum[i]);
                sb.append(',');
            }
            sb.append(this.count);
            sb.append(',');
            for (int i=0; i<this.idList.size(); i++) {
                sb.append(this.idList.get(i));
                if (i<this.idList.size()-1) {
                    sb.append(',');
                }
            }
            return sb.toString();
        }
    }

    /**
     * Mapper that takes in a stream of pisxel ids with its RGB values,
     * computes for each pixel point its closest center,
     * and emit a key-value pair where key is center found for the pixel point,
     * and value is info data of this pixel point
     */
    private static class KMeansMapper
            extends Mapper<LongWritable, Text, CenterPixel, SumCountTuple> {
        private List<double[]> centersRgb;
        private CenterPixel outKey = new CenterPixel();
        private SumCountTuple outValue = new SumCountTuple();

        /**
         * This function loads centers from distributed cache before mapping process
         *
         * @param context context of mapreduce job
         * @throws IOException
         * @throws InterruptedException
         */
        @Override
        public void setup(Context context) throws IOException, InterruptedException {
            int k = Integer.parseInt(context.getConfiguration().get("numClusters"));
            // initialize centers for each mapper
            this.centersRgb = new ArrayList<double[]>();

            Path[] files = DistributedCache.getLocalCacheFiles(context.getConfiguration());
            int i = 0;
            for (Path p : files) {
                String line;
                BufferedReader rdr = new BufferedReader(new FileReader(p.toString()));
                try {
                    while ((line = rdr.readLine()) != null) {
                        String[] centerStrArray = line.split("\t")[0].split(",");
                        // center value format: Rvalue,Gvalue,Bvalue
                        if (centerStrArray.length < 3) {
                            throw new RuntimeException("Invalid center value!");
                        }
                        double[] rgb = new double[3];
                        for (int j=0; j<3; j++) {
                            rgb[j] = Double.parseDouble(centerStrArray[j]);
                        }

                        this.centersRgb.add(rgb);
                    }
                } finally {
                    rdr.close();
                }
            }
            // assert number of centers equals to number of clusters set for the kmeans clustering
            assert (this.centersRgb.size()==k);
        }

        /**
         * This function implements map step. It will compare each input pixel RGB value to RGB of centers and assign
         * a closest center to this pixel point
         *
         * @param key Key
         * @param value RGB value with its id number, sperated by comma. e.g.: 1,244,102,19
         * @param context Context of this job
         * @throws IOException
         * @throws InterruptedException
         */
        @Override
        public void map(LongWritable key, Text value, Context context) throws IOException, InterruptedException {
            //input format: id,R,G,B
            String[] line = value.toString().split(",");
            if (line.length < 4) {
                throw new RuntimeException("Invalid input data format!");
            }
            long id = Long.parseLong(line[0]);
            double[] rgb = new double[3];
            for(int i=0; i<3; i++) {
                rgb[i] = Double.parseDouble(line[i+1]);
            }

            // set mapper's output value
            this.outValue.setRgbSum(rgb);
            this.outValue.setCount(1);
            List<Long> idList = new ArrayList<Long>();
            idList.add(id);
            this.outValue.setIdList(idList);

            // compare distances of the pixel to all centers and find its closest center pixel
            int centerIndex = 0;
            double minDist = Double.POSITIVE_INFINITY;
            for (int i=0; i<this.centersRgb.size(); i++) {
                double curDist = computeDist(rgb, this.centersRgb.get(i));
                if (curDist < minDist) {
                    minDist = curDist;
                    centerIndex = i;
                }
            }
            // assign the pixel to cluster of its closest center
            this.outKey.setRgb(this.centersRgb.get(centerIndex));
            context.write(this.outKey, this.outValue);
        }

        /**
         * Compute Euclidean distance of two RGB values
         *
         * @param a RGB value 1
         * @param b RGB value 2
         * @return Euclidean distance of two RGB values
         */
        private double computeDist(double[] a, double[] b) {
            double res = 0.0;
            for(int i=0; i<3; i++) {
                res += Math.pow(a[i]-b[i], 2);
            }
            return Math.sqrt(res);
        }
    }

    /**
     * Combiner that aggregate output from Mapper
     */
    private static class KMeansCombiner
            extends Reducer<CenterPixel, SumCountTuple, CenterPixel, SumCountTuple> {
        /**
         * This function aggregates output from Mapper. It will compute sum of RGB values
         * on each RGB dimension as well as count of pixel points over
         * each group grouped by centers
         *
         * @param key pixel instance
         * @param values values
         * @param context context of job
         * @throws IOException
         * @throws InterruptedException
         */
        @Override
        public void reduce(CenterPixel key, Iterable<SumCountTuple> values, Context context)
                           throws IOException, InterruptedException {
            SumCountTuple outValue = new SumCountTuple();
            long count = 0;
            double[] rgbSum = new double[3];
            List<Long> idList = new ArrayList<Long>();
            for (SumCountTuple value : values) {
                for (int i=0; i<3; i++) {
                    rgbSum[i] += value.getRgbSum()[i];
                }
                count += value.getCount();
                idList.addAll(value.getIdList());
            }
            outValue.setRgbSum(rgbSum);
            outValue.setCount(count);
            outValue.setIdList(idList);

            context.write(key, outValue);
        }
    }

    /**
     * Reducer that computes new centers of clusters by computing average RGB values in each cluster
     */
    private static class KMeansReducer
            extends Reducer<CenterPixel, SumCountTuple, CenterPixel, Text> {
        private long cnt = 0;
        private double[] newCenter = new double[3];
        private List<Long> idList = new ArrayList<Long>();
        private CenterPixel outKey = new CenterPixel();
        private Text outValue = new Text();

        /**
         * This function aggregates pixel points collected for each center. It will comput sum of RGB values
         * and count of pixel points in this cluster. If no pixel points
         * in that cluster then it will randomly generate one center.
         *
         * @param key center pixel instance
         * @param values aggregation of pixel points
         * @param context context of job
         * @throws IOException
         * @throws InterruptedException
         */
        @Override
        public void reduce(CenterPixel key, Iterable<SumCountTuple> values, Context context)
                throws IOException, InterruptedException {
            for(SumCountTuple value : values) {
                // compute sum of RGB values in each cluster
                double[] rgb = value.getRgbSum();
                for (int i=0; i<3; i++) {
                    this.newCenter[i] += rgb[i];
                }
                // compute total number of RGB values in each cluster
                cnt += value.getCount();
                // collect ids of RGB points of each cluster
                idList.addAll(value.getIdList());
            }
        }

        /***
         * This function will compute average RGB values in a cluster as the new center for this cluster and
         * output result to HDFS
         *
         * @param context context of job
         * @throws IOException
         * @throws InterruptedException
         */
        protected void cleanup(Context context) throws IOException, InterruptedException {
            StringBuilder stringBuilder = new StringBuilder();
            Random random = new Random();
            if (cnt > 0) {
                for (int i=0; i<3; i++) {
                    this.newCenter[i] /= (double)cnt;
                }
            } else {
                // if the center has no points assigned to it, then generate a new center randomly.
                for(int i=0; i<3; i++) {
                    this.newCenter[i] = random.nextInt(255);
                }
            }

            this.outKey.setRgb(this.newCenter);
            // collect ids of pixel points belonging to this cluster
            for (long id : this.idList) {
                stringBuilder.append(id).append(",");
            }
            //delete trailing comma
            stringBuilder.deleteCharAt(stringBuilder.length() - 1);
            this.outValue.set(stringBuilder.toString());
            context.write(this.outKey, this.outValue);
        }
    }

    /**
     * Given paths to two groups of centers, compute their distance in frobenius norm
     * @param nextCentersPath Path containing centers of next iteration
     * @param curCentersPath Path of centers of current iteration
     * @param k number of clusters
     * @return distance of two group of centers
     * @throws IOException
     */
    private double computeCentersDist(Path nextCentersPath, Path curCentersPath, int k) throws IOException {
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

            BufferedReader br = new BufferedReader(new InputStreamReader(fs.open(path)));

            String line;
            try {
                while ((line = br.readLine()) != null) {
                    String[] rgbStr = line.split("\t")[0].split(",");
                    assert (rgbStr.length == 3);
                    double[] rgb = new double[3];
                    for (int i=0; i<3; i++) {
                        rgb[i] = Double.parseDouble(rgbStr[i]);
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
     * Create a Job that will run K-means in one iteration
     *
     * @param inputPath The path containing the input pixels
     * @param outputPath The path for the result of this job
     * @param centersPath The path containing center pixels which need to be distributed to distributed cache
     * @param k number of clusters set for kmeans clustering
     * @param cnt count of iterations
     * @return a Job instance
     * @throws IOException
     * @throws InterruptedException
     * @throws ClassNotFoundException
     */
    private Job submitJob (Path inputPath, Path outputPath, Path centersPath, int k, int cnt)
            throws IOException, InterruptedException, ClassNotFoundException {
        Job job = new Job(this.conf, "K-Means Pixel Clustering on iteration " + cnt);
        // set driver class, mapper class, combiner class, and reducer class
        job.setJarByClass(KMeansDrive.class);
        job.setMapperClass(KMeansMapper.class);
        job.setCombinerClass(KMeansCombiner.class);
        job.setReducerClass(KMeansReducer.class);

        FileInputFormat.addInputPath(job, inputPath);
        FileOutputFormat.setOutputPath(job, outputPath);
        job.setOutputKeyClass(CenterPixel.class);
        job.setMapOutputValueClass(SumCountTuple.class);
        job.setOutputValueClass(Text.class);
        job.setNumReduceTasks(k);
        job.getConfiguration().set("numClusters", String.valueOf(k));
        job.getConfiguration().set("iterationCounter", String.valueOf(cnt));

        // distribute centers to distributed cache
        DistributedCache.addCacheFile(centersPath.toUri(), job.getConfiguration());

        //submit job and immediately return, rather than waiting for completion
        log.info("runing iteration " + cnt);
        job.submit();
        return job;
    }


    /**
     * Start the program
     *
     * @param inputDirStr The directory to the input rgb data
     * @param outputDirStr The directory for the results to be output to
     * @param centersDirStr The path for to the centers of kmeans clustering
     * @param numClusters The number of clusters set for kmeans clustering
     * @throws IOException,
     * @throws InterruptedException
     * @throws ClassNotFoundException
     */
    public boolean start(String inputDirStr, String outputDirStr, String centersDirStr, int numClusters)
            throws IOException, InterruptedException, ClassNotFoundException {
        // set initial parameters
        int cnt = 0;
        double dist = 1.0;
        Path inputPath = new Path(inputDirStr);
        Path outputPath = new Path(outputDirStr + "/" + cnt);
        Path centersPath = new Path(centersDirStr);

        while(cnt< NUM_ITERS && dist> TOL ) {
            log.info("start iteration " + cnt);
            Job job = submitJob(inputPath, outputPath, centersPath, numClusters, cnt);
            // wait for job to complete
            while (!job.isComplete()) {
                Thread.sleep(5000);
            }

            if (job.isSuccessful()) {
                // if this iteration succeed then update dist, centersPath, outputPath, and cnt
                dist = computeCentersDist(outputPath, centersPath, numClusters);
                centersPath = outputPath;
                outputPath = new Path(outputDirStr + "/" + (++cnt));
            } else {
                log.info("job failed in iteration " + cnt);
                return false;
            }

            log.info("dist: " + dist);
            log.info("centers path " + centersPath.toString());
            log.info("output path" + outputPath.toString());
        }

        return true;
    }

    /**
     * Main method for running the PixelCLusteringProgram. This takes four parameters:
     * 1. input path: path to files containing RGB values on HDFS,
     *    where each line of input represents a pixel point,
     *    each line is comma seperated formatted in the way: id,Rvalue,Gvalue,Bvalue.
     * 2. output path: path for the output on HDFS
     * 3. centers path: path to files containing k initial centers on HDFS,
     *    where each line represents a center pixel which is formatted in the way: Rvalue,Gvalue,Bvalue
     * 4. number of clusters set for kmeans clustering
     * @param args Command line arguments
     * @throws Exception
     */
    public static void main( String[] args ) throws Exception {
        if (args.length < 4) {
            throw new IllegalArgumentException("Illegal arguments! " +
                    "Usage: KMeansHadoop <input path> <output path> <centers path> <numClusters>");
        }
        String inputDirStr = args[0]; // path to the input files on HDFS
        String outputDirStr = args[1]; // path for the output on HDFS
        String centersDirStr = args[2]; // path to initial centers on HDFS
        int numClusters = Integer.parseInt(args[3]); // number of clusters

        PixelClusteringProgram listener = new PixelClusteringProgram();

        log.info("running K-means clustering on input directory: " + inputDirStr);
        boolean isSuccesful = listener.start(inputDirStr, outputDirStr, centersDirStr, numClusters);

        System.exit(isSuccesful? 0 : 1);
    }
}
