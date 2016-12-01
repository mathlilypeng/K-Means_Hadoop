package lily.lab.mpred_demo;

import java.io.IOException;

import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Reducer;

public class KMeansReducer extends Reducer<PixelWritableComparable, PixelWritableComparable, PixelWritableComparable, Text> {
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