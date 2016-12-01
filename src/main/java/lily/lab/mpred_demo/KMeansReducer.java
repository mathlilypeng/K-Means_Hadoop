package lily.lab.mpred_demo;

import java.io.IOException;

import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Reducer;

public class KMeansReducer extends Reducer<PixelWritableComparable, PixIndexTuple, PixelWritableComparable, Text> {
    StringBuilder sb = new StringBuilder();
    long cnt = 0;
    double[] newCenter = new double[4];
    PixelWritableComparable outKey = new PixelWritableComparable();
    Text outValue = new Text();
    
    @Override
    public void reduce(PixelWritableComparable key, Iterable<PixIndexTuple> values, 
			Context context) throws IOException, InterruptedException {
        for(PixIndexTuple value : values) {
            cnt++;
            int id = value.getID();
            sb.append(String.valueOf(id));
            sb.append(',');
            double[] rgba = value.getPixel();
            for (int i=0; i<rgba.length; i++) {
                newCenter[i] += rgba[i];
            }
        }
    }
    
    protected void cleanup(Context context) throws IOException, InterruptedException {
        if (cnt > 0) {
            for (int i=0; i<newCenter.length; i++) {
                newCenter[i] /= (double)cnt;
            }
            sb.deleteCharAt(sb.length()-1);
        } else {
            for(int i=0; i<newCenter.length; i++) {
                newCenter[i] = Math.random();
            }
        }
        outKey.setPixel(newCenter);
        outValue.set(sb.toString());
        context.write(outKey, outValue);
    }
}
