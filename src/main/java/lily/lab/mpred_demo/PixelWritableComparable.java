package lily.lab.mpred_demo;

import java.io.DataInput;
import java.io.DataOutput;
import java.io.IOException;
import java.util.Arrays;

import org.apache.hadoop.io.WritableComparable;

import javax.annotation.Nonnull;

public class PixelWritableComparable implements WritableComparable<PixelWritableComparable> {
    /** id of pixel */
    private long id = -1;
    /** rgba value of pixel */
    private double[] rgba = new double[4];

    public void setPixel(double[] pixel) {
        if (pixel.length != 4) {
            System.err.println("Invalid pixel value!");
            System.exit(2);
        }
        System.arraycopy(pixel, 0, rgba, 0, pixel.length);
    }
    
    public double[] getPixel() {
        return rgba;
    }

    public void setID(int id) {
        this.id = id;
    }

    public long getID() {
        return id;
    }

    public void readFields(DataInput in) throws IOException {
        // Read the data out in the order it is written
        for (int i=0; i<rgba.length; i++) {
            rgba[i] = in.readDouble();
        }
    }
    
    public void write(DataOutput out) throws IOException {
        // Write the data out in the order it is read
        for (double aRgba : rgba) {
            out.writeDouble(aRgba);
        }
    }
    
    public String toString() {
        StringBuilder sb = new StringBuilder();
        sb.append(id);
        sb.append(',');
        for (int i=0; i<rgba.length-1; i++) {
            sb.append(String.valueOf(rgba[i]));
            sb.append(',');
        }
        sb.append(rgba[rgba.length-1]);
        return sb.toString();
    }
    
    public int compareTo(@Nonnull PixelWritableComparable o) {
        double[] thisValue = this.rgba;
        double[] thatValue = o.rgba;
        for(int i=0; i<thisValue.length; i++) {
            if(thisValue[i] != thatValue[i]) {
                return ((Double)thisValue[i]).compareTo((Double)thatValue[i]);
            }
        }
        return 0;
    }
    
    public int hashCode() {
        return Arrays.hashCode(rgba);
    }
}