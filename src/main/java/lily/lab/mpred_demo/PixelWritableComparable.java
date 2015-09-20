package lily.lab.mpred_demo;

import java.io.DataInput;
import java.io.DataOutput;
import java.io.IOException;
import java.util.Arrays;

import org.apache.hadoop.io.WritableComparable;

public class PixelWritableComparable implements WritableComparable<PixelWritableComparable> {
	private double[] rgba = new double[4];
	
	public void setPixel(double[] pix) {
		if (pix.length != 4) {
			System.err.println("Error arises in calling setPixel function!");
			System.exit(-1);
		}
		System.arraycopy(pix, 0, rgba, 0, pix.length);
	}
	
	public double[] getPixel() {
		return rgba;
	}
	
	public void readFields(DataInput in) throws IOException {
		// Read the data out in the order it is written
		for (int i=0; i<rgba.length; i++)
			rgba[i] = in.readDouble();		
	}
	
	public void write(DataOutput out) throws IOException {
		// Write the data out in the order it is read
		for(int i=0; i<rgba.length; i++)
			out.writeDouble(rgba[i]);
	}
	
	public String toString() {
		StringBuilder sb = new StringBuilder();
		for (int i=0; i<rgba.length-1; i++) {
			sb.append(String.valueOf(rgba[i]));
			sb.append(',');
		}
		sb.append(rgba[rgba.length-1]);
		
		return sb.toString();
	}
	
	public int compareTo(PixelWritableComparable o) {
		double[] thisValue = this.rgba;
		double[] thatValue = o.rgba;
		
		for(int i=0; i<thisValue.length; i++) {
			if(thisValue[i] != thatValue[i])
				return ((Double)thisValue[i]).compareTo((Double)thatValue[i]);
		}
		
		return 0;
	}
	
	public int hashCode() {
		return Arrays.hashCode(rgba);
	}
}