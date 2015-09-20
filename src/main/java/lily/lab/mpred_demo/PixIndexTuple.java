package lily.lab.mpred_demo;

import java.io.DataInput;
import java.io.DataOutput;
import java.io.IOException;

import org.apache.hadoop.io.Writable;

public class PixIndexTuple implements Writable {
	private double[] rgba = new double[4];
	private int id;
	
	public void setPixel(double[] pix) {
		if (pix.length != 4) {
			System.err.println("The argument value for setPixel in PixIndexTuple is not valid.");
			System.exit(-1);
		}
		System.arraycopy(pix, 0, rgba, 0, pix.length);
	}
	
	public double[] getPixel() {
		return rgba;
	}
	
	public void setID(int id) {
		this.id = id;
	}
	
	public int getID() {
		return id;
	}
	
	public void readFields(DataInput in) throws IOException {
		// Read the data out in the order it is written
		id = in.readInt();
		for (int i=0; i<rgba.length; i++)
			rgba[i] = in.readDouble();		
	}
	
	public void write(DataOutput out) throws IOException {
		// Write the data out in the order it is read
		out.writeInt(id);
		for(int i=0; i<rgba.length; i++)
			out.writeDouble(rgba[i]);
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
	
}