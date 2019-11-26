//import com.sun.deploy.util.ArrayUtil;

import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.file.*;
//import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.*;

public class Streams {
    /**
     * The 2 first bytes of the binary files are m resp. n dimensions
     * Rel path example:
     *  dir: "files\\streams\\bin\\"
     *  name: "bin_file_test.dat"
     * */
    protected int _m; // m dim of f_data
    protected int _n; // n_dim of f_data
    protected float[] _f_matrix;
    private byte[] _b_matrix;
    protected String _rel_path;

    Streams(){ }

    Streams(String rel_path){
        this._rel_path = rel_path;
        this.read_buff(rel_path);
    }

    public float byteBuffToFloatBuff(byte[] array) {
        ByteBuffer buffer = ByteBuffer.wrap(array);
        return buffer.getFloat();
    }

    public  byte[] floatBufftoByteBuff(float[] f_data){
        int size = f_data.length;
        byte[] b_data = new byte[size*Float.BYTES];
        for(int i = 0; i < size; ++i){
            byte[] frame = ByteBuffer.allocate(Float.BYTES).putFloat(f_data[i]).array();
            for(int j = 0; j < Float.BYTES; ++j){
                b_data[j + i * Float.BYTES] = frame[j];
            }
        }
        return b_data;
    }

    private void read_buff(String from_path){
        try{
            int f_size = Float.BYTES;
            this._b_matrix = Files.readAllBytes(Paths.get(from_path));
            // where the converted data will be stored
            this._f_matrix = new float[(this._b_matrix.length-2*f_size)/f_size];
            byte[] b_frame = new byte[f_size];

            for(int j = 0; j < f_size; ++j){
                b_frame[j] = this._b_matrix[j]; // [j + 0 * f_size]
            }
            this._m = (int)byteBuffToFloatBuff(b_frame);

            for(int j = 0; j < f_size; ++j){
                b_frame[j] = this._b_matrix[j + f_size]; // [j + 1 * f_size]
            }
            this._n = (int)byteBuffToFloatBuff(b_frame);

            for(int i = 2; i < this._f_matrix.length + 2; ++i){
                for(int j = 0; j < f_size; ++j){
                    b_frame[j] = this._b_matrix[j + i * f_size];
                }
                this._f_matrix[i-2] = byteBuffToFloatBuff(b_frame);
            }
        } catch(IOException ex){ ex.printStackTrace(); }
    }

    protected void write_buff(String to_path){
        try{
            float[] dims_data = {(float)this._m, (float)this._n};
            byte[] bytes_dims = floatBufftoByteBuff(dims_data);
            byte[] bytes_matrix = floatBufftoByteBuff(this._f_matrix);
            Files.write(Paths.get(to_path), bytes_dims);
            Files.write(Paths.get(to_path), bytes_matrix, StandardOpenOption.APPEND);
        } catch(IOException ex){ ex.printStackTrace(); }
    }
}
