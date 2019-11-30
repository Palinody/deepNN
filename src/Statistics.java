public class Statistics {

    public static Matrix getAverage(Matrix data){
        /**
         * We take the average over the examples
         * of each feature. We get a horizontal
         * Matrix of dim(1, n) where each element
         * along n is the average of each feature
         * */
        Matrix avg = data.vSum();
        avg.elemOperation(1F/data.getM(), '*');
        return avg;
    }

    public static Matrix getSigma(Matrix data){
        /**
         * sigma = sqrt(SUM{x_i - x_mean}² / (n-1))
         * */
        Matrix data_cpy = new Matrix(data);
        // X - X_avg
        data_cpy.vBroadcast(Statistics.getAverage(data), '-');
        // SUM{(X - X_avg)²}
        data_cpy.selfMultMat(data_cpy);
        Matrix summed = data_cpy.vSum();
        // sqrt(SUM{(X - X_avg)²} / (n - 1))
        summed.elemOperation((data.getM()-1), '*');
        Function.sqrt(summed);
        return summed;
    }

    public static void normalize_standard(Matrix data){
        /**
         * Works well for populations that are normally distributed
         * standard score : X' = (X - µ) / sigma
         * */
        Matrix mu = Statistics.getAverage(data);
        Matrix sigma = Statistics.getSigma(data);
        data.vBroadcast(mu, '-');
        data.vBroadcast(sigma, '/');
    }

    public static void normalize_by_feature_min_max(Matrix data, float a, float b){
        /**
         * Min-Max feature scaling [0-1] : X' = (X - Xmin) / (Xmax - Xmin)
         * Generalized [a, b] : X' = a + (X - Xmin)(b - a) / (Xmax - Xmin)
         * */
        Matrix Xmax = data.max(1);
        Matrix Xmin = data.min(1);
        Matrix Xmax_sub_Xmin = Xmax.subMat(Xmin);
        if(a != 0 || b != 1){
            data.vBroadcast(Xmin, '-');
            data.elemOperation(b-a, '*');
            data.vBroadcast(Xmax_sub_Xmin, '/');
            data.elemOperation(a, '+');
        } else {
            data.vBroadcast(Xmin, '-');
            data.vBroadcast(Xmax_sub_Xmin, '/');
        }
    }

    public static void normalize_min_max(Matrix data, float a, float b){
        /**
         * Min-Max feature scaling [0-1] : X' = (X - Xmin) / (Xmax - Xmin)
         * Generalized [a, b] : X' = a + (X - Xmin)(b - a) / (Xmax - Xmin)
         * */
        // get overall max/min
        float Xmax = data.max(-1).get(0, 0);
        float Xmin = data.min(-1).get(0, 0);
        float Xmax_sub_Xmin = Xmax - Xmin;
        if(a != 0 || b != 1){
            data.elemOperation(Xmin, '-');
            data.elemOperation((b-a)/Xmax_sub_Xmin, '*');
            data.elemOperation(a, '+');
        } else {
            data.elemOperation(Xmin, '-');
            data.elemOperation(1F/Xmax_sub_Xmin, '*');
        }
    }
    /**
     * Converts each digit of a dataset into a one-hot vector
     * which dimension is based on the absolute difference
     * in between max and min of the dataset.
     * E.g.: if dataset consists of integers [-2, 10]
     * -> |10 - (-2)| + 1 = 13 -> 13 classes
     * @param data : vertical vector of digits
     * @return res : each line -> one-hot vector
     */
    public static Matrix digits_matrix_to_one_hot_matrix(Matrix data){
        float min = data.min(-1).get(0, 0);
        float max = data.max(-1).get(0, 0);
        int one_hot_length = (int)Math.abs(max - min) + 1;
        Matrix res = new Matrix(data.getM(), one_hot_length);

        for(int i = 0; i < data.getM(); ++i){
            // we need to add |min| to get indexes >= 0
            res.set(i, (int)(data.get(i, 0)+Math.abs(min)), 1F);
        }
        return res;
    }

    public static Matrix getVMR(Matrix data){
        /**
         * Variance to Mean Ratio (VMR) measures whether
         * some data is clustered (not dispersed) or not
         * generally used for positive data. User may find
         * useful to normalize [0, 1] before using VMR
         *
         * |   Distribution	                |   VMR	      |   interpretation
         * -----------------------------------------------------------------
         * | constant random variable	    |   VMR = 0	  | not dispersed
         * | binomial distribution	        | 0 < VMR < 1 | under-dispersed
         * | Poisson distribution	        |   VMR = 1	  |
         * | negative binomial distribution |	VMR > 1	  | over-dispersed
         *
         * D = sigma² / µ
         * */
        Matrix sigma = Statistics.getSigma(data);
        Matrix mu = Statistics.getAverage(data);
        Function.pow(sigma, 2);
        sigma.selfDivMat(mu);
        return sigma;
    }
}
