public class Function{

    public static void sigmoid(Matrix data){
        for(int i = 0; i < data.getM(); ++i){
            for(int j = 0; j < data.getN(); ++j){
                float value = data.get(i, j);
                data.set(i, j, 1F / (1F + (float)Math.exp(-value)));
            }
        }
    }

    public static void reLU(Matrix data){
        for(int i = 0; i < data.getM(); ++i){
            for(int j = 0; j < data.getN(); ++j){
                float value = data.get(i, j);
                data.set(i, j, Math.max(0, value));
            }
        }
    }

    public static void tanh(Matrix data){
        for(int i = 0; i < data.getM(); ++i){
            for(int j = 0; j < data.getN(); ++j){
                float value = data.get(i, j);
                data.set(i, j, (float)Math.tanh(value));
            }
        }
    }

    public static void softmax(Matrix data){
        /**
         * We shift the logit 'data' by the max
         * of each example of the output layer
         * in order to avoid NaN and stabilize softmax
         */
        // exp(zj) / SUM(zj)
        // accumulate the sum terms of exp in buffer
        Matrix buffer = new Matrix(data.getM(), 1);
        Matrix shifted_entry = new Matrix(data);
        Matrix max_vector = data.max(0);
        shifted_entry.hBroadcast(max_vector, '-');

        for(int i = 0; i < data.getM(); ++i){
            float temp = 0;
            for(int j = 0; j < data.getN(); ++j){
                float value = (float)Math.exp(shifted_entry.get(i, j));
                data.set(i, j, value);
                temp += value;
            }
            buffer.set(i, 0, temp);
        }
        data.hBroadcast(buffer, '/');
    }

    public static Matrix der_sigmoid(Matrix data){
        //sig(x) * (1F - sig(x));
        Matrix res = new Matrix(data);
        for(int i = 0; i < res.getM(); ++i){
            for(int j = 0; j < res.getN(); ++j){
                float value = res.get(i, j);
                res.set(i, j, value * (1F - value));
            }
        }
        return res;
    }

    public static Matrix der_reLU(Matrix data){
        Matrix res = new Matrix(data);
        for(int i = 0; i < res.getM(); ++i){
            for(int j = 0; j < res.getN(); ++j){
                float value = (res.get(i, j) <= 0) ? 0 : 1;
                res.set(i, j, value);
            }
        }
        return res;
    }

    public static Matrix der_tanh(Matrix data){
        Matrix res = new Matrix(data);
        for(int i = 0; i < res.getM(); ++i){
            for(int j = 0; j < res.getN(); ++j){
                float value = res.get(i, j);
                res.set(i, j, 1F - value*value); // tanh(x)' = 1 - tanh²(x)
            }
        }
        return res;
    }

    public static Matrix der_softmax(Matrix data, Matrix labels){
        /**
         * Compute softmax derivative
         * d(softmax)/dz = 1_{y=c} * softmax(z)_i - softmax(z)²_i
         * where 1_{y=c} are the labels. labels and data -> same dims
         * */
        Matrix lhs = new Matrix(data);
        Matrix rhs = new Matrix(data);

        rhs.selfMultMat(lhs);
        lhs.selfMultMat(labels);

        lhs.selfSubMat(rhs);
        return lhs;
    }

    public static Matrix der_linear(Matrix data){
        Matrix res = new Matrix(data.getM(), data.getN(), 1F);
        return res;
    }
    ///////////////////////////////////////////////
    //                  COST FUNCTIONS           //
    ///////////////////////////////////////////////

    public static Matrix MSE(Matrix predictions, Matrix labels){
        Matrix res = predictions.subMat(labels);
        res.selfMultMat(res);
        res.vSum();
        res.elemOperation(1F/(2F*(float)predictions.getM()), '*');
        return res;
    }

    public static Matrix bCE(Matrix predictions, Matrix labels){
        /** binary Cross Entropy */
        // SUM{ -[(1 - labels) * log(1 - prediction) + labels * log(prediction)] }
        // term1 = (1 - labels) * log(1 - prediction)
        // term2 = labels * log(prediction)

        Matrix labels_cpy = new Matrix(labels);
        labels_cpy.elemLeftOperation(1, '-');
        Matrix pred_cpy = new Matrix(predictions);
        pred_cpy.elemLeftOperation(1, '-');
        Function.log(pred_cpy);
        //term1
        labels_cpy.selfMultMat(pred_cpy);

        Matrix pred_cpy2 = new Matrix(predictions);
        Function.log(pred_cpy2);
        //term2
        pred_cpy2.selfMultMat(labels);

        labels_cpy.selfAddMat(pred_cpy2);
        // finally...
        labels_cpy.elemOperation(-1, '*');
        // summing along batch dimension
        Matrix cost = labels_cpy.vSum();
        return cost.elemOperation(1F/(float)predictions.getM(), '*');
    }

    public static Matrix CE(Matrix predictions, Matrix labels){
        /** Cross Entropy */
        // - labels * log(predictions)
        Matrix predictions_cpy = new Matrix(predictions);
        Function.log(predictions_cpy);
        predictions_cpy.selfMultMat(labels);
        predictions_cpy.elemOperation(-1, '*');

        Matrix cost = predictions_cpy.vSum();
        return cost.elemOperation(1F/(float)predictions.getM(), '*');
    }
    ///////////////////////////////////////////////
    //       COST FUNCTIONS DERIVATIVES          //
    ///////////////////////////////////////////////

    public static Matrix der_MSE(Matrix predictions, Matrix labels){
        /** Mean Squared Error derivative */
        Matrix res = predictions.subMat(labels);
        return res;
    }

    public static Matrix der_bCE(Matrix predictions, Matrix labels){
        /** binary Cross Entropy derivative */
        // (1 - labels)/(1 - prediction) - labels/prediction

        /* step1 */
        // getting (1 - labels)/(1 - prediction)
        /* step2 */
        // getting labels/predictions
        /* step3 */
        // getting (1 - labels)/(1 - prediction) - labels/prediction

        /* step 1 */
        Matrix num1 = new Matrix(labels);
        Matrix den1 = new Matrix(predictions);
        num1.elemLeftOperation(1, '-');
        den1.elemLeftOperation(1, '-');
        num1.selfDivMat(den1);
        /* step 2 */
        Matrix num2 = new Matrix(labels);
        num2.selfDivMat(predictions);
        /* step 3 */
        num1.selfSubMat(num2);
        return num1;
    }

    public static Matrix der_CE(Matrix predictions, Matrix labels){
        /** Cross Entropy derivative */
        // - labels / predictions
        Matrix labels_cpy = new Matrix(labels);
        labels_cpy.selfDivMat(predictions);
        labels_cpy.elemOperation(-1, '*');
        return labels_cpy;
    }

    ///////////////////////////////////////////////
    //                  MISCELLANEOUS            //
    ///////////////////////////////////////////////

    public static void log(Matrix data){
        for(int i = 0; i < data.getM(); ++i){
            for(int j = 0; j < data.getN(); ++j){
                float value = data.get(i, j) + 1e-8F;
                data.set(i, j, (float)Math.log(value));
            }
        }
    }

    public static void sqrt(Matrix data){
        for(int i = 0; i < data.getM(); ++i){
            for(int j = 0; j < data.getN(); ++j){
                float value = data.get(i, j);
                data.set(i, j, (float)Math.sqrt(value));
            }
        }
    }

    public static void pow(Matrix data, int coeff){
        for(int i = 0; i < data.getM(); ++i){
            for(int j = 0; j < data.getN(); ++j){
                float value = data.get(i, j);
                data.set(i, j, (float)Math.pow(value, coeff));
            }
        }
    }
}
