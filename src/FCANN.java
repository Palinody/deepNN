
public class FCANN{
    private int[] _model;
    private Matrix[] _gradients_list; // L - 1 gradients
    private Matrix[] _layers_list; // L layers
    private Matrix[] _weights_list; // L - 1 weights
    private Matrix[] _bias_list; // L - 1 biases

    private int _batch;
    private int _layers;

    private Matrix LABELS;
    private String COST_FUNC;
    private String HIDDEN_ACTIVATION;
    private String OUTPUT_ACTIVATION;

    FCANN(int[] model, int batch,
          String hidden_activation, String output_activation,
          String distr, String cost_function){
        /* cost function:
                MSE (Mean Squared Error),
                bCE (binary Cross Entropy),
                CE (Cross Entropy)
        * */
        _model = model; // list of number of neurons in each layer (including input layer)
        _layers = model.length; // total number of layers in net
        _batch = batch;

        _gradients_list = new Matrix[_layers-1]; // gradients from 1st hidden layer to output layer
        _layers_list = new Matrix[_layers]; // array that contains the layers data (including input layer)
        _weights_list = new Matrix[_layers-1]; // weights data
        _bias_list = new Matrix[_layers-1]; // bias data

        for(int l = 0; l < _layers-1; ++l){
            _gradients_list[l] = new Matrix(_batch, _model[l+1]); // useless to initialize them because references are going to be copied
            _layers_list[l] = new Matrix(_batch, _model[l]);
            _weights_list[l] = new Matrix(_model[l], _model[l+1], distr);
            _bias_list[l] = new Matrix(1, _model[l+1], distr);
        }
        // need to add output layer manually because there is one weight matrix less than layer matrix
        _layers_list[_layers-1] = new Matrix(_batch, _model[_layers-1]);

        LABELS = new Matrix(_batch, _model[_layers-1]);
        COST_FUNC = cost_function;
        HIDDEN_ACTIVATION = hidden_activation;
        OUTPUT_ACTIVATION = output_activation;
    }
    ///////////////////////////////////////////////
    //              GETTER/SETTER                //
    ///////////////////////////////////////////////

    public void setData(Matrix inputs, Matrix labels){
        _layers_list[0].copy(inputs);
        LABELS.copy(labels);
    }

    public float getError(){
        /* currently:
         *      MSE
         *      binary cross-entropy
         * */
        Matrix error_vector = new Matrix();
        // output -> horizontal vector
        switch (COST_FUNC) {
            case "MSE":
                error_vector = Function.MSE(_layers_list[_layers - 1], LABELS);
                break;
            case "bCE":
                error_vector = Function.bCE(_layers_list[_layers - 1], LABELS);
                break;
            case "CE":
                error_vector = Function.CE(_layers_list[_layers - 1], LABELS);
                break;
        }
        int n = error_vector.getN();
        error_vector = error_vector.hSum();
        return error_vector.get(0, 0) / (float)n;
    }

    Matrix getPrediction(){
        return _layers_list[_layers-1];
    }
    ///////////////////////////////////////////////
    //                  METHODS                  //
    ///////////////////////////////////////////////

    public void feedForward(){
        switch (HIDDEN_ACTIVATION) {
            case "relu":
                for (int l = 1; l < _layers - 1; ++l) {
                    _layers_list[l] = _layers_list[l - 1].dotProd(_weights_list[l - 1]);
                    _layers_list[l].vBroadcast(_bias_list[l - 1], '+');
                    Function.reLU(_layers_list[l]);
                }
                break;
            case "sigmoid":
                for (int l = 1; l < _layers - 1; ++l) {
                    _layers_list[l] = _layers_list[l - 1].dotProd(_weights_list[l - 1]);
                    _layers_list[l].vBroadcast(_bias_list[l - 1], '+');
                    Function.sigmoid(_layers_list[l]);
                }
                break;
            case "tanh":
                for (int l = 1; l < _layers - 1; ++l) {
                    _layers_list[l] = _layers_list[l - 1].dotProd(_weights_list[l - 1]);
                    _layers_list[l].vBroadcast(_bias_list[l - 1], '+');
                    Function.tanh(_layers_list[l]);
                }
                break;
            case "linear":
                for (int l = 1; l < _layers - 1; ++l) {
                    _layers_list[l] = _layers_list[l - 1].dotProd(_weights_list[l - 1]);
                    _layers_list[l].vBroadcast(_bias_list[l - 1], '+');
                }
                break;
            default:
                break;
        }
        // computing output layer separately
        _layers_list[_layers-1] = _layers_list[_layers-2].dotProd(_weights_list[_layers-2]);
        _layers_list[_layers-1].vBroadcast(_bias_list[_layers-2], '+');
        switch (OUTPUT_ACTIVATION) {
            case "relu":
                Function.reLU(_layers_list[_layers - 1]);
                break;
            case "sigmoid":
                Function.sigmoid(_layers_list[_layers - 1]);
                break;
            case "tanh":
                Function.tanh(_layers_list[_layers - 1]);
                break;
            case "softmax":
                Function.softmax(_layers_list[_layers - 1]);
                break;
            case "linear":
                break;
            default: // linear activation
                break;
        }

    }

    public void backProp(){
        // gradient of output layer
        Matrix layer_derivative = new Matrix();
        switch (OUTPUT_ACTIVATION) {
            case "relu":
                layer_derivative = Function.der_reLU(_layers_list[_layers - 1]);
                break;
            case "sigmoid":
                layer_derivative = Function.der_sigmoid(_layers_list[_layers - 1]);
                break;
            case "tanh":
                layer_derivative = Function.der_tanh(_layers_list[_layers - 1]);
                break;
            case "softmax":
                layer_derivative = Function.der_softmax(_layers_list[_layers - 1], LABELS);
                break;
            case "linear": // linear der -> useless matrix of ones
                // layer_derivative = Function.der_linear(_layers_list[_layers - 1]);
                break;
            default:
                break;
        }

        switch (COST_FUNC) {
            case "MSE":
                if(OUTPUT_ACTIVATION != "linear") {
                    _gradients_list[_layers - 2] = Function.der_MSE(_layers_list[_layers - 1], LABELS).multMat(layer_derivative);
                } else { _gradients_list[_layers - 2] = Function.der_MSE(_layers_list[_layers - 1], LABELS); }
                break;
            case "bCE":
                if(OUTPUT_ACTIVATION != "linear") {
                    _gradients_list[_layers - 2] = Function.der_bCE(_layers_list[_layers - 1], LABELS).multMat(layer_derivative);
                } else { _gradients_list[_layers - 2] = Function.der_bCE(_layers_list[_layers - 1], LABELS); }
                break;
            case "CE":
                if(OUTPUT_ACTIVATION != "linear") {
                    _gradients_list[_layers - 2] = Function.der_CE(_layers_list[_layers - 1], LABELS).multMat(layer_derivative);
                } else { _gradients_list[_layers - 2] = Function.der_CE(_layers_list[_layers - 1], LABELS); }
                break;
            default:
                break;
        }

        switch (HIDDEN_ACTIVATION) {
            case "relu":
                for(int l = _layers-3; l >= 0; --l){
                    layer_derivative = Function.der_reLU(_layers_list[l+1]);
                    _gradients_list[l] = _gradients_list[l+1].dotProdTranspose(_weights_list[l+1]).multMat(layer_derivative);
                }
                break;
            case "sigmoid":
                for(int l = _layers-3; l >= 0; --l){
                    layer_derivative = Function.der_sigmoid(_layers_list[l+1]);
                    _gradients_list[l] = _gradients_list[l+1].dotProdTranspose(_weights_list[l+1]).multMat(layer_derivative);
                }
                break;
            case "tanh":
                for(int l = _layers-3; l >= 0; --l){
                    layer_derivative = Function.der_tanh(_layers_list[l+1]);
                    _gradients_list[l] = _gradients_list[l+1].dotProdTranspose(_weights_list[l+1]).multMat(layer_derivative);
                }
                break;
            case "linear":
                for(int l = _layers-3; l >= 0; --l){
                    layer_derivative = Function.der_linear(_layers_list[l+1]);
                    _gradients_list[l] = _gradients_list[l+1].dotProdTranspose(_weights_list[l+1]).multMat(layer_derivative);
                }
                break;
            default:
                break;
        }
    }

    public void gradientDescent(float alpha, float lambd){
        /*
         * alpha : learning rate
         * lambd : Tikhonov coefficient (weight decay)
         * */
        if(lambd == 0){
            float batch_correction = 1F / (float)_batch * alpha;
            // updating the weights so the index reference is the index of the weights
            for(int l = _layers-2; l >= 0; --l){
                Matrix dJ_dW = _layers_list[l].transposeDotProd(_gradients_list[l]);
                Matrix dJ_db = _gradients_list[l].vSum();

                dJ_dW.elemOperation(batch_correction, '*');
                dJ_db.elemOperation(batch_correction, '*');

                _weights_list[l].selfSubMat(dJ_dW);
                _bias_list[l].selfSubMat(dJ_db);
            }
        } else {
            // updating the weights so the index reference is the index of the weights
            for(int l = _layers-2; l >= 0; --l){
                Matrix dJ_dW = _layers_list[l].transposeDotProd(_gradients_list[l]);
                Matrix dJ_db = _gradients_list[l].vSum();

                dJ_dW.elemOperation(1F/(float)_batch, '*');
                dJ_db.elemOperation(1F/(float)_batch, '*');

                Matrix W_cpy = new Matrix(_weights_list[l]);
                Matrix b_cpy = new Matrix(_bias_list[l]);
                W_cpy.elemOperation(lambd, '*');
                b_cpy.elemOperation(lambd, '*');

                dJ_dW.selfAddMat(W_cpy);
                dJ_db.selfAddMat(b_cpy);

                _weights_list[l].selfSubMat(dJ_dW);
                _bias_list[l].selfSubMat(dJ_db);
            }
        }
    }
    ///////////////////////////////////////////////
    //              VISUALIZATION                //
    ///////////////////////////////////////////////

    public void printModelDims(){
        /*
         * Shows only the dimensions of the model
         * IN:
         * Example {2, 3, 4}, batch = 4
         * OUT:
         * layer(0): (4, 2) | weight(0): (2, 3) | bias(0): (1, 3)
         * layer(1): (4, 3) | weight(1): (3, 4) | bias(1): (1, 4)
         * layer(2): (4, 4)
         * */
        for(int l = 0; l < _layers-1; ++l){
            System.out.printf("gradient(%d): (%d, %d)", l+1, _gradients_list[l].getM(), _gradients_list[l].getN());
            System.out.printf(" | layer(%d): (%d, %d)", l, _layers_list[l].getM(), _layers_list[l].getN());
            System.out.printf(" | weight(%d): (%d, %d)", l, _weights_list[l].getM(), _weights_list[l].getN());
            System.out.printf(" | bias(%d): (%d, %d)\n", l, _bias_list[l].getM(), _bias_list[l].getN());
        }
        System.out.printf("layer(%d): (%d, %d)\n", _layers-1, _layers_list[_layers-1].getM(), _layers_list[_layers-1].getN());
    }
    void printModel(char data){
        /*
         * Shows the matrices
         * */
        if(data == 'w'){
            for(int l = 0; l < _layers-1; ++l){
                System.out.printf("weight(%d): (%d, %d)\n", l, _weights_list[l].getM(), _weights_list[l].getN());
                _weights_list[l].print();
            }
        } else if(data == 'b'){
            for(int l = 0; l < _layers-1; ++l) {
                System.out.printf("bias(%d): (%d, %d)\n", l, _bias_list[l].getM(), _bias_list[l].getN());
                _bias_list[l].print();
            }
        } else if(data == 'l'){
            for(int l = 0; l < _layers; ++l) {
                System.out.printf("layer(%d): (%d, %d)\n", l, _layers_list[l].getM(), _layers_list[l].getN());
                _layers_list[l].print();
            }
        }  else if(data == 'g'){
            for(int l = 0; l < _layers-1; ++l) {
                System.out.printf("gradient(%d): (%d, %d)\n", l+1, _gradients_list[l].getM(), _gradients_list[l].getN());
                _gradients_list[l].print();
            }
        } else {
            System.out.println("printModel() takes w (weights), b (bias), l (layer) or g (gradients) as argument");
        }
    }

    public void printModelMem(){
        int temp = 0;
        for(int l = 0; l < _layers-1; ++l){
            // float_size * ((gradients) + (layers) + (weights) + (bias))
            temp += 4 * (_batch * _model[l+1] + _batch * _model[l] + _model[l] * _model[l+1] + _model[l+1]);
        }
        // last layer
        temp += 4 * (_batch * _model[_layers-1]);

        if(temp >= 1e9){ float temp_f = (float)temp * 1e-9F; System.out.printf("Mem. alloc.: %.3f Gbytes", temp_f);}
        else if(temp >= 1e6){ float temp_f = (float)temp * 1e-6F; System.out.printf("Mem. alloc.: %.3f Mbytes", temp_f);}
        else if(temp >= 1e3){ float temp_f = (float)temp * 1e-3F; System.out.printf("Mem. alloc.: %.3f Kbytes", temp_f);}
        else {System.out.printf("Mem. alloc.: %d bytes", temp);}
    }
}
