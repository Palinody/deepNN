class CSVParser {
    private String _csv_path;

    // total number of lines/cols of csv
    private int _csv_lines = 0;
    private int _csv_cols;
    // where we want to start parsing data
    // after we got the csv total lines/rows
    private int _from_i;
    private int _to_i;
    private int _from_j;
    private int _to_j;

    CSVParser(String from_path){
        _csv_path = from_path;
        String line = "";
        String splitBy = ",";
        int temp_csv_cols = -1;
        try{
            BufferedReader br = new BufferedReader(new FileReader(_csv_path));
            while((line = br.readLine()) != null){
                String[] example = line.split(splitBy);
                if(_csv_lines == 0){
                    _csv_cols = example.length;
                    temp_csv_cols = example.length;
                } else if (temp_csv_cols != example.length){
                    System.out.printf("Line %d has a different number of columns\n", _csv_lines);
                }
                ++_csv_lines;
            }
        } catch(IOException e){
            e.printStackTrace();
        }
    }

    public Matrix getMatrix(int from_i, int to_i, int from_j, int to_j){
        _from_i = from_i;
        _to_i = to_i;
        _from_j = from_j;
        _to_j = to_j;
        Matrix csv_slice = new Matrix (_to_i-_from_i, _to_j-_from_j);

        String line = "";
        String splitBy = ",";
        int rows_counter = 0;
        try{
            BufferedReader br = new BufferedReader(new FileReader(_csv_path));
            while((line = br.readLine()) != null && rows_counter < _to_i){
                // skips the lines we are not interested in
                if(rows_counter < _from_i){ ++rows_counter; continue; }

                String[] example = line.split(splitBy);
                for(int j = _from_j; j < _to_j; ++j){
                    csv_slice.set(rows_counter-_from_i, j-_from_j, Float.valueOf(example[j]));
                }
                ++rows_counter;
            }
            return csv_slice;
        } catch(IOException e){
            e.printStackTrace();
        }
        return csv_slice;
    }

    public int getCSVRows(){ return _csv_lines; }
    public int getCSVCols(){ return _csv_cols; }
}

public class Main {

    public static void XOR_example(){
        int batch_size = 4;
        int input_nodes = 2;
        int output_nodes = 1;

        int[] model = {input_nodes, 2, output_nodes};
        FCANN fcann = new FCANN(model, batch_size,
                "relu", "sigmoid",
                "xavier", "bCE");
        Matrix X = new Matrix(batch_size, input_nodes);
        Matrix Y = new Matrix(batch_size, output_nodes);

        // training loop
        for(int epoch = 0; epoch < 10000; ++epoch)
        {
            // get data from stream here
            // for(int line = 0; line < batch; ++line)
            {
                // MSE end bCE example benchmark database (X - Y)

                float[] intput_data = {1, 0, 0, 1, 1, 1, 0, 0};
                float[] target_data = {1, 1, 0, 0};

                X.copy(intput_data, batch_size, input_nodes);
                Y.copy(target_data, batch_size, output_nodes);

                fcann.setData(X, Y);
            }
            fcann.feedForward();
            fcann.backProp();
            fcann.gradientDescent( 0.1F, 0F);

            float curr_cost = fcann.getError();

            if(epoch % 100 == 0){
                for(int n = 0; n < (int)(curr_cost*10); ++n){
                    System.out.print("-");
                }
                System.out.println("> (" + curr_cost + ")");
            }

        }
        System.out.println("Model accuracy: " + fcann.evaluate(X, Y));

        fcann.printModelDistr('w', 10, 4);
        fcann.printModelMem();
    }
    public static void cross_entropy_example(){
        int batch_size = 4;
        int input_nodes = 4;
        int output_nodes = 4;

        int[] model = {input_nodes, 4, output_nodes};
        FCANN fcann = new FCANN(model, batch_size,
                "relu", "softmax",
                "xavier", "CE");
        Matrix X = new Matrix(batch_size, input_nodes);
        Matrix Y = new Matrix(batch_size, output_nodes);

        // training loop
        for(int epoch = 0; epoch < 10000; ++epoch)
        {
            // get data from stream here
            // for(int line = 0; line < batch; ++line)
            {
                /* cross entropy example benchmark database (X - Y)
                {1, 1, 1, 1} = {0, 0, 0, 1}
                {1, 1, 1, 0} = {0, 0, 1, 0}
                {1, 1, 0, 0} = {0, 1, 0, 0}
                {1, 0, 0, 0} = {1, 0, 0, 0}
                * */
                float[] intput_data = {1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 0, 0, 1, 0, 0, 0};
                float[] target_data = {0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0};

                X.copy(intput_data, batch_size, input_nodes);
                Y.copy(target_data, batch_size, output_nodes);
                Statistics.normalize_min_max(X, -1, 1);

                fcann.setData(X, Y);
            }
            fcann.feedForward();
            fcann.backProp();
            fcann.gradientDescent( 0.1F, 0F);

            float curr_cost = fcann.getError();

            if(epoch % 100 == 0){
                for(int n = 0; n < (int)(curr_cost*10); ++n){
                    System.out.print("-");
                }
                System.out.println("> (" + curr_cost + ")");
            }

        }
        System.out.println("Model accuracy: " + fcann.evaluate(X, Y));

        fcann.printModelDistr('w', 5, 4);
        fcann.printModelMem();
    }

    public static void print_mnist_image(Matrix data, Matrix labels, Matrix predictions, int from_line, int to_line){
        /**
         * uncomment the nested loop if you want to print the images of the misclassified data in console
         * I commented it because there was too many misclassified examples for the whole dataset
         * However, to specify a range in the dataset use the from_line/to_line parameters and hopefully
         * there wont be too many misclassified examples to print
         * */
        int[] wrong_predictions_counter = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
        for(int line = from_line; line < to_line; ++line){
            float label = labels.get(line, 0);
            float prediction = predictions.get(line, 0);
            if(label != prediction){
                ++wrong_predictions_counter[(int)label];
                /*
                System.out.println("Image label: " + label + " | predicted -> " + prediction);
                // following nested loop prints one image
                for(int i = 0; i < 28; ++i){
                    for(int j = 0; j < 28; ++j){
                        float pix_val = data.get(line, j + i * 28);
                        if(pix_val < 85){
                            System.out.print("*");
                        } else if(pix_val >= 85 && pix_val < 170){
                            System.out.print(".");
                        }
                        else {
                            System.out.print(" ");
                        }
                    }
                    System.out.println();
                }
                System.out.println();
                 */
            }
        }
        System.out.println("Wrong predictions counter: ");
        for(int i = 0; i < wrong_predictions_counter.length; ++i){
            System.out.println(i + ": " + wrong_predictions_counter[i]);
        }
    }

    public static void mnist_example() {
        /**
         * TODO
         * save necessary binaries
         * print the images in console (done)
         *
         * */
        int batch_size = 32;
        int input_nodes = 784;
        int output_nodes = 10;

        /**
         * MODEL SETUP
         * */
        int[] model = {input_nodes, 64, 64, output_nodes};
        FCANN fcann = new FCANN(model, batch_size,
                "relu", "sigmoid",
                "xavier", "MSE");

        /**
         * LOADING TEST DATA
         * */
        String csv_path_test = "files\\datasets\\csv\\mnist\\mnist_test.csv";
        CSVParser cr_test = new CSVParser(csv_path_test);
        // inputs not normalized
        Matrix TEST_DATA_INPUTS_RAW = cr_test.getMatrix(0, cr_test.getCSVRows(), 1, cr_test.getCSVCols());
        // outputs still need to be converted to one-hot vectors
        Matrix TEST_DATA_LABELS_RAW = cr_test.getMatrix(0, cr_test.getCSVRows(), 0, 1);
        // copying the raw data to not override it
        Matrix TEST_DATA_INPUTS_NORMALIZED = new Matrix(TEST_DATA_INPUTS_RAW);
        Statistics.normalize_min_max(TEST_DATA_INPUTS_NORMALIZED, -1, 1);
        Matrix TEST_DATA_LABELS_ONE_HOT_VECT = Statistics.digits_matrix_to_one_hot_matrix(TEST_DATA_LABELS_RAW);

        /**
         * LOADING TRAIN DATA
         * */
        String csv_path_train = "files\\datasets\\csv\\mnist\\mnist_train.csv";
        CSVParser cr_train = new CSVParser(csv_path_train);
        // inputs not normalized
        Matrix TRAIN_DATA_INPUTS_RAW = cr_train.getMatrix(0, cr_train.getCSVRows(), 1, cr_train.getCSVCols());
        // outputs still need to be converted to one-hot vectors
        Matrix TRAIN_DATA_LABELS_RAW = cr_train.getMatrix(0, cr_train.getCSVRows(), 0, 1);
        // copying the raw data to not override it
        Matrix TRAIN_DATA_INPUTS_NORMALIZED = new Matrix(TRAIN_DATA_INPUTS_RAW);
        Statistics.normalize_min_max(TRAIN_DATA_INPUTS_NORMALIZED, -1, 1);
        Matrix TRAIN_DATA_LABELS_ONE_HOT_VECT = Statistics.digits_matrix_to_one_hot_matrix(TRAIN_DATA_LABELS_RAW);

        Matrix BATCH_INPUT = new Matrix(batch_size, TRAIN_DATA_INPUTS_NORMALIZED.getN());
        Matrix BATCH_LABELS = new Matrix(batch_size, TRAIN_DATA_LABELS_ONE_HOT_VECT.getN());
        // training loop
        for (int epoch = 0; epoch < 5; ++epoch) {
            // get data from stream here
            for (int line = 0; line < cr_train.getCSVRows(); line += batch_size) {
                try{
                    BATCH_INPUT.copy_slice(TRAIN_DATA_INPUTS_NORMALIZED, line, 0);
                    BATCH_LABELS.copy_slice(TRAIN_DATA_LABELS_ONE_HOT_VECT, line, 0);
                } catch (Exception e) {
                    BATCH_INPUT.copy_slice(TRAIN_DATA_INPUTS_NORMALIZED, cr_train.getCSVRows()-line, 0);
                    BATCH_LABELS.copy_slice(TRAIN_DATA_LABELS_ONE_HOT_VECT, cr_train.getCSVRows()-line, 0);
                }

                fcann.setData(BATCH_INPUT, BATCH_LABELS);

                fcann.feedForward();
                fcann.backProp();
                fcann.gradientDescent(0.05F, 0F);

                float curr_cost = fcann.getError();

                if (line % 1000 == 0) {
                    for (int n = 0; n < (int) (curr_cost * 10000); ++n) {
                        System.out.print("-");
                    }
                    System.out.println("> (" + curr_cost + " | " + epoch + " | " + line + ")");
                }
            }
            fcann.printModelMem();
        }
        /**
         * EVALUATING TRAIN DATA
         * */
        // finally evaluate the model over the whole dataset and get an accuracy [0-1]
        float score_train = fcann.evaluate(TRAIN_DATA_INPUTS_NORMALIZED, TRAIN_DATA_LABELS_ONE_HOT_VECT);
        System.out.printf("Model accuracy (training set) %.2f %%\n", score_train*100F);
        Matrix prediction_train = fcann.getPrediction(TRAIN_DATA_INPUTS_NORMALIZED);
        Matrix prediction_train_index = prediction_train.indexOfMax(0);
        System.out.println("WRONG PREDICTIONS (TRAIN)");
        print_mnist_image(TRAIN_DATA_INPUTS_RAW, TRAIN_DATA_LABELS_RAW, prediction_train_index,0, prediction_train.getM());
        System.out.println(prediction_train.getM() + " objects");
        /**
         * EVALUATING TEST DATA
         * */
        float score_test = fcann.evaluate(TEST_DATA_INPUTS_NORMALIZED, TEST_DATA_LABELS_ONE_HOT_VECT);
        System.out.printf("Model accuracy (test set) %.2f %%\n", score_test*100F);
        Matrix prediction_test = fcann.getPrediction(TEST_DATA_INPUTS_NORMALIZED);
        Matrix prediction_test_index = prediction_test.indexOfMax(0);
        System.out.println("WRONG PREDICTIONS (TEST)");
        print_mnist_image(TEST_DATA_INPUTS_RAW, TEST_DATA_LABELS_RAW, prediction_test_index,0, prediction_test.getM());
        System.out.println(prediction_test.getM() + " objects");
    }


    public static void main(String[] args){
        long start_s = System.currentTimeMillis();

        //cross_entropy_example();
        //XOR_example();
        // String path = "files\\streams\\bin\\bin_file_test.dat";
        mnist_example();

        float end_s = (float)(System.currentTimeMillis() - start_s) * 1e-3F;
        System.out.printf("time: %.1f s\r", end_s);
    }
}
