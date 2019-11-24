public class Main {

    public static void XOR_example(){
        int batch_size = 4;
        int input_nodes = 2;
        int output_nodes = 1;

        int[] model = {input_nodes, 4, output_nodes};
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

        Matrix y_hat = fcann.getPrediction();
        y_hat.print();
        Y.print();

        fcann.printModelMem();
    }

    public static void cross_entropy_example(){
        int batch_size = 4;
        int input_nodes = 4;
        int output_nodes = 4;

        int[] model = {input_nodes, output_nodes};
        FCANN fcann = new FCANN(model, batch_size,
                "None", "softmax",
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

        Matrix y_hat = fcann.getPrediction();
        y_hat.print();
        Y.print();

        fcann.printModelMem();
    }

    public static void main(String[] args){
        long startTime = System.nanoTime();

        cross_entropy_example();

        long timeEstimation = System.nanoTime() - startTime;
        System.out.println("\nEstimated time: " + timeEstimation * 1e-9 + " s");
    }
}
