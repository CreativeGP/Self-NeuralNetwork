/* ========================================================================
   $File: $
   $Date: $
   $Revision: $
   $Creator: Creative GP $
   $Notice: (C) Copyright 2018 by Creative GP. All Rights Reserved. $
   ======================================================================== */

double error(NeuralNet &net, vector<double> teacher) {
    // Update the network
    update_network(&net);

    double sum = 0;
    // E(y1,...,yK) = ∑Kk(tk−yk)2/2
    for (int k = 0;
         k < net.back().size();
         ++k)
    {
        sum += pow((teacher[k] - net.back()[k].value), 2) / 2;
    }

    return sum;
}

#define object error

double LEARNING_RATE = 30;
double LEARNING_BIAS_RATE = 20;
double OBJECT_ACC = 0.000001;

void fit_once(NeuralNet& net, vector<double> teacher) {
    vector<double> last_error_by_in;
    for (int i = net.size()-1; i > 0; --i) {
        for (int j = 0; j < net[i].size(); ++j) {
            double error_by_out;
            if (i == net.size()-1)
                error_by_out = net[i][j].value - teacher[j];
            else {
                for (int e = 0; e < net[i+1].size(); ++e)
                    error_by_out += last_error_by_in[e] * net[i+1][e].weights[j];
            }
            
            double out_by_in = d_s(net[i][j].in);
            last_error_by_in.clear();
            last_error_by_in.push_back(error_by_out * out_by_in);
            
            double in_by_weights;
            for (int k = 0; k < net[i][j].weights.size(); ++k) {
//                cout << "Layer " << i << ", Neuron " << j << ", Weight " << k << endl;
                in_by_weights = net[i-1][k].value;
                net[i][j].weights[k] -= LEARNING_RATE * error_by_out * out_by_in * in_by_weights;
                net[i][j].bias -= LEARNING_BIAS_RATE * error_by_out * out_by_in * in_by_weights;
            }
        }
//        break;
    }
}

// TODO: Make it easier to configure!
void fit(NeuralNet& network, vector<double> teacher, bool show) {
    time_t start, end;
    time(&start);


    while (error(network, teacher) > OBJECT_ACC) {
        // if (error(network, teacher) < 0.01) LEARNING_RATE *= 0.3;
        // if (error(network, teacher) < 0.001) LEARNING_RATE *= 0.01;
//        LEARNING_RATE = ;
        
        fit_once(network, teacher);
        // update_network(&network);

        if (show) {
            show_network(network);
            cout << endl;
            cout << "Error: " << error(network, teacher) << endl;
        }
//        usleep(1000000 * 0.5);
    }

    // update_network(&network, true);
    zero_network(&network);
    update_network(&network);

    if (show) {
        show_network(network);
        time(&end);
        std::cout << "duration = " << end - start << "sec.\n";
    }
}
