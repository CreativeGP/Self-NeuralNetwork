/* ========================================================================
   $File: $
   $Date: $
   $Revision: $
   $Creator: Creative GP $
   $Notice: (C) Copyright 2018 by Creative GP. All Rights Reserved. $
   ======================================================================== */

// 計算結果をまとめて、モデルを返す
// value, in は 0 固定
NeuralNet get_model(vector<NeuralNet>& nets) {
    NeuralNet result = {
        {
        }
    };

    for (int layer = 0;
         layer < nets[0].size();
         ++layer)
    {
        if (layer == 0) {
            input(nets[0][layer].size(), &result);
        } else {
            dense(nets[0][layer].size(), &result);
        }
//        result.push_back();
        for (int n = 0;
             n < nets[0][layer].size();
             ++n)
        {
            // average

// struct Neuron {
//     double value, in, bias;
//     vector<Neuron*> backs;
//     vector<Neuron*> nexts;
//     vector<double> weights;
// };
            // Neuron ave;
            // ave.value = 0;
            // ave.in = 0;

            // ave.backs = result[layer][n].backs;
            // ave.nexts = result[layer][n].nexts;

            double bias_sum = 0;
            vector<double> weights_sum(nets[0][layer][n].weights.size());
            for (int net = 0;
                 net < nets.size();
                 ++net)
            {
                bias_sum += nets[net][layer][n].bias;

                for (int w = 0;
                     w < nets[net][layer][n].weights.size();
                     ++w)
                {
                    weights_sum[w] += nets[net][layer][n].weights[w];
                }
            }
            
            for (int w = 0;
                 w < nets[0][layer][n].weights.size();
                 ++w)
            {
                result[layer][n].weights[w] = weights_sum[w] / nets.size();
//                ave.weights.push_back(weights_sum[w] / nets.size());
            }
            
            result[layer][n].bias = bias_sum / nets.size();
//            ave.bias = bias_sum / nets.size();

//            result[layer].push_back(ave);
//            result[layer][n] = ave;
        }
    }

    return result;
}

NeuralNet get_trained_network(
    vector<double> inputs,
    vector<double> teacher,
    int layers=2, int denses=4)
{
    NeuralNet network = {
        {
        }
    };

    input(inputs.size(), &network);
    set_input(inputs, &network);

    rep (i, layers)
        dense(denses, &network);

    dense(teacher.size(), &network);

    show_network(network);

    fit(network, teacher, false);

    return network;
}
