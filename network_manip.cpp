/* ========================================================================
   $File: $
   $Date: $
   $Revision: $
   $Creator: Creative GP $
   $Notice: (C) Copyright 2018 by Creative GP. All Rights Reserved. $
   ======================================================================== */

string show (Neuron &n) {
    stringstream ss;
    ss << n.value << "[";
    for (auto d : n.weights) ss << d << ", ";
    ss << "]" << n.bias << "/";
    return ss.str();
}

void show_network (NeuralNet &net, bool value_only=true) {
    // Don't show the connection because this is a dense neural network.
    for (auto layer : net) {
        for (auto u : layer) {
            if (value_only) {
                cout << u.value << " ";
            } else {
                cout << show(u) << " ";
            }
        }
        cout << endl;
    }
    cout << endl;
}

void set_input(vector<double> inputs, NeuralNet *net) {
    if ((*net)[0].size() != inputs.size()) return;

    for (int i = 0; i < inputs.size(); ++i) {
        (*net)[0][i].in = (*net)[0][i].value = inputs[i];
    }
}

void input(int layer_num, NeuralNet *net) {
    rep (i, layer_num)
        (*net)[0].push_back({
                0, 0, 0,
                {},
                {},
                {}
            });

}

void dense(int layer_num, NeuralNet *net) {

    int back_layer_num = net->back().size();
    vector<Neuron*> back_layer = {};
    for (int i = 0; i < back_layer_num; ++i)
        back_layer.push_back(& net->back()[i]);
    
    vector<Neuron*> this_layer = {};
    net->push_back(vector<Neuron>(layer_num, Neuron {}));
    for (int i = 0; i < layer_num; ++i) {
        net->back()[i].value = 0;
        net->back()[i].backs = back_layer;
        // This is not random!!!
        rep(j, back_layer_num)
            net->back()[i].weights.push_back(r());

        this_layer.push_back(& net->back()[i]);
    }

    for (int i = 0; i < back_layer_num; ++i)
        (*prev(net->end(), 2))[i].nexts = this_layer;
}

double zero_network(NeuralNet *net) {
    for (int l = 1; l < net->size(); ++l) {
        for (auto &e : (*net)[l]) {
            e.value = 0;
            e.in = 0;
        }
    }
}

double calculate_value(Neuron& n, bool forcibly = false) {
    if (n.backs.size() != 0) {
        vector<double> values;
        for (auto e : n.backs) {
            // Don't calculate already calculated value.
            if (e->value == 0 || forcibly) calculate_value(*e, forcibly);
            values.push_back(e->value);
        }

        auto tmp = inner_product(values.begin(), values.end(), n.weights.begin(), 0.0f);
        if (forcibly) {
            v__print(values);
            v__print(n.weights);
        }
        tmp += n.bias;
        n.in = tmp; // This isn't applied by sigmoid function
        tmp = s(tmp);
        n.value = tmp;
        return tmp;
    } else {
        return n.value;
    }
}

void update_network(NeuralNet* net, bool forcibly = false) {
    for (auto &e : net->back())
        calculate_value(e, forcibly);
}
