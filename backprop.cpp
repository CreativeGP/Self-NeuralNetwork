/* ========================================================================
   $File: $
   $Date: $
   $Revision: $
   $Creator: Creative GP $
   $Notice: (C) Copyright 2018 by Creative GP. All Rights Reserved. $
   ======================================================================== */


#include <iostream>
#include <iomanip>
#include <sstream>

#include <cstdlib>
#include <cmath>

#include <iterator>
#include <algorithm>
#include <set>
#include <unordered_set>
#include <vector>
#include <map>
#include <stack>
#include <queue>
#include <utility>
#include <numeric>
#include <random>
#include <limits>

#include <unistd.h>

#define ll long
#define ull unsigned long long

#define gt(a, b) (((a)>(b))?(a):(b))
#define ls(a, b) (((a)<(b))?(a):(b))

#define rep(i, n) for(int i = 0; i < (int)(n); i++)

using namespace std;

random_device rnd;

double r()
{
    return (double)rnd() / numeric_limits<uint32_t>::max();
}

double s(double x)
{
    return 1.0 / (1.0 + exp(-x));
}

double d_s(double x)
{
    return s(x) * (1 - s(x));
}


struct Neuron {
    double value, in;
    vector<Neuron*> backs;
    vector<Neuron*> nexts;
    vector<double> weights;
};

Neuron i1, i2, h11, h12, h13, h21, h22, h23, v1, v2, v3;

string show(Neuron &n) {
    stringstream ss;
    ss << n.value << "[";
    for (auto d : n.weights) ss << d << ", ";
    ss << "]";
    return ss.str();
}

void show_network (vector<vector<Neuron>> &net) {
    // Don't show the connection because this is a dense neural network.
    for (auto layer : net) {
        for (auto u : layer) {
            cout << show(u) << " ";
        }
        cout << endl;
    }
}

void set_input(vector<vector<Neuron>> &net, vector<double> inputs) {
    if (net[0].size() != inputs.size()) return;

    for (int i = 0; i < inputs.size(); ++i) {
        net[0][i].in = net[0][i].value = inputs[i];
    }
}

void dense(int layer_num, vector<vector<Neuron>> *net) {

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

double calculate_value(Neuron& n, bool forcibly = false) {
    if (n.backs.size() != 0) {
        vector<double> values;
        for (auto e : n.backs) {
            // Don't calculate already calculated value.
            if (e->value == 0 && !forcibly) calculate_value(*e);
            values.push_back(e->value);
        }

        auto tmp = inner_product(values.begin(), values.end(), n.weights.begin(), 0.0f);
        n.in = tmp; // This isn't applied by sigmoid function
        tmp = s(tmp);
        n.value = tmp;
        return tmp;
    } else {
        return n.value;
    }
}

void update_network(vector<vector<Neuron>>& net, bool forcibly = false) {
    for (auto &e : net.back())
        calculate_value(e, forcibly);
}

double error(vector<vector<Neuron>>& net, vector<double> teacher) {
    // Update the network
    update_network(net);

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

double LEARNING_RATE = 1;
double OBJECT_ACC = 0.00001;

void fit(vector<vector<Neuron>>& net, vector<double> teacher) {
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
                in_by_weights = net[i-1][k].value;
                net[i][j].weights[k] -= LEARNING_RATE * error_by_out * out_by_in * in_by_weights;
            }
        }
//        break;
    }
}

int main() {
/*
  make network (dence)
  i1 h11 h21 v1
  i2 h12 h22 v2
     h13 h23 v3
*/

    vector<vector<Neuron>> network = {
        {
            {
                r(),
                {},
                {},
                {}
            },
            {
                r(),
                {},
                {},
                {}
            }
        }
    };


    dense(4, &network);
    dense(4, &network);

    vector<double> teacher = {0.3, 0.8};
//    set_input(network, {0.1, 0.9});

    cout << "inputs: " << endl;
    double gtmp; cin >> gtmp;
    cout << "Network input {" << gtmp <<"}: " << endl;
    rep(i, gtmp) {
        double tmp; cin >> tmp;
        network[0][i].value = tmp;
    }


    cout << "outputs: " << endl;
    cin >> gtmp;
    cout << "Want to be {" << gtmp << "}: " << endl;
    rep(i, gtmp) {
        double tmp; cin >> tmp;
        teacher[i] = tmp;
    }

    dense(gtmp, &network);

    cout << error(network, {1, 1}) << endl;
    show_network(network);


    while (error(network, teacher) > OBJECT_ACC) {
        // if (error(network, teacher) < 0.01) LEARNING_RATE *= 0.3;
        // if (error(network, teacher) < 0.001) LEARNING_RATE *= 0.01;
//        LEARNING_RATE = ;
        
        fit(network, teacher);
        update_network(network);
        show_network(network);

        cout << endl;
        cout << "Error: " << error(network, teacher) << endl;
//        usleep(1000000 * 0.5);
    }

    update_network(network, true);
    show_network(network);
             
    return 0;
}
