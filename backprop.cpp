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


struct Neuron {
    double value;
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
        net->back()[i].weights = vector<double>(back_layer_num, r());

        this_layer.push_back(& net->back()[i]);
    }

    for (int i = 0; i < back_layer_num; ++i)
        (*prev(net->end(), 2))[i].nexts = this_layer;
}

double calculate_value(Neuron& n) {
    if (n.backs.size() != 0) {
        vector<double> values;
        for (auto e : n.backs) {
            // Don't calculate already calculated value.
            if (e->value == 0) calculate_value(*e);
            values.push_back(e->value);
        }

        auto tmp = inner_product(values.begin(), values.end(), n.weights.begin(), 0.0f);
        tmp = s(tmp);
        n.value = tmp;
        return tmp;
    } else {
        return n.value;
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

    dense(3, &network);
    dense(3, &network);
    dense(2, &network);

    cout << calculate_value(network[3][0])  << endl;
    cout << calculate_value(network[3][1])  << endl;

    show_network(network);
             
    return 0;
}
