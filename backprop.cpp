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

double calculate_value(Neuron& n) {
    if (n.backs.size() != 0) {
        vector<double> values;
        for (auto e : n.backs) {
            calculate_value(*e);
            values.push_back(e->value);
        }
        auto tmp = inner_product(values.begin(), values.end(), n.weights.begin(), 0.0f);
        n.value = tmp;
        return tmp;
    } else {
        return n.value;
    }

    
    // stack<vector<Neuron*>> sta;
    // sta.push(n.backs);

    // Neuron& last = n;
    // while (!sta.empty()) {
    //     auto e = sta.top(); sta.top();
    //     e = inner_product(e.begin(), e.end(), last.weights.begin(), 0.0f);
    // }
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
        },

        {
            {
                0,
                {},
                {},
                {r(), r()}
            },
            {
                0,
                {},
                {},
                {r(), r()}
            },
            {
                0,
                {},
                {},
                {r(), r()}
            }
        },

        {
            {
                0,
                {},
                {},
                {r(), r(), r()}
            },
            {
                0,
                {},
                {},
                {r(), r(), r()}
            },
            {
                0,
                {},
                {},
                {r(), r(), r()}
            }
        },

        {
            {
                0,
                {},
                {},
                {r(), r(), r()}
            },
            {
                0,
                {},
                {},
                {r(), r(), r()}
            }
        },
    };

    network[0][0].nexts = {&network[1][0], &network[1][1], &network[1][2]};
    network[0][1].nexts = {&network[1][0], &network[1][1], &network[1][2]};

    network[1][0].nexts = {&network[2][0], &network[2][1], &network[2][2]};
    network[1][0].backs = {&network[0][0], &network[0][1]};
    network[1][1].nexts = {&network[2][0], &network[2][1], &network[2][2]};
    network[1][1].backs = {&network[0][0], &network[0][1]};
    network[1][2].nexts = {&network[2][0], &network[2][1], &network[2][2]};
    network[1][2].backs = {&network[0][0], &network[0][1]};

    network[2][0].nexts = {&network[2][0], &network[2][1], &network[2][2]};
    network[2][0].backs = {&network[1][0], &network[1][1], &network[1][2]};
    network[2][1].nexts = {&network[2][0], &network[2][1], &network[2][2]};
    network[2][1].backs = {&network[1][0], &network[1][1], &network[1][2]};
    network[2][2].nexts = {&network[2][0], &network[2][1], &network[2][2]};
    network[2][2].backs = {&network[1][0], &network[1][1], &network[1][2]};

    network[3][0].nexts = {};
    network[3][0].backs = {&network[2][0], &network[2][1], &network[2][2]};
    network[3][1].nexts = {};
    network[3][1].backs = {&network[2][0], &network[2][1], &network[2][2]};

    cout << calculate_value(network[3][0])  << endl;

    show_network(network);
             
    return 0;
}
