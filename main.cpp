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
#include <time.h>

#define ll long
#define ull unsigned long long

#define gt(a, b) (((a)>(b))?(a):(b))
#define ls(a, b) (((a)<(b))?(a):(b))

#define rep(i, n) for(int i = 0; i < (int)(n); i++)

using namespace std;


void SetPrint(set<int>& rseti) {
    for (set<int>::iterator iti = rseti.begin(); iti != rseti.end(); iti++) { printf("%3d ", *iti); }
    printf("\n");
}

template<class T> void VectorPrint(vector<T>& vi) {
    cout << "{";
    for (T e : vi)
        cout << e << " ";
    cout << "}" << endl;
}

#define v__print VectorPrint 
#define s__print VectorPrint 


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
    double value, in, bias;
    vector<Neuron*> backs;
    vector<Neuron*> nexts;
    vector<double> weights;
};

typedef vector<vector<Neuron>> NeuralNet;

#include "network_manip.cpp"
#include "fit.cpp"
#include "convenient.cpp"

int main() {
/*
  make network (dence)
  i1 h11 h21 v1
  i2 h12 h22 v2
  h13 h23 v3
*/

    // TODO: update関数の挙動を確かめる

    vector<NeuralNet> xor_nets = {
        get_trained_network({0, 0}, {0}),
        get_trained_network({0, 1}, {1}),
        get_trained_network({1, 0}, {1}),
        get_trained_network({1, 1}, {0})
    };

    for (auto e : xor_nets) {
        show_network(e, false);
    }

    // zero_network(&xor_nets[3]);
    // show_network(xor_nets[3]);


    NeuralNet xor_model = get_model(xor_nets);
    show_network(xor_model, false);
//    cout << 'a' << endl;
//    show_network(xor_model, false);

    // set_input({0, 0}, &xor_model);
    // zero_network(&xor_model);
    // update_network(&xor_model);
    // show_network(xor_model);
    
    // set_input({1, 0}, &xor_model);
    // zero_network(&xor_model);
    // update_network(&xor_model);
    // show_network(xor_model);

    // set_input({0, 1}, &xor_model);
    // zero_network(&xor_model);
    // update_network(&xor_model);
    // show_network(xor_model);

    set_input({1, 1}, &xor_model);
    zero_network(&xor_model);
    update_network(&xor_model);
    show_network(xor_model);

    return 0;
}
