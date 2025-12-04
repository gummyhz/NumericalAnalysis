#include <stdio.h>
#include <vector>
#include <fstream>
using namespace std;

template <typename T>
vector<T> read_vector(const string& filename) {
    ifstream file(filename);
    if (!file) throw runtime_error("Failed to open: " + filename);
    vector<T> data;
    T val;
    while (file >> val) data.push_back(val);
    return data;
}