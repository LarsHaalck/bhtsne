#include "tsne.h"
#include <iostream>
#include <memory>

int main()
{
    int no_dims = 2;
    double perplexity = 40;
    double theta = 0.5;
    int rand_seed = 2;
    int max_iter = 1000;
    int D = 1000; // initial dims?
    int N = 10000; // num data points
    auto data = std::vector<double>(N * D); // contains data points
    auto Y = std::vector<double>(N * no_dims); // contains res
    for (int i = 0; i < static_cast<int>(data.size()); i++)
    {
        data[i] = i;
    }
    tsne::TSNE::run(data, N, D, Y, no_dims, perplexity, theta, rand_seed, true, max_iter);
}