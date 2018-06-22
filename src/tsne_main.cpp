#include "tsne.h"
#include <iostream>
#include <memory>

int main()
{
    auto tsne = std::make_unique<tsne::TSNE>();
    int no_dims = 2;
    double perplexity = 5;
    double theta = 0.5;
    int rand_seed = -1;
    int max_iter = 1000;
    int D = 5000; // initial dims?
    int N = 20000; // num data points
    //data is vector of n*d datapoints
    //y = res?
    auto data = std::vector<double>(N * D);
    auto Y = std::vector<double>(N * no_dims);
    for (int i = 0; i < (int) data.size(); i++) {
        data[i] = i;
    }
    tsne->run(data, N, D, Y, no_dims, perplexity, theta, rand_seed, false, max_iter);
}
