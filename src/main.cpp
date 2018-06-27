#include "tsne.h"
#include <iostream>
#include <memory>
#include <omp.h>
int main()
{
    omp_set_num_threads(4);
    int no_dims = 2;
    double perplexity = 10.0;
    double theta = 1.0;
    int max_iter = 1000;
    int D = 5000; // initial dims?
    int N = 1000; // num data points
    auto data = std::make_shared<std::vector<double>>(N * D); // contains data points
    auto Y = std::make_shared<std::vector<double>>(N * no_dims); // contains res
    for (int i = 0; i < static_cast<int>(data->size()); i++)
    {
        (*data)[i] = i;
    }
    tsne::TSNE::run(data, N, D, Y, no_dims, perplexity, theta, 200.0, false, max_iter);
}
