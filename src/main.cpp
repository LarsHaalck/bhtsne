#include "tsne.h"
#include <iostream>
#include <memory>
//#include <omp.h>
int main()
{
    //omp_set_num_threads(2);
    int no_dims = 2;
    double perplexity = 50.0;
    double theta = 1.0;
    int max_iter = 2000;
    int D = 10000; // initial dims?
    int N = 5000; // num data points
    auto data = std::vector<double>(N * D); // contains data points
    auto Y = std::vector<double>(N * no_dims); // contains res
    for (int i = 0; i < static_cast<int>(data.size()); i++)
        data[i] = i;
    tsne::TSNE::run(data, N, D, Y, no_dims, perplexity, theta, 200.0, false, max_iter);
}
