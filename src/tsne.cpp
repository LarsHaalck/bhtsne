/*
 *
 * Copyright (c) 2014, Laurens van der Maaten (Delft University of Technology)
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 * 1. Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 * 2. Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 * 3. All advertising materials mentioning features or use of this software
 *    must display the following acknowledgement:
 *    This product includes software developed by the Delft University of Technology.
 * 4. Neither the name of the Delft University of Technology nor the names of
 *    its contributors may be used to endorse or promote products derived from
 *    this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY LAURENS VAN DER MAATEN ''AS IS'' AND ANY EXPRESS
 * OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES
 * OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO
 * EVENT SHALL LAURENS VAN DER MAATEN BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
 * SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 * PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR
 * BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING
 * IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY
 * OF SUCH DAMAGE.
 *
 */

#include "tsne.h"
#include "sptree.h"
#include "vptree.h"

#include <chrono>
#include <functional>
#include <iostream>
#include <limits>
#include <cmath>
#include <algorithm>
#include <numeric>

namespace tsne
{
// Perform t-SNE
void TSNE::run(std::vector<double>& X, int N, int D, std::vector<double>& Y,
    int no_dims, double perplexity, double theta, int rand_seed, bool skip_random_init,
    int max_iter, int stop_lying_iter, int mom_switch_iter)
{
    // Set random seed
    if (!skip_random_init)
    {
        if (rand_seed >= 0)
        {
            std::cout << "Using random seed: " << rand_seed << std::endl;
            srand((unsigned int)rand_seed);
        }
        else
        {
            std::cout << "Using current time as random seed..." << std::endl;
            srand(time(NULL));
        }
    }

    // Determine whether we are using an exact algorithm
    if (N - 1 < 3 * perplexity)
    {
        std::cout <<"Perplexity too large for the number of data points!" << std::endl;
        std::exit(1);
    }
    std::cout << "Using no_dims = " << no_dims << ", perplexity = " << perplexity
        << ", and theta = " << theta << std::endl;

    // time execution time
    std::chrono::time_point<std::chrono::system_clock> start_time, end_time;

    // Set learning parameters
    double total_time = 0.0;
    double momentum = 0.5;
    double final_momentum = 0.8;
    double eta = 200.0;

    // Allocate some memory
    auto dY = std::vector<double>(N * no_dims);
    auto uY = std::vector<double>(N * no_dims);
    auto gains = std::vector<double>(N * no_dims);

    // Normalize input data (to prevent numerical problems)
    std::cout << "Computing input similarities..." << std::endl;
    start_time = std::chrono::system_clock::now();

    zeroMean(X, N, D);

    double max_X = *std::max_element(std::begin(X), std::end(X), detail::abs_compare);
    max_X = std::abs(max_X);
    for(int i = 0; i < N * D; i++) X[i] /= max_X;

    // Compute input similarities for approximate t-SNE
    auto row_P = std::vector<int>();
    auto col_P = row_P;
    auto val_P = std::vector<double>();
    // Compute asymmetric pairwise input similarities
    computeGaussianPerplexity(
        X, N, D, row_P, col_P, val_P, perplexity, static_cast<int>(3 * perplexity));

    // Symmetrize input similarities
    symmetrizeMatrix(row_P, col_P, val_P, N);

    double sum_P = std::accumulate(std::begin(val_P), std::begin(val_P) + row_P[N],
        0.0);
    for (int i = 0; i < row_P[N]; i++)
        val_P[i] /= sum_P;
    end_time = std::chrono::system_clock::now();

    // Lie about the P-values
    for (int i = 0; i < row_P[N]; i++)
        val_P[i] *= 12.0;

    // Initialize solution (randomly)
    if (skip_random_init != true)
    {
        #pragma omp parallel for
        for (int i = 0; i < N * no_dims; i++)
            Y[i] = randn() * 0.0001;
    }

    double elapsed_seconds = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count() / 1000.0;
    double sparsity = static_cast<double>(row_P[N]) / (N * N);
    std::cout << "Input similarities computed in " << elapsed_seconds
        << " seconds (sparsity = " << sparsity << ")!. Learning embedding..."
        << std::endl;
    start_time = std::chrono::system_clock::now();

    // Perform main training loop
    for (int iter = 0; iter < max_iter; iter++)
    {
        // Compute (approximate) gradient
        computeGradient(row_P, col_P, val_P, Y, N, no_dims, dY, theta);

        // Update gains and
        // Perform gradient update (with momentum and gains)
        #pragma omp parallel for
        for (int i = 0; i < N * no_dims; i++)
        {
            gains[i] = (detail::sign(dY[i]) != detail::sign(uY[i]))
                ? (gains[i] + 0.2) : (gains[i] * 0.8);
            if (gains[i] < 0.01)
                gains[i] = 0.01;
            uY[i] = momentum * uY[i] - eta * gains[i] * dY[i];
            Y[i] = Y[i] + uY[i];
        }

        // Make solution zero-mean
        zeroMean(Y, N, no_dims);

        // Stop lying about the P-values after a while, and switch momentum
        if (iter == stop_lying_iter)
        {
            for (int i = 0; i < row_P[N]; i++)
                val_P[i] /= 12.0;
        }
        if (iter == mom_switch_iter)
            momentum = final_momentum;

        // Print out progress
        if ((iter % 50 == 0 || iter == max_iter - 1))
        {
            end_time = std::chrono::system_clock::now();
            double C = 0.0;
            std::cout << "error" << std::endl;
            C = evaluateError(row_P, col_P, val_P, Y, N, no_dims, theta);
            if (iter == 0)
                std::cout << "Iteration: " << iter << ", error is " << C << std::endl;
            else
            {
                elapsed_seconds = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count() / 1000.0;
                total_time += elapsed_seconds;
                std::cout << "Iteration: " << iter << ", error is " << C
                << " (50 iterations in " << elapsed_seconds << " seconds)" << std::endl;
            }
            start_time = std::chrono::system_clock::now();
        }
    }
    end_time = std::chrono::system_clock::now();
    elapsed_seconds = std::chrono::duration_cast<std::chrono::seconds>(end_time - start_time).count() / 1000.0;
    total_time += elapsed_seconds;

    std::cout << "Fitting performed in " << total_time << " seconds." << std::endl;
}

// Compute gradient of the t-SNE cost function (using Barnes-Hut algorithm)
void TSNE::computeGradient(const std::vector<int>& row_P,
    const std::vector<int>& col_P, const std::vector<double>& val_P,
    std::vector<double>& Y, int N, int D, std::vector<double>& dC, double theta)
{
    // Construct space-partitioning tree on current map
    auto tree = std::make_unique<SPTree>(D, Y, N);

    // Compute all terms required for t-SNE gradient
    auto pos_f = std::vector<double>(N * D);
    auto neg_f = std::vector<double>(N * D);

    // was tree->computeEdgeForces(inp_row_P, inp_col_P, inp_val_P, N, pos_f); before
    // data in sptree equals Y here
    auto local_Q = std::vector<double>(N);
    #pragma omp parallel for
    for (int n = 0; n < N; n++)
    {
        //computeEdgeForces
        int ind1 = n * D;
        for (int i = row_P[n]; i < row_P[n + 1]; i++)
        {
            // Compute pairwise distance and Q-value
            double dist = 1.0;
            int ind2 = col_P[i] * D;
            for (int d = 0; d < D; d++)
            {
                double temp = Y[ind1 + d] - Y[ind2 + d];
                dist += temp * temp;
            }
            dist = val_P[i] / dist;

            // Sum positive force
            for (int d = 0; d < D; d++)
                pos_f[ind1 + d] += dist * (Y[ind1 + d] - Y[ind2 + d]);
        }

        double current_Q = 0.0;
        tree->computeNonEdgeForces(n, theta, neg_f, n * D, current_Q);
        local_Q[n] = current_Q;
    }

    double sum_Q = 0.0;
    //#pragma omp parallel for reduction(+:sum_Q)
    for (int n = 0; n < N; n++)
        sum_Q += local_Q[n];

    // Compute final t-SNE gradient
    #pragma omp parallel for
    for (int i = 0; i < N * D; i++)
        dC[i] = pos_f[i] - (neg_f[i] / sum_Q);
}

// Evaluate t-SNE cost function (approximately)
double TSNE::evaluateError(const std::vector<int>& row_P, const std::vector<int>& col_P,
    const std::vector<double>& val_P, const std::vector<double>& Y, int N, int D,
    double theta)
{
    // Get estimate of normalization term
    auto tree = std::make_unique<SPTree>(D, Y, N);
    auto buff = std::vector<double>(D);
    double sum_Q = 0.0;
    for (int n = 0; n < N; n++)
        tree->computeNonEdgeForces(n, theta, buff, 0, sum_Q);

    // Loop over all edges to compute t-SNE error
    double C = 0.0;
    #pragma omp parallel for reduction(+:C)
    for (int n = 0; n < N; n++)
    {
        int ind1 = n * D;
        for (int i = row_P[n]; i < row_P[n + 1]; i++)
        {
            double Q = 0.0;
            int ind2 = col_P[i] * D;
            for (int d = 0; d < D; d++)
            {
                double temp = Y[ind1 + d] - Y[ind2 + d];
                Q += temp * temp;
            }
            Q = (1.0 / (1.0 + Q)) / sum_Q;

            double min = std::numeric_limits<float>::min();
            C += val_P[i] * std::log((val_P[i] + min) / (Q + min));
        }
    }
    return C;
}

// Compute input similarities with a fixed perplexity using ball trees
void TSNE::computeGaussianPerplexity(const std::vector<double>& X, int N, int D,
    std::vector<int>& row_P, std::vector<int>& col_P, std::vector<double>& val_P,
    double perplexity, int K)
{
    if (perplexity > K)
        std::cout << "Perplexity should be lower than K!" << std::endl;

    // Allocate the memory we need
    row_P = std::vector<int>(N + 1);
    col_P = std::vector<int>(N * K);
    val_P = std::vector<double>(N * K);

    row_P[0] = 0;
    for (int n = 0; n < N; n++)
        row_P[n + 1] = row_P[n] + K;

    // Build ball tree on data set
    auto tree = std::make_unique<VpTree>();;
    std::vector<DataPoint> obj_X(N);
    for (int n = 0; n < N; n++)
        obj_X[n] = DataPoint(D, n,
            std::vector<double>(std::begin(X) + n * D, std::begin(X) + n * D + D));
    tree->create(obj_X);

    // Loop over all points to find nearest neighbors
    std::cout << "Building tree..." << std::endl;
    #pragma omp parallel for
    for (int n = 0; n < N; n++)
    {
        // Find nearest neighbors
        auto cur_P = std::vector<double>(K);
        std::vector<DataPoint> indices;
        std::vector<double> distances;

        tree->search(obj_X[n], K + 1, indices, distances);

        // Initialize some variables for binary search
        bool found = false;
        double beta = 1.0;
        double min_beta = -std::numeric_limits<double>::max();
        double max_beta = std::numeric_limits<double>::max();
        double tol = 1e-5;

        // Iterate until we found a good perplexity
        int iter = 0;
        double sum_P;
        while (!found && iter < 200)
        {
            // Compute Gaussian kernel row
            for (int m = 0; m < K; m++)
                cur_P[m] = std::exp(-beta * distances[m + 1] * distances[m + 1]);

            // Compute entropy of current row
            sum_P = std::numeric_limits<double>::min();
            for (int m = 0; m < K; m++)
                sum_P += cur_P[m];
            double H = 0.0;
            for (int m = 0; m < K; m++)
                H += beta * (distances[m + 1] * distances[m + 1] * cur_P[m]);
            H = (H / sum_P) + std::log(sum_P);

            // Evaluate whether the entropy is within the tolerance level
            double Hdiff = H - std::log(perplexity);
            if (Hdiff < tol && -Hdiff < tol)
                found = true;
            else
            {
                if (Hdiff > 0)
                {
                    min_beta = beta;
                    if (max_beta == std::numeric_limits<double>::max()
                        || max_beta == -std::numeric_limits<double>::max())
                        beta *= 2.0;
                    else
                        beta = (beta + max_beta) / 2.0;
                }
                else
                {
                    max_beta = beta;
                    if (min_beta == -std::numeric_limits<double>::max()
                        || min_beta == std::numeric_limits<double>::max())
                        beta /= 2.0;
                    else
                        beta = (beta + min_beta) / 2.0;
                }
            }

            // Update iteration counter
            iter++;
        }

        // Row-normalize current row of P and store in matrix
        for (int m = 0; m < K; m++)
            cur_P[m] /= sum_P;
        for (int m = 0; m < K; m++)
        {
            col_P[row_P[n] + m] = indices[m + 1].index();
            val_P[row_P[n] + m] = cur_P[m];
        }
    }
}

// Symmetrizes a sparse matrix
void TSNE::symmetrizeMatrix(std::vector<int>& row_P, std::vector<int>& col_P,
    std::vector<double>& val_P, int N)
{
    // Count number of elements and row counts of symmetric matrix
    auto row_counts = std::vector<int>(N);
    for (int n = 0; n < N; n++)
    {
        for (int i = row_P[n]; i < row_P[n + 1]; i++)
        {
            // Check whether element (col_P[i], n) is present
            bool present = false;
            for (int m = row_P[col_P[i]]; m < row_P[col_P[i] + 1]; m++)
            {
                if (col_P[m] == n)
                    present = true;
            }
            if (present)
                row_counts[n]++;
            else
            {
                row_counts[n]++;
                row_counts[col_P[i]]++;
            }
        }
    }
    int no_elem = 0;
    for (int n = 0; n < N; n++)
        no_elem += row_counts[n];

    // Allocate memory for symmetrized matrix
    auto sym_row_P = std::vector<int>(N + 1);
    auto sym_col_P = std::vector<int>(no_elem);
    auto sym_val_P = std::vector<double>(no_elem);

    // Construct new row indices for symmetric matrix
    sym_row_P[0] = 0;
    for (int n = 0; n < N; n++)
        sym_row_P[n + 1] = sym_row_P[n] + (int)row_counts[n];

    // Fill the result matrix
    auto offset = std::vector<int>(N);
    for (int n = 0; n < N; n++)
    {
        for (int i = row_P[n]; i < row_P[n + 1]; i++)
        { // considering element(n, col_P[i])

            // Check whether element (col_P[i], n) is present
            bool present = false;
            for (int m = row_P[col_P[i]]; m < row_P[col_P[i] + 1]; m++)
            {
                if (col_P[m] == n)
                {
                    present = true;
                    if (n <= col_P[i])
                    { // make sure we do not add elements twice
                        sym_col_P[sym_row_P[n] + offset[n]] = col_P[i];
                        sym_col_P[sym_row_P[col_P[i]] + offset[col_P[i]]] = n;
                        sym_val_P[sym_row_P[n] + offset[n]] = val_P[i] + val_P[m];
                        sym_val_P[sym_row_P[col_P[i]] + offset[col_P[i]]]
                            = val_P[i] + val_P[m];
                    }
                }
            }

            // If (col_P[i], n) is not present, there is no addition involved
            if (!present)
            {
                sym_col_P[sym_row_P[n] + offset[n]] = col_P[i];
                sym_col_P[sym_row_P[col_P[i]] + offset[col_P[i]]] = n;
                sym_val_P[sym_row_P[n] + offset[n]] = val_P[i];
                sym_val_P[sym_row_P[col_P[i]] + offset[col_P[i]]] = val_P[i];
            }

            // Update offsets
            if (!present || (present && n <= col_P[i]))
            {
                offset[n]++;
                if (col_P[i] != n)
                    offset[col_P[i]]++;
            }
        }
    }

    // Divide the result by two
    for (int i = 0; i < no_elem; i++)
        sym_val_P[i] /= 2.0;

    row_P = std::move(sym_row_P);
    col_P = std::move(sym_col_P);
    val_P = std::move(sym_val_P);
}

// Compute squared Euclidean distance matrix
void TSNE::computeSquaredEuclideanDistance(const std::vector<double>& X, int N, int D,
    std::vector<double>& DD)
{
    int XnD = 0;
    for (int n = 0; n < N; ++n, XnD += D)
    {
        int XmD = XnD + D;
        int curr_elem = n * N + n;
        int curr_elem_sym = curr_elem + N;
        for (int m = n + 1; m < N; ++m, XmD += D, curr_elem_sym += N)
        {
            DD[++curr_elem] = 0.0;
            for (int d = 0; d < D; ++d)
            {
                DD[curr_elem] += (X[XnD + d] - X[XmD + d]) * (X[XnD + d] - X[XmD + d]);
            }
            DD[curr_elem_sym] = DD[curr_elem];
        }
    }
}

// Makes data zero-mean
void TSNE::zeroMean(std::vector<double>& X, int N, int D)
{
    // Compute data mean
    auto mean = std::vector<double>(D);
    int nD = 0;
    for (int n = 0; n < N; n++)
    {
        for (int d = 0; d < D; d++)
        {
            mean[d] += X[nD + d];
        }
        nD += D;
    }
    for (int d = 0; d < D; d++)
    {
        mean[d] /= static_cast<double>(N);
    }

    // Subtract data mean
    nD = 0;
    for (int n = 0; n < N; n++)
    {
        for (int d = 0; d < D; d++)
        {
            X[nD + d] -= mean[d];
        }
        nD += D;
    }
}

// Generates a Gaussian random number
double TSNE::randn()
{
    double x, y, radius;
    do
    {
        x = 2 * (std::rand() / ((double)RAND_MAX + 1)) - 1;
        y = 2 * (std::rand() / ((double)RAND_MAX + 1)) - 1;
        radius = (x * x) + (y * y);
    } while ((radius >= 1.0) || (radius == 0.0));
    radius = std::sqrt(-2 * std::log(radius) / radius);
    x *= radius;
    y *= radius;
    return x;
}
}
