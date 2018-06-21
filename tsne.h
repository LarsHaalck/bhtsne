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

#ifndef TSNE_TSNE_H
#define TSNE_TSNE_H

#include <vector>
#include <cstdlib> // for std::exit(1), and rand(
namespace tsne
{
namespace detail
{
static inline double sign(double x) { return (x == 0.0 ? 0.0 : (x < 0.0 ? -1.0 : 1.0)); }
static inline bool abs_compare(double a, double b) { return (std::abs(a) < std::abs(b)); }
}

class TSNE
{
public:
    void run(std::vector<double>& X, int N, int D, std::vector<double>& Y, int no_dims,
        double perplexity, double theta, int rand_seed, bool skip_random_init,
        int max_iter = 1000, int stop_lying_iter = 250, int mom_switch_iter = 250);
    //bool load_data(double** data, int* n, int* d, int* no_dims, double* theta,
    //    double* perplexity, int* rand_seed, int* max_iter);
    //void save_data(double* data, int* landmarks, double* costs, int n, int d);
    static void symmetrizeMatrix(std::vector<int>& row_P, std::vector<int>& col_P,
        std::vector<double>& val_P, int N);

private:
    void computeGradient(const std::vector<double>& P, const std::vector<int>& inp_row_P,
        const std::vector<int>& inp_col_P, const std::vector<double>& inp_val_P,
        std::vector<double>& Y, int N, int D, std::vector<double>& dC, double theta);
    void computeExactGradient(const std::vector<double>& P, const std::vector<double>& Y,
        int N, int D, std::vector<double>& dC);
    double evaluateError(const std::vector<double>& P, const std::vector<double>& Y,
        int N, int D);
    double evaluateError(const std::vector<int>& row_P, const std::vector<int>& col_P,
        const std::vector<double>& val_P, const std::vector<double>& Y, int N, int D,
        double theta);
    void computeGaussianPerplexity(const std::vector<double>& X, int N, int D,
        std::vector<double>& P, double perplexity);
    void computeGaussianPerplexity(const std::vector<double>& X, int N, int D,
        std::vector<int>& row_P, std::vector<int>& col_P, std::vector<double>& val_P,
        double perplexity, int K);
    void computeSquaredEuclideanDistance(const std::vector<double>& X, int N, int D,
        std::vector<double>& DD);
    void zeroMean(std::vector<double>& X, int N, int D);
    double randn();


};
}

#endif // TSNE_TSNE_H
