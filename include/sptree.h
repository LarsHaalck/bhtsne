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

#ifndef TSNE_SPTREE_H
#define TSNE_SPTREE_H

#include "cell.h"

#include <cmath>
#include <memory>
#include <vector>

namespace tsne
{

class SPTree
{
private:
    // Fixed constants
    static const int QT_NODE_CAPACITY = 1;

    // Properties of this node in the tree
    int dimension;
    bool is_leaf;
    int size;
    int cum_size;

    // Axis-aligned bounding box stored as a center with half-dimensions to represent the
    // boundaries of this quad tree
    std::unique_ptr<Cell> boundary;

    // Indices in this space-partitioning tree node, corresponding center-of-mass, and
    // list of all children
    std::vector<double> data;
    std::vector<double> center_of_mass;
    int index[QT_NODE_CAPACITY];

    // Children
    std::vector<std::unique_ptr<SPTree>> children;
    int no_children;

public:
    SPTree(int D, const std::vector<double>& inp_data, int N);
    SPTree(int D, const std::vector<double>& inp_data,
        const std::vector<double>& inp_corner, const std::vector<double>& inp_width);
    SPTree(int D, const std::vector<double>& inp_data, int N,
        const std::vector<double>& inp_corner, const std::vector<double>& inp_width);

    bool insert(int new_index);
    void subdivide();
    bool isCorrect();
    int getDepth();
    void computeNonEdgeForces(int point_index, double theta, std::vector<double>& neg_f,
        int neg_offset, double& sum_Q);
private:
    void init(int D, const std::vector<double>& inp_data,
        const std::vector<double>& inp_corner, const std::vector<double>& inp_width);
    void fill(int N);
};
}

#endif // TSNE_SPTREE_H
