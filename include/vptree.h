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

/* This code was adopted with minor modifications from Steve Hanov's great tutorial at
 * http://stevehanov.ca/blog/index.php?id=130 */

#ifndef TSNE_VPTREE_H
#define TSNE_VPTREE_H

#include <algorithm>
#include <cmath>
#include <chrono>
#include <limits>
#include <memory>
#include <numeric>
#include <random>
#include <queue>

#include "datapoint.h"

namespace tsne
{
namespace detail
{
    double eucl_dist(const DataPoint& t1, const DataPoint& t2);
    double jaccard_dist(const DataPoint& t1, const DataPoint& t2);
}

typedef double (*func)(const DataPoint&, const DataPoint&);

class VpTreeBase
{
public:
    virtual void create(const std::vector<DataPoint>&) = 0;
    virtual void search(
        const DataPoint&, int, std::vector<DataPoint>&, std::vector<double>&)
        = 0;
    virtual ~VpTreeBase() {}
};

template <func dist_func>
class VpTree : public VpTreeBase
{
public:
    VpTree();

    // Function to create a new VpTree from data
    void create(const std::vector<DataPoint>& items);

    // Function that uses the tree to find the k nearest neighbors of target
    void search(const DataPoint& target, int k, std::vector<DataPoint>& results,
        std::vector<double>& distances);

private:
    std::default_random_engine m_gen;
    std::vector<DataPoint> m_items;

    // Single node of a VP tree
    // (has a point and radius; left children are closer to point than radius)
    struct Node
    {
        int index; // index of point in node
        double threshold; // radius(?)
        std::shared_ptr<Node> left; // points closer by than threshold
        std::shared_ptr<Node> right; // points farther away than threshold

        Node()
            : index(0)
            , threshold(0.0)
            , left(nullptr)
            , right(nullptr)
        {
        }
    };

    std::shared_ptr<Node> m_root;

    // An item on the intermediate result queue
    struct HeapItem
    {
        int index;
        double dist;

        HeapItem(int a_index, double a_dist)
            : index(a_index)
            , dist(a_dist)
        {
        }
        bool operator<(const HeapItem& o) const { return dist < o.dist; }
    };

    // Distance comparator for use in std::nth_element
    struct DistanceComparator
    {
        const DataPoint& item;

        DistanceComparator(const DataPoint& a_item)
            : item(a_item)
        {
        }
        bool operator()(const DataPoint& a, const DataPoint& b)
        {
            return dist_func(item, a) < dist_func(item, b);
        }
    };

    // Function that (recursively) fills the tree
    std::shared_ptr<Node> buildFromPoints(int lower, int upper);

    // Helper function that searches the tree
    void search(std::shared_ptr<Node> node, const DataPoint& target, int k,
        std::priority_queue<HeapItem>& heap, double& tau);
};

extern template class VpTree<detail::eucl_dist>;
extern template class VpTree<detail::jaccard_dist>;
}

#endif // TSNE_VPTREE_H
