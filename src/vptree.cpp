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

#include "vptree.h"

namespace tsne
{

// Function to create a new VpTree from data
void VpTree::create(const std::vector<DataPoint>& items)
{
    m_items = items; // make copy, because this object modifies by swapping in items
    m_root = buildFromPoints(0, items.size());
}

// Function that uses the tree to find the k nearest neighbors of target
void VpTree::search(const DataPoint& target, int k, std::vector<DataPoint>& results,
    std::vector<double>& distances)
{

    // Use a priority queue to store intermediate results on
    std::priority_queue<HeapItem> heap;

    // Variable that tracks the distance to the farthest point in our results
    double tau = std::numeric_limits<double>::max();

    // Perform the search
    search(m_root, target, k, heap, tau);

    // Gather final results
    results.clear();
    distances.clear();
    while (!heap.empty())
    {
        results.push_back(m_items[heap.top().index]);
        distances.push_back(heap.top().dist);
        heap.pop();
    }

    // Results are in reverse order
    std::reverse(std::begin(results), std::end(results));
    std::reverse(std::begin(distances), std::end(distances));
}

// Function that (recursively) fills the tree
std::shared_ptr<VpTree::Node> VpTree::buildFromPoints(int lower, int upper)
{
    if (upper == lower) // indicates that we're done here!
        return nullptr;

    // Lower index is center of current node
    auto node = std::make_shared<Node>();
    node->index = lower;

    if (upper - lower > 1) // if we did not arrive at leaf yet
    {
        // Choose an arbitrary point and move it to the start
        int i = (int)((double)rand() / RAND_MAX * (upper - lower - 1)) + lower;
        std::swap(m_items[lower], m_items[i]);

        // Partition around the median distance
        int median = (upper + lower) / 2;
        std::nth_element(m_items.begin() + lower + 1, m_items.begin() + median,
            m_items.begin() + upper, DistanceComparator(m_items[lower]));

        // Threshold of the new node will be the distance to the median
        node->threshold = eucl_dist(m_items[lower], m_items[median]);

        // Recursively build tree
        node->index = lower;
        node->left = buildFromPoints(lower + 1, median);
        node->right = buildFromPoints(median, upper);
    }

    // Return result
    return node;
}

// Helper function that searches the tree
void VpTree::search(std::shared_ptr<Node> node, const DataPoint& target, int k,
    std::priority_queue<HeapItem>& heap, double& tau)
{
    if (node == nullptr)
        return; // indicates that we're done here

    // Compute distance between target and current node
    double dist = eucl_dist(m_items[node->index], target);

    // If current node within radius tau
    if (dist < tau)
    {
        if (static_cast<int>(heap.size()) == k)
            heap.pop(); // remove furthest node from result list (if we already have k
                        // results)
        heap.push(HeapItem(node->index, dist)); // add current node to result list
        if (static_cast<int>(heap.size()) == k)
            tau = heap.top().dist; // update value of tau (farthest point in result)
    }

    // Return if we arrived at a leaf
    if (node->left == nullptr && node->right == nullptr)
    {
        return;
    }

    // If the target lies within the radius of ball
    if (dist < node->threshold)
    {
        // if there can still be neighbors inside the ball, recursively search left
        // child first
        if (dist - tau <= node->threshold)
            search(node->left, target, k, heap, tau);

        // if there can still be neighbors outside the ball, recursively search right
        // child
        if (dist + tau >= node->threshold)
            search(node->right, target, k, heap, tau);

        // If the target lies outsize the radius of the ball
    }
    else
    {
        // if there can still be neighbors outside the ball, recursively search right
        // child first
        if (dist + tau >= node->threshold)
            search(node->right, target, k, heap, tau);

        // if there can still be neighbors inside the ball, recursively search left child
        if (dist - tau <= node->threshold)
            search(node->left, target, k, heap, tau);
    }
}

double VpTree::eucl_dist(const DataPoint& t1, const DataPoint& t2)
{
    double dd = 0.0;
    auto it1 = t1.getIt();
    auto it2 = t2.getIt();
    double diff;
    for (int d = 0; d < t1.getDim(); d++)
    {
        diff = *(it1 + d) - *(it2 + d);
        dd += diff * diff;
    }
    return std::sqrt(dd);
}
}
