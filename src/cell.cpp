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

#include "cell.h"
#include <cmath>
#include <numeric>
#include <algorithm>

namespace tsne
{
Cell::Cell(int inp_dimension)
    : m_dimension(inp_dimension)
    , m_corner()
    , m_width()
    , m_max_width(0)
{
}

Cell::Cell(int inp_dimension, std::shared_ptr<std::vector<double>> inp_corner,
    std::shared_ptr<std::vector<double>> inp_width)
    : m_dimension(inp_dimension)
    , m_corner(std::move(inp_corner))
    , m_width(std::move(inp_width))
    , m_max_width(0)
{
    calculateMaxWidth();
}

Cell::Cell(int inp_dimension, const std::vector<double>& inp_corner,
    const std::vector<double>& inp_width)
    : m_dimension(inp_dimension)
    , m_corner(std::make_shared<std::vector<double>>(inp_corner))
    , m_width(std::make_shared<std::vector<double>>(inp_width))
    , m_max_width(0)
{
    calculateMaxWidth();
}

void Cell::calculateMaxWidth()
{
    for (int d = 0; d < m_dimension; d++)
        m_max_width = std::max(m_max_width, (*m_width)[d]);
}

// Checks whether a point lies in a cell
bool Cell::containsPoint(std::shared_ptr<std::vector<double>> point, int offset) const
{
    for (int d = 0; d < m_dimension; d++)
    {
        if (std::abs((*m_corner)[d] - (*point)[d + offset]) > (*m_width)[d])
            return false;
    }
    return true;
}
}
