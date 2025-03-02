#include <iostream>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <algorithm>
#include <set>
#include <stack>
#include "Graph.hpp"

using namespace std;

Graph::Graph(const Graph &graph) {
    nodeIndex = graph.nodeIndex;
    zeroDegreeNodes = graph.zeroDegreeNodes;
    nextUnusedId = graph.nextUnusedId;
    mapping = graph.mapping;
    edgeBuffer = new vector<uint32_t>(*(graph.edgeBuffer));
    if (graph.mapping) {
        idToPos = new unordered_map<uint32_t, uint32_t>(*(graph.idToPos));
        posToId = new vector<uint32_t>(*(graph.posToId));
    }
}

Graph& Graph::operator=(const Graph &graph) {
    if (this != &graph) {
        nodeIndex = graph.nodeIndex;
        zeroDegreeNodes = graph.zeroDegreeNodes;
        nextUnusedId = graph.nextUnusedId;
        mapping = graph.mapping;
        edgeBuffer = new vector<uint32_t>(*(graph.edgeBuffer));
        if (graph.mapping) {
            idToPos = new unordered_map<uint32_t, uint32_t>(*(graph.idToPos));
            posToId = new vector<uint32_t>(*(graph.posToId));
        }
    }
    return *this;
}

Graph::~Graph() {
    delete edgeBuffer;
    if (mapping) {
        delete idToPos;
        delete posToId;
    }
}

Graph::GraphTraversal::GraphTraversal(const Graph &graph) {
    curNode = NONE;
    curEdgeOffset = NONE;
    graph.getNextNode(*this);
}

Graph::GraphTraversal::GraphTraversal(const Graph &graph, const uint32_t &node) {
    graph.goToNode(node, *this);
}

/* Could be potentially replaced by structures that are updated on the fly as degrees change,
 * but is pretty fast and memory efficient nonetheless, even on huge graphs */
void Graph::getMaxNodeDegree(uint32_t &node, uint32_t &maxDegree, const uint32_t &bound) const {
    node = NONE;
    maxDegree = 0;
    for (uint32_t i = 0 ; i < nodeIndex.size() ; i++) {
        if (!nodeIndex[i].removed && nodeIndex[i].edges > maxDegree) {
            node = (!mapping ? i : posToId->at(i));
            maxDegree = nodeIndex[i].edges;
            if (bound != NONE && maxDegree >= bound) {
                return;
            }
        }
    }
}

/* Optimized for short funnel detection to not necesssarily return the min degree,
 * if it is less than 3 */
void Graph::getMinDegree(uint32_t &minDegree) const {
    minDegree = NONE;
    for (uint32_t i = 0 ; i < nodeIndex.size() ; i++) {
        if (!nodeIndex[i].removed && nodeIndex[i].edges && nodeIndex[i].edges < minDegree) {
            minDegree = nodeIndex[i].edges;
            if (minDegree < 3) {
                return;
            }
        }
    }
}

uint32_t Graph::getTotalEdges() const {
    uint32_t count = 0;
    for (uint32_t pos = 0 ; pos < nodeIndex.size() ; pos++) {
        if (!nodeIndex[pos].removed) {
            count += nodeIndex[pos].edges;
        }
    }
    return count;
}

uint32_t Graph::getNumberOfDegreeNeighbors(const uint32_t &node, const uint32_t &degree, const uint32_t &atLeast) const {
    uint32_t count = 0;
    uint32_t pos = (!mapping ? node : idToPos->at(node));
    uint32_t neighborCount = nodeIndex[pos].edges;
    uint32_t nextNodeOffset = (pos == nodeIndex.size()-1 ? edgeBuffer->size() : nodeIndex[pos+1].offset);
    for (uint32_t i = nodeIndex[pos].offset ; i < nextNodeOffset && neighborCount ; i++) {
        uint32_t nPos = (!mapping ? (*edgeBuffer)[i] : idToPos->at((*edgeBuffer)[i]));
        if (!nodeIndex[nPos].removed) {
            neighborCount--;
            if (nodeIndex[nPos].edges == degree) {
                count++;
                if (atLeast && count == atLeast) {
                    break;
                }
            }
        }
    }
    return count;
}

/* Does not include neighbors at distance 2 that can also be found at distance 1.
 * If the last 2 optional arguments are given, it also counts the number of neighbors
 * at distance 2 with degree less than 'degree' */
void Graph::getNeighborsAtDistance2(const uint32_t &node, unordered_set<uint32_t> &neighbors, const uint32_t &degree, uint32_t *count) const {
    if (degree != NONE && count != NULL) {
        *count = 0;
    }
    uint32_t pos = (!mapping ? node : idToPos->at(node));
    uint32_t neighborCount = nodeIndex[pos].edges;
    uint32_t nextNodeOffset = (pos == nodeIndex.size()-1 ? edgeBuffer->size() : nodeIndex[pos+1].offset);
    for (uint32_t i = nodeIndex[pos].offset ; i < nextNodeOffset && neighborCount ; i++) {
        uint32_t nPos = (!mapping ? (*edgeBuffer)[i] : idToPos->at((*edgeBuffer)[i]));
        if (!nodeIndex[nPos].removed) {
            neighborCount--;
            uint32_t neighborCount2 = nodeIndex[nPos].edges;
            uint32_t nextNodeOffset2 = (nPos == nodeIndex.size()-1 ? edgeBuffer->size() : nodeIndex[nPos+1].offset);
            for (uint32_t j = nodeIndex[nPos].offset ; j < nextNodeOffset2 && neighborCount2 ; j++) {
                uint32_t id = (*edgeBuffer)[j];
                uint32_t nPos2 = (!mapping ? id : idToPos->at(id));
                if (!nodeIndex[nPos2].removed) {
                    neighborCount2--;
                    if (id != node && !edgeExists(id, node)) {
                        neighbors.insert(id);
                        if (degree != NONE && count != NULL && nodeIndex[nPos2].edges < degree) {
                            (*count)++;
                        }
                    }
                }
            }
        }
    }
}


void Graph::remove(const uint32_t &node, ReduceInfo &reduceInfo, const bool &removeZeroDegreeNodes) {
    remove(std::vector<uint32_t>(1, node), reduceInfo, false, NULL, removeZeroDegreeNodes);
}

/* Rebuild structures, completely removing nodes that are marked as removed
 * and collecting zero degree nodes */
void Graph::rebuild(ReduceInfo &reduceInfo) {
    if (!reduceInfo.nodesRemoved) {
        return;
    }
    vector<NodeInfo> nodeIndex;
    uint32_t newNodes = (this->nodeIndex.size() > reduceInfo.nodesRemoved ? this->nodeIndex.size() - reduceInfo.nodesRemoved : this->nodeIndex.size());
    uint32_t newEdges = this->getTotalEdges();
    nodeIndex.reserve(newNodes);
    vector<uint32_t> *edgeBuffer = new vector<uint32_t>();
    edgeBuffer->reserve(newEdges);
    unordered_map<uint32_t, uint32_t> *idToPos = new unordered_map<uint32_t, uint32_t>();
    vector<uint32_t> *posToId = new vector<uint32_t>();
    posToId->reserve(newNodes);
    uint32_t offset = 0;

    for (uint32_t pos = 0 ; pos < this->nodeIndex.size() ; pos++) {
        if (this->nodeIndex[pos].removed) {
            continue;
        }
        uint32_t node = (!mapping ? pos : (*this->posToId)[pos]);
        if (!this->nodeIndex[pos].edges) {
            //cout << "Found node " << node << " with no edges at rebuilding\n";
            zeroDegreeNodes.push_back(node);
            continue;
        }
        uint32_t edges = 0;
        uint32_t nextNodeOffset = (pos == this->nodeIndex.size()-1 ? this->edgeBuffer->size() : this->nodeIndex[pos+1].offset);
        /* Don't add neighbors that are marked as removed */
        for (uint32_t i = this->nodeIndex[pos].offset ; i < nextNodeOffset ; i++) {
            uint32_t nPos = (!this->mapping ? (*this->edgeBuffer)[i] : this->idToPos->at((*this->edgeBuffer)[i]));
            if (!this->nodeIndex[nPos].removed) {
                edgeBuffer->push_back((*this->edgeBuffer)[i]);
                edges++;
            }
            if (edges == this->nodeIndex[pos].edges) {
                break;
            }
        }
        assert(edges > 0);
        idToPos->insert({node, nodeIndex.size()});
        posToId->push_back(node);
        nodeIndex.push_back(Graph::NodeInfo(offset, edges));
        offset += edges;
    }
    this->mapping = true;
    if (this->idToPos != NULL) {
        delete this->idToPos;
    }
    if (this->posToId != NULL) {
        delete this->posToId;
    }
    this->idToPos = idToPos;
    this->posToId = posToId;
    this->nodeIndex = nodeIndex;
    delete this->edgeBuffer;
    this->edgeBuffer = edgeBuffer;
    reduceInfo.nodesRemoved = 0;
}

void Graph::rebuildFromNodes(unordered_set<uint32_t> &nodes) {
    zeroDegreeNodes.clear();
    if (!nodes.size()) {
        nodeIndex.clear();
        edgeBuffer->clear();
        if (mapping) {
            idToPos->clear();
            posToId->clear();
        }
        return;
    }
    vector<NodeInfo> nodeIndex;
    nodeIndex.reserve(nodes.size());
    vector<uint32_t> *edgeBuffer = new vector<uint32_t>();
    unordered_map<uint32_t, uint32_t> *idToPos = new unordered_map<uint32_t, uint32_t>();
    vector<uint32_t> *posToId = new vector<uint32_t>();
    posToId->reserve(nodes.size());
    uint32_t offset = 0;

    for (auto node: nodes) {
        uint32_t edges = 0;
        uint32_t pos = (!mapping ? node : this->idToPos->at(node));
        uint32_t nextNodeOffset = (pos == this->nodeIndex.size()-1 ? this->edgeBuffer->size() : this->nodeIndex[pos+1].offset);
        /* Don't add neighbors that are marked as removed or are outside of nodes struct */
        for (uint32_t i = this->nodeIndex[pos].offset ; i < nextNodeOffset ; i++) {
            uint32_t nPos = (!this->mapping ? (*this->edgeBuffer)[i] : this->idToPos->at((*this->edgeBuffer)[i]));
            if (!this->nodeIndex[nPos].removed || nodes.find((*this->edgeBuffer)[i]) != nodes.end()) {
                edgeBuffer->push_back((*this->edgeBuffer)[i]);
                edges++;
            }
            if (edges == this->nodeIndex[pos].edges) {
                break;
            }
        }
        if (!edges) {
            zeroDegreeNodes.push_back(node);
        } else {
            idToPos->insert({node, nodeIndex.size()});
            posToId->push_back(node);
            nodeIndex.push_back(Graph::NodeInfo(offset, edges));
            offset += edges;
        }
    }
    this->mapping = true;
    if (this->idToPos != NULL) {
        delete this->idToPos;
    }
    if (this->posToId != NULL) {
        delete this->posToId;
    }
    this->idToPos = idToPos;
    this->posToId = posToId;
    this->nodeIndex = nodeIndex;
    delete this->edgeBuffer;
    this->edgeBuffer = edgeBuffer;
}

/* Contract 'nodes' and 'neighbors' to a single node.
 * It is taken for granted that the only neighbors of 'nodes' are 'neighbors' */
uint32_t Graph::contractToSingleNode(const vector<uint32_t> &nodes, const vector<uint32_t> &neighbors, ReduceInfo &reduceInfo) {
    uint32_t newNode = nextUnusedId;
    assert(++nextUnusedId != 0);
    assert(!mapping || mapping && idToPos->find(newNode) == idToPos->end());
    set<uint32_t> newNeighbors;
    for (auto it = neighbors.begin() ; it != neighbors.end() ; it++) {
        GraphTraversal graphTraversal(*this, *it);
        while (graphTraversal.curEdgeOffset != NONE) {
            uint32_t neighbor = (*edgeBuffer)[graphTraversal.curEdgeOffset];
            if (find(nodes.begin(), nodes.end(), neighbor) == nodes.end() && find(neighbors.begin(), neighbors.end(), neighbor) == neighbors.end()) {
                if (newNeighbors.insert(neighbor).second) {
                    replaceNeighbor(neighbor, *it, newNode);
                    uint32_t pos = (!mapping ? neighbor : idToPos->at(neighbor));
                    nodeIndex[pos].edges++;
                }
            }
            getNextEdge(graphTraversal);
        }
    }
    reduceInfo.nodesRemoved--;
    if (!newNeighbors.size()) {
        zeroDegreeNodes.push_back(newNode);
    } else {
        uint32_t offset = edgeBuffer->size();
        edgeBuffer->reserve(edgeBuffer->size() + newNeighbors.size());
        copy(newNeighbors.begin(), newNeighbors.end(), back_inserter(*edgeBuffer));
        nodeIndex.push_back(NodeInfo(offset, newNeighbors.size()));
        if (mapping) {
            idToPos->insert({newNode, nodeIndex.size() - 1});
            posToId->push_back(newNode);
        }
    }
    return newNode;
}

void Graph::replaceNeighbor(const uint32_t &node, const uint32_t &oldNeighbor, const uint32_t &newNeighbor) {
    uint32_t offset;
    offset = findEdgeOffset(node, oldNeighbor);
    assert(offset != NONE);
    uint32_t pos = (!mapping ? node : idToPos->at(node));
    uint32_t endOffset = (pos == nodeIndex.size()-1 ? edgeBuffer->size() : nodeIndex[pos+1].offset);
    auto it = edgeBuffer->begin();
    move(it + offset + 1, it + endOffset, it + offset);
    (*edgeBuffer)[endOffset-1] = newNeighbor;
}

uint32_t Graph::getNextNodeWithIdenticalNeighbors(const uint32_t &previousNode, const vector<uint32_t> &neighbors) const {
    uint32_t pos = (!mapping ? previousNode : idToPos->at(previousNode));
    for (pos = pos+1 ; pos < nodeIndex.size() ; pos++) {
        if (!nodeIndex[pos].removed && nodeIndex[pos].edges == neighbors.size()) {
            uint32_t neighborCount = neighbors.size();
            uint32_t nextNodeOffset = (pos == nodeIndex.size()-1 ? edgeBuffer->size() : nodeIndex[pos+1].offset);
            for (uint32_t offset = nodeIndex[pos].offset ; offset  < nextNodeOffset && neighborCount; offset++) {
                uint32_t nPos = (!mapping ? (*edgeBuffer)[offset] : idToPos->at((*edgeBuffer)[offset]));
                if (!nodeIndex[nPos].removed && find(neighbors.begin(), neighbors.end(), (*edgeBuffer)[offset]) != neighbors.end()) {
                    neighborCount--;
                }
            }
            if (!neighborCount) {
                return (!mapping ? pos : posToId->at(pos));
            }
        }
    }
    return NONE;
}

uint32_t Graph::getNodeWithOneUncommonNeighbor(const vector<uint32_t> &neighbors, uint32_t &uncommonNeighbor) const {
    uncommonNeighbor = NONE;
    for (uint32_t pos = 0 ; pos < nodeIndex.size() ; pos++) {
        if (!nodeIndex[pos].removed && nodeIndex[pos].edges == 3 || nodeIndex[pos].edges == 4) { // Optimization for 3-4 structures reduction
            uint32_t node = (!mapping ? pos : posToId->at(pos));
            vector<uint32_t> newNeighbors;
            gatherNeighbors(node, newNeighbors);
            if (setsHaveKUncommonElements(neighbors, newNeighbors, 1, uncommonNeighbor)) {
                return node;
            }
        }
    }
    return NONE;
}

void Graph::getExtendedGrandchildren(Graph::GraphTraversal &graphTraversal, unordered_set<uint32_t> &extendedGrandchildren, bool *isUnconfined, const bool &stopAtFirst) const {
    while (graphTraversal.curEdgeOffset != NONE) {
        uint32_t neighbor = (*edgeBuffer)[graphTraversal.curEdgeOffset];
        uint32_t outerNeighbor;
        bool exactlyOne;
        getOuterNeighbor(graphTraversal.curNode, neighbor, outerNeighbor, exactlyOne);
        if (isUnconfined != NULL && outerNeighbor == NONE) {
            *isUnconfined = true;
            break;
        } else if (exactlyOne) {
            extendedGrandchildren.insert(outerNeighbor);
            if (stopAtFirst) {
                return;
            }
        }
        getNextEdge(graphTraversal);
    }
}

uint32_t Graph::getOptimalNodeTheta3(const uint32_t initialMaxDegreeNode, const uint32_t &initialMaxDegree) const {
    uint32_t maxDegree = initialMaxDegree;
    uint32_t maxDegreeNode = initialMaxDegreeNode;
    if (maxDegree >= 4) {
        uint32_t maxNeighborsAtDistance2 = 0;
        GraphTraversal graphTraversal(*this, maxDegreeNode);
        while (graphTraversal.curNode != NONE) {
            unordered_set<uint32_t> neighborsAtDistance2;
            getNeighborsAtDistance2(graphTraversal.curNode, neighborsAtDistance2);
            if (neighborsAtDistance2.size() > maxNeighborsAtDistance2) {
                maxDegreeNode = graphTraversal.curNode;
                maxNeighborsAtDistance2 = neighborsAtDistance2.size();
            }
            getNextNode(graphTraversal);
        }
    }
    return maxDegreeNode;
}

uint32_t Graph::getOptimalDegree4Node() const {
    uint32_t node = getOptimalDegree4Node1();
    if (node == NONE) {
        node = getOptimalDegree4Node2();
        if (node == NONE) {
            uint32_t maxNodeWithCond = NONE;
            uint32_t maxNode = NONE;
            getOptimalDegree4Node3(maxNodeWithCond, maxNode);
            if (maxNodeWithCond != NONE) {
                node = maxNodeWithCond;
            } else {
                node = maxNode;
            }
        }
    }
    //cout << "branching on optimal degree 4 node " << node << "\n";
    return node;
}

uint32_t Graph::getOptimalDegree4Node1() const {
    GraphTraversal graphTraversal(*this);
    while (graphTraversal.curNode != NONE) {
        if (getNodeDegree(graphTraversal.curNode) == 4) {
            vector<uint32_t> neighborsV;
            gatherNeighbors(graphTraversal.curNode, neighborsV);
            for (uint32_t neighborV: neighborsV) {
                if (getNodeDegree(neighborV) == 3) {
                    vector<uint32_t> neighborsN;
                    gatherNeighbors(neighborV, neighborsN);
                    if (edgeExists(neighborsN[0], neighborsN[1]) || edgeExists(neighborsN[0], neighborsN[2]) || edgeExists(neighborsN[1], neighborsN[2])) {
                        return graphTraversal.curNode;
                    }
                }
            }
        }
        getNextNode(graphTraversal);
    }
    return NONE;
}

uint32_t Graph::getOptimalDegree4Node2() const {
    GraphTraversal graphTraversal(*this);
    while (graphTraversal.curNode != NONE) {
        if (getNodeDegree(graphTraversal.curNode) == 4) {
            vector<uint32_t> neighborsV;
            gatherNeighbors(graphTraversal.curNode, neighborsV);
            for (uint32_t neighborV: neighborsV) {
                if (getNodeDegree(neighborV) == 3) {
                    vector<uint32_t> neighborsN;
                    gatherNeighbors(neighborV, neighborsN);
                    for (uint32_t neighborN: neighborsN) {
                        if (neighborN != graphTraversal.curNode && getNodeDegree(neighborN) == 4) {
                            return graphTraversal.curNode;
                        }
                    }
                }
            }
        }
        getNextNode(graphTraversal);
    }
    return NONE;
}

void Graph::getOptimalDegree4Node3(uint32_t &maxNodeWithCond, uint32_t &maxNode) const {
    uint32_t maxDegree3NeighborsWithCond = NONE;
    maxNodeWithCond = NONE;
    uint32_t maxDegree3Neighbors = NONE;
    maxNode = NONE;
    GraphTraversal graphTraversal(*this);
    while (graphTraversal.curNode != NONE) {
        if (getNodeDegree(graphTraversal.curNode) == 4) {
            vector<uint32_t> neighborsV;
            gatherNeighbors(graphTraversal.curNode, neighborsV);
            uint32_t degree3Neighbors = 0;
            for (uint32_t neighborV: neighborsV) {
                if (getNodeDegree(neighborV) == 3) {
                    degree3Neighbors++;
                }
            }
            if (maxDegree3Neighbors == NONE || degree3Neighbors > maxDegree3Neighbors) {
                maxDegree3Neighbors = degree3Neighbors;
                maxNode = graphTraversal.curNode;
            }
            if (maxDegree3NeighborsWithCond == NONE || degree3Neighbors > maxDegree3NeighborsWithCond) {
                for (uint32_t neighborV: neighborsV) {
                    if (getNodeDegree(neighborV) == 4) {
                        vector<uint32_t> neighborsN;
                        gatherNeighbors(neighborV, neighborsN);
                        uint32_t a, b, c;
                        uint32_t v = graphTraversal.curNode;
                        uint32_t i = 0;
                        for (uint32_t n: neighborsN) {
                            if (n != v) {
                                if (i == 0) {
                                    a = n;
                                } else if (i == 1) {
                                    b = n;
                                } else if (i == 2) {
                                    c = n;
                                }
                                i++;
                            }
                        }
                        bool ab = edgeExists(a, b);
                        bool bc = edgeExists(b, c);
                        bool ac = edgeExists(a, c);
                        bool va = edgeExists(v, a);
                        bool vb = edgeExists(v, b);
                        bool vc = edgeExists(v, c);
                        if (!va && !vb && !vc && (ab && !bc && !ac || bc && !ab && !ac || ac && !ab && !bc)) {
                            maxDegree3NeighborsWithCond = degree3Neighbors;
                            maxNodeWithCond = graphTraversal.curNode;
                        }
                    }
                }
            }
        }
        getNextNode(graphTraversal);
    }
}

bool Graph::get4Cycle(vector<uint32_t> &optimalCycle) const {
    optimalCycle.clear();
    uint32_t maxDegree3NodesCount = NONE;
    GraphTraversal graphTraversal(*this);
    while (graphTraversal.curNode != NONE) {
        if (getNodeDegree(graphTraversal.curNode) == 4) {
            vector<uint32_t> neighbors;
            gatherNeighbors(graphTraversal.curNode, neighbors);
            uint32_t b, d;
            for (uint32_t i = 0 ; i < 6 ; i++) {
                switch (i) {
                    case 0:
                        b = neighbors[0];
                        d = neighbors[1];
                        break;
                    case 1:
                        b = neighbors[0];
                        d = neighbors[2];
                        break;
                    case 2:
                        b = neighbors[0];
                        d = neighbors[3];
                        break;
                    case 3:
                        b = neighbors[1];
                        d = neighbors[2];
                        break;
                    case 4:
                        b = neighbors[1];
                        d = neighbors[3];
                        break;
                    case 5:
                        b = neighbors[2];
                        d = neighbors[3];
                        break;
                    default:
                        assert(false);
                }
                vector<uint32_t> commonNeighbors;
                getCommonNeighbors(b, d, commonNeighbors);
                for (auto c: commonNeighbors) {
                    if (c == graphTraversal.curNode) {
                        continue;
                    }
                    uint32_t degree3NodesCount = 0;
                    if (getNodeDegree(b) == 3) {
                        degree3NodesCount++;
                    }
                    if (getNodeDegree(c) == 3) {
                        degree3NodesCount++;
                    }
                    if (getNodeDegree(d) == 3) {
                        degree3NodesCount++;
                    }
                    if (maxDegree3NodesCount == NONE || degree3NodesCount > maxDegree3NodesCount) {
                        maxDegree3NodesCount = degree3NodesCount;
                        optimalCycle.clear();
                        optimalCycle.push_back(graphTraversal.curNode);
                        optimalCycle.push_back(b);
                        optimalCycle.push_back(c);
                        optimalCycle.push_back(d);
                        if (maxDegree3NodesCount == 3) {
                            return true;
                        }
                    }
                }
            }
        }
        getNextNode(graphTraversal);
    }
    return maxDegree3NodesCount != NONE;
}

bool Graph::get4CycleTheta3(vector<uint32_t> &optimalCycle) const {
    optimalCycle.clear();
    uint32_t maxDegree3NodesCount = NONE;
    GraphTraversal graphTraversal(*this);
    while (graphTraversal.curNode != NONE) {
        vector<uint32_t> neighbors;
        gatherNeighbors(graphTraversal.curNode, neighbors);
        uint32_t b, d;
        for (uint32_t i = 0 ; i < neighbors.size() ; i++) {
            for (uint32_t j = i+1 ; j < neighbors.size() ; j++) {
                b = neighbors[i];
                d = neighbors[j];
                vector<uint32_t> commonNeighbors;
                getCommonNeighbors(b, d, commonNeighbors);
                for (auto c: commonNeighbors) {
                    if (c == graphTraversal.curNode) {
                        continue;
                    }
                    uint32_t degreeA = getNodeDegree(graphTraversal.curNode);
                    uint32_t degreeB = getNodeDegree(b);
                    uint32_t degreeC = getNodeDegree(c);
                    uint32_t degreeD = getNodeDegree(d);
                    if (degreeA == 3 && degreeC == 3 || degreeB == 3 && degreeD == 3) {
                        optimalCycle.clear();
                        optimalCycle.push_back(graphTraversal.curNode);
                        optimalCycle.push_back(b);
                        optimalCycle.push_back(c);
                        optimalCycle.push_back(d);
                        return true;
                    }
                    uint32_t degree3NodesCount = 0;
                    if (degreeA == 3) {
                        degree3NodesCount++;
                    }
                    if (degreeB == 3) {
                        degree3NodesCount++;
                    }
                    if (degreeC == 3) {
                        degree3NodesCount++;
                    }
                    if (degreeD == 3) {
                        degree3NodesCount++;
                    }
                    if (maxDegree3NodesCount == NONE || degree3NodesCount > maxDegree3NodesCount) {
                        maxDegree3NodesCount = degree3NodesCount;
                        optimalCycle.clear();
                        optimalCycle.push_back(graphTraversal.curNode);
                        optimalCycle.push_back(b);
                        optimalCycle.push_back(c);
                        optimalCycle.push_back(d);
                    }
                }
            }
        }
        getNextNode(graphTraversal);
    }
    return maxDegree3NodesCount != NONE;
}

uint32_t Graph::getEffectiveNodeMeasure(const uint32_t &bound) const {
    uint32_t measure = 0;
    GraphTraversal graphTraversal(*this);
    while (graphTraversal.curNode != NONE) {
        uint32_t degree = getNodeDegree(graphTraversal.curNode);
        if (degree > 2) {
            measure += (degree - 2);
        }
        if (bound != NONE && measure > bound) {
            break;
        }
        getNextNode(graphTraversal);
    }
    return measure;
}

bool Graph::getEffectiveNodeOrOptimalFunnel(uint32_t &effectiveNode, uint32_t &nodeV, uint32_t &nodeA) const {
    effectiveNode = nodeV = nodeA = NONE;
    uint32_t measure = getEffectiveNodeMeasure();
    vector<Funnel> funnels;
    Funnel fourFunnel;
    if (getFunnels(funnels, &measure, &effectiveNode, &fourFunnel)) {
        if (effectiveNode != NONE) {
            return true;
        } else {
            nodeV = fourFunnel.v;
            nodeA = fourFunnel.a;
            return true;
        }
    } else {
        for (uint32_t i = 0 ; i < 3 ; i++) {
            for (auto &funnel: funnels) {
                bool optimal = false;
                if (!i && getNodeDegree(funnel.a) >= 4 || i == 1 && isInTriangle(funnel.a)) {
                    optimal = true;
                }
                else if (i == 2) {
                    unordered_set<uint32_t> nodes;
                    nodes.insert(funnel.v);
                    nodes.insert(funnel.a);
                    unordered_set<uint32_t> neighbors;
                    gatherAllNeighbors(nodes, neighbors);
                    for (auto neighbor: neighbors) {
                        if (getNodeDegree(neighbor) >= 4) {
                            optimal = true;
                            break;
                        }
                    }
                } else if (i == 3) {
                    Graph newGraph = *this;
                    vector<uint32_t> neighborsA;
                    gatherNeighbors(funnel.a, neighborsA);
                    neighborsA.push_back(funnel.a);
                    ReduceInfo dummy;
                    newGraph.remove(neighborsA, dummy);
                    optimal = newGraph.isFineInstance();
                }
                if (optimal) {
                    nodeV = fourFunnel.v;
                    nodeA = fourFunnel.a;
                    return true;
                }
            }
        }
    }
    return true;
}

bool Graph::getGoodFunnel(uint32_t &node1, uint32_t &node2) const {
    vector<Funnel> funnels;
    if (getFunnels(funnels)) {
        Funnel &funnel = funnels.back();
        node1 = funnel.a;
        node2 = funnel.v;
        //cout << "branching on good funnel ";
        //funnel.print();
        return true;
    } else {
        for (uint32_t i = 0 ; i < 2 ; i++) {
            for (auto &funnel: funnels) {
                uint32_t bDegree = getNodeDegree(funnel.b);
                uint32_t cDegree = getNodeDegree(funnel.c);
                if (!i && bDegree == 4 && cDegree == 4 || i && (bDegree == 4 || cDegree == 4)) {
                    node1 = funnel.a;
                    node2 = funnel.v;
                    //cout << "branching on good funnel ";
                    //funnel.print();
                    return true;
                }
            }
        }
    }
    return false;
}

bool Graph::getGoodFunnelTheta5(uint32_t &node1, uint32_t &node2) const {
    GraphTraversal graphTraversal(*this);
    while (graphTraversal.curNode != NONE) {
        uint32_t nodeV = graphTraversal.curNode;
        uint32_t vDegree = getNodeDegree(nodeV);
        if (vDegree == 3) {
            //cout << "nodeV " << nodeV << endl;
            vector<uint32_t> neighborsV;
            gatherNeighbors(nodeV, neighborsV);
            for (uint32_t i = 0 ; i < neighborsV.size() ; i++) {
                uint32_t nodeA = neighborsV[i];
                uint32_t nodeB, nodeC;
                if (i == 0) {
                    nodeB = neighborsV[1];
                    nodeC = neighborsV[2];
                } else if (i == 1) {
                    nodeB = neighborsV[0];
                    nodeC = neighborsV[2];
                } else if (i == 2) {
                    nodeB = neighborsV[0];
                    nodeC = neighborsV[1];
                } else {
                    nodeB = neighborsV[0];
                    nodeC = neighborsV[1];
                }
                if (edgeExists(nodeB, nodeC))  {
                    vector<uint32_t> neighborsA;
                    gatherNeighbors(nodeA, neighborsA);
                    for (auto neighborA: neighborsA) {
                        if (neighborA != nodeV && getNodeDegree(neighborA) == 5) {
                            vector<uint32_t> commonNeighbors;
                            getCommonNeighbors(neighborA, nodeV, commonNeighbors, 3);
                            if (commonNeighbors.size() >= 3) {
                                node1 = nodeA;
                                node2 = nodeV;
                                //cout << "branching on theta-5 good funnel " << node1 << "-" << node2 << "\n";
                                return true;
                            }
                        }
                    }
                }
            }
        }
        getNextNode(graphTraversal);
    }
    return false;
}

bool Graph::getFunnels(vector<Funnel> &funnels, const uint32_t *measure, uint32_t *effectiveNode, Funnel *fourFunnel) const {
    GraphTraversal graphTraversal(*this);
    while (graphTraversal.curNode != NONE) {
        uint32_t nodeV = graphTraversal.curNode;
        uint32_t vDegree = getNodeDegree(nodeV);
        if (vDegree == 3 || vDegree == 4) {
            vector<uint32_t> neighborsV;
            gatherNeighbors(nodeV, neighborsV);
            for (uint32_t i = 0 ; i < neighborsV.size() ; i++) {
                uint32_t nodeA = neighborsV[i];
                uint32_t nodeB, nodeC, nodeD;
                if (i == 0) {
                    nodeB = neighborsV[1];
                    nodeC = neighborsV[2];
                    nodeD = (neighborsV.size() == 3 ? NONE : neighborsV[3]);
                } else if (i == 1) {
                    nodeB = neighborsV[0];
                    nodeC = neighborsV[2];
                    nodeD = (neighborsV.size() == 3 ? NONE : neighborsV[3]);
                } else if (i == 2) {
                    nodeB = neighborsV[0];
                    nodeC = neighborsV[1];
                    nodeD = (neighborsV.size() == 3 ? NONE : neighborsV[3]);
                } else {
                    nodeB = neighborsV[0];
                    nodeC = neighborsV[1];
                    nodeD = (neighborsV.size() == 3 ? NONE : neighborsV[2]);
                }
                if (edgeExists(nodeB, nodeC) && (vDegree == 3 || vDegree == 4 && edgeExists(nodeB, nodeD) && edgeExists(nodeC, nodeD)))  {
                    if (measure == NULL) {
                        funnels.push_back(Funnel(nodeA, nodeB, nodeC, nodeD, nodeV));
                        if (vDegree == 4 || (getNodeDegree(nodeA) == 4 && (getNodeDegree(nodeB) == 4 || getNodeDegree(nodeC) == 4))) {
                            return true;
                        }
                    } else {
                        uint32_t degree3NodesCount;
                        if (getNodeDegree(nodeA) == 3) {
                            degree3NodesCount++;
                        }
                        if (getNodeDegree(nodeB) == 3) {
                            degree3NodesCount++;
                        }
                        if (getNodeDegree(nodeC) == 3) {
                            degree3NodesCount++;
                        }
                        if (vDegree == 4 && getNodeDegree(nodeD) == 3) {
                            degree3NodesCount++;
                        }
                        if (degree3NodesCount >= 3) {
                            Graph newGraph = *this;
                            unordered_set<uint32_t> extendedGrandchildren;
                            goToNode(nodeV, graphTraversal);
                            getExtendedGrandchildren(graphTraversal, extendedGrandchildren);
                            extendedGrandchildren.insert(nodeV);
                            std::unordered_set<uint32_t> neighbors;
                            gatherAllNeighbors(extendedGrandchildren, neighbors);
                            unordered_set<uint32_t> *smaller = &neighbors;
                            unordered_set<uint32_t> *bigger = &extendedGrandchildren;
                            if (smaller->size() > bigger->size()) {
                                std::unordered_set<uint32_t> *tmp = smaller;
                                smaller = bigger;
                                bigger = tmp;
                            }
                            smaller->insert(bigger->begin(), bigger->end());
                            ReduceInfo dummy;
                            newGraph.remove(vector<uint32_t>(smaller->begin(), smaller->end()), dummy);
                            if (newGraph.getEffectiveNodeMeasure(*measure - 20) <= *measure - 20) {
                                *effectiveNode = nodeV;
                                return true;
                            }
                        }
                        Funnel funnel(nodeA, nodeB, nodeC, nodeD, nodeV);
                        if (vDegree == 4) {
                            *fourFunnel = funnel;
                            return true;
                        } else {
                            funnels.push_back(funnel);
                        }
                    }
                }
            }
        }
        getNextNode(graphTraversal);
    }
    return false;
}

bool Graph::getGoodPair(uint32_t &node1, uint32_t &node2, vector<uint32_t> &commonNeighbors) const {
    for (uint32_t pos1 = 0 ; pos1 < nodeIndex.size() ; pos1++) {
        if (nodeIndex[pos1].removed) {
            continue;
        }
        for (uint32_t pos2 = pos1+1 ; pos2 < nodeIndex.size() ; pos2++) {
            if (nodeIndex[pos2].removed) {
                continue;
            }
            if (nodeIndex[pos1].edges == 5 || nodeIndex[pos2].edges == 5) {
                uint32_t nodeA = (!mapping ? pos1 : posToId->at(pos1));
                uint32_t nodeB = (!mapping ? pos2 : posToId->at(pos2));
                if (edgeExists(nodeA, nodeB)) {
                    continue;
                }
                commonNeighbors.clear();
                getCommonNeighbors(nodeA, nodeB, commonNeighbors);
                if (commonNeighbors.size() >= 3) {
                    node1 = nodeA;
                    node2 = nodeB;
                    //cout << "branching on a good pair " << node1 << "-" << node2 << "\n";
                    return true;
                }
            }
        }
    }
    return false;
}

uint32_t Graph::getOptimalDegree5Node() const {
    uint32_t node = NONE;
    vector<uint32_t> nodes;
    for (uint32_t pos = 0 ; pos < nodeIndex.size() ; pos++) {
        if (nodeIndex[pos].removed || nodeIndex[pos].edges != 5) {
            continue;
        }
        uint32_t node = (!mapping ? pos : posToId->at(pos));
        nodes.push_back(node);
        if (getNumberOfDegreeNeighbors(node, 3, 1) >= 1) {
            //cout << "Branching on optimal degree-5 node: at least 1 degree 3 neighbor " << node << "\n";
            return node;
        }
    }
    for (auto node: nodes) {
        if (nodeIsEffective(node)) {
            //cout << "Branching on optimal degree-5 node: effective node " << node << "\n";
            return node;
        }
        std::unordered_set<uint32_t> extendedGrandchildren;
        GraphTraversal graphTraversal(*this, node);
        getExtendedGrandchildren(graphTraversal, extendedGrandchildren, NULL, true);
        if (extendedGrandchildren.size()) {
            //cout << "Branching on optimal degree-5 node: extended grandchildren " << node << "\n";
            return node;
        }
    }
    return node;
}

uint32_t Graph::nodeIsEffective(const uint32_t &node) const {
    vector<uint32_t> neighbors;
    gatherNeighbors(node, neighbors);
    neighbors.push_back(node);
    unordered_set<uint32_t> neighborsAtDistance2;
    getNeighborsAtDistance2(node, neighborsAtDistance2);
    uint32_t f = 0;
    for (auto it = neighborsAtDistance2.begin() ; it != neighborsAtDistance2.end() ; it++) {
        for (auto neighbor: neighbors) {
            if (edgeExists(*it, neighbor)) {
                f++;
            }
        }
    }
    uint32_t g = 0;
    for (auto it = neighborsAtDistance2.begin() ; it != neighborsAtDistance2.end() ; it++) {
        g += (5 - getNodeDegree(*it) + f - neighborsAtDistance2.size());
    }

    uint32_t k4 = getNumberOfDegreeNeighbors(node, 4);
    uint32_t k5 = getNumberOfDegreeNeighbors(node, 5);
    bool effective = false;
    if (k4 == 0 && k5 == 5) {
        if (f >= 14 && g >= 0 || f >= 12 && g >= 3 || f >= 10 && g >= 5) {
            effective = true;
        }
    } else if (k4 == 1 && k5 == 4) {
        if (f >= 13 && g >= 0 || f >= 11 && g >= 2) {
            effective = true;
        }
    } else if (k4 == 2 && k5 == 3) {
        if (f >= 12 && g >= 0 || f >= 10 && g >= 2) {
            effective = true;
        }
    } else if (k4 == 3 && k5 == 2 && f >= 11 && g >= 0) {
        effective = true;
    } else if (k4 == 4 && k5 == 1) {
        if (f >= 12 && g >= 0 || f >= 10 && g >= 1) {
            effective = true;
        }
    } else if (k4 == 5 && k5 == 0 && f >= 10 && g >= 0) {
        effective = true;
    }
    return effective;
}

bool Graph::isInTriangle(const uint32_t &node) const {
    vector<uint32_t> neighbors;
    for (uint32_t i = 0 ; i < neighbors.size() ; i++) {
        for (uint32_t j = i+1 ; j < neighbors.size() ; j++) {
            if (edgeExists(neighbors[i], neighbors[j])) {
                return true;
            }
        }
    }
    return false;
}

bool Graph::isFineInstance() const {
    GraphTraversal graphTraversal(*this);
    bool fineInstance = false;
    while (graphTraversal.curNode != NONE) {
        uint32_t degree = getNodeDegree(graphTraversal.curNode);
        if (degree >= 4) {
            fineInstance = true;
        } else if (degree == 1) {
            return false;
        }
        getNextNode(graphTraversal);
    }
    return fineInstance;
}

uint32_t Graph::getGoodNode(unordered_map<uint32_t, vector<uint32_t>* > &ccToNodes) const {
    return NONE;
    for (auto &cc: ccToNodes) {
        vector<uint32_t> &nodes = *(cc.second);
        if (nodes.size() < 9) {
            continue;
        }
        uint32_t start = 26;
        if (nodes.size() < 29) {
            start = nodes.size() - 3;
        }
        for (uint32_t node: nodes) {
            assert(!nodeIndex[getPos(node)].removed);
            if (getNodeDegree(node) < 5) {
                continue;
            }
            for (uint32_t size = start ; size >= 6 ; size--) {
                vector<Traversal *> frontier;
                unordered_set<uint32_t> set;
                frontier.push_back(new Traversal(node, *this));
                set.insert(node);
                //cout << "size " << size << " node " << node << endl;
                uint32_t goodNode = getGoodNode(frontier, set, nodes, size);
                if (goodNode != NONE) {
                    //cout << "Found good node for good set of size " << size << "\n";
                    return goodNode;
                }
            }
        }
    }
    return NONE;
}

uint32_t Graph::getGoodNode(vector<Traversal *> &frontier, unordered_set<uint32_t> &set, vector<uint32_t> &nodes, const uint32_t &size) const {
    Traversal *traversal = frontier[0];
    if (traversal->index == NONE) {
        return NONE;
    }
    while (true) {
        auto it = traversal->set.begin();
        std::advance(it, traversal->index);
        uint32_t neighbor = *it;
        if (set.find(neighbor) == set.end()) {
            frontier.push_back(new Traversal(neighbor, *this, frontier[frontier.size()-1]));
            set.insert(neighbor);
            traversal = frontier[frontier.size()-1];

            if (set.size() == size) {
                unordered_set<uint32_t> neighbors;
                gatherAllNeighbors(set, neighbors, 3);
                if (neighbors.size() == 3) {
                    return *(neighbors.begin());
                } else {
                    delete traversal;
                    frontier.pop_back();
                    set.erase(neighbor);
                    if (!advance(frontier, &traversal, *this, set)) {
                        return NONE;
                    }
                }
            }
        } else {
            if (!advance(frontier, &traversal, *this, set)) {
                return NONE;
            }
        }
    }
    return true;
}

void Graph::getOptimalShortEdge(const uint32_t &degree, uint32_t &finalNode1, uint32_t &finalNode2, vector<uint32_t> &finalContainer) const {
    finalNode1 = NONE;
    finalNode2 = NONE;
    vector<uint32_t> commonNeighbors;
    bool done = false;
    for (uint32_t pos = 0 ; pos < nodeIndex.size() && !done ; pos++) {
        if (nodeIndex[pos].removed) {
            continue;
        }
        if (nodeIndex[pos].edges == degree) {
            uint32_t neighborCount = degree;
            uint32_t nextNodeOffset = (pos == nodeIndex.size()-1 ? edgeBuffer->size() : nodeIndex[pos+1].offset);
            for (uint32_t offset = nodeIndex[pos].offset ; offset < nextNodeOffset && neighborCount; offset++) {
                uint32_t nPos = (!mapping ? (*edgeBuffer)[offset] : idToPos->at((*edgeBuffer)[offset]));
                if (nodeIndex[nPos].removed) {
                    continue;
                }
                neighborCount--;
                if (nodeIndex[nPos].edges == degree || (degree == 6 && nodeIndex[nPos].edges == 5)) {
                    uint32_t node1 = (!mapping ? pos : posToId->at(pos));
                    uint32_t node2 = (!mapping ? nPos : posToId->at(nPos));
                    if (node1 < node2) {
                        getCommonNeighbors(node1, node2, commonNeighbors);
                        if ((degree == 6 && commonNeighbors.size() >= 3 || (degree == 7 || degree == 8) && commonNeighbors.size() >= 4) &&
                        commonNeighbors.size() > finalContainer.size()) {
                            finalContainer.clear();
                            finalContainer.insert(finalContainer.end(), commonNeighbors.begin(), commonNeighbors.end());
                            finalNode1 = node1;
                            finalNode2 = node2;
                            if (finalContainer.size() == degree - 1) {
                                done = true;
                                break;
                            }
                        }
                        commonNeighbors.clear();
                    }
                }
            }
        }
    }
}

void Graph::getCommonNeighbors(const uint32_t &node1, const uint32_t &node2, vector<uint32_t> &commonNeighbors, const uint32_t &atLeast) const {
    uint32_t pos1 = (!mapping ? node1 : idToPos->at(node1));
    uint32_t pos2 = (!mapping ? node2 : idToPos->at(node2));
    assert(!nodeIndex[pos1].removed && !nodeIndex[pos2].removed);
    uint32_t count = 0;
    vector<uint32_t> neighbors1;
    gatherNeighbors(node1, neighbors1);
    for (auto &neighbor: neighbors1) {
        if (edgeExists(neighbor, node2)) {
            commonNeighbors.push_back(neighbor);
            if (atLeast && ++count == atLeast) {
                return;
            }
        }
    }
}

/* Connect 'node' with 'nodes'. Since moving elements and a vector reallocation is possible,
 * the mirror edges are not added. It is more performant to add them manually with another call
 * to this function, along with any other edges. (Optimization for branching on edges) */
void Graph::addEdges(const uint32_t node, const vector<uint32_t> &nodes) {
    uint32_t pos = (!mapping ? node : idToPos->at(node));
    set<uint32_t> neighbors;
    vector<uint32_t> removedNeighbors;
    gatherNeighborsWithRemoved(node, neighbors, removedNeighbors);
    uint32_t space = neighbors.size() + removedNeighbors.size();
    neighbors.insert(nodes.begin(), nodes.end());
    uint32_t finalNeighborCount = neighbors.size();
    if (neighbors.size() <= space) {
        while (neighbors.size() < space) {
            uint32_t removed = removedNeighbors.back();
            removedNeighbors.pop_back();
            neighbors.insert(removed);
        }
        copy(neighbors.begin(), neighbors.end(), edgeBuffer->begin() + nodeIndex[pos].offset);
    } else {
        auto it = neighbors.begin();
        uint32_t insertedElements = 0;
        uint32_t nextNodeOffset = (pos == nodeIndex.size()-1 ? edgeBuffer->size() : nodeIndex[pos+1].offset);
        for (uint32_t offset = nodeIndex[pos].offset ; offset < nextNodeOffset ; offset++) {
            (*edgeBuffer)[offset] = *it;
            it++;
            insertedElements++;
        }
        uint32_t addition = (neighbors.size() - insertedElements);
        edgeBuffer->reserve(edgeBuffer->size() + addition);
        edgeBuffer->insert(edgeBuffer->begin() + nextNodeOffset, it, neighbors.end());
        for (uint32_t i = pos + 1 ; i < nodeIndex.size() ; i++) {
            nodeIndex[i].offset += addition;
        }
    }
    nodeIndex[pos].edges = finalNeighborCount;
}

void Graph::collectZeroDegreeNodes() {
    for (uint32_t pos = 0 ; pos < nodeIndex.size() ; pos++) {
        if (!nodeIndex[pos].removed && !nodeIndex[pos].edges) {
            zeroDegreeNodes.push_back((!mapping ? pos : posToId->at(pos)));
            nodeIndex[pos].removed = true;
        }
    }
}

bool Graph::getArticulationPoints(unordered_set<uint32_t> &vertexCut, vector<uint32_t> &component1, vector<uint32_t> &component2, bool &actualComponent1, bool &connected) const {
    struct Value {
        Value(const uint32_t &visit) : visit(visit), low(visit) {}
        uint32_t visit;
        uint32_t low;
    };

    struct Instance {
        Instance(const uint32_t &node, const uint32_t &parentNode, const Graph &graph) : graphTraversal(graph, node), parentNode(parentNode), dfsChildren(0) {}
        GraphTraversal graphTraversal;
        uint32_t parentNode;
        uint32_t dfsChildren;
    };

    vertexCut.clear();
    component1.clear();
    component2.clear();

    Graph::GraphTraversal graphTraversal(*this);
    if (graphTraversal.curNode == NONE) {
        return false;
    }
    unordered_set<uint32_t> articulationPoints;
    unordered_map<uint32_t, Value> exploredSet;
    stack<Instance> frontier;
    uint32_t visit = 0;
    uint32_t root = NONE;
    if (exploredSet.find(graphTraversal.curNode) == exploredSet.end()) {
        root = graphTraversal.curNode;
        frontier.push(Instance(graphTraversal.curNode, NONE, *this));
        while (!frontier.empty()) {
            uint32_t node = frontier.top().graphTraversal.curNode;
            exploredSet.insert({node, Value(visit)});
            visit++;
            bool newCall;
            do {
                node = frontier.top().graphTraversal.curNode;
                Graph::GraphTraversal &neighbors = frontier.top().graphTraversal;
                newCall = false;
                while (neighbors.curEdgeOffset != NONE) {
                    uint32_t neighbor = (*edgeBuffer)[neighbors.curEdgeOffset];
                    if (neighbor != node) {
                        auto it = exploredSet.find(neighbor);
                        if (it == exploredSet.end()) {
                            newCall = true;
                            frontier.top().dfsChildren++;
                            frontier.push(Instance(neighbor, node, *this));
                            break;
                        } else if (it->second.visit < exploredSet.find(node)->second.visit) {
                            auto it1 = exploredSet.find(node);
                            if (it->second.visit < it1->second.low) {
                                it1->second.low = it->second.visit;
                            }
                        }
                    }
                    getNextEdge(neighbors);
                }
                if (!newCall) {
                    uint32_t parentNode = frontier.top().parentNode;
                    if (frontier.size() > 1) {
                        auto it1 = exploredSet.find(parentNode);
                        auto it2 = exploredSet.find(node);
                        assert(it1 != exploredSet.end() && it2 != exploredSet.end());
                        if (it2->second.low < it1->second.low) {
                            it1->second.low = it2->second.low;
                        }
                        if (parentNode != root && it2->second.low >= it1->second.visit && articulationPoints.insert(parentNode).second) {
                            //cout << "Articulation point " << parentNode << "\n";
                            unordered_set<uint32_t> cut;
                            cut.insert(parentNode);
                            if (checkSeparation(cut, component1, component2, actualComponent1)) {
                                vertexCut = cut;
                                //cout << "Articulation point " << parentNode << "\n";
                                return true;
                            }
                        }
                        frontier.pop();
                        if (frontier.top().graphTraversal.curEdgeOffset != NONE) {
                            getNextEdge(frontier.top().graphTraversal);
                        }
                    } else if (frontier.size() == 1) {
                        if (frontier.top().dfsChildren > 1 && articulationPoints.insert(frontier.top().graphTraversal.curNode).second) {
                            //cout << "Root articulation point " << frontier.top().graphTraversal.curNode << "\n";
                            unordered_set<uint32_t> cut;
                            cut.insert(frontier.top().graphTraversal.curNode);
                            if (checkSeparation(cut, component1, component2, actualComponent1)) {
                                vertexCut = cut;
                                //cout << "Root articulation point " << frontier.top().graphTraversal.curNode << "\n";
                                return true;
                            }
                        }
                        frontier.pop();
                        break;
                    }
                }
            } while (!newCall);
        }
    }
    if (exploredSet.size() != getNodeCountWithEdges()) {
        connected = false;
    } else {
        connected = true;
    }
    return false;
}

void Graph::addPalmTreeArc(unordered_map<uint32_t, vector<uint32_t> > &palmTree, const uint32_t &node, const uint32_t &neighbor) {
    auto res = palmTree.find(node);
    if (res == palmTree.end()) {
        vector<uint32_t> tmp;
        tmp.push_back(neighbor);
        palmTree.insert({node, tmp});
    } else {
        res->second.push_back(neighbor);
    }
}


bool Graph::getSeparatingPairs(unordered_set<uint32_t> &vertexCut, vector<uint32_t> &component1, vector<uint32_t> &component2, bool &actualComponent1) const {
    struct Instance {
        Instance(const uint32_t &node, const uint32_t &parentNode, const Graph &graph) : graphTraversal(graph, node), parentNode(parentNode) {}
        GraphTraversal graphTraversal;
        uint32_t parentNode;
    };

    Graph::GraphTraversal graphTraversal(*this);
    if (graphTraversal.curNode == NONE) {
        return false;
    }
    vertexCut.clear();
    component1.clear();
    component2.clear();
    unordered_map<uint32_t, uint32_t> visitToId;
    unordered_map<uint32_t, Value1> exploredSet;
    stack<Instance> frontier;
    unordered_set<uint32_t> flags;
    unordered_map<uint32_t, uint32_t> fathers;
    unordered_map<uint32_t, uint32_t> sons;
    unordered_map<uint32_t, vector<uint32_t> > palmTree1;
    unordered_map<uint32_t, vector<uint32_t> > palmTree2;
    unordered_map<uint32_t, vector<uint32_t> > bothPalmTrees;
    unordered_map<uint32_t, unordered_set<uint32_t> > descendants;
    uint32_t visit = 0;
    uint32_t numberOfNodes = 0;
    if (exploredSet.find(graphTraversal.curNode) == exploredSet.end()) {
        frontier.push(Instance(graphTraversal.curNode, NONE, *this));
        while (!frontier.empty()) {
            uint32_t node = frontier.top().graphTraversal.curNode;
            visit++;
            exploredSet.insert({node, Value1(visit)});
            visitToId.insert({visit, node});
            sons.insert({visit, 0});
            numberOfNodes++;
            bool newCall;
            do {
                node = frontier.top().graphTraversal.curNode;
                Graph::GraphTraversal &neighbors = frontier.top().graphTraversal;
                newCall = false;
                while (neighbors.curEdgeOffset != NONE) {
                    uint32_t neighbor = (*edgeBuffer)[neighbors.curEdgeOffset];
                    auto it = exploredSet.find(neighbor);
                    if (it == exploredSet.end()) {
                        addPalmTreeArc(palmTree1, node, neighbor);
                        addPalmTreeArc(bothPalmTrees, node, neighbor);
                        fathers.insert({neighbor, node});
                        newCall = true;
                        frontier.push(Instance(neighbor, node, *this));
                        break;
                    } else if (neighbor != node && it->second.visit < exploredSet.find(node)->second.visit || flags.find(node) != flags.end()) {
                        addPalmTreeArc(palmTree2, node, neighbor);
                        addPalmTreeArc(bothPalmTrees, node, neighbor);
                        auto it1 = exploredSet.find(node);
                        if (it->second.visit < it1->second.low1) {
                            it1->second.low2 = it1->second.low1;
                            it1->second.low1 = it->second.visit;
                        } else if (it->second.visit > it1->second.low1) {
                            if (it->second.visit < it1->second.low2) {
                                it1->second.low2 = it->second.visit;
                            }
                        }
                    } else if (neighbor == node && flags.find(node) == flags.end()) {
                        flags.insert(node);
                    }
                    getNextEdge(neighbors);
                }
                if (!newCall) {
                    uint32_t parentNode = frontier.top().parentNode;
                    if (parentNode == NONE) {
                        assert(frontier.size() == 1);
                        frontier.pop();
                        break;
                    }
                    auto it1 = exploredSet.find(parentNode);
                    auto it2 = exploredSet.find(node);
                    assert(it1 != exploredSet.end() && it2 != exploredSet.end());
                    it1->second.nd += (1 + it2->second.nd);
                    auto d1 = descendants.find(parentNode);
                    if (d1 == descendants.end()) {
                        descendants.insert({parentNode, unordered_set<uint32_t>()});
                        d1 = descendants.find(parentNode);
                    }
                    d1->second.insert(node);
                    auto d2 = descendants.find(node);
                    if (d2 != descendants.end()) {
                        d1->second.insert(d2->second.begin(), d2->second.end());
                    }
                    if (it2->second.low1 < it1->second.low1) {
                        it1->second.low2 = it1->second.low1;
                        if (it2->second.low2 < it1->second.low1) {
                            it1->second.low2 = it2->second.low2;
                        }
                        it1->second.low1 = it2->second.low1;
                    } else if (it2->second.low1 == it1->second.low1) {
                        if (it2->second.low2 < it1->second.low2) {
                            it1->second.low2 = it2->second.low2;
                        }
                    } else if (it2->second.low1 < it1->second.low2) {
                        it1->second.low2 = it2->second.low1;
                    }
                    frontier.pop();
                    if (!frontier.size()) {
                        break;
                    }
                    if (frontier.top().graphTraversal.curEdgeOffset != NONE) {
                        getNextEdge(frontier.top().graphTraversal);
                    }
                }
            } while (!newCall);
        }
    }
    if (getSeparatingPairs2(vertexCut, component1, component2, actualComponent1, numberOfNodes, palmTree1, palmTree2, bothPalmTrees, exploredSet, fathers, sons, descendants, visitToId)) {
        return true;
    }
    return false;
}

    bool Graph::getSeparatingPairs2(unordered_set<uint32_t> &vertexCut, vector<uint32_t> &component1, vector<uint32_t> &component2, bool &actualComponent1,
                               const uint32_t &numberOfNodes, unordered_map<uint32_t, vector<uint32_t> > &palmTree1, unordered_map<uint32_t, vector<uint32_t> > &palmTree2,
                               unordered_map<uint32_t, vector<uint32_t> > &bothPalmTrees, unordered_map<uint32_t, Value1> &oldExploredSet, unordered_map<uint32_t, uint32_t> &fathers,
                               unordered_map<uint32_t, uint32_t> &sons, unordered_map<uint32_t, unordered_set<uint32_t> > &descendants, unordered_map<uint32_t, uint32_t> &visitToId) const {

    struct Edge {
        uint32_t node;
        uint32_t neighbor;

        Edge(const uint32_t &node, const uint32_t &neighbor) : node(node), neighbor(neighbor) {}
    };

    struct Value2 {
        Value2(const uint32_t &visit) : visit(visit), low1(visit), low2(visit), high(visit) {}
        uint32_t visit;
        uint32_t low1;
        uint32_t low2;
        uint32_t high;
    };

    struct Instance {
        Instance(const uint32_t &node) : node(node), curNeighborIndex(0) {}
        uint32_t node;
        uint32_t curNeighborIndex;
    };

    vector<vector<Edge> > buckets(2 * numberOfNodes + 1, vector<Edge>());
    assert(buckets.size() == 2 * numberOfNodes + 1);
    for (auto &arc: bothPalmTrees) {
        uint32_t v = arc.first;
        for (auto w: arc.second) {
            auto res = palmTree2.find(v);
            bool edgeExists = false;
            if (res != palmTree2.end()) {
                for (auto n: res->second) {
                    if (n == w) {
                        edgeExists = true;
                        buckets[2 * oldExploredSet.find(w)->second.visit + 1].push_back(Edge(v, w));
                    }
                }
            }
            if (!edgeExists) {
                auto itW = oldExploredSet.find(w);
                if (itW->second.low2 < oldExploredSet.find(v)->second.visit) {
                    buckets[2 * itW->second.low1].push_back(Edge(v, w));
                } else {
                    buckets[2 * itW->second.low1 + 1].push_back(Edge(v, w));
                }
            }
        }
    }
    unordered_map<uint32_t, vector<uint32_t> > sortedPalmTree;
    for (auto &bucket: buckets) {
        for (auto &edge: bucket) {
            auto it = sortedPalmTree.find(edge.node);
            if (it != sortedPalmTree.end()) {
                it->second.push_back(edge.neighbor);
            } else {
                vector<uint32_t> tmp;
                tmp.push_back(edge.neighbor);
                sortedPalmTree.insert({edge.node, tmp});
            }
        }
    }

    uint32_t visit = numberOfNodes;
    unordered_map<uint32_t, Value2> exploredSet;
    stack<Instance> frontier;
    uint32_t root = visitToId.find(1)->second;
    if (exploredSet.find(root) == exploredSet.end()) {
        frontier.push(Instance(root));
        while (!frontier.empty()) {
            uint32_t node = frontier.top().node;
            exploredSet.insert({node, Value2(visit - oldExploredSet.find(node)->second.nd)});
            bool newCall;
            do {
                node = frontier.top().node;
                uint32_t &curNeighborIndex = frontier.top().curNeighborIndex;
                newCall = false;
                while (curNeighborIndex < sortedPalmTree.find(node)->second.size()) {
                    uint32_t neighbor = sortedPalmTree.find(node)->second.at(curNeighborIndex);
                    auto it = exploredSet.find(neighbor);
                    if (it == exploredSet.end()) {
                        newCall = true;
                        frontier.push(Instance(neighbor));
                        break;
                    } else {
                        auto itV = exploredSet.find(node);
                        if (neighbor != root) {
                            auto itW = exploredSet.find(fathers.find(neighbor)->second);
                            if (itV->second.visit > itW->second.high) {
                                itW->second.high = itV->second.visit;
                            }
                        }
                        auto itW = exploredSet.find(neighbor);
                        if (itW->second.visit < itV->second.low1) {
                            itV->second.low2 = itV->second.low1;
                            itV->second.low1 = itW->second.visit;
                        } else if (itW->second.visit > itV->second.low1 && itW->second.visit < itV->second.low2) {
                            itV->second.low2 = itW->second.visit;
                        }
                    }
                    curNeighborIndex++;
                }
                if (!newCall) {
                    uint32_t neighbor = frontier.top().node;
                    frontier.pop();
                    if (!frontier.size()) {
                        break;
                    }
                    node = frontier.top().node;
                    auto itVSons = sons.find(exploredSet.find(node)->second.visit);
                    auto itW = exploredSet.find(neighbor);
                    if (itVSons->second == 0) {
                        itVSons->second = itW->second.visit;
                    }
                    visit--;
                    auto itV = exploredSet.find(node);
                    if (itW->second.low1 < itV->second.low1) {
                        itV->second.low2 = itV->second.low1;
                        if (itW->second.low2 < itV->second.low2) {
                            itV->second.low2 = itW->second.low2;
                        }
                        itV->second.low1 = itW->second.low1;
                    } else if (itW->second.low1 == itV->second.low1) {
                        if (itW->second.low2 < itV->second.low2) {
                            itV->second.low2 = itW->second.low2;
                        }
                    } else if (itW->second.low1 < itV->second.low2) {
                        itV->second.low2 = itW->second.low1;
                    }
                    frontier.top().curNeighborIndex++;
                }
            } while (!newCall);
        }
    }

    for (auto &edge: sortedPalmTree) {
        uint32_t b = edge.first;
        uint32_t visitB = exploredSet.find(b)->second.visit;
        for (auto r: edge.second) {
            auto itR = exploredSet.find(r);
            uint32_t low1R = itR->second.low1;
            uint32_t low2R = itR->second.low2;
            if (low2R >= visitB) {
                for (auto &a: exploredSet) {
                    if (a.second.visit < visitB && low1R == a.second.visit) {
                        auto descendantsR = descendants.find(r);
                        for (auto &node: exploredSet) {
                            if (node.first != a.first && node.first != b && node.first != r && (descendantsR == descendants.end() || descendantsR->second.find(node.first) == descendantsR->second.end())) {
                                //cout << "Separating pair " << a.first << "-" << b << endl;
                                unordered_set<uint32_t> cut;
                                cut.insert(a.first);
                                cut.insert(b);
                                if (checkSeparation(cut, component1, component2, actualComponent1)) {
                                    vertexCut = cut;
                                    //cout << "Separating pair " << a.first << "-" << b << endl;
                                    return true;
                                }
                                break;
                            }
                        }
                    }
                }
            }
        }
    }
    return false;
}

bool Graph::getSeparatingTriplets(unordered_set<uint32_t> &vertexCut, vector<uint32_t> &component1, vector<uint32_t> &component2, bool &actualComponent1) const {
    return false;
    for (uint32_t pos1 = 0 ; pos1 < nodeIndex.size() ; pos1++) {
        if (nodeIndex[pos1].removed || !nodeIndex[pos1].edges) {
            continue;
        }
        for (uint32_t pos2 = pos1+1 ; pos2 < nodeIndex.size() ; pos2++) {
            if (nodeIndex[pos2].removed || !nodeIndex[pos2].edges) {
                continue;
            }
            for (uint32_t pos3 = pos2+1 ; pos3 < nodeIndex.size() ; pos3++) {
                if (nodeIndex[pos3].removed || !nodeIndex[pos3].edges) {
                    continue;
                }
                unordered_set<uint32_t> excludedNodes;
                excludedNodes.insert((!mapping ? pos1 : posToId->at(pos1)));
                excludedNodes.insert((!mapping ? pos2 : posToId->at(pos2)));
                excludedNodes.insert((!mapping ? pos3 : posToId->at(pos3)));
                if (checkSeparation(excludedNodes, component1, component2, actualComponent1)) {
                    vertexCut = excludedNodes;
                    return true;
                }
            }
        }
    }
    return false;
}

bool Graph::checkSeparation(const unordered_set<uint32_t> &cut, vector<uint32_t> &component1, vector<uint32_t> &component2, bool &actualComponent1) const {
    if (!buildCC(cut, component1, component2) || !component1.size() || !component2.size()) {
        //cout << "more than 2 cc" << endl;
        component1.clear();
        component2.clear();
        return false;
    }
    if (cut.size() < 3) {
        uint32_t degreeSum1 = 0;
        for (auto node: component1) {
            uint32_t degree = getNodeDegree(node);
            for (auto z: cut) {
                if (edgeExists(z, node)) {
                    degree--;
                }
            }
            if (degree >= 3) {
                degreeSum1 += degree;
            }
        }

        uint32_t degreeSum2 = 0;
        for (auto node: component2) {
            uint32_t degree = getNodeDegree(node);
            for (auto z: cut) {
                if (edgeExists(z, node)) {
                    degree--;
                }
            }
            if (degree >= 3) {
                degreeSum2 += degree;
                if (degreeSum2 >= degreeSum1) {
                    break;
                }
            }
        }
        if (degreeSum1 <= degreeSum2) {
            actualComponent1 = true;
        } else {
            actualComponent1 = false;
        }
        return true;
    } else {
        if (component1.size() <= 24) {
            actualComponent1 = true;
        } else if (component2.size() <= 24) {
            actualComponent1 = false;
        } else {
            return false;
        }
        vector<uint32_t> *c = (actualComponent1 ? &component1 : &component2);
        uint32_t degreeSum = 0;
        for (auto node: *c) {
            degreeSum += getNodeDegree(node);
            if (degreeSum >= 17) {
                return true;
            }
        }
        return false;
    }
}

/* Only builds 2 CCs, optimized for separation set detection */
bool Graph::buildCC(const unordered_set<uint32_t> &excludedNodes, vector<uint32_t> &component1, vector<uint32_t> &component2) const {
    component1.clear();
    component2.clear();
    uint32_t component = 0;
    unordered_set<uint32_t> exploredSet;
    stack<uint32_t> frontier;
    Graph::GraphTraversal graphTraversal(*this);
    while (graphTraversal.curNode != NONE) {
        if (excludedNodes.find(graphTraversal.curNode) == excludedNodes.end() && exploredSet.insert(graphTraversal.curNode).second) {
            if (component == 2) {
                return false;
            }
            vector<uint32_t> *componentNodes = (component == 0 ? &component1 : &component2);
            componentNodes->push_back(graphTraversal.curNode);
            frontier.push(graphTraversal.curNode);
            while (!frontier.empty()) {
                uint32_t node = frontier.top();
                frontier.pop();
                Graph::GraphTraversal neighbors(*this, node);
                while (neighbors.curEdgeOffset != NONE) {
                    node = (*edgeBuffer)[neighbors.curEdgeOffset];
                    if (excludedNodes.find(node) == excludedNodes.end() && exploredSet.insert(node).second) {
                        frontier.push(node);
                        componentNodes->push_back(node);
                    }
                    getNextEdge(neighbors);
                }
            }
            component++;
        }
        getNextNode(graphTraversal);
    }
    return true;
}

uint32_t Graph::getNodeCountWithEdges() const {
    uint32_t count = 0;
    for (auto &node: nodeIndex) {
        if (!node.removed && node.edges) {
            count++;
        }
    }
    return count;
}

void Graph::print(bool direction) const {
    cout << "\nNodes: " << nodeIndex.size() << " Edges: " << edgeBuffer->size() / 2 << "\n";
    for (uint32_t pos = 0 ; pos < nodeIndex.size() ; pos++) {
        if (!nodeIndex[pos].removed) {
            uint32_t node = (!mapping ? pos : (*posToId)[pos]);
            if (!nodeIndex[pos].edges) {
                cout << node << "\n";
            }
            uint32_t nextNodeOffset = (pos == nodeIndex.size()-1 ? edgeBuffer->size() : nodeIndex[pos+1].offset);
            for (uint32_t offset = nodeIndex[pos].offset ; offset < nextNodeOffset ; offset++) {
                uint32_t nPos = (!mapping ? (*edgeBuffer)[offset] : idToPos->at((*edgeBuffer)[offset]));
                if (!nodeIndex[nPos].removed && (direction || !direction && node < (*edgeBuffer)[offset])) {
                    cout << node << "\t" << (*edgeBuffer)[offset] << "\n";
                }
            }
        }
    }
    /*cout << "Zero degree nodes: \n";
    for (uint32_t i = 0 ; i < zeroDegreeNodes.size() ; i++) {
        cout << zeroDegreeNodes[i] << "\n";
    }*/
}

void Graph::printWithGraphTraversal(bool direction) const {
    GraphTraversal graphTraversal(*this);
    while (graphTraversal.curNode != NONE) {
        while (graphTraversal.curEdgeOffset != NONE) {
            if (direction || !direction && graphTraversal.curNode < (*edgeBuffer)[graphTraversal.curEdgeOffset]) {
                cout << graphTraversal.curNode << "\t" << (*edgeBuffer)[graphTraversal.curEdgeOffset] << "\n";
            }
            getNextEdge(graphTraversal);
        }
        getNextNode(graphTraversal);
    }
    cout << "Zero degree nodes: \n";
    for (uint32_t i = 0 ; i < zeroDegreeNodes.size() ; i++) {
        cout << zeroDegreeNodes[i] << "\n";
    }
}

void Graph::printEdgeCounts() const {
    for (uint32_t pos = 0 ; pos < nodeIndex.size() ; pos++) {
        if (nodeIndex[pos].removed) {
            continue;
        }
        uint32_t node = (!mapping ? pos : (*posToId)[pos]);
        cout << "Node " << node << " has " << nodeIndex[pos].edges << " edges\n";
    }
}

/* Build graph from file, include both edge directions, keep them sorted.
 * Zero degree nodes are included in nodeIndex, but are makred as removed.
 * They are also held on zeroDegreeNodes seperate structure */
Graph::Graph(const string &inputFile, const bool &checkIndependentSet) : mapping(false), idToPos(NULL), posToId(NULL) {
    /* Open graph input file */
    FILE *f;
    f = fopen(inputFile.c_str(), "r");
    if (f == NULL) {
        cerr << "Error in opening input file " << inputFile << endl;
        exit(EXIT_FAILURE);
    }
    char buf[MAXLINE];

    /* Skip info lines */
    for (int i = 0 ; i < 3 ; i++) {
        if (fgets(buf, MAXLINE, f) == NULL) {
            cerr << "Error in parsing input file " << inputFile << endl;
            exit(EXIT_FAILURE);
        }
    }

    /* Get total node and edge numbers */
    char *ptr = strstr(buf, "Nodes:");
    uint32_t nodes = atoi(ptr + 7);
    ptr = strstr(buf, "Edges:");
    uint32_t edges = atoi(ptr + 7);
    if (fgets(buf, MAXLINE, f) == NULL) {
        cerr << "Error in parsing input file " << inputFile << endl;
        exit(EXIT_FAILURE);
    }

    try {
        nodeIndex.reserve(nodes);
    }
    catch (const length_error &le) {
        cerr << "NodeIndex length error: " << le.what() << endl;
        exit(EXIT_FAILURE);
    }

    edgeBuffer = new vector<uint32_t>();
    try {
        edgeBuffer->reserve(edges * 2);
    }
    catch (const length_error &le) {
        cerr << "EdgeBuffer length error: " << le.what() << endl;
        exit(EXIT_FAILURE);
    }

    /* Hold reverse direction edges temporarily */
    vector<vector<uint32_t> > reverseEdges(nodes);

    /* Build graph with both edge directions, keep them sorted */
    uint32_t sourceNode, targetNode, previousNode, straightEdges, offset;
    previousNode = NONE;
    straightEdges = 0;
    offset = 0;
    while(fgets(buf, MAXLINE, f) != NULL) {
        parseNodeIDs(buf, &sourceNode, &targetNode);
        if (sourceNode >= nodes || targetNode >= nodes) {
            cerr << "Error: received a node id equal or larger than the total nodes number specified at the beginning of the file" << endl;
            exit(EXIT_FAILURE);
        }
        /* Add smaller nodes than first source node */
        if (previousNode == NONE && sourceNode) {
            this->fill(sourceNode, checkIndependentSet);
        }
        if (previousNode != sourceNode && previousNode != NONE) {
            /* At new source node, add previous nodes' (including those who don't appear as source nodes) smaller neighbors */
            for (uint32_t node = previousNode + 1 ; node <= sourceNode ; node++) {
                uint32_t previousEdges = straightEdges + reverseEdges[node - 1].size();
                nodeIndex.push_back(Graph::NodeInfo(offset, previousEdges));
                if (!previousEdges && !checkIndependentSet) {
                    zeroDegreeNodes.push_back(nodeIndex.size()-1);
                    nodeIndex[nodeIndex.size()-1].removed = true;
                }
                offset += previousEdges;
                edgeBuffer->insert(edgeBuffer->end(), reverseEdges[node].begin(), reverseEdges[node].end());
                straightEdges = 0;
                if (node < sourceNode) {
                    //cout << "in between missing node " << node << endl;
                }
            }
        }
        /* Add source node's bigger neighbors */
        edgeBuffer->push_back(targetNode);
        straightEdges++;
        reverseEdges[targetNode].push_back(sourceNode);
        previousNode = sourceNode;
    }
    /* Add final source node's info */
    uint32_t previousEdges = straightEdges + reverseEdges[previousNode].size();
    nodeIndex.push_back(Graph::NodeInfo(offset, previousEdges));
    if (!previousEdges && !checkIndependentSet) {
        zeroDegreeNodes.push_back(nodeIndex.size()-1);
        nodeIndex[nodeIndex.size()-1].removed = true;
    }
    offset += previousEdges;
    /* Add smaller neighbors of the nodes left that don't appear as source nodes */
    for (uint32_t missingNode = nodeIndex.size() ; missingNode < nodes ; missingNode++) {
        //cout << "end missing node " << missingNode << endl;
        previousEdges = reverseEdges[missingNode].size();
        nodeIndex.push_back(Graph::NodeInfo(offset, previousEdges));
        if (!previousEdges && !checkIndependentSet) {
            zeroDegreeNodes.push_back(nodeIndex.size()-1);
            nodeIndex[nodeIndex.size()-1].removed = true;
        }
        offset += previousEdges;
        edgeBuffer->insert(edgeBuffer->end(), reverseEdges[missingNode].begin(), reverseEdges[missingNode].end());
    }
    nextUnusedId = nodeIndex.size();
    fclose(f);
}

void Graph::fill(const uint32_t &size, const bool &checkIndependentSet) {
    //cout << "start missing nodes from " << nodeIndex.size() << " to " << size << endl;
    while (nodeIndex.size() < size) {
        nodeIndex.push_back(NodeInfo(edgeBuffer->size(), 0));
        if (!checkIndependentSet) {
            zeroDegreeNodes.push_back(nodeIndex.size()-1);
            nodeIndex[nodeIndex.size()-1].removed = true;
        }
    }
}

void Graph::parseNodeIDs(char *buf, uint32_t *sourceNode, uint32_t *targetNode) {
    uint32_t *node = sourceNode;
    for (int i = 0 ; i < 2 ; i++) {
        uint32_t id = 0;
        for (; *buf != ' ' && *buf != '\t' && *buf != '\n' && *buf != '\0' ; buf++) {
            id = id * 10 + (*buf - '0');
        }
        *node = id;
        node = targetNode;
        for (; *buf == ' ' || *buf == '\t' ; buf++);
    }
}
