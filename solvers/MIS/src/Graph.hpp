#ifndef GRAPH_H
#define GRAPH_H

#include <iostream>
#include <string>
#include <vector>
#include <unordered_map>
#include <unordered_set>
#include <set>
#include <algorithm>
#include <assert.h>
#include "Util.hpp"

#define MAXLINE 1024

class Graph {

friend class ControlUnit;
friend class Reductions;
friend class Alg;
class NodeInfo;
struct Traversal;

public:
    struct GraphTraversal;
    Graph(const std::string &inputFile, const bool &checkIndependentSet);
    Graph() : mapping(false), idToPos(NULL), posToId(NULL) {}
    Graph(const Graph &graph);
    Graph& operator=(const Graph &graph);

    uint32_t getNodeCountWithEdges() const;

    uint32_t getPos(const uint32_t &node) const {
        return (!mapping ? node : idToPos->at(node));
    }

    uint32_t getNode(const uint32_t &pos) const {
        return (!mapping ? pos : posToId->at(pos));
    }

    ~Graph();

    void setIdToPos(std::unordered_map<uint32_t, uint32_t> *idToPos) {
        if (this->idToPos != NULL) {
            delete this->idToPos;
        }
        this->idToPos = idToPos;
    }

    void setPosToId(std::vector<uint32_t> *posToId) {
        if (this->posToId != NULL) {
            delete this->posToId;
        }
        this->posToId = posToId;
    }

    void setMapping(const bool &mapping) {
        this->mapping = mapping;
    }

    uint32_t getNodeDegree(const uint32_t &node) const {
        uint32_t pos = (!mapping ? node : idToPos->at(node));
        assert(!nodeIndex[pos].removed);
        return nodeIndex[pos].edges;
    }

    bool getArticulationPoints(std::unordered_set<uint32_t> &vertexCut, std::vector<uint32_t> &component1, std::vector<uint32_t> &component2, bool &actualComponent1, bool &connected) const;
    bool getSeparatingPairs(std::unordered_set<uint32_t> &vertexCut, std::vector<uint32_t> &component1, std::vector<uint32_t> &component2, bool &actualComponent1) const;
    bool getSeparatingTriplets(std::unordered_set<uint32_t> &vertexCut, std::vector<uint32_t> &component1, std::vector<uint32_t> &component2, bool &actualComponent1) const;
    uint32_t getOptimalNodeTheta3(const uint32_t initialMaxDegreeNode, const uint32_t &initialMaxDegree) const;
    bool getEffectiveNodeOrOptimalFunnel(uint32_t &effectiveNode, uint32_t &nodeV, uint32_t &nodeA) const;
    bool get4CycleTheta3(std::vector<uint32_t> &optimalCycle) const;
    uint32_t getEffectiveNodeMeasure(const uint32_t &bound = NONE) const;
    bool isInTriangle(const uint32_t &node) const;
    bool isFineInstance() const;
    uint32_t getOptimalDegree4Node() const;
    bool get4Cycle(std::vector<uint32_t> &cycle) const;
    bool getGoodFunnel(uint32_t &node1, uint32_t &node2) const;
    bool getGoodFunnelTheta5(uint32_t &node1, uint32_t &node2) const;
    bool getGoodPair(uint32_t &node1, uint32_t &node2, std::vector<uint32_t> &commonNeighbors) const;
    uint32_t getOptimalDegree5Node() const;
    uint32_t nodeIsEffective(const uint32_t &node) const;
    uint32_t getGoodNode(std::unordered_map<uint32_t, std::vector<uint32_t>* > &ccToNodes) const;
    void collectZeroDegreeNodes();
    void addEdges(const uint32_t node, const std::vector<uint32_t> &nodes);
    void getNeighborsAtDistance2(const uint32_t &node, std::unordered_set<uint32_t> &neighbors, const uint32_t &degree = NONE, uint32_t *count = NULL) const;
    uint32_t getNumberOfDegreeNeighbors(const uint32_t &node, const uint32_t &degree, const uint32_t &atLeast = 0) const;
    void getCommonNeighbors(const uint32_t &node1, const uint32_t &node2, std::vector<uint32_t> &container, const uint32_t &atLeast= 0) const;
    void getOptimalShortEdge(const uint32_t &degree, uint32_t &finalNode1, uint32_t &finalNode2, std::vector<uint32_t> &finalSet) const;
    void getExtendedGrandchildren(Graph::GraphTraversal &graphTraversal, std::unordered_set<uint32_t> &extendedGrandchildren, bool *isUnconfined = NULL, const bool &stopAtFirst = false) const;
    void getMaxNodeDegree(uint32_t &node, uint32_t &maxDegree, const uint32_t &bound = NONE) const;
    void getMinDegree(uint32_t &minDegree) const;
    uint32_t getTotalEdges() const;
    void remove(const uint32_t &node, ReduceInfo &reduceInfo, const bool &removeZeroDegreeNodes = false);
    void rebuild(ReduceInfo &reduceInfo);
    void rebuildFromNodes(std::unordered_set<uint32_t> &nodes);
    void buildNDegreeSubgraph(const uint32_t &degree, Graph &subgraph);
    uint32_t contractToSingleNode(const std::vector<uint32_t> &nodes, const std::vector<uint32_t> &neighbors, ReduceInfo &reduceInfo);
    uint32_t getNodeWithOneUncommonNeighbor(const std::vector<uint32_t> &neighbors, uint32_t &uncommonNeighbor) const;
    uint32_t getNextNodeWithIdenticalNeighbors(const uint32_t &previousNode, const std::vector<uint32_t> &neighbors) const;
    void replaceNeighbor(const uint32_t &node, const uint32_t &oldNeighbor, const uint32_t &newNeighbor);
    void print(bool direction) const;
    void printWithGraphTraversal(bool direction) const;
    void printEdgeCounts() const;

    /* Mark selected nodes as removed and reduce their neighbors' neighbor count.
     * fullComponent should be set to true when the nodes to be removed belong in the same component, and
     * that component has no other nodes (e.g. when removing line graphs). */
    template <typename Container>
    void remove(const Container &nodes, ReduceInfo &reduceInfo, const bool &fullComponent = false, std::unordered_set<uint32_t> *candidateNodes = NULL, const bool &removeZeroDegreeNodes = false) {
        for (auto it = nodes.begin() ; it != nodes.end() ; it++) {
            //std::cout << "removing " << *it << std::endl;
            uint32_t pos = (!mapping ? *it : idToPos->at(*it));
            if (!nodeIndex[pos].removed) {
                reduceInfo.nodesRemoved++;
                if (!fullComponent) {
                    uint32_t nextNodeOffset = (pos == nodeIndex.size()-1 ? edgeBuffer->size() : nodeIndex[pos+1].offset);
                    for (uint32_t offset = nodeIndex[pos].offset ; offset < nextNodeOffset ; offset++) {
                        uint32_t neighbor = (*edgeBuffer)[offset];
                        uint32_t nPos = (!mapping ? neighbor : idToPos->at(neighbor));
                        if (!nodeIndex[nPos].removed) {
                            nodeIndex[nPos].edges--;
                            if (removeZeroDegreeNodes && !nodeIndex[nPos].edges) {
                                zeroDegreeNodes.push_back(neighbor);
                                nodeIndex[nPos].removed = true;
                            }
                            if (find(std::next(it, 1), nodes.end(), neighbor) == nodes.end() &&
                            candidateNodes != NULL && (nodeIndex[nPos].edges == 2 || nodeIndex[nPos].edges == 3) && nPos < pos) {
                                candidateNodes->insert(neighbor);
                            }
                        }
                    }
                }
                //nodeIndex[pos].edges = 0;
                nodeIndex[pos].removed = true;
            }
        }
    }

    template <typename Container, typename OriginalNodesContainer = std::unordered_set<uint32_t> >
    bool gatherNeighbors(const uint32_t &node, Container &neighbors, const OriginalNodesContainer *nodes = NULL, const uint32_t &maxNeighbors = NONE) const {
        uint32_t pos = (!mapping ? node : idToPos->at(node));
        uint32_t neighborCount = nodeIndex[pos].edges;
        uint32_t nextNodeOffset = (pos == nodeIndex.size()-1 ? edgeBuffer->size() : nodeIndex[pos+1].offset);
        for (uint32_t offset = nodeIndex[pos].offset ; offset < nextNodeOffset && neighborCount; offset++) {
            uint32_t nPos = (!mapping ? (*edgeBuffer)[offset] : idToPos->at((*edgeBuffer)[offset]));
            if (!nodeIndex[nPos].removed && (nodes == NULL || nodes->find((*edgeBuffer)[offset]) == nodes->end())) {
                neighbors.insert(neighbors.end(), (*edgeBuffer)[offset]);
                if (maxNeighbors != NONE && neighbors.size() > maxNeighbors) {
                    return false;
                }
                neighborCount--;
            }
        }
        return true;
    }

    template <typename Container, typename ContainerRemoved = std::vector<uint32_t> >
    bool gatherNeighborsWithRemoved(const uint32_t &node, Container &neighbors, ContainerRemoved &removedNeighbors) const {
        uint32_t pos = (!mapping ? node : idToPos->at(node));
        uint32_t nextNodeOffset = (pos == nodeIndex.size()-1 ? edgeBuffer->size() : nodeIndex[pos+1].offset);
        for (uint32_t offset = nodeIndex[pos].offset ; offset < nextNodeOffset ; offset++) {
            uint32_t nPos = (!mapping ? (*edgeBuffer)[offset] : idToPos->at((*edgeBuffer)[offset]));
            if (!nodeIndex[nPos].removed) {
                neighbors.insert(neighbors.end(), (*edgeBuffer)[offset]);
            } else {
                removedNeighbors.insert(removedNeighbors.end(), (*edgeBuffer)[offset]);
            }
        }
        return true;
    }

    template <typename Container, typename OriginalNodesContainer>
    void gatherAllNeighbors(OriginalNodesContainer &nodes, Container &neighbors, const uint32_t &maxNeighbors = NONE) const {
        for (auto &node: nodes) {
            if (!gatherNeighbors(node, neighbors, &nodes, maxNeighbors)) {
                return;
            }
        }
    }

    template <typename Container1, typename Container2>
    uint32_t getNumberOfEdgesBetweenSets(const Container1 &set1, const Container2 &set2) const {
        uint32_t count = 0;
        for (auto i: set1) {
            for (auto j: set2) {
                if (edgeExists(i, j)) {
                    count++;
                }
            }
        }
        return count;
    }

    bool edgeExists(const uint32_t &node, const uint32_t &neighbor) const {
        return (findEdgeOffset(node, neighbor) != NONE);
    }

    /* Check whether a particular edge exists with binary search,
     * return neighbor's offset in edge buffer */
    uint32_t findEdgeOffset(const uint32_t &node, const uint32_t &neighbor) const {
        uint32_t pos = (!mapping ? node : idToPos->at(node));
        uint32_t nPos = (!mapping ? neighbor : idToPos->at(neighbor));
        assert(!nodeIndex[pos].removed && !nodeIndex[nPos].removed);
        uint32_t offset = nodeIndex[pos].offset;
        uint32_t endOffset = (pos == nodeIndex.size()-1 ? edgeBuffer->size() - 1 : nodeIndex[pos+1].offset - 1);
        if (offset == endOffset+1) {
            return NONE;
        }
        uint32_t startIndex = 0;
        uint32_t endIndex = endOffset - offset;
        uint32_t index = (endIndex - startIndex) / 2;
        while (startIndex != endIndex) {
            if ((*edgeBuffer)[offset + startIndex + index] == neighbor) {
                return offset + startIndex + index;
            } else if ((*edgeBuffer)[offset + startIndex + index] < neighbor) {
                startIndex += index + 1;
            } else {
                if (!index) {
                    return NONE;
                }
                endIndex = startIndex + index - 1;
            }
            index = (endIndex - startIndex) / 2;
        }
        return ((*edgeBuffer)[offset + startIndex + index] == neighbor ? offset + startIndex + index : NONE);
    }

    /* Return the first "outer neighbor of 'neighbor' at 'node'", and a flag of whether its the only one */
    void getOuterNeighbor(const uint32_t &node, const uint32_t &neighbor, uint32_t &outerNeighbor, bool &exactlyOne) const {
        outerNeighbor = NONE;
        exactlyOne = false;
        bool found = false;
        GraphTraversal graphTraversal(*this, neighbor);
        while (graphTraversal.curEdgeOffset != NONE) {
            uint32_t extendedGrandchild = (*edgeBuffer)[graphTraversal.curEdgeOffset];
            if (extendedGrandchild != node && !edgeExists(extendedGrandchild, node)) {
                if (!found) {
                    found = true;
                    outerNeighbor = extendedGrandchild;
                } else {
                    return;
                }
            }
            getNextEdge(graphTraversal);
        }
        if (found) {
            exactlyOne = true;
        }
    }

    template <typename Container>
    bool isIndependentSet(const Container &set, uint32_t *node1 = NULL, uint32_t *node2 = NULL) const {
        for (auto i = set.begin() ; i != set.end() ; i++) {
            for (auto j = std::next(i, 1) ; j != set.end() ; j++) {
                if (edgeExists(*i, *j)) {
                    if (node1 != NULL && node2 != NULL) {
                        *node1 = *i;
                        *node2 = *j;
                    }
                    return false;
                }
            }
        }
        return true;
    }

    /* For graph traversal through functions */
    struct GraphTraversal {
    public:
        GraphTraversal(const Graph &graph);
        GraphTraversal(const Graph &graph, const uint32_t &node);
        GraphTraversal(const uint32_t &curNode, const uint32_t &curEdgeOffset) : curNode(curNode), curEdgeOffset(curEdgeOffset) {}

        uint32_t curNode;
        uint32_t curEdgeOffset;
    };

    /* Return the next node in a graph traversal, ignoring removed or zero-degree ones.
     * Initialise its first edge in the traversal as its first non-removed neighbor. */
    void getNextNode(GraphTraversal &graphTraversal) const {
        bool firstTime = true;
        uint32_t pos;
        if (graphTraversal.curNode == NONE) {
            pos = NONE;
        } else {
            pos = (!mapping ? graphTraversal.curNode : idToPos->at(graphTraversal.curNode));
        }
        while (firstTime || pos == NONE || !nodeIndex[pos].edges) {
            if (pos == NONE && nodeIndex.size() || pos != NONE && pos < nodeIndex.size() - 1) {
                pos = (pos == NONE ? 0 : pos+1);
                if (nodeIndex[pos].removed) {
                    continue;
                }
                graphTraversal.curNode = (!mapping ? pos : (*posToId)[pos]);
                uint32_t offset = getFirstValidNeighborOffset(pos);
                if ((graphTraversal.curEdgeOffset = offset) != NONE) {
                    firstTime = false;
                }
            }
            else {
                graphTraversal.curNode = NONE;
                graphTraversal.curEdgeOffset = NONE;
                break;
            }
        }
    }

    /* Return next edge in a graph traversal, ignoring removed nodes */
    void getNextEdge(GraphTraversal &graphTraversal) const {
        assert(graphTraversal.curNode != NONE && graphTraversal.curEdgeOffset != NONE);
        uint32_t pos = (!mapping ? graphTraversal.curNode : idToPos->at(graphTraversal.curNode));
        uint32_t nextNodeOffset = (pos == nodeIndex.size()-1 ? edgeBuffer->size() : nodeIndex[pos+1].offset);
        graphTraversal.curEdgeOffset++;
        bool validNeighbor = false;
        for ( ; graphTraversal.curEdgeOffset  < nextNodeOffset ; graphTraversal.curEdgeOffset++) {
            uint32_t nPos = (!mapping ? (*edgeBuffer)[graphTraversal.curEdgeOffset] : idToPos->at((*edgeBuffer)[graphTraversal.curEdgeOffset]));
            if (!nodeIndex[nPos].removed) {
                validNeighbor = true;
                break;
            }
        }
        if (!validNeighbor) {
            graphTraversal.curEdgeOffset = NONE;
        }
    }

    /* Jump to a node */
    void goToNode(const uint32_t &node, GraphTraversal &graphTraversal) const {
        graphTraversal.curNode = node;
        uint32_t pos = (!mapping ? node : idToPos->at(node));
        graphTraversal.curEdgeOffset = getFirstValidNeighborOffset(pos);
    }

    /* Get current node's next valid neighbor, with backtracking in vector */
    static bool advance(std::vector<Graph::GraphTraversal> &frontier, Graph::GraphTraversal &graphTraversal, const Graph &graph, std::unordered_set<uint32_t> *set = NULL) {
        bool validNeighbor = false;
        while(!validNeighbor) {
            graphTraversal = frontier.back();
            graph.getNextEdge(graphTraversal);
            frontier.back() = graphTraversal;
            if (graphTraversal.curEdgeOffset != NONE) {
                validNeighbor = true;
            } else {
                graphTraversal = frontier.back();
                if (set != NULL) {
                    set->erase(graphTraversal.curNode);
                }
                frontier.pop_back();
                if (frontier.empty()) {
                    return false;
                }
            }
        }
        return true;
    }

    /* Get current node's next valid neighbor, with backtracking in vector */
    static bool advance(std::vector<Traversal *> &frontier, Traversal **traversal, const Graph &graph, std::unordered_set<uint32_t> &set) {
        bool validNeighbor = false;
        while(!validNeighbor) {
            *traversal = frontier.back();
            (*traversal)->next();
            if ((*traversal)->index != NONE) {
                validNeighbor = true;
            } else {
                set.erase((*traversal)->sourceNode);
                delete *traversal;
                frontier.pop_back();
                if (frontier.empty()) {
                    return false;
                }
            }
        }
        return true;
    }

private:
    struct Funnel;

    void static parseNodeIDs(char *buf, uint32_t *sourceNode, uint32_t *targetNode);
    void fill(const uint32_t &size, const bool &checkIndependentSet);
    bool getFunnels(std::vector<Funnel> &funnels, const uint32_t *measure = NULL, uint32_t *effectiveNode = NULL, Funnel *fourFunnel = NULL) const;
    uint32_t getGoodNode(std::vector<Traversal *> &frontier, std::unordered_set<uint32_t> &set, std::vector<uint32_t> &nodes, const uint32_t &size) const;
    uint32_t getOptimalDegree4Node1() const;
    uint32_t getOptimalDegree4Node2() const;
    void getOptimalDegree4Node3(uint32_t &maxNodeWithCond, uint32_t &maxNode) const;
    bool checkSeparation(const std::unordered_set<uint32_t> &cut, std::vector<uint32_t> &component1, std::vector<uint32_t> &component2, bool &actualComponent1) const;
    bool buildCC(const std::unordered_set<uint32_t> &excludedNodes, std::vector<uint32_t> &component1, std::vector<uint32_t> &component2) const;
    void static addPalmTreeArc(std::unordered_map<uint32_t, std::vector<uint32_t> > &palmTree, const uint32_t &node, const uint32_t &neighbor);
    struct Value1 {
        Value1(const uint32_t &visit) : visit(visit), low1(NONE), low2(visit), nd(0) {}
        uint32_t visit;
        uint32_t low1;
        uint32_t low2;
        uint32_t nd;
    };
    bool getSeparatingPairs2(std::unordered_set<uint32_t> &vertexCut, std::vector<uint32_t> &component1, std::vector<uint32_t> &component2, bool &actualComponent1,
                               const uint32_t &numberOfNodes, std::unordered_map<uint32_t, std::vector<uint32_t> > &palmTree1, std::unordered_map<uint32_t, std::vector<uint32_t> > &palmTree2,
                               std::unordered_map<uint32_t, std::vector<uint32_t> > &bothPalmTrees, std::unordered_map<uint32_t, Value1> &oldExploredSet, std::unordered_map<uint32_t, uint32_t> &fathers, std::unordered_map<uint32_t, uint32_t> &sons, std::unordered_map<uint32_t, std::unordered_set<uint32_t> > &descendants, std::unordered_map<uint32_t, uint32_t> &visitToId) const;



    struct NodeInfo {
    public:
        NodeInfo(const uint32_t &offset, const uint32_t &edges) : offset(offset), edges(edges), removed(false) {}

        uint32_t offset; // Offset of neighbors in edgeBuffer
        uint32_t edges;
        bool removed;
    };

    struct Traversal {
        Traversal(const uint32_t &node, const Graph &graph, Traversal *previous = NULL) {
            sourceNode = node;
            if (previous != NULL) {
                auto it = previous->set.begin();
                std::advance(it, previous->index);
                while (++it != previous->set.end()) {
                    set.insert(*it);
                }
                set.erase(node);
            }
            graph.gatherNeighbors(node, set);
            if (!set.size()) {
                index = NONE;
            } else {
                index = 0;
            }
        }

        void next() {
            if (++index == set.size()) {
                index = NONE;
            }
        }

        std::set<uint32_t> set;
        uint32_t index;
        uint32_t sourceNode;
    };

    struct Funnel {
        Funnel(const uint32_t &a, const uint32_t &b, const uint32_t &c, const uint32_t &d, const uint32_t &v) : a(a), b(b), c(c), d(d), v(v) {}
        Funnel() : a(NONE), b(NONE), c(NONE), d(NONE) {}
        void print() const {
            std::cout << a << "-" << v << "-{" << b << "," << c;
            if (d != NONE) {
                std::cout << "," << d;
            }
            std::cout << "}\n";
        }

        uint32_t a;
        uint32_t b;
        uint32_t c;
        uint32_t d;
        uint32_t v;
    };

    /* Rerturn edge buffer offset of first non-removed neighbor of node at pos */
    uint32_t getFirstValidNeighborOffset(const uint32_t &pos) const {
        bool validNeighbor = false;
        uint32_t offset = nodeIndex[pos].offset;
        uint32_t nextNodeOffset = (pos == nodeIndex.size()-1 ? edgeBuffer->size() : nodeIndex[pos+1].offset);
        for ( ; offset < nextNodeOffset ; offset++) {
            uint32_t nPos = (!mapping ? (*edgeBuffer)[offset] : idToPos->at((*edgeBuffer)[offset]));
            if (!nodeIndex[nPos].removed) {
                validNeighbor = true;
                break;
            }
        }
        return (validNeighbor ? offset : NONE);
    }

    std::vector<NodeInfo> nodeIndex;
    std::vector<uint32_t> *edgeBuffer; // Pointer to avoid copying at Graph::rebuild
    std::vector<uint32_t> zeroDegreeNodes;
    uint32_t nextUnusedId;

    /* Optional mappers, when ids are not equal to the equivalent vector index position */
    bool mapping;
    std::unordered_map<uint32_t, uint32_t> *idToPos;
    std::vector<uint32_t> *posToId;
};

#endif
