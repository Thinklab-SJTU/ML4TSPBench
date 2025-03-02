#ifndef REDUCTIONS_H
#define REDUCTIONS_H

#include "Graph.hpp"
#include "Mis.hpp"

class Reductions {
public:
    Reductions(Graph &graph, Mis &mis) : graph(graph), mis(mis) {}
    ~Reductions();
    void run(const uint32_t &theta);
    ReduceInfo &getReduceInfo() {
        return reduceInfo;
    }
    std::unordered_map<uint32_t, std::vector<uint32_t>* > &getCCToNodes() {
        return ccToNodes;
    }

private:
    /* Check whether subset is a subset of node's neighbors */
    static bool isSubsetOfNeighbors(const std::vector<Graph::GraphTraversal> &subset, const uint32_t &node, const Graph &graph) {
        uint32_t pos = graph.getPos(node);
        uint32_t count = subset.size();
        uint32_t nextNodeOffset = (pos == graph.nodeIndex.size()-1 ? graph.edgeBuffer->size() : graph.nodeIndex[pos+1].offset);
        for (uint32_t offset = graph.nodeIndex[pos].offset ; count && offset < nextNodeOffset ; offset++) {
            if (find((*graph.edgeBuffer)[offset], subset)) {
                count--;
            }
        }
        return !count;
    }

    static bool find(const uint32_t &node, const std::vector<Graph::GraphTraversal> &set) {
        for (uint32_t i = 0 ; i < set.size() ; i++) {
            if (set[i].curNode == node) {
                return true;
            }
        }
        return false;
    }

    static bool nodeDegreesEqualTo(std::vector<uint32_t> &nodes, const uint32_t &degree, const Graph &graph) {
        for (auto node : nodes) {
            if (graph.getNodeDegree(node) != degree) {
                return false;
            }
        }
        return true;
    }

    void reduce6(const uint32_t &theta);
    void reduce5(const uint32_t &theta);
    void reduce4(const uint32_t &theta);
    void reduce3(const uint32_t &theta);
    bool removeDesks();
    bool removeDominatedNodes(const uint32_t &theta);
    bool removeDominatedNodes2(const uint32_t &theta);
    bool removeShortFunnels(const uint32_t &theta);
    void removeEasyInstances(const uint32_t &theta);
    void findMis(const std::vector<uint32_t> &cc);
    bool removeUnconfinedNodes();
    void removeUnconfinedNodes2();
    bool foldCompleteKIndependentSets(const uint32_t &theta, std::unordered_set<uint32_t> **oldCandidateNodes, std::unordered_set<uint32_t> **newCandidateNodes, const bool &theta4 = false);
    void foldCompleteKIndependentSets2(const uint32_t &theta, const bool &checkAllNodes, std::unordered_set<uint32_t> &oldCandidateNodes, std::unordered_set<uint32_t> &newCandidateNodes, const bool &theta4 = false);
    void removeLineGraphs(const uint32_t &theta);
    bool findClique(std::vector<Graph::GraphTraversal> &clique, std::vector<Graph::GraphTraversal> *previousClique, const uint32_t &cliqueSize);
    void findMisInComponent(const std::vector<uint32_t> &cc);
    void buildCC();
    void printCC() const;
    void printCCSizes() const;
    static void swap(std::unordered_set<uint32_t> **p1, std::unordered_set<uint32_t> **p2);

    Graph &graph;
    Mis &mis;
    ReduceInfo reduceInfo;
    std::unordered_map<uint32_t, std::vector<uint32_t>* > ccToNodes;
};

#endif
