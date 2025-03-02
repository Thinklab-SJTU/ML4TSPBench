#ifndef ALG_H
#define ALG_H

#include "Graph.hpp"
#include "Reductions.hpp"
#include "Mis.hpp"


class Alg {
    struct SearchNode;

public:
    Alg(const std::string &inputFile, const bool &checkIndependentSet);
    ~Alg();
    void run();
    const std::vector<SearchNode *> &getSearchTree() const {
        return searchTree;
    }
    void print() const;

private:
    struct BranchingRule {
    public:
        enum class Type {
            MAX_DEGREE, SHORT_EDGE, OPTNODE, GOOD_FUNNEL, GOOD_PAIR, FOUR_CYCLE, OPT4NODE, EFFECTIVE_NODE, CUT, CUT_RIGHT_1, CUT_RIGHT_2, DONE
        };

        Type type;
        uint32_t node1;
        uint32_t node2;
        std::vector<uint32_t> container;


        BranchingRule() : type(Type::MAX_DEGREE), node1(NONE), node2(NONE) {}
        void clear() {
            node1 = NONE;
            node2 = NONE;
            container.clear();
        }

        void choose(const Graph &graph, Reductions &reductions, uint32_t &theta, SearchNode *searchNode) {
            uint32_t maxDegree = 0;
            uint32_t maxDegreeNode = NONE;
            graph.getMaxNodeDegree(maxDegreeNode, maxDegree);
            bool run = true;
            while (run) {
                //std::cout << "theta " << theta << " max degree " << maxDegree << "\n";
                run = false;
                switch (theta) {
                    case 8:
                    case 7:
                    case 6:
                        if (maxDegree > theta) {
                            type = Type::MAX_DEGREE;
                        } else if (maxDegree == theta) {
                            graph.getOptimalShortEdge(theta, node1, node2, container);
                            if (node1 != NONE) {
                                type = Type::SHORT_EDGE;
                                //std::cout << "optimal short edge " << node1 << " " << node2 << " with size " << container.size() << "\n";
                            } else {
                                type = Type::OPTNODE;
                                getOptimalNode(graph, theta);
                            }
                        } else {
                            theta = maxDegree;
                            if (theta < 3) {
                                theta = 3;
                            } else if (theta == 5 && searchNode->handleCuts()) {
                                return;
                            }
                            if (graph.nodeIndex.size()) {
                                reductions.run(theta);
                                graph.getMaxNodeDegree(maxDegreeNode, maxDegree);
                                run = true;
                            } else {
                                type = Type::DONE;
                            }
                        }
                        break;
                    case 5:
                        if (!maxDegree) {
                            type = Type::DONE;
                        } else if (maxDegree >= 6) {
                            type = Type::MAX_DEGREE;
                        } else if (graph.getGoodFunnelTheta5(node1, node2)) {
                            type = Type::GOOD_FUNNEL;
                        } else if (graph.getGoodPair(node1, node2, container)) {
                            type = Type::GOOD_PAIR;
                        } else if (maxDegree >= 5) {
                            node1 = graph.getOptimalDegree5Node();
                            type = Type::OPTNODE;
                        } else if (maxDegree < theta) {
                            theta = maxDegree;
                            if (theta < 3) {
                                theta = 3;
                            }
                            if (graph.nodeIndex.size()) {
                                reductions.run(theta);
                                graph.getMaxNodeDegree(maxDegreeNode, maxDegree);
                                run = true;
                            } else {
                                type = Type::DONE;
                            }
                        }
                        break;
                    case 4: {
                        type = Type::MAX_DEGREE;
                        if (maxDegree >= 5) {
                            uint32_t goodNode = graph.getGoodNode(reductions.getCCToNodes());
                            if (goodNode != NONE) {
                                maxDegreeNode = goodNode;
                            }
                        } else if (graph.getGoodFunnel(node1, node2)) {
                            type = Type::GOOD_FUNNEL;
                        } else if (graph.get4Cycle(container)) {
                            type = Type::FOUR_CYCLE;
                            //std::cout << "branching on optimal cycle " << container[0] << "-" << container[1] << "-" << container[2] << "-" << container[3] << "\n";
                        } else if (maxDegree == theta) {
                            node1 = graph.getOptimalDegree4Node();
                            assert(node1 != NONE);
                            type = Type::OPT4NODE;
                        }
                        else if (maxDegree < theta) {
                            theta = 3;
                            if (graph.nodeIndex.size()) {
                                reductions.run(theta);
                                graph.getMaxNodeDegree(maxDegreeNode, maxDegree);
                                run = true;
                            } else {
                                type = Type::DONE;
                            }
                        } else {
                            assert(false);
                        }
                        break;
                    }
                    case 3:
                        if (!maxDegree) {
                            type = Type::DONE;
                        }
                        else {
                            uint32_t effectiveNode = NONE;
                            uint32_t nodeV = NONE;
                            uint32_t nodeA = NONE;
                            graph.getEffectiveNodeOrOptimalFunnel(effectiveNode, nodeV, nodeA);
                            if (effectiveNode != NONE) {
                                node1 = effectiveNode;
                                type = Type::EFFECTIVE_NODE;
                            } else if (nodeV != NONE && nodeA != NONE) {
                                node1 = nodeA;
                                node2 = nodeV;
                                type = Type::GOOD_FUNNEL;
                            } else if (graph.get4CycleTheta3(container)) {
                                type = Type::FOUR_CYCLE;
                            }
                            else {
                                node1 = graph.getOptimalNodeTheta3(maxDegreeNode, maxDegree);
                                type = Type::MAX_DEGREE;
                            }
                        }
                        break;
                    case 0:
                        break;
                    default:
                        assert(false);
                }
            }
            if (type == Type::MAX_DEGREE) {
                node1 = maxDegreeNode;
            }
        }

        void getOptimalNode(const Graph &graph, const uint32_t &theta) {
            bool found = false;
            uint32_t count = 0;
            Graph::GraphTraversal graphTraversal(graph);
            while (graphTraversal.curNode != NONE) {
                if (graph.getNodeDegree(graphTraversal.curNode) == theta) {
                    count++;
                    uint32_t node = graphTraversal.curNode;
                    switch (theta) {
                        case 8: {
                            uint32_t degree8Neighbors = graph.getNumberOfDegreeNeighbors(node, 8);
                            if (degree8Neighbors <= 7) {
                                found = true;
                            } else if (degree8Neighbors == 8 && getMeasure(graph, node) >= 36) {
                                found = true;
                            }
                            break;
                        }
                        case 7: {
                            std::unordered_set<uint32_t> extendedGrandchildren;
                            graph.getExtendedGrandchildren(graphTraversal, extendedGrandchildren, NULL, true);
                            if (extendedGrandchildren.size()) {
                                found = true;
                            } else {
                                uint32_t degree7Neighbors = graph.getNumberOfDegreeNeighbors(node, 7);
                                if (degree7Neighbors <= 5) {
                                    found = true;
                                } else {
                                    uint32_t measure = getMeasure(graph, node);
                                    if (degree7Neighbors == 6 && measure >= 22 - 2 * graph.getNumberOfDegreeNeighbors(node, 3) - graph.getNumberOfDegreeNeighbors(node, 4) ||
                                    degree7Neighbors == 7 && measure >= 26) {
                                        found = true;
                                    }
                                }
                            }
                            break;
                        }
                        case 6: {
                            if (graph.getNumberOfDegreeNeighbors(node, 3) >= 1) {
                                found = true;
                            } else {
                                uint32_t degree6Neighbors = graph.getNumberOfDegreeNeighbors(node, 6);
                                if (degree6Neighbors <= 3) {
                                    found = true;
                                } else {
                                    uint32_t degree5Neighbors = graph.getNumberOfDegreeNeighbors(node, 5);
                                    if (degree6Neighbors == 4 && degree5Neighbors <= 1) {
                                        found = true;
                                    } else {
                                        std::unordered_set<uint32_t> neighborsAtDistance2;
                                        uint32_t count = 0;
                                        graph.getNeighborsAtDistance2(node, neighborsAtDistance2, theta, &count);
                                        uint32_t measure = getMeasure(graph, node, &neighborsAtDistance2);
                                        measure += count;
                                        if (degree6Neighbors == 4 && degree5Neighbors == 2 && measure >= 17 ||
                                        degree6Neighbors == 5 && graph.getNumberOfDegreeNeighbors(node, 4) == 1 && measure >= 18 ||
                                        degree6Neighbors == 5 && degree5Neighbors == 1 && measure >= 19 ||
                                        degree6Neighbors == 6 && measure >= 22) {
                                            found = true;
                                        }
                                    }
                                }
                            }
                            break;
                        }
                        default:
                            assert(false);
                    }
                    if (found) {
                        break;
                    }
                }
                graph.getNextNode(graphTraversal);
            }
            assert(found);
            node1 = graphTraversal.curNode;
        }

        static uint32_t getMeasure(const Graph &graph, const uint32_t &node, std::unordered_set<uint32_t> *neighborsAtDistance2Ptr = NULL) {
            std::vector<uint32_t> neighbors;
            graph.gatherNeighbors(node, neighbors);
            neighbors.push_back(node);
            std::unordered_set<uint32_t> neighborsAtDistance2;
            if (neighborsAtDistance2Ptr == NULL) {
                graph.getNeighborsAtDistance2(node, neighborsAtDistance2);
                neighborsAtDistance2Ptr = &neighborsAtDistance2;
            }
            uint32_t f = graph.getNumberOfEdgesBetweenSets(neighbors, *neighborsAtDistance2Ptr);
            return 2 * f - neighborsAtDistance2Ptr->size();
        }

    };

    struct SearchNode {
    public:
        SearchNode(const SearchNode &searchNode, const uint32_t &parent = NONE);
        SearchNode(const std::string &inputFile, const bool &checkIndependentSet) : id(NONE), graph(inputFile, checkIndependentSet), reductions(new Reductions(graph, mis)), parent(NONE), leftChild(NONE), rightChild(NONE), finalMis(NULL), hasCut(false), cutIsDone(false) {
            uint32_t maxDegreeNode;
            graph.getMaxNodeDegree(maxDegreeNode, theta);
            if (theta > 8) {
                theta = 8;
            } else if (theta < 3) {
                theta = 3;
            }
        }
        ~SearchNode();
        const Graph &getGraph() const {
            return graph;
        }
        void print() const;
        bool handleCuts();

        uint32_t id;
        uint32_t theta;
        BranchingRule branchingRule;
        Graph graph;
        Mis mis; // Current mis and hypernodes
        Reductions *reductions;
        uint32_t parent;
        uint32_t leftChild;
        uint32_t rightChild;
        std::vector<uint32_t> *finalMis; // Final mis of children search nodes, no hypernodes

        bool hasCut;
        std::unordered_set<uint32_t> cut;
        std::vector<uint32_t> c1, c2;
        bool actualComponent1;
        bool cutIsDone;
    };


    void chooseMaxMis(SearchNode *parent) {
        //std::cout << "no next child" << std::endl;
        SearchNode *leftChild = searchTree[parent->leftChild];
        SearchNode *rightChild = searchTree[parent->rightChild];
        assert(parent->rightChild == searchTree.size() - 1 && parent->leftChild == searchTree.size() - 2);
        parent->id = leftChild->id;
        parent->finalMis = leftChild->finalMis;
        std::vector<uint32_t> *min = rightChild->finalMis;
        if (rightChild->finalMis->size() > parent->finalMis->size()) {
            parent->id = rightChild->id;
            parent->finalMis = rightChild->finalMis;
            min = leftChild->finalMis;
        }
        delete min;
        delete searchTree.back();
        searchTree.pop_back();
        delete searchTree.back();
        searchTree.pop_back();
    }

    void concatMis(SearchNode *parent) {
        SearchNode *leftChild = searchTree[parent->leftChild];
        SearchNode *rightChild = searchTree[parent->rightChild];
        assert(parent->rightChild == searchTree.size() - 1 && parent->leftChild == searchTree.size() - 2);
        for (auto i: *leftChild->finalMis) {
            for (auto j: *rightChild->finalMis) {
                if (i == j) {
                    assert(0);
                }
            }
        }
        parent->finalMis = leftChild->finalMis;
        parent->finalMis->insert(parent->finalMis->end(), rightChild->finalMis->begin(), rightChild->finalMis->end());
        parent->graph.collectZeroDegreeNodes();
        parent->mis.unfoldHypernodes(parent->graph.zeroDegreeNodes, *(parent->finalMis));

        delete rightChild->finalMis;
        delete searchTree.back();
        searchTree.pop_back();
        delete searchTree.back();
        searchTree.pop_back();
    }

    void chooseCutBranch(SearchNode *parent) {
        SearchNode *leftChild = searchTree[parent->leftChild];
        SearchNode *rightChild = searchTree[parent->rightChild];
        assert(parent->rightChild == searchTree.size() - 1 && parent->leftChild == searchTree.size() - 2);
        if (leftChild->finalMis->size() == rightChild->finalMis->size()) {
            parent->branchingRule.type = BranchingRule::Type::CUT_RIGHT_1;
        } else { //if (leftChild->finalMis->size() > rightChild->finalMis->size()) {
            parent->branchingRule.type = BranchingRule::Type::CUT_RIGHT_2;
        }/* else {
            assert(false);
        }*/
        delete rightChild->finalMis;
        delete searchTree.back();
        searchTree.pop_back();
        parent->cutIsDone = true;
    }

    void branchLeft(const BranchingRule &branchingRule, SearchNode *searchNode, Mis &parentMis) const {
        switch (branchingRule.type) {
            case BranchingRule::Type::MAX_DEGREE:
            case BranchingRule::Type::OPTNODE:
            case BranchingRule::Type::OPT4NODE:
            case BranchingRule::Type::EFFECTIVE_NODE: {
                searchNode->graph.remove(branchingRule.node1, searchNode->reductions->getReduceInfo());
                break;
            }
            case BranchingRule::Type::SHORT_EDGE: {
                std::vector<uint32_t> shortEdge;
                shortEdge.push_back(branchingRule.node1);
                shortEdge.push_back(branchingRule.node2);
                searchNode->graph.remove(shortEdge, searchNode->reductions->getReduceInfo());
                break;
            }
            case BranchingRule::Type::GOOD_FUNNEL: {
                branchOnExtendedGranchildren(branchingRule, searchNode);
                break;
            }
            case BranchingRule::Type::GOOD_PAIR: {
                std::vector<uint32_t> goodPair;
                goodPair.push_back(branchingRule.node1);
                goodPair.push_back(branchingRule.node2);
                searchNode->graph.remove(goodPair, searchNode->reductions->getReduceInfo());
                break;
            }
            case BranchingRule::Type::FOUR_CYCLE: {
                std::vector<uint32_t> nonAdjacent;
                nonAdjacent.push_back(branchingRule.container[0]);
                nonAdjacent.push_back(branchingRule.container[2]);
                searchNode->graph.remove(nonAdjacent, searchNode->reductions->getReduceInfo());
                break;
            }
            case BranchingRule::Type::CUT: {
                //std::cout << "left cut" << std::endl;
                std::vector<uint32_t> *component1 = (searchNode->actualComponent1 ? &(searchNode->c1) : &(searchNode->c2));
                std::vector<uint32_t> *component2 = (searchNode->actualComponent1 ? &(searchNode->c2) : &(searchNode->c1));
                component2->insert(component2->end(), searchNode->cut.begin(), searchNode->cut.end());
                searchNode->graph.remove(*component2, searchNode->reductions->getReduceInfo());
                for (auto c: searchNode->cut) {
                    component2->pop_back();
                }
                std::unordered_set<uint32_t> nodesInComponent;
                nodesInComponent.insert(component1->begin(), component1->end());
                searchNode->graph.rebuildFromNodes(nodesInComponent);
                searchNode->mis.getMis().clear();
                searchNode->mis.removeSubsequentNodes(nodesInComponent);
                parentMis.removeHypernodes(searchNode->mis.getHypernodeToInnerNode());
                break;
            }
            default:
                assert(false);
        }
        uint32_t dummy, maxDegree;
        uint32_t oldTheta = searchNode->theta;
        searchNode->graph.getMaxNodeDegree(dummy, maxDegree, oldTheta);
        if (maxDegree < oldTheta) {
            if (maxDegree >= 3) {
                searchNode->theta = maxDegree;
            } else {
                searchNode->theta = 3;
            }
        }
    }

    void branchRight(BranchingRule &branchingRule, SearchNode *searchNode, Mis &parentMis) const {
        //std::cout << "right\n";
        switch (branchingRule.type) {
            case BranchingRule::Type::MAX_DEGREE:
            case BranchingRule::Type::OPTNODE:
            case BranchingRule::Type::EFFECTIVE_NODE: {
                branchOnExtendedGranchildren(branchingRule, searchNode);
                break;
            }
            case BranchingRule::Type::SHORT_EDGE: {
                branchingRule.container.push_back(branchingRule.node1);
                branchingRule.container.push_back(branchingRule.node2);
                searchNode->graph.remove(branchingRule.container, searchNode->reductions->getReduceInfo());
                std::vector<uint32_t> neighbors1, neighbors2;
                searchNode->graph.gatherNeighbors(branchingRule.node1, neighbors1);
                searchNode->graph.gatherNeighbors(branchingRule.node2, neighbors2);
                std::unordered_map<uint32_t, uint32_t> &subsequentNodes = searchNode->mis.getSubsequentNodes();
                for (auto neighbor1: neighbors1) {
                    subsequentNodes.insert({neighbor1, branchingRule.node2});
                    searchNode->graph.addEdges(neighbor1, neighbors2);
                }
                for (auto neighbor2: neighbors2) {
                    subsequentNodes.insert({neighbor2, branchingRule.node1});
                    searchNode->graph.addEdges(neighbor2, neighbors1);
                }
                break;
            }
            case BranchingRule::Type::GOOD_FUNNEL: {
                searchNode->mis.getMis().push_back(branchingRule.node2);
                std::vector<uint32_t> neighbors;
                searchNode->graph.gatherNeighbors(branchingRule.node2, neighbors);
                neighbors.push_back(branchingRule.node2);
                searchNode->graph.remove(neighbors, searchNode->reductions->getReduceInfo());
                break;
            }
            case BranchingRule::Type::GOOD_PAIR: {
                searchNode->graph.remove(branchingRule.container, searchNode->reductions->getReduceInfo());
                break;
            }
            case BranchingRule::Type::FOUR_CYCLE: {
                std::vector<uint32_t> nonAdjacent;
                nonAdjacent.push_back(branchingRule.container[1]);
                nonAdjacent.push_back(branchingRule.container[3]);
                searchNode->graph.remove(nonAdjacent, searchNode->reductions->getReduceInfo());
                break;
            }
            case BranchingRule::Type::OPT4NODE: {
                std::vector<uint32_t> neighbors;
                searchNode->graph.gatherNeighbors(branchingRule.node1, neighbors);
                neighbors.push_back(branchingRule.node1);
                searchNode->mis.getMis().push_back(branchingRule.node1);
                searchNode->graph.remove(neighbors, searchNode->reductions->getReduceInfo());
                break;
            }
            case BranchingRule::Type::CUT: {
                //std::cout << "right cut" << std::endl;
                std::vector<uint32_t> *component1 = (searchNode->actualComponent1 ? &(searchNode->c1) : &(searchNode->c2));
                std::vector<uint32_t> *component2 = (searchNode->actualComponent1 ? &(searchNode->c2) : &(searchNode->c1));
                std::set<uint32_t> neighbors;
                searchNode->graph.gatherAllNeighbors(searchNode->cut, neighbors);
                component2->insert(component2->end(), searchNode->cut.begin(), searchNode->cut.end());
                searchNode->graph.remove(*component2, searchNode->reductions->getReduceInfo());
                for (auto c: searchNode->cut) {
                    component2->pop_back();
                }
                searchNode->graph.remove(neighbors, searchNode->reductions->getReduceInfo());
                std::unordered_set<uint32_t> nodesInComponent;
                nodesInComponent.insert(component1->begin(), component1->end());
                for (auto n: neighbors) {
                    nodesInComponent.erase(n);
                }
                searchNode->graph.rebuildFromNodes(nodesInComponent);
                searchNode->mis.getMis().clear();
                searchNode->mis.removeSubsequentNodes(nodesInComponent);
                parentMis.removeHypernodes(searchNode->mis.getHypernodeToInnerNode());
                break;
            }
            case BranchingRule::Type::CUT_RIGHT_1: {
                //std::cout << "right cut 1" << std::endl;
                std::vector<uint32_t> *component1 = (searchNode->actualComponent1 ? &(searchNode->c1) : &(searchNode->c2));
                std::vector<uint32_t> *component2 = (searchNode->actualComponent1 ? &(searchNode->c2) : &(searchNode->c1));
                searchNode->graph.remove(*component1, searchNode->reductions->getReduceInfo());
                std::unordered_set<uint32_t> nodesInComponent;
                nodesInComponent.insert(component2->begin(), component2->end());
                nodesInComponent.insert(searchNode->cut.begin(), searchNode->cut.end());
                searchNode->graph.rebuildFromNodes(nodesInComponent);
                searchNode->mis.getMis().clear();
                searchNode->mis.removeSubsequentNodes(nodesInComponent);
                parentMis.removeHypernodes(searchNode->mis.getHypernodeToInnerNode());
                break;
            }
            case BranchingRule::Type::CUT_RIGHT_2: {
                //std::cout << "right cut 2" << std::endl;
                std::vector<uint32_t> *component1 = (searchNode->actualComponent1 ? &(searchNode->c1) : &(searchNode->c2));
                std::vector<uint32_t> *component2 = (searchNode->actualComponent1 ? &(searchNode->c2) : &(searchNode->c1));
                component1->insert(component1->end(), searchNode->cut.begin(), searchNode->cut.end());
                searchNode->graph.remove(*component1, searchNode->reductions->getReduceInfo());
                for (auto c: searchNode->cut) {
                    component1->pop_back();
                }
                std::unordered_set<uint32_t> nodesInComponent;
                nodesInComponent.insert(component2->begin(), component2->end());
                searchNode->graph.rebuildFromNodes(nodesInComponent);
                searchNode->mis.getMis().clear();
                searchNode->mis.removeSubsequentNodes(nodesInComponent);
                parentMis.removeHypernodes(searchNode->mis.getHypernodeToInnerNode());
                break;
            }
            default:
                assert(false);
        }
        uint32_t dummy, maxDegree;
        uint32_t oldTheta = searchNode->theta;
        searchNode->graph.getMaxNodeDegree(dummy, maxDegree, oldTheta);
        if (maxDegree < oldTheta) {
            if (maxDegree >= 3) {
                searchNode->theta = maxDegree;
            } else {
                searchNode->theta = 3;
            }
        }
    }

    void branchOnExtendedGranchildren(const BranchingRule &branchingRule, SearchNode *searchNode) const {
        std::unordered_set<uint32_t> extendedGrandchildren;
        Graph::GraphTraversal graphTraversal(searchNode->graph, branchingRule.node1);
        searchNode->graph.getExtendedGrandchildren(graphTraversal, extendedGrandchildren);
        extendedGrandchildren.insert(branchingRule.node1);
        std::vector<uint32_t> &mis = searchNode->mis.getMis();
        mis.insert(mis.end(), extendedGrandchildren.begin(), extendedGrandchildren.end());
        std::unordered_set<uint32_t> neighbors;
        searchNode->graph.gatherAllNeighbors(extendedGrandchildren, neighbors);

        std::unordered_set<uint32_t> *smaller = &neighbors;
        std::unordered_set<uint32_t> *bigger = &extendedGrandchildren;
        if (smaller->size() > bigger->size()) {
            std::unordered_set<uint32_t> *tmp = smaller;
            smaller = bigger;
            bigger = tmp;
        }
        smaller->insert(bigger->begin(), bigger->end());
        searchNode->graph.remove(std::vector<uint32_t>(smaller->begin(), smaller->end()), searchNode->reductions->getReduceInfo());
    }

    std::vector<SearchNode *> searchTree;
};


#endif
