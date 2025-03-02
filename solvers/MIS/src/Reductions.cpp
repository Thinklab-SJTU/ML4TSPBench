#include <iostream>
#include <vector>
#include <stack>
#include <unordered_map>
#include <algorithm>
#include <unordered_set>
#include <iomanip>
#include <iterator>
#include "Reductions.hpp"

using namespace std;

void Reductions::run(const uint32_t &theta) {
    if (!graph.nodeIndex.size()) {
        return;
    }
    switch(theta) {
        case 8:
        case 7:
        case 6:
            reduce6(theta);
            break;
        case 5:
            reduce5(theta);
            break;
        case 4:
            reduce4(theta);
            break;
        case 3:
            reduce3(theta);
            break;
        default:
            assert(false);
    }
}

void Reductions::reduce6(const uint32_t &theta) {
    bool firstTime = true;
    unordered_set<uint32_t> *oldCandidateNodes = new unordered_set<uint32_t>();
    unordered_set<uint32_t> *newCandidateNodes = new unordered_set<uint32_t>();
    do {
        removeUnconfinedNodes();
    } while (foldCompleteKIndependentSets(theta, &oldCandidateNodes, &newCandidateNodes));
    delete oldCandidateNodes;
    delete newCandidateNodes;
    buildCC();
    removeLineGraphs(theta);
    graph.rebuild(reduceInfo);
}

void Reductions::reduce5(const uint32_t &theta) {
    unordered_set<uint32_t> *oldCandidateNodes = new unordered_set<uint32_t>();
    unordered_set<uint32_t> *newCandidateNodes = new unordered_set<uint32_t>();
    do {
        do {
            removeUnconfinedNodes();
        } while (foldCompleteKIndependentSets(theta, &oldCandidateNodes, &newCandidateNodes));
    } while (removeShortFunnels(theta));
    delete oldCandidateNodes;
    delete newCandidateNodes;
    buildCC();
    removeEasyInstances(theta);
    removeLineGraphs(theta);
    graph.rebuild(reduceInfo);
}

void Reductions::reduce4(const uint32_t &theta) {
    unordered_set<uint32_t> *oldCandidateNodes = new unordered_set<uint32_t>();
    unordered_set<uint32_t> *newCandidateNodes = new unordered_set<uint32_t>();
    do {
        do {
            do {
                do {
                    foldCompleteKIndependentSets(theta, &oldCandidateNodes, &newCandidateNodes);
                } while (removeDominatedNodes(theta));
            } while (removeUnconfinedNodes());
        } while (foldCompleteKIndependentSets(theta, &oldCandidateNodes, &newCandidateNodes, true));
    } while (removeShortFunnels(theta));
    delete oldCandidateNodes;
    delete newCandidateNodes;
    buildCC();
    removeEasyInstances(theta);
    removeLineGraphs(theta);
    graph.rebuild(reduceInfo);
}

void Reductions::reduce3(const uint32_t &theta) {
    unordered_set<uint32_t> *oldCandidateNodes = new unordered_set<uint32_t>();
    unordered_set<uint32_t> *newCandidateNodes = new unordered_set<uint32_t>();
    do {
        do {
            do {
                do {
                    do {
                        removeDominatedNodes(theta);
                    } while (foldCompleteKIndependentSets(theta, &oldCandidateNodes, &newCandidateNodes));
                } while (removeUnconfinedNodes());
            } while (foldCompleteKIndependentSets(theta, &oldCandidateNodes, &newCandidateNodes, true));
        } while (removeShortFunnels(theta));
    } while (removeDesks());
    delete oldCandidateNodes;
    delete newCandidateNodes;
    buildCC();
    removeEasyInstances(theta);
    removeLineGraphs(theta);
    graph.rebuild(reduceInfo);
}

bool Reductions::removeDominatedNodes(const uint32_t &theta) {
    //cout << "\n**Performing dominated nodes reduction**" << endl;
    ReduceInfo old = reduceInfo;
    removeDominatedNodes2(theta);
    if (old.nodesRemoved == reduceInfo.nodesRemoved) {
        //cout << "No nodes removed." << endl;
        return false;
    }
    do {
        old = reduceInfo;
        removeDominatedNodes2(theta);
    } while (old.nodesRemoved != reduceInfo.nodesRemoved);
    return true;
}

bool Reductions::removeDominatedNodes2(const uint32_t &theta) {
    for (uint32_t pos1 = 0 ; pos1 < graph.nodeIndex.size() ; pos1++) {
        if (!graph.nodeIndex[pos1].edges || graph.nodeIndex[pos1].removed || theta == 3 && graph.nodeIndex[pos1].edges != 1) {
            continue;
        }
        for (uint32_t pos2 = pos1+1 ; pos2 < graph.nodeIndex.size() ; pos2++) {
            if (graph.nodeIndex[pos2].removed || graph.nodeIndex[pos1].edges > graph.nodeIndex[pos2].edges) {
                continue;
            }
            uint32_t node1 = graph.getNode(pos1);
            vector<uint32_t> neighbors1;
            uint32_t nextNodeOffset = (pos2 == graph.nodeIndex.size()-1 ? graph.edgeBuffer->size() : graph.nodeIndex[pos2+1].offset);
            graph.gatherNeighbors(node1, neighbors1);
            vector<uint32_t>::iterator begin = graph.edgeBuffer->begin();
            vector<uint32_t>::iterator end = graph.edgeBuffer->begin();
            std::advance(begin, graph.nodeIndex[pos2].offset);
            std::advance(end, nextNodeOffset);
            if (isSubsetOf(neighbors1, begin, end)) {
                uint32_t node2 = graph.getNode(pos2);
                //cout << "Dominated node " << node2 << "\n";
                graph.remove(node2, reduceInfo, (theta == 3 ? true : false));
                if (!graph.nodeIndex[pos1].edges) {
                    break;
                }
            }
        }
    }
    return true;
}

bool Reductions::removeDesks() {
    Graph::GraphTraversal graphTraversal(graph);
    while (graphTraversal.curNode != NONE) {
        bool valid = true;
        if (graph.getNodeDegree(graphTraversal.curNode) >= 3) {
            vector<uint32_t> neighbors;
            graph.gatherNeighbors(graphTraversal.curNode, neighbors);
            if (valid) {
                uint32_t b, d;
                for (uint32_t i = 0 ; i < neighbors.size() ; i++) {
                    b = neighbors[i];
                    if (graph.getNodeDegree(b) < 3) {
                        continue;
                    }
                    for (uint32_t j = i+1 ; j < neighbors.size() ; j++) {
                        d = neighbors[j];
                        if (graph.getNodeDegree(d) < 3) {
                            continue;
                        }
                        vector<uint32_t> commonNeighbors;
                        graph.getCommonNeighbors(b, d, commonNeighbors);
                        for (auto c: commonNeighbors) {
                            if (c == graphTraversal.curNode || graph.getNodeDegree(c) < 3) {
                                continue;
                            }
                            if (!graph.edgeExists(graphTraversal.curNode, c) && !graph.edgeExists(b, d)) {
                                set<uint32_t> neighborsAC;
                                set<uint32_t> neighborsBD;
                                graph.gatherNeighbors(graphTraversal.curNode, neighborsAC);
                                graph.gatherNeighbors(c, neighborsAC);
                                graph.gatherNeighbors(b, neighborsBD);
                                graph.gatherNeighbors(d, neighborsBD);
                                if (neighborsAC.size() <= 4 && neighborsBD.size() <= 4) {
                                    for (auto n: neighborsAC) {
                                        if (neighborsBD.find(n) != neighborsBD.end()) {
                                            valid = false;
                                            break;
                                        }
                                    }
                                    if (valid) {
                                        //cout << "reducing desk " << graphTraversal.curNode << "-" << b << "-" << c << "-" << d << "\n";
                                        graph.remove(neighborsAC, reduceInfo);
                                        graph.remove(neighborsBD, reduceInfo);
                                        std::unordered_map<uint32_t, uint32_t> &subsequentNodes = mis.getSubsequentNodes();
                                        for (auto neighborAC: neighborsAC) {
                                            subsequentNodes.insert({neighborAC, b});
                                            subsequentNodes.insert({neighborAC, d});
                                            graph.addEdges(neighborAC, vector<uint32_t>(neighborsBD.begin(), neighborsBD.end()));
                                        }
                                        for (auto neighborBD: neighborsBD) {
                                            subsequentNodes.insert({neighborBD, graphTraversal.curNode});
                                            subsequentNodes.insert({neighborBD, c});
                                            graph.addEdges(neighborBD, vector<uint32_t>(neighborsAC.begin(), neighborsAC.end()));
                                        }
                                        return true;
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
        graph.getNextNode(graphTraversal);
    }
    return false;
}


bool Reductions::removeShortFunnels(const uint32_t &theta) {
    //cout << "\n**Performing short funnels reduction**" << endl;
    uint32_t minDegree = NONE;
    graph.getMinDegree(minDegree);
    if (minDegree > 4) {
        return false;
    }
    Graph::GraphTraversal graphTraversal(graph);
    while (graphTraversal.curNode != NONE) {
        uint32_t nodeV = graphTraversal.curNode;
        if (graph.getNodeDegree(nodeV) == 3) {
            //cout << "nodeV " << nodeV << endl;
            vector<uint32_t> neighborsV;
            graph.gatherNeighbors(nodeV, neighborsV);
            for (uint32_t i = 0 ; i < 3 ; i++) {
                uint32_t nodeA = neighborsV[i];
                uint32_t nodeB, nodeC;
                if (i == 0) {
                    nodeB = neighborsV[1];
                    nodeC = neighborsV[2];
                } else if (i == 1) {
                    nodeB = neighborsV[0];
                    nodeC = neighborsV[2];
                } else {
                    nodeB = neighborsV[0];
                    nodeC = neighborsV[1];
                }
                if ((theta == 5 && graph.getNodeDegree(nodeA) == minDegree || theta == 4 && graph.getNodeDegree(nodeA) <= 4 || theta == 3) && graph.edgeExists(nodeB, nodeC)) {
                    bool shortFunnel = false;
                    vector<uint32_t> neighborsA;
                    graph.gatherNeighbors(nodeA, neighborsA);
                    if (theta == 5 && minDegree == 3 || theta == 4) {
                        uint32_t edges;
                        if (theta == 5) {
                            edges = 1;
                        } else if (theta == 4) {
                            edges = graph.getNodeDegree(nodeA) - 2;
                        }
                        for (auto &neighbor: neighborsA) {
                            if (neighbor == nodeV) {
                                continue;
                            }
                            if (graph.edgeExists(neighbor, nodeB) || graph.edgeExists(neighbor, nodeC)) {
                                if (--edges) {
                                    shortFunnel = true;
                                    break;
                                }
                            }
                        }
                    } else if (theta == 5 && minDegree == 4) {
                        uint32_t countB = 0;
                        uint32_t countC = 0;
                        uint32_t *count;
                        uint32_t target;
                        for (auto &neighbor: neighborsA) {
                            count = &countB;
                            target = nodeB;
                            for (uint32_t i = 0 ; i < 2 ; i++) {
                                if (graph.edgeExists(neighbor, target)) {
                                    (*count)++;
                                    if ((*count) == 2) {
                                        shortFunnel = true;
                                        break;
                                    }
                                }
                                count = &countC;
                                target = nodeC;
                            }
                            if (shortFunnel) {
                                break;
                            }
                        }
                    } else if (theta == 3) {
                        uint32_t count = 0;
                        uint32_t atmost = graph.getNodeDegree(nodeA);
                        shortFunnel = true;
                        for (auto &neighbor: neighborsA) {
                            if (neighbor == nodeV) {
                                continue;
                            }
                            if (!graph.edgeExists(neighbor, nodeB)) {
                                if (++count > atmost) {
                                    shortFunnel = false;
                                    break;
                                }
                            }
                            if (!graph.edgeExists(neighbor, nodeC)) {
                                if (++count > atmost) {
                                    shortFunnel = false;
                                    break;
                                }
                            }
                        }
                    }
                    if (shortFunnel) {
                        //cout << "short funnel " << nodeA << "-" << nodeV << "-{" << nodeB << "," << nodeC << "}" << endl;
                        vector<uint32_t> container;
                        container.push_back(nodeA);
                        container.push_back(nodeV);
                        graph.remove(container, reduceInfo);
                        uint32_t target = nodeB;
                        for (uint32_t i = 0 ; i < 2 ; i++) {
                            container.clear();
                            for (auto &neighbor: neighborsA) {
                                if (neighbor == nodeV) {
                                    continue;
                                }
                                if (!graph.edgeExists(neighbor, target)) {
                                    container.push_back(neighbor);
                                    graph.addEdges(neighbor, vector<uint32_t>(1, target));
                                }
                            }
                            graph.addEdges(target, container);
                            target = nodeC;
                        }
                        auto &subsequentNodes = mis.getSubsequentNodes();
                        for (auto &neighbor: neighborsA) {
                            if (neighbor == nodeV) {
                                continue;
                            }
                            subsequentNodes.insert({neighbor, nodeV});
                        }
                        subsequentNodes.insert({nodeB, nodeA});
                        subsequentNodes.insert({nodeC, nodeA});
                        return true;
                    }
                }
            }
        }
        graph.getNextNode(graphTraversal);
    }
    return false;
}

void Reductions::removeEasyInstances(const uint32_t &theta) {
    vector<unordered_map<uint32_t, vector<uint32_t>* >::iterator> removedCCs;
    for (auto it = ccToNodes.begin() ; it != ccToNodes.end() ; it++) {
        vector<uint32_t> *cc = it->second;
            if (theta == 5 && cc->size() <= 28 || theta == 4 && cc->size() <= 23 || theta == 3 && cc->size() <= 20) {
                //cout << "Removing easy instance component " << it->first << "\n";
                findMis(*cc);
                graph.remove(*cc, reduceInfo, true);
                removedCCs.push_back(it);
            }
        }
    if (removedCCs.empty()) {
        //cout << "No nodes removed" << endl;
    } else {
        for (auto cc : removedCCs) {
            delete cc->second;
            ccToNodes.erase(cc);
        }
    }
}

/* Find mis via brute force. For easy instances */
void Reductions::findMis(const vector<uint32_t> &cc) {
    vector<uint32_t> independentSet;
    vector<uint32_t> maximumIndependentSet;
    vector<uint32_t> frontier; // cc array indexes
    vector<unordered_set<uint32_t> > removedNodes;
    frontier.push_back(0);
    removedNodes.push_back(unordered_set<uint32_t>());
    while (frontier.size()) {
        uint32_t node = cc[frontier.back()];
        Graph::GraphTraversal graphTraversal(graph, node);
        while (graphTraversal.curEdgeOffset != NONE) {
            removedNodes.back().insert((*graph.edgeBuffer)[graphTraversal.curEdgeOffset]);
            graph.getNextEdge(graphTraversal);
        }
        uint32_t newNode = NONE;
        uint32_t newIndex;
        bool found = false;
        for (newIndex = frontier.back()+1 ; newIndex < cc.size() ; newIndex++) {
            if (cc.size() - newIndex + frontier.size() <= maximumIndependentSet.size()) {
                break;
            }
            newNode = cc[newIndex];
            if (removedNodes.back().find(newNode) == removedNodes.back().end()) {
                found = true;
                break;
            }
        }
        if (found) {
            frontier.push_back(newIndex);
            removedNodes.push_back(removedNodes.back());
        } else {
            if (frontier.size() > maximumIndependentSet.size()) {
                maximumIndependentSet = frontier;
            }
            while (frontier.size()) {
                newIndex = frontier.back() + 1;
                frontier.pop_back();
                removedNodes.pop_back();
                found = false;
                for (; newIndex < cc.size() ; newIndex++) {
                    if (cc.size() - newIndex + frontier.size() <= maximumIndependentSet.size()) {
                        break;
                    }
                    newNode = cc[newIndex];
                    if (!removedNodes.size() || removedNodes.size() && removedNodes.back().find(newNode) == removedNodes.back().end()) {
                        found = true;
                        break;
                    }
                }
                if (found) {
                    frontier.push_back(newIndex);
                    removedNodes.push_back((removedNodes.size() ? removedNodes.back() : unordered_set<uint32_t>()));
                    break;
                }
            }
        }
    }
    vector<uint32_t> &mis = this->mis.getMis();
    for (auto &i: maximumIndependentSet) {
        mis.push_back(cc[i]);
    }
}

bool Reductions::foldCompleteKIndependentSets(const uint32_t &theta, unordered_set<uint32_t> **oldCandidateNodes, unordered_set<uint32_t> **newCandidateNodes, const bool &theta4) {
    //cout << "\n**Performing K-Independent set folding reduction**" << endl;
    (*oldCandidateNodes)->clear();
    ReduceInfo old = reduceInfo;
    foldCompleteKIndependentSets2(theta, true, **oldCandidateNodes, **newCandidateNodes, theta4);
    swap(oldCandidateNodes, newCandidateNodes);
    if (old.nodesRemoved == reduceInfo.nodesRemoved) {
        return false;
    }
    do {
        old = reduceInfo;
        foldCompleteKIndependentSets2(theta, false, **oldCandidateNodes, **newCandidateNodes, theta4);
        swap(oldCandidateNodes, newCandidateNodes);
    } while (old.nodesRemoved != reduceInfo.nodesRemoved);
    return true;
}


bool Reductions::removeUnconfinedNodes() {
    //cout << "\n**Performing unconfined nodes reduction**" << endl;
    ReduceInfo old = reduceInfo;
    removeUnconfinedNodes2();
    if (old.nodesRemoved == reduceInfo.nodesRemoved) {
        return false;
    }
    do {
        old = reduceInfo;
        removeUnconfinedNodes2();
    } while (old.nodesRemoved != reduceInfo.nodesRemoved);
    return true;
}

void Reductions::foldCompleteKIndependentSets2(const uint32_t &theta, const bool &checkAllNodes, unordered_set<uint32_t> &oldCandidateNodes, unordered_set<uint32_t> &newCandidateNodes, const bool &theta4) {
    newCandidateNodes.clear();
    Graph::GraphTraversal graphTraversal(graph);
    auto it = oldCandidateNodes.begin();
    uint32_t node = (checkAllNodes ? graphTraversal.curNode : (it == oldCandidateNodes.end() ? NONE : *it));
    while (node != NONE) {
        uint32_t k, end;
        if (theta >= 6) {
            k = 1;
            end = 2;
        }
        else if (theta == 5) {
            k = 2;
            end = 2;
        } else if ((theta == 4 || theta == 3) && !theta4) {
            k = 1;
            end = 1;
        } else if (theta == 4 && theta4) {
            k = 2;
            end = 3;
        } else if (theta == 3 && theta4) {
            k = 2;
            end = 2;
        }
        for (; k <= end ; k++) {
            if ((checkAllNodes || (!checkAllNodes && !graph.nodeIndex[graph.getPos(*it)].removed)) &&
            graph.getNodeDegree(node) == k+1 && k != 3 || k == 3 && (graph.getNodeDegree(node) == 3 || graph.getNodeDegree(node) == 4)) {
                vector<uint32_t> nodes;
                vector<uint32_t> neighbors;
                nodes.push_back(node);
                graph.gatherNeighbors(node, neighbors);
                uint32_t secondNode = NONE;
                if (k == 2 || k == 3) {
                    secondNode = graph.getNextNodeWithIdenticalNeighbors(node, neighbors);
                    if (secondNode != NONE) {
                        nodes.push_back(secondNode);
                    }
                    else {
                        continue;
                    }
                }
                if (k == 3) {
                    uint32_t thirdNode;
                    uint32_t degree = graph.getNodeDegree(node);
                    if (degree == 3) {
                        uint32_t uncommonNeighbor;
                        thirdNode = graph.getNodeWithOneUncommonNeighbor(neighbors, uncommonNeighbor);
                        neighbors.push_back(uncommonNeighbor);
                    } else if (degree == 4) {
                        thirdNode = graph.getNextNodeWithIdenticalNeighbors(secondNode, neighbors);
                    } else {
                        assert(false);
                    }
                    if (thirdNode != NONE) {
                        nodes.push_back(thirdNode);
                    } else {
                        continue;
                    }
                }
                vector<uint32_t> &mis = this->mis.getMis();
                if (graph.isIndependentSet(neighbors)) {
                    uint32_t newNode = graph.contractToSingleNode(nodes, neighbors, reduceInfo);
                    this->mis.markHypernode(newNode, nodes, neighbors);
                } else if (theta != 4) {
                    mis.insert(mis.end(), nodes.begin(), nodes.end());
                }
                neighbors.insert(neighbors.end(), nodes.begin(), nodes.end());
                graph.remove(neighbors, reduceInfo, false, &newCandidateNodes);
                break;
            }
        }
        if (checkAllNodes) {
            graph.getNextNode(graphTraversal);
            node = graphTraversal.curNode;
        } else {
            it++;
            node = (it == oldCandidateNodes.end() ? NONE : *it);
        }
    }
}

void Reductions::removeUnconfinedNodes2() {
    Graph::GraphTraversal graphTraversal(graph);
    while (graphTraversal.curNode != NONE) {
        bool isUnconfined = false;
        unordered_set<uint32_t> extendedGrandchildren;
        graph.getExtendedGrandchildren(graphTraversal, extendedGrandchildren, &isUnconfined);
        if (isUnconfined || !graph.isIndependentSet(extendedGrandchildren)) {
            graph.remove(graphTraversal.curNode, reduceInfo);
        }
        graph.getNextNode(graphTraversal);
    }
}

void Reductions::removeLineGraphs(const uint32_t &theta) {
    //cout << "\n**Performing line graph reduction**" << endl;
    vector<unordered_map<uint32_t, vector<uint32_t>* >::iterator> removedCCs;
    for (auto it = ccToNodes.begin() ; it != ccToNodes.end() ; it++) {
        vector<uint32_t> *cc = it->second;
        vector<Graph::GraphTraversal> clique1;
        vector<Graph::GraphTraversal> clique2;
        clique1.reserve(5);
        clique2.reserve(5);
        uint32_t checks;
        if (theta >= 6) {
            checks = 3;
        } else {
            checks = 1;
        }
        for (uint32_t i = 0 ; i < checks ; i++) {
            uint32_t nodes, degree, clique1Size, clique2Size;
            if (theta >= 6) {
                switch (i) {
                    case 0:
                        nodes = 10;
                        degree = 6;
                        clique1Size = clique2Size = 4;
                        break;
                    case 1:
                        nodes = 15;
                        degree = 8;
                        clique1Size = clique2Size = 5;
                        break;
                    case 2:
                        nodes = 20;
                        degree = 7;
                        clique1Size = 5;
                        clique2Size = 4;
                        break;
                }
            } else if (theta == 5) {
                nodes = 12;
                degree = 5;
                clique1Size = 4;
                clique2Size = 3;
            } else if (theta <= 4) {
                nodes = 6;
                degree = 4;
                clique1Size = clique2Size = 3;
            }
            if (cc->size() == nodes && nodeDegreesEqualTo(*cc, degree, graph)) {
                bool foundCliques = true;
                for (auto node : *cc) {
                    clique1.clear();
                    clique2.clear();
                    clique1.push_back(Graph::GraphTraversal(graph, node));
                    if (!findClique(clique1, NULL, clique1Size)) {
                        foundCliques = false;
                        break;
                    }
                    clique2.push_back(Graph::GraphTraversal(graph, node));
                    if (!findClique(clique2, &clique1, clique2Size)) {
                        foundCliques = false;
                        break;
                    }
                }
                if (foundCliques) {
                    //cout << "Removing component " << it->first << "\n";
                    findMisInComponent(*cc);
                    graph.remove(*cc, reduceInfo, true);
                    removedCCs.push_back(it);
                    break;
                } else {
                    break;
                }
            }
        }
    }
    if (removedCCs.empty()) {
        //cout << "No nodes removed" << endl;
    } else {
        for (auto cc : removedCCs) {
            delete cc->second;
            ccToNodes.erase(cc);
        }
    }
}

void Reductions::findMisInComponent(const vector<uint32_t> &cc) {
    unordered_set<uint32_t> removedNodes;
    for (auto node : cc) {
        if (removedNodes.find(node) == removedNodes.end()) {
            mis.getMis().push_back(node);
            Graph::GraphTraversal graphTraversal(graph, node);
            while (graphTraversal.curEdgeOffset != NONE) {
                removedNodes.insert((*graph.edgeBuffer)[graphTraversal.curEdgeOffset]);
                graph.getNextEdge(graphTraversal);
            }
        }
    }
}

bool Reductions::findClique(vector<Graph::GraphTraversal> &clique, vector<Graph::GraphTraversal> *previousClique, const uint32_t &cliqueSize) {
    Graph::GraphTraversal graphTraversal = clique[0];
    if (graphTraversal.curNode == NONE) {
        return false;
    }
    while (clique.size() < cliqueSize) {
        uint32_t neighbor = (*graph.edgeBuffer)[graphTraversal.curEdgeOffset];
        bool existsInPreviousClique = (previousClique == NULL ? false : find(neighbor, *previousClique));
        if (!existsInPreviousClique && !find(neighbor, clique) && isSubsetOfNeighbors(clique, neighbor, graph)) {
            graph.goToNode(neighbor, graphTraversal);
            clique.push_back(graphTraversal);
        } else {
            if (!Graph::advance(clique, graphTraversal, graph)) {
                return false;
            }
        }
    }
    return true;
}

void Reductions::buildCC() {
    for (auto &cc: ccToNodes) {
        delete cc.second;
    }
    ccToNodes.clear();
    uint32_t component = 0;
    unordered_set<uint32_t> exploredSet;
    stack<uint32_t> frontier;
    Graph::GraphTraversal graphTraversal(graph);
    while (graphTraversal.curNode != NONE) {
        if (exploredSet.insert(graphTraversal.curNode).second) {
            vector<uint32_t> *componentNodes = new vector<uint32_t>();
            componentNodes->push_back(graphTraversal.curNode);
            frontier.push(graphTraversal.curNode);
            bool big = false;
            while (!frontier.empty()) {
                uint32_t node = frontier.top();
                frontier.pop();
                Graph::GraphTraversal neighbors(graph, node);
                while (neighbors.curEdgeOffset != NONE) {
                    node = (*graph.edgeBuffer)[neighbors.curEdgeOffset];
                    if (exploredSet.insert(node).second) {
                        frontier.push(node);
                        if (!big) {
                            componentNodes->push_back(node);
                        }
                        else if (componentNodes->size() > 20) {
                            big = true;
                        }
                    }
                    graph.getNextEdge(neighbors);
                }
            }
            if (!big) {
                ccToNodes.insert({component, componentNodes});
                component++;
            } else {
                delete componentNodes;
            }
        }
        graph.getNextNode(graphTraversal);
    }
}

Reductions::~Reductions() {
    for (auto &cc : ccToNodes) {
        delete cc.second;
    }
}

void Reductions::printCC() const {
    for (auto &cc : ccToNodes) {
        cout << "\nCC " << cc.first << ":\n";
    }
}

void Reductions::printCCSizes() const {
    for (auto &cc : ccToNodes) {
        cout << "CC " << cc.first << " size: " << (cc.second)->size() << "\n";
    }
}

void Reductions::swap(unordered_set<uint32_t> **p1, unordered_set<uint32_t> **p2) {
    unordered_set<uint32_t> *tmp = *p1;
    *p1 = *p2;
    *p2 = tmp;
}
