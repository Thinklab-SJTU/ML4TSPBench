#include "Mis.hpp"
#include <iostream>
using namespace std;

string Mis::misOutputFile = ".mis";

/* Insert hypernode, mark any inner hypernodes' outerLevel as false */
void Mis::markHypernode(const uint32_t &hypernode, const vector<uint32_t> &nodes, const vector<uint32_t> &neighbors) {
    const vector<uint32_t> *container = &nodes;
    for (uint32_t i = 0 ; i < 2 ; i++) {
        for (auto node: *container) {
            auto it = hypernodeToInnernode.find(node);
            if (it != hypernodeToInnernode.end()) {
                it->second.outerLevel = false;
            }
        }
        container = &neighbors;
    }
    assert(hypernodeToInnernode.insert({hypernode, Innernode(nodes, neighbors)}).second);
}

void Mis::unfoldHypernodes(vector<uint32_t> &zeroDegreeNodes, vector<uint32_t> &finalMis) {
    finalMis.reserve(zeroDegreeNodes.size() + mis.size());
    expandIncludedNodes(mis, finalMis);
    expandIncludedNodes(zeroDegreeNodes, finalMis);
    expandExcludedNodes(finalMis);
}

void Mis::print(vector<uint32_t> &finalMis) {
    cout <<"\nWriting Maximum Independent Set to file " << Mis::misOutputFile << endl;
    cout << "Maximum Independent Set size: " << finalMis.size() << "\n";
    FILE *f;
    f = fopen(Mis::misOutputFile.c_str(), "w");
    if (f == NULL) {
        cerr << "Error in opening mis output file " << Mis::misOutputFile << endl;
        exit(EXIT_FAILURE);
    }
    fprintf(f, "Maximum Independent Set size: %ld\n", finalMis.size());
    for (auto node : finalMis) {
        fprintf(f, "%" PRIu32 "\n", node);
    }
    fclose(f);
}

void Mis::expandExcludedNodes(vector<uint32_t> &finalMis) {
    while (!hypernodeToInnernode.empty()) {
        auto hypernode = hypernodeToInnernode.begin();
        while (!hypernode->second.outerLevel) {
            hypernode++;
        }
        for (auto node : hypernode->second.nodes) {
            auto nestedHypernode = hypernodeToInnernode.find(node);
            if (nestedHypernode != hypernodeToInnernode.end()) {
                assert(!nestedHypernode->second.outerLevel);
                nestedHypernode->second.outerLevel = true;
            }
            mis.push_back(node);
        }
        for (auto node : hypernode->second.neighbors) {
            auto innerHypernode = hypernodeToInnernode.find(node);
            if (innerHypernode != hypernodeToInnernode.end()) {
                assert(!innerHypernode->second.outerLevel);
                innerHypernode->second.outerLevel = true;
            }
        }
        hypernodeToInnernode.erase(hypernode);
        expandIncludedNodes(mis, finalMis);
    }
}

void Mis::removeHypernodes(unordered_set<uint32_t> &nodes) {
    unordered_set<uint32_t> toRemove;
    for (auto &h: hypernodeToInnernode) {
        bool marked = false;
        for (auto n: h.second.nodes) {
            if (hypernodeToInnernode.find(n) == hypernodeToInnernode.end() && nodes.find(n) == nodes.end()) {
                toRemove.insert(h.first);
                marked = true;
                break;
            }
        }
        if (marked) {
            continue;
        }
        for (auto n: h.second.neighbors) {
            if (hypernodeToInnernode.find(n) == hypernodeToInnernode.end() && nodes.find(n) == nodes.end()) {
                toRemove.insert(h.first);
                break;
            }
        }
    }

    /* Also remove outer hypernodes of hypernodes to be removed */
    while (toRemove.size()) {
        auto n = *toRemove.begin();
        hypernodeToInnernode.erase(n);
        for (auto &h: hypernodeToInnernode) {
            bool marked = false;
            for (auto i: h.second.nodes) {
                if (i == n) {
                    toRemove.insert(h.first);
                    marked = true;
                    break;
                }
            }
            if (!marked) {
                for (auto i: h.second.neighbors) {
                    if (i == n) {
                        toRemove.insert(h.first);
                        break;
                    }
                }
            }
        }
        toRemove.erase(n);
    }
}

void Mis::removeHypernodes(const unordered_map<uint32_t, Innernode> &hypernodes) {
    for (auto h: hypernodes) {
        hypernodeToInnernode.erase(h.first);
    }
}

void Mis::removeSubsequentNodes(unordered_set<uint32_t> &nodes) {
    removeHypernodes(nodes);
    unordered_set<uint32_t> toRemove;
    for (auto s: subsequentNodes) {
        if (nodes.find(s.second) == nodes.end()) {
            toRemove.insert(s.first);
        }
    }
    for (auto i: toRemove) {
        subsequentNodes.erase(i);
    }
}

void Mis::printAll(vector<uint32_t> &zeroDegreeNodes) const {
    cout << "\nMis:\n";
    for (auto node: mis) {
        cout << node << "\n";
    }
    cout << "\nZero degree nodes:\n";
    for (auto node: zeroDegreeNodes) {
        cout << node << "\n";
    }

    cout << "\nSubsequent nodes:\n";
    for (auto &subsequent: subsequentNodes) {
        cout << subsequent.first << " -> " << subsequent.second << "\n";
    }


    cout << "\nHypernodes:\n";
    for (auto &hypernode: hypernodeToInnernode) {
        cout << hypernode.first << ", outer " << hypernode.second.outerLevel << "\n";
        for (auto node: hypernode.second.nodes) {
            cout << " node  " << node << "\n";
        }
        for (auto node: hypernode.second.neighbors) {
            cout << " neighbor " << node << "\n";
        }
    }
}
