#ifndef MIS_H
#define MIS_H
#include <stdint.h>
#include <inttypes.h>
#include <vector>
#include <unordered_map>
#include <unordered_set>
#include <assert.h>
#include <iostream>
#include <algorithm>

struct Innernode {
public:
    Innernode(const std::vector<uint32_t> &nodes, const std::vector<uint32_t> &neighbors) : nodes(nodes), neighbors(neighbors), outerLevel(true) {}

    std::vector<uint32_t> nodes;
    std::vector<uint32_t> neighbors;
    bool outerLevel;
};

class Mis {
public:
    std::vector<uint32_t> &getMis() {
        return mis;
    }
    void setMisOutputFile(std::string misOutputFile) {
        Mis::misOutputFile = misOutputFile;
    }
    void markHypernode(const uint32_t &hypernode, const std::vector<uint32_t> &nodes, const std::vector<uint32_t> &neighbors);
    void unfoldHypernodes(std::vector<uint32_t> &zeroDegreeNodes, std::vector<uint32_t> &finalMis);
    std::unordered_map<uint32_t, uint32_t> &getSubsequentNodes() {
        return subsequentNodes;
    }
    std::unordered_map<uint32_t, Innernode> &getHypernodeToInnerNode() {
        return hypernodeToInnernode;
    }
    void removeSubsequentNodes(std::unordered_set<uint32_t> &nodes);
    void removeHypernodes(std::unordered_set<uint32_t> &nodes);
    void removeHypernodes(const std::unordered_map<uint32_t, Innernode> &hypernodes);
    void static print(std::vector<uint32_t> &finalMis);
    void printAll(std::vector<uint32_t> &zeroDegreeNodes) const;

private:
    void expandIncludedNodes(std::vector<uint32_t> &set, std::vector<uint32_t> &finalMis) {
        for (uint32_t i = 0 ; i < set.size() ; i++) {
            auto subsequent = subsequentNodes.find(set[i]);
            if (subsequent != subsequentNodes.end()) {
                if (std::find(set.begin(), set.end(), subsequent->second) != set.end() || std::find(finalMis.begin(), finalMis.end(), subsequent->second) != finalMis.end()) {
                    subsequentNodes.erase(subsequent);
                } else {
                    set.push_back(subsequent->second);
                }
            }
            auto res = hypernodeToInnernode.find(set[i]);
            if (res == hypernodeToInnernode.end()) {
                finalMis.push_back(set[i]);
            } else {
                assert(res->second.outerLevel);
                for (auto node : res->second.neighbors) {
                    auto innerHypernode = hypernodeToInnernode.find(node);
                    if (innerHypernode != hypernodeToInnernode.end()) {
                        assert(!innerHypernode->second.outerLevel);
                        innerHypernode->second.outerLevel = true;
                    }
                    set.push_back(node);
                }
                for (auto node : res->second.nodes) {
                    auto innerHypernode = hypernodeToInnernode.find(node);
                    if (innerHypernode != hypernodeToInnernode.end()) {
                        assert(!innerHypernode->second.outerLevel);
                        innerHypernode->second.outerLevel = true;
                    }
                }
                hypernodeToInnernode.erase(res);
            }
        }
        set.clear();
    }

    void expandExcludedNodes(std::vector<uint32_t> &finalMis);

    std::vector<uint32_t> mis;
    std::unordered_map<uint32_t, Innernode> hypernodeToInnernode;

    /* Including one key node in the mis, results in including the mapped node as well,
     * These could be hypernodes, regular nodes, or mixed */
    std::unordered_map<uint32_t, uint32_t> subsequentNodes;
    static std::string misOutputFile;
};
#endif
