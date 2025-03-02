#ifndef UTIL_H
#define UTIL_H
#include <stdint.h>
#include <iostream>
#include <vector>

#define NONE UINT32_MAX

bool isSubsetOf(const std::vector<uint32_t> &container1, std::vector<uint32_t>::iterator &begin, std::vector<uint32_t>::iterator &end);
bool setsHaveKUncommonElements(const std::vector<uint32_t> &container1, const std::vector<uint32_t> &container2, const uint32_t &k, uint32_t &uncommonElement);

struct ReduceInfo {
public:
    ReduceInfo() : nodesRemoved(0) {}
    uint32_t nodesRemoved;

    void print(ReduceInfo *old = NULL) const {
        std::cout << "Nodes removed " << nodesRemoved - (old == NULL ? 0 : old->nodesRemoved) << std::endl;
    }
};

#endif
