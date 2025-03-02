#include "Util.hpp"
#include <algorithm>

using namespace std;

/* Vectors should be sorted by default. Optimization check of container sizes should
 * be done from caller (since there might be removed nodes in the container) */
bool isSubsetOf(const vector<uint32_t> &container1, vector<uint32_t>::iterator &begin, vector<uint32_t>::iterator &end) {
    for (auto &i: container1) {
        if (!binary_search(begin, end, i)) {
            return false;
        }
    }
    return true;
}

bool setsHaveKUncommonElements(const vector<uint32_t> &container1, const vector<uint32_t> &container2, const uint32_t &k, uint32_t &uncommonElement) {
    uint32_t uncommonElements = 0;
    const vector<uint32_t> *containerPtr = &container1;
    auto begin = container2.begin();
    auto end = container2.end();
    for (uint32_t it = 0 ; it < 2 ; it++) {
        for (auto &i: *containerPtr) {
            if (!binary_search(begin, end, i)) {
                uncommonElements++;
                uncommonElement = i;
                if (uncommonElements > k) {
                    return false;
                }
            }
        }
        containerPtr = &container2;
        begin = container1.begin();
        end = container1.end();
    }
    return (uncommonElements == k);
}
