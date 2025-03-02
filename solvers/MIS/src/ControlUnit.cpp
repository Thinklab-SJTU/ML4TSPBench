#include "ControlUnit.hpp"
#include <iostream>
#include <string>
#include <cstring>
#include <cstdlib>

using namespace std;

int main(int argc, char **argv) {
    string graphInputFile;
    bool checkIndependentSet = false;
    for (int i=1 ; i < argc ; i++) {
        if (!strcmp(argv[i], "-check")) {
            checkIndependentSet = true;
        } else {
            graphInputFile = argv[i];
        }
   }
   if (!graphInputFile.compare("")) {
       cerr << "Error: No graph input file specified" << endl;
       exit(EXIT_FAILURE);
   }
   ControlUnit controlUnit(graphInputFile, checkIndependentSet);
   if (!checkIndependentSet) {
       controlUnit.run();
   } else {
       controlUnit.checkIndependentSet(graphInputFile + ".mis");
   }
}

void ControlUnit::run() {
    alg.run();
}

void ControlUnit::checkIndependentSet(const string &misInputFile) const {
    /* Build mis from input file */
    vector<uint32_t> mis;
    FILE *f;
    f = fopen(misInputFile.c_str(), "r");
    if (f == NULL) {
        cerr << "Error in opening MIS input file " << misInputFile << endl;
        exit(EXIT_FAILURE);
    }
    char buf[MAXLINE];
    if (fgets(buf, MAXLINE, f) == NULL) {
        cerr << "Error in parsing MIS input file " << misInputFile << endl;
        exit(EXIT_FAILURE);
    }
    while(fgets(buf, MAXLINE, f) != NULL) {
        uint32_t node = 0;
        for (uint32_t i = 0; buf[i] != ' ' && buf[i] != '\t' && buf[i] != '\n' ; i++) {
            node = node * 10 + (buf[i] - '0');
        }
        mis.push_back(node);
    }
    uint32_t node1, node2;
    unordered_set<uint32_t> set;
    for (auto node: mis) {
        if (!set.insert(node).second) {
            cout << "NOT a set! Node " << node << " exists more than once" << endl;
        }
    }

    bool independentSet = alg.getSearchTree()[0]->getGraph().isIndependentSet(mis, &node1, &node2);
    if (independentSet) {
        cout << "Valid independent set\n";
    } else {
        cout << "NOT an independent set!\nConflict between nodes " << node1 << " and " << node2 << endl;
    }
    fclose(f);
}
