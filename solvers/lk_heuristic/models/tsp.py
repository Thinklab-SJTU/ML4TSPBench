import math
import logging
import random
from itertools import permutations
from solvers.lk_heuristic.models.edge import Edge
from solvers.lk_heuristic.models.tour import Tour
from solvers.lk_heuristic.models.node import NodePivot


class Tsp:
    """
    The TSP class represents the travelling salesman problem inputs and helper methods for the problem.
    """

    # set the precision of the gain
    # this is required when computing the gain in "simmetric" tours, to avoid incorrect calculations when using just "> 0"
    gain_precision = 0.01

    def __init__(self, nodes, cost_function, shuffle=False, backtracking=(5, 5), reduction_level=4, reduction_cycle=4, tour_type="cycle", logging_level=logging.INFO):
        """
        The TSP input is a list of nodes which will be used as input to build a tour and a cost function to build the cost matrix.

        :param nodes: the list of nodes for the tsp
        :type nodes: list
        :param cost_function: the function used to compute the cost between two nodes
        :type cost_function: function
        :param shuffle: a boolean value indicating if nodes shall be shuffled before building the tour
        :type shuffle: boolean
        :param backtracking: the number of closest neighbors to use in each backtracking level.
        :type backtracking: tuple
        :param reduction_level: the starting level in current optimization cycle where reducted edges will start being considered
        :type reduction_level: int
        :param reduction_cycle: the number of optimization cycles when reducted edges will start being considered
        :type reduction_cycle: int
        :param tour_type: the type of the tour (either 'path' or 'cycle')
        :type tour_type: str
        :param logging_level: the level for logging messages
        :type logging_level: int
        """

        # setup the logger
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging_level)

        # initialize nodes
        self.nodes = nodes

        # initialize the tour type
        self.tour_type = tour_type

        # initialize the tour using the nodes and the tour type
        self.tour = Tour(self.nodes, t=tour_type)

        # shuffle the tour nodes
        if shuffle:
            self.tour.shuffle()

        # a boolean indicating if tour nodes shall be shuffled at every restart of the optimization loop
        # shuffle will not be applied when a double bridge move is found using lk1 optimizer
        self.shuffle = True

        # initialize the cost matrix after having the nodes initialized into the tour object using the defined cost_function
        self.cost_matrix = {}
        self.set_cost_matrix(cost_function)

        # initialize tour cost after having the cost matrix initialized
        self.tour.set_cost(self.cost_matrix)

        # solution methods avaliable to solve the problem
        self.methods = {
            "bf_improve": self.bf_improve,
            "nn_improve": self.nn_improve,
            'lk1_improve': self.lk1_improve,
            "lk2_improve": self.lk2_improve,
        }

        # init the set of solutions
        # the set of solutions will help to avoid repeated tours analysis
        # it is a hash of a tuple of node sequences
        self.solutions = set()

        # init a dict of closest neighbors
        # each node will have a maximum number of closest neighbors defined by the cost between them. It is required to use the maximum value of the backtracking.
        self.closest_neighbors = {}
        self.set_closest_neighboors(max_neighbors=max(backtracking))

        # init the bactracking levels
        # each value represent the number of neighbors to try in each search level
        self.backtracking = backtracking

        # init the reduction starting level, starting cycle and reduction edges set
        # this set contains the non-intersected edges at a maximum reduction levels
        # the set is initialized empty and populated after first local optima is found
        self.reduction_level = reduction_level
        self.reduction_cycle = reduction_cycle
        self.reduction_edges = set()

        # init the improvement cycles count, updated everytime a local optima is found
        # this parameter is only relevant when running many improvement cycles, used to apply reduction and double bridge moves
        self.cycles = 0

        # gains values collected during the search, the best gain value and the double bridge gain
        self.close_gains = []
        self.best_close_gain = 0
        self.double_bridge_gain = 0

    def set_cost_matrix(self, cost_func):
        """
        Compute the cost matrix using tsp nodes and a predefined cost function. The matrix is in the form of a dictionary with (row,column) values as keys of the dict

        :param cost_func: the cost function to be used to compute the cost matrix
        :type cost_func: function
        """

        # index is used to avoid repeated computation
        index = 0
        for i in range(len(self.nodes)):
            for j in range(index, len(self.nodes)):
                # get node 1 and node2
                n1 = self.nodes[i]
                n2 = self.nodes[j]

                # compute the cost between nodes
                # for hamiltonian path, pivot nodes are used and edges containing those nodes will have zero cost
                cost = 0
                if not (type(n1) == NodePivot or type(n2) == NodePivot):
                    cost = cost_func(n1, n2)

                # update cost matrix
                self.cost_matrix[(n1.id, n2.id)] = cost
                self.cost_matrix[(n2.id, n1.id)] = cost  # simmetric value

            # update index for next iteration
            index += 1

    def set_closest_neighboors(self, max_neighbors):
        """
        Get the closest neighbors of each tsp node using the cost_matrix values. The return will be a dict of each node as key and the value is a list of closest nodes. The number of values to be returned inside the list is defined by the max_neighboors variable.

        This function, combined with the get_best_neighbors, is the lookahead refinement from LK paper, where the best swaps from the closest neighbors of a node direct the search and speed up performance.

        :param max_neighboors: the maximum number of closest neighbors to collect
        :type max_neighboors: int
        :return: returns a dict mapping each node with its closest neighbors
        :rtype: list
        """

        # loop through each node
        for node1 in self.tour.nodes:

            # get all edge costs for each possible neighbor
            neighbors = [(node2, self.cost_matrix[(node1.id, node2.id)]) for node2 in self.tour.nodes if node2 != node1]

            # sort the neighbors based on the cost value and get the smallest ones
            neighbors_min = sorted(neighbors, key=lambda x: x[1])[:max_neighbors]

            # populate neighbor list with the best neighbors
            self.closest_neighbors[node1] = [neighbor[0] for neighbor in neighbors_min]

    def get_best_neighbors(self, t2, t1=None):
        """
        Get the best tuple of nodes (t3,t4) by computing the gain of swapping the closest neighbors of t2 (i.e, swapping (t3,t4) with (t2,t3)). This is the lookahead refinement where the nodes to be selected for executing the swap are sorted/selected by best gain. Node t1 is used in this function to check if the swap (t1,t2,t3,t4) is valid. If t1 is not provided, the swap is done without checking if it is valid.

        :param t2: the t2 node from where best neighbors shall be found
        :type t2: Node
        :param t1: the t1 node (neighbor of t2 that makes the broken edge)
        :type t1: Node
        :return: a dict with keys (t3,t4) and value is the gain
        :rtype: dict
        """

        # a dict storing the best neighbors from t2
        best_neighbors = {}

        # loop through each closest neighbor of t2 (since looping throgh every neighbor would be time consuming)
        for t3 in self.closest_neighbors[t2]:
            # loop through each possible t4
            for t4 in (t3.pred, t3.succ):
                # check if swap is valid
                if (t1):
                    if (self.tour.is_swap_feasible(t1, t2, t3, t4)):
                        # compute the gain
                        best_neighbors[(t3, t4)] = self.cost_matrix[(t3.id, t4.id)] - self.cost_matrix[t2.id, t3.id]
                # compute the gain not checking for valid swaps
                else:
                    # compute the gain
                    best_neighbors[(t3, t4)] = self.cost_matrix[(t3.id, t4.id)] - self.cost_matrix[t2.id, t3.id]

        # returns a list of (key,value) pairs of the max values of gain.
        return sorted(best_neighbors.items(), key=lambda x: x[1], reverse=True)

    def lk1_feasible_search(self, level, gain, swap_func, t1, t2, t3, t4, broken_edges, joined_edges):
        """
        This is the main search loop of LK Heuristic, trying to find broken and joined edges in such way that a feasible tour with lower cost is found. The search is recursivelly called while potential nodes exists. 

        The search store both the gain value of closing the loop and the swap memory of the executed swaps. This memory is used when exiting the recursive call, so that is possible to select the best gain found in the search by undoing the executed swaps until the best gain position.

        This function may be called from the unfeasible swap function. In this case, the starting swap function will be a special function that turns an unfeasible tour into a feasible one. 

        :param level: the current level of the search to compare with backtracking levels and reduction level
        :type level: int
        :param gain: the current gain from last LK step
        :type gain: float
        :param swap_func: the swap function to be used at nodes (t1, t2, t3, t4)
        :type swap_func: str
        :param t1: the tail node of the first broken edge in the 2-opt swap
        :type t1: Node2D
        :param t2: the head node of the first broken edge in the 2-opt swap
        :type t2: Node2D
        :param t3: the tail node of the second broken edge in the 2-opt swap
        :type t3: Node2D
        :param t4: the head node of the second broken edge in the 2-opt swap
        :type t4: Node2D
        :param broken_edges: the set of broken edges
        :type broken_edges: set
        :param joined_edges: the set of joined edges
        :type joined_edges: set
        """

        # step 4(a) at LK Paper is omitted here since the feasibility criterion is done when searching for the best neighbors in a swap. When entering this function, step 4(a) was already checked by the get best neighbors function.

        # set xi edge (the new edge to be broken)
        broken_edge = Edge(t3, t4)
        broken_cost = self.cost_matrix[(t3.id, t4.id)]

        # apply the reduction refinement from LK Paper
        # reduction is not applied in current optimization cycle if level is greater than a certain level
        # reduction is not applied if a certain amount of optimization cycles were not run yet
        if (level >= self.reduction_level and self.cycles <= self.reduction_cycle and broken_edge in self.reduction_edges):
            return

        # update the sets of broken and joined edges using the swap nodes
        broken_edges.add(Edge(t1, t2))
        joined_edges.add(Edge(t2, t3))

        # execute the swap based on the swap function to be executed
        # when coming from previous feasible swap, a feasible swap is reaplied
        # when coming from a previous unfeasible swap, the node between t2 and t3 is applied
        if swap_func == "swap_feasible":
            self.tour.swap_feasible(t1, t2, t3, t4)
        elif swap_func == "swap_node_between_t2_t3":
            self.tour.swap_node_between_t2_t3(t1, t2, t3, t4)

        # set yi edge to close the tour (instead of continuing exploration)
        # also check that close joined edge is valid (disjoint and not already in tour)
        joined_close_edge = Edge(t4, t1)
        joined_close_cost = self.cost_matrix[(t4.id, t1.id)]
        joined_close_valid = joined_close_edge not in self.tour.edges and joined_close_edge not in broken_edges

        # compute the gain of closing the tour and add it to close gain list
        # also update the best gain found so far for closing the tour
        close_gain = gain + (broken_cost - joined_close_cost)
        self.close_gains.append(close_gain)
        self.best_close_gain = close_gain if close_gain > self.best_close_gain else self.best_close_gain

        # get the number of backtracked neighbors at current level (defaults to 1 if no value is defined at backtracking parameter)
        curr_backtracking = 1
        if (level <= len(self.backtracking) - 1):
            curr_backtracking = self.backtracking[level]

        # yi is selected based on best neighbors of t4
        # next_y_head is the selected yi head node (at the end of the yi)
        # next_x_head is the xi+1 head node (at the end of the xi+1)
        # this is step 4(b) at LK Paper (with the lookahead refinement - 2.B)
        for (next_y_head, next_x_head), _ in self.get_best_neighbors(t4, t1)[:curr_backtracking]:

            # set yi edge
            joined_edge = Edge(t4, next_y_head)
            joined_cost = self.cost_matrix[(t4.id, next_y_head.id)]

            # compute gain for exploration (i.e, if not closing the tour)
            explore_gain = gain + (broken_cost - joined_cost)

            # disjoint criteria (xi can't be previously joined and yi can't be previously broken)
            # it is also required to check if broken edge is not repeated and if joined edge is not already in tour
            # this is step 4(c) in LK Paper
            disjoint_criteria = False
            if broken_edge not in broken_edges and broken_edge not in joined_edges:
                if joined_edge not in self.tour.edges and joined_edge not in broken_edges:
                    disjoint_criteria = True

            # gain criteria (gain must be positive)
            # this is step 4(d) in LK Paper
            gain_criteria = False
            if explore_gain > self.gain_precision:
                gain_criteria = True

            # xi+1 criteria (next x must be possible to be broken)
            # next x can be broken if it follows the disjoint criteria and is not repeated (already broken)
            # this is step 4(e) in LK Paper
            next_xi_criteria = False
            next_broken_edge = Edge(next_y_head, next_x_head)
            if (next_broken_edge not in broken_edges and next_broken_edge not in joined_edges):
                next_xi_criteria = True

            # checking the 3 required criterias (4c, 4d and 4e), as mentioned in step 4(b) in LK Paper
            if disjoint_criteria and gain_criteria and next_xi_criteria:

                # check for repeated tours (checkout refinement - 2.A)
                if (hash(tuple([node.succ.id for node in self.tour.nodes])) in self.solutions):
                    return

                # checking if closing the tour will lead to a better gain than continuing exploration
                # if close is better than explore, ends the loop
                # if explore is better than close, call the search loop with new nodes
                # this is step 4(f) at LK Paper
                if (close_gain > explore_gain and close_gain >= self.best_close_gain and close_gain > self.gain_precision and joined_close_valid):

                    # update the sets of broken and joined edges
                    broken_edges.add(broken_edge)
                    joined_edges.add(joined_close_edge)

                    # end the loop if closing is better than exploring
                    return

                else:

                    # continue the exploration, if it is better than closing the tour
                    # this is the return to step 4 with i = i + 1 at LK Paper
                    return self.lk1_feasible_search(level + 1, explore_gain, "swap_feasible", t1, t4, next_y_head, next_x_head, broken_edges, joined_edges)

    def lk1_unfeasible_search(self, gain, t1, t2, t3, t4, broken_edges, joined_edges):
        """
        This is the alternative search loop of LK Heuristic at level 1, when the swap of the nodes t1, t2, t3, t4 leads to an unfeasible tour. LK suggest that in this condition, a special search shall be made looking for nodes t5, t6 (t7 and t8, if required) in such way that the unfeasible tour turns into a feasible one. If a feasible tour is found with better gain, a call for the main loop search is made. This is the step 6(b) in LK Paper.

        :param gain: the current gain from last LK step
        :type gain: float
        :param t1: the tail node of the first broken edge in the 2-opt swap
        :type t1: Node2D
        :param t2: the head node of the first broken edge in the 2-opt swap
        :type t2: Node2D
        :param t3: the tail node of the second broken edge in the 2-opt swap
        :type t3: Node2D
        :param t4: the head node of the second broken edge in the 2-opt swap
        :type t4: Node2D
        :param broken_edges: the set of broken edges
        :type broken_edges: set
        :param joined_edges: the set of joined edges
        :type joined_edges: set
        """

        # update the sets of broken and joined edges using the swap nodes
        # adding x1 and y1
        broken_edges.add(Edge(t1, t2))
        joined_edges.add(Edge(t2, t3))

        # the broken edge (x2) that will lead to 2 separated tours
        broken_edge_1 = Edge(t3, t4)
        broken_cost_1 = self.cost_matrix[(t3.id, t4.id)]

        # execute the unfeasible swap (creating two separated tours)
        # append a dummy gain value for the unfeasible swap
        # this is the step represented at Fig. 4(a) in LK Paper
        self.tour.swap_unfeasible(t1, t2, t3, t4)
        self.close_gains.append(-1)

        # get the backtracking at second level
        curr_backtracking = 1
        if (len(self.backtracking) - 1 >= 1):
            curr_backtracking = self.backtracking[1]

        # looking for the best neighbors of t4 at second level of backtracking
        for (t5, t6), _ in self.get_best_neighbors(t4)[:curr_backtracking]:

            # set y2 edge
            joined_edge_1 = Edge(t4, t5)
            joined_cost_1 = self.cost_matrix[(t4.id, t5.id)]

            # compute gain for exploration (not closing the tour but accepting the exploring node t5)
            explore_gain = gain + (broken_cost_1 - joined_cost_1)

            # gain criteria (gain must be positive)
            # the gain criteria shall be validated even if using an unfeasible tour
            gain_criteria = False
            if explore_gain > self.gain_precision:
                gain_criteria = True

            # checking if (t5,t6) is a valid choice
            # t5 and t6 must be different from t1,t2,t3,t4 (it must lie between those nodes)
            valid_nodes = False
            if (t5 != t1 and t5 != t2 and t5 != t3 and t5 != t4):
                if (t6 != t1 and t6 != t2 and t6 != t3 and t6 != t4):
                    valid_nodes = True

            # checking for valid node selection with positive gain
            if valid_nodes and gain_criteria:

                # set x3 edge
                broken_edge_2 = Edge(t5, t6)
                broken_cost_2 = self.cost_matrix[(t5.id, t6.id)]

                # a boolean checking if t5 is between t1-t4 segment
                t5_between_t1_t4 = False

                # checking the order of the segment t1-t4
                t1_after_t4 = t4.succ == t1

                # validate if t5 is between t1-t4
                if t1_after_t4:
                    t5_between_t1_t4 = self.tour.between(t1, t5, t4)
                else:
                    t5_between_t1_t4 = self.tour.between(t4, t5, t1)

                # t5 condition of being between t1 and t4
                if (t5_between_t1_t4):

                    # checking if t5 swap is feasible
                    if (self.tour.is_swap_feasible(t1, t4, t5, t6)):

                        # get the backtracking at third level
                        curr_backtracking = 1
                        if (len(self.backtracking) - 1 >= 2):
                            curr_backtracking = self.backtracking[2]

                        # looking for (t7,t8)
                        for (t7, t8), _ in self.get_best_neighbors(t6)[:curr_backtracking]:

                            # set y3 edge
                            joined_edge_2 = Edge(t6, t7)
                            joined_cost_2 = self.cost_matrix[(t6.id, t7.id)]

                            # update the exploration gain
                            explore_gain += (broken_cost_2 - joined_cost_2)

                            # gain criteria (gain must be positive)
                            gain_criteria = False
                            if explore_gain > self.gain_precision:
                                gain_criteria = True

                            # a boolean checking if t7 is between t2-t3 segment
                            t7_between_t2_t3 = False

                            # checking the order of the segment t2-t3
                            t2_after_t3 = t3.succ == t2

                            # validate if t7 is between t2-t3
                            if t2_after_t3:
                                t7_between_t2_t3 = self.tour.between(t2, t7, t3)
                            else:
                                t7_between_t2_t3 = self.tour.between(t3, t7, t2)

                            # checking if (t7,t8) is a valid choice
                            # t7 and t8 must be different from t2,t3 (it must lie between those nodes)
                            valid_nodes = False
                            if (t7 != t2 and t7 != t3 and t8 != t2 and t8 != t3):
                                valid_nodes = True

                            # (t7,t8) will define the x4 broken edge. LK suggest to select the greater x4 value possible from t7. This is something already done by the get_best_neighbors function, since it selected the maximum delta between (t8,t7) - (t6,t7)

                            # validating gain criteria and if selected nodes are valid
                            if gain_criteria and valid_nodes and t7_between_t2_t3:

                                # set xi edge (the new edge to be broken)
                                broken_edge_3 = Edge(t7, t8)

                                # execute the swap of t5 being between t1-t4 and append a dummy value
                                # for this special swap, t1-t4 is a subtour of an unfeasible tour
                                self.tour.swap_feasible(t1, t4, t5, t6, is_subtour=True)
                                self.close_gains.append(-1)

                                # update the sets of broken and joined edges
                                broken_edges.add(broken_edge_1)
                                broken_edges.add(broken_edge_2)
                                broken_edges.add(broken_edge_3)
                                joined_edges.add(joined_edge_1)
                                joined_edges.add(joined_edge_2)

                                # continue the exploration with feasible search, applied with a special swap function
                                # this is the return to step 4 with i = i + 1 at LK Paper
                                return self.lk1_feasible_search(4, explore_gain, "swap_node_between_t2_t3", t1, t6, t7, t8, broken_edges, joined_edges)

                # if t5 is between t2 and t3, t6 will be joined to t1 to close the loop
                else:

                    # update the sets of broken and joined edges
                    broken_edges.add(broken_edge_1)
                    broken_edges.add(broken_edge_2)
                    joined_edges.add(joined_edge_1)

                    # continue the exploration, if it is better than closing the tour
                    # this is the return to step 4 with i = i + 1 at LK Paper
                    return self.lk1_feasible_search(3, explore_gain, "swap_node_between_t2_t3", t1, t4, t5, t6, broken_edges, joined_edges)

        # if no improvement was found in the unfeasible search, undo the unfeasible swap
        self.tour.restore()

    def lk1_double_bridge_search(self, max_tests=100):
        """
        Execute a random search for a double bridge move that is feasible and results in a better tour. If this double bridge move is found, the move is executed, tour edges are updated and the double bridge gain for this move is stored into class variable

        :param max_tests: the maximum number of double bridge tests to be made
        :type max_tests: int
        :return: the execution of the double bridge move
        :rtype: None
        """

        # getting the edges that can be broken (using only edges not in reduction set)
        search_edges = list(self.tour.edges.difference(self.reduction_edges))

        # the double bridge swap requires 4 edges
        if len(search_edges) >= 4:

            # loop until max number of tests are reached
            for _ in range(max_tests):

                # shuffle the edges
                random.shuffle(search_edges)

                # get the broken edges
                broken_edge_1 = search_edges[0]
                broken_edge_2 = search_edges[1]
                broken_edge_3 = search_edges[2]
                broken_edge_4 = search_edges[3]

                # try to get the double bridge swap configuration
                double_bridge_nodes = self.tour.is_swap_double_bridge(broken_edge_1.n1, broken_edge_1.n2, broken_edge_2.n1, broken_edge_2.n2, broken_edge_3.n1, broken_edge_3.n2, broken_edge_4.n1, broken_edge_4.n2)

                # if configuration is found continue the double bridge move
                if double_bridge_nodes:

                    # get all nodes of the swap operation
                    t1 = double_bridge_nodes[0]
                    t2 = double_bridge_nodes[1]
                    t3 = double_bridge_nodes[2]
                    t4 = double_bridge_nodes[3]
                    t5 = double_bridge_nodes[4]
                    t6 = double_bridge_nodes[5]
                    t7 = double_bridge_nodes[6]
                    t8 = double_bridge_nodes[7]

                    # compute the broken costs
                    broken_cost_1 = self.cost_matrix[(t1.id, t2.id)]
                    broken_cost_2 = self.cost_matrix[(t3.id, t4.id)]
                    broken_cost_3 = self.cost_matrix[(t5.id, t6.id)]
                    broken_cost_4 = self.cost_matrix[(t7.id, t8.id)]

                    # compute the joined costs
                    joined_cost_1 = self.cost_matrix[(t1.id, t4.id)]
                    joined_cost_2 = self.cost_matrix[(t2.id, t3.id)]
                    joined_cost_3 = self.cost_matrix[(t5.id, t8.id)]
                    joined_cost_4 = self.cost_matrix[(t6.id, t7.id)]

                    # compute the gain
                    gain = (broken_cost_1 + broken_cost_2 + broken_cost_3 + broken_cost_4) - (joined_cost_1 + joined_cost_2 + joined_cost_3 + joined_cost_4)

                    # checking if the gain is positive
                    if (gain > self.gain_precision):

                        # execute double bridge swap
                        self.tour.swap_double_bridge(t1, t2, t3, t4, t5, t6, t7, t8)

                        # update the double bridge gain
                        self.double_bridge_gain = gain

                        # update tour edges
                        self.tour.edges.remove(Edge(t1, t2))
                        self.tour.edges.remove(Edge(t3, t4))
                        self.tour.edges.remove(Edge(t5, t6))
                        self.tour.edges.remove(Edge(t7, t8))
                        self.tour.edges.add(Edge(t1, t4))
                        self.tour.edges.add(Edge(t2, t3))
                        self.tour.edges.add(Edge(t5, t8))
                        self.tour.edges.add(Edge(t6, t7))

                        # break the loop
                        break

    def lk1_main(self):
        """
        Execute the main loop of LK Heuristic to optimize the tour. This is a single run of LK loop, which tries to find a better tour considering current tour being analyzed.

        :return: a boolean value indicating if an improve was found
        :rtype: boolean
        """

        # loop through each node in the tour, to apply the optimization
        # LK Heuristic will loop over all tour nodes as initial nodes (first node is explored in total)
        # this is step 2 and 6(e) in LK Paper
        for t1 in self.tour.get_nodes():

            # loop through each neighboorhood node from initial node (second node is also explored as total, so at least 2-opt is always tested for all nodes)
            # this is step 2 and 6(d) in LK Paper
            for t2 in (t1.pred, t1.succ):

                # get the break edge (x1) and the cost
                # this is step 2 in LK Paper
                broken_edge = Edge(t1, t2)
                broken_cost = self.cost_matrix[(t1.id, t2.id)]

                # loop through the best possible nodes (t3,t4) from t2 instead of looping through all nodes. The number of best nodes is defined by the backtracking parameter.
                # this is step 3 and 6(c) in LK Paper (with additional lookahead refinement - 2.B in LK paper)
                # there's also step 4(a) in the get best neighbors, since the function checks if swap is valid for the nodes t1,t2,t3,t4
                for (t3, t4), _ in self.get_best_neighbors(t2)[:self.backtracking[0]]:

                    # get the joined edge (y1) and its cost
                    joined_edge = Edge(t3, t2)
                    joined_cost = self.cost_matrix[(t3.id, t2.id)]

                    # compute the gain
                    gain = broken_cost - joined_cost

                    # check if joined edge is not inside the existing tour
                    # check if gain is positive
                    # this is step 3 in LK Paper
                    if joined_edge not in self.tour.edges and gain > self.gain_precision:

                        # initialize broken and joined edges set
                        broken_edges = set()
                        joined_edges = set()

                        # execute the search loop
                        # if swap is feasible, execute the normal search
                        # if swap is unfeasible (2 separated tours), execute specific search from step 6(b) in LK Paper
                        if (self.tour.is_swap_feasible(t1, t2, t3, t4)):
                            self.lk1_feasible_search(1, gain, "swap_feasible", t1, t2, t3, t4, broken_edges, joined_edges)
                        elif (self.tour.is_swap_unfeasible(t1, t2, t3, t4)):
                            self.lk1_unfeasible_search(gain, t1, t2, t3, t4, broken_edges, joined_edges)

                        # check if a new gain was found
                        if self.close_gains:

                            # check if best gain is positive
                            # this is step 5 of LK Paper
                            if (max(self.close_gains) > 0):

                                # get the index of the best gain found
                                # LK search is done until gain is positive, but the best gain may not be the last one found
                                best_index = self.close_gains.index(max(self.close_gains))

                                # update tour edges joined and removed until best swap (which matches the index of the best gain)
                                for i in range(best_index + 1):
                                    (n1, n2, n3, n4, _) = self.tour.swap_stack[i]
                                    self.tour.edges.remove(Edge(n1, n2))
                                    self.tour.edges.remove(Edge(n3, n4))
                                    self.tour.edges.add(Edge(n2, n3))
                                    self.tour.edges.add(Edge(n4, n1))

                                # undo executed swaps until best gain index
                                self.tour.restore((len(self.close_gains) - 1) - best_index)

                                # compute the new cost for the new tour
                                self.tour.set_cost(self.cost_matrix)

                                # reset close gains
                                self.close_gains.clear()

                                # return true if improvement was found
                                return True

                            else:

                                # reset close gains
                                self.close_gains.clear()

                                # if best gain found during search is negative, restore the initial tour (since no improvement was found, but swaps were made)
                                self.tour.restore()

        # if no improvement is found, return false
        return False

    def lk1_improve(self):
        """
        The improve loop using Lin-Kernighan Heuristic. The loop calls main LK optimizer everytime a better tour is found (which will try to optimize the new best tour using the same optimizer). The loop ends when no improvement is found, such that current tour will be the local optimal.

        After a local optimal is found, the preparation for 2 refinements are done:
        *for reduction refinement, edges of new tour is intersected with last tour edges. The intersection is used for the refinement during the main improvement loop.
        *for double-bridge refinement, after a certain amount of improvement cycles are run, the non-reduction edges are used to perform a search for a double-bridge move
        """

        # number of new tours achieved
        tour_count = 1

        # boolean checking if a improved tour was found
        improved = True

        # log the starting tour cost
        self.logger.debug(f"Starting tour cost: {self.tour.cost:.3f}")

        # loop until no improvement is found at TSP tour
        # this is the step 5 in LK Paper
        while improved:

            # compute the improvement
            improved = self.lk1_main()

            # get executed swaps count
            total_swaps = len(self.tour.swap_stack)
            feasible_swaps = len([swap for swap in self.tour.swap_stack if swap[-1] == "swap_feasible"])
            unfeasible_swaps = len([swap for swap in self.tour.swap_stack if swap[-1] != "swap_feasible"])

            # log the current tour cost
            self.logger.debug(f"Current tour '{tour_count}' cost: {self.tour.cost:.3f} / gain: {self.best_close_gain:.3f} / swaps: {total_swaps} / feasible swaps: {feasible_swaps} / unfeasible swaps: {unfeasible_swaps}")

            # update tour count
            tour_count += 1

            # reset tsp parameters for next iteration
            self.tour.swap_stack.clear()
            self.best_close_gain = 0

        # update improvement cycle count
        self.cycles += 1

        # add the current solution string to the set (direct and reversed order)
        self.solutions.add(hash(tuple([node.succ.id for node in self.tour.nodes])))
        self.solutions.add(hash(tuple([node.pred.id for node in self.tour.nodes])))

        # initialize or update the reduction edges by intersecting reduction edges with new tour edges
        self.reduction_edges = set(self.tour.edges) if self.cycles == 1 else self.reduction_edges.intersection(self.tour.edges)

        # execute the double bridge refinement after certain amount of optimization cycles
        if self.cycles >= self.reduction_cycle:

            # search for a double bridge move
            self.lk1_double_bridge_search()

            # check if a good double bridge move was found
            if (self.double_bridge_gain > 0):

                # log the current tour cost
                self.logger.info(
                    f"Double bridge move found: cost: {self.tour.cost:.3f} / gain: {self.double_bridge_gain:.3f}")

                # reset the double brige gain
                self.double_bridge_gain = 0

                # reset the shuffle value (for next iteration the new double bridge tour will be used)
                self.shuffle = False

            else:

                # if double bridge was not found, next improvement loop will shuffle the tour
                self.shuffle = True

    def lk2_select_broken_edge(self, gain, t1, t2, t3, t4, broken_edges, joined_edges):
        """
        This function is used to select the next edge to be broken at current configuration of the LK optimization procedure. After that, the function will swap the tour nodes (t1,t2,t3,t4) and return a boolean value indicating if the swap resulted in a valid better tour.

        If the swap results in a valid worse tour, the call to select a new joined edge will be made to continue the exploration of new edges. This new exploration will try to swap the closed edge (t4,t1) with a new potencial better edge.

        The nodes t1, t2, t3 and t4 represents the swap in current configuration. When having sequence of swaps in the same tour, LK paper will mention those nodes based on the swap number, like (t2i, t2i+1, t2i+2...), where "i" is the swap number.

        :param gain: the current gain from last LK step
        :type gain: float
        :param t1: the tail node of the first broken edge in the 2-opt swap
        :type t1: Node2D
        :param t2: the head node of the first broken edge in the 2-opt swap
        :type t2: Node2D
        :param t3: the tail node of the second broken edge in the 2-opt swap
        :type t3: Node2D
        :param t4: the head node of the second broken edge in the 2-opt swap
        :type t4: Node2D
        :param broken_edges: the set of broken edges
        :type broken_edges: set
        :param joined_edges: the set of joined edges
        :type joined_edges: set
        :return: a boolean indicating if improvement was found
        :rtype: boolean
        """

        # get the edge to be broken and its cost
        broken_edge = Edge(t3, t4)
        broken_cost = self.cost_matrix[(t3.id, t4.id)]

        # the neighboor can't be t1 (this results in an invalid tour)
        # disjoint criteria (broken edge can't be previouslly at joined edges)
        # check if broken_edge is not already inside broken_edges set, because an edge can't be broken two times, this would lead to incorrect calculation
        if t1 != t4 and broken_edge not in joined_edges and broken_edge not in broken_edges:

            # check if swap is valid
            if (self.tour.is_swap_feasible(t1, t2, t3, t4)):

                    # execute the swap
                self.tour.swap_feasible(t1, t2, t3, t4)

                # add the broken edge to the set of broken edges
                broken_edges.add(broken_edge)

                # build the join edge from neighboor to last node (closing the tour)
                joined_edge = Edge(t4, t1)
                joined_cost = self.cost_matrix[(t4.id, t1.id)]

                # compute the current gain value
                curr_gain = gain + (broken_cost - joined_cost)

                # if gain is positive, restart the search with new tour
                # this is a simplification of LK from Helsgaun paper. LK suggest to continue the search, while Helsgaun suggest to simplify the algorithm
                if curr_gain > self.gain_precision:

                    # add joined edge
                    joined_edges.add(joined_edge)

                    # a sanity check here
                    old_tour_cost = self.tour.cost
                    self.tour.set_cost(self.cost_matrix)  # this shall not be erased in the future!
                    delta_tour_cost = abs(old_tour_cost - self.tour.cost)
                    delta_gain = abs(delta_tour_cost - curr_gain)

                    if (delta_gain > self.gain_precision):
                        self.logger.debug("delta gain error")

                    if (old_tour_cost < self.tour.cost):
                        self.logger.debug("cost error")

                    # return true value meaning an improvement was found
                    return True

                # if gain is negative, try to find a new best edge to add to form a new best tour
                else:

                    # try to find new joined edge
                    return self.lk2_select_joined_edge(curr_gain, t1, t4, broken_edges, joined_edges)

        # boolean value indicating no improvement was found
        return False

    def lk2_select_joined_edge(self, gain, t1, t4, broken_edges, joined_edges):
        """
        This function is used to select the next edge to be joined at current configuration of the LK optimization procedure. A false boolean is returned if a new joined edge was not found. If a replacement is found, the call for the selection of a new broken edge is made. 

        The selection function will "undo" the last joined edge (t4,t1) from previous "select_broken_edge" procedure, which closed the tour, by selecting a new potencial "node" so that (t1,t4) is broken and a new joined edge (t4,node) will be tested.

        :param gain: the current gain from last LK step
        :type gain: float
        :param t1: the head node of the last joined edge
        :type t1: Node2D
        :param t4: the tail node of the last joined edge
        :type t4: Node2D
        :param broken_edges: the set of broken edges
        :type broken_edges: set
        :param joined_edges: the set of joined edges
        :type joined_edges: set
        :return: a boolean indicating if improvement was found
        :rtype: boolean
        """

        # get the cost of the last joined edge (t4,t1)
        broken_cost = self.cost_matrix[(t4.id, t1.id)]

        # loop through the closest possible node from t4 instead of looping through all nodes
        for (node, neighbor_node), _ in self.get_best_neighbors(t4, t1):

            # create the edge and get the edge cost
            joined_edge = Edge(t4, node)
            joined_cost = self.cost_matrix[(t4.id, node.id)]

            # compute the new gain value
            curr_gain = gain + (broken_cost - joined_cost)

            # check if edge to be added is not in broken edges (disjoint criteria)
            # check if edge to be added is not already inside the tour
            # check if gain is positive
            if joined_edge not in broken_edges and joined_edge not in self.tour.edges and curr_gain > self.gain_precision:

                # add the edge to joined edges
                joined_edges.add(joined_edge)

                # try to select an edge to be broken
                return self.lk2_select_broken_edge(curr_gain, t1, t4, node, neighbor_node, broken_edges, joined_edges)

        # boolean value indicating no improvement was found
        return False

    def lk2_main(self):
        """
        Execute the main loop of LK Heuristic to optimize the tour. This is a single run of LK loop, which tries to find a better tour considering current tour being analyzed.

        :return: a boolean value indicating if an improve was found
        :rtype: boolean
        """

        # loop through each node in the tour, to apply the optimization
        # LK Heuristic will loop over all tour nodes as initial nodes (first node is explored in total)
        for t1 in self.tour.get_nodes():

            # loop through each neighboorhood node from initial node (second node is also explored as total, so at least 2-opt is always tested for all nodes)
            for t2 in (t1.pred, t1.succ):

                # get the break edge and the cost
                broken_edge = Edge(t1, t2)
                broken_cost = self.cost_matrix[(t1.id, t2.id)]

                # loop through the best possible nodes (t3,t4) from t2 instead of looping through all nodes
                for (t3, t4), _ in self.get_best_neighbors(t2, t1):

                    # get the joined edge and its cost
                    joined_edge = Edge(t3, t2)
                    joined_cost = self.cost_matrix[(t3.id, t2.id)]

                    # compute the gain
                    gain = broken_cost - joined_cost

                    # check if joined edge is not inside the existing tour
                    # check if gain is positive
                    if joined_edge not in self.tour.edges and gain > self.gain_precision:

                        # append the edges to each edges set
                        broken_edges = set([broken_edge])
                        joined_edges = set([joined_edge])

                        # try to select the next break edge and get a possibly new tour
                        if self.lk2_select_broken_edge(gain, t1, t2, t3, t4, broken_edges, joined_edges):

                            # reset the swap stack
                            self.tour.swap_stack.clear()

                            # return true value
                            return True
                        else:
                            # if new best tour was not found, restore the initial tour to continue the loop (if tour was not swapped, it will remain the same)
                            self.tour.restore()

        # if no improvement is found, return false
        return False

    def lk2_improve(self):
        """
        The improve loop using Lin-Kernighan Heuristic. The loop calls main LK optimizer everytime a better tour is found (which will try to optimize the new best tour using the same optimizer). The loop ends when no improvement is found, such that current tour will be the local optimal
        """

        # number of tours tested
        tour_count = 1

        # boolean checking if a improved tour was found
        improved = True

        # log the starting tour cost
        self.logger.debug(f"Starting tour cost: {self.tour.cost:.3f}")

        # loop until no improvement is found at TSP tour
        while improved:

            # compute the improvement
            improved = self.lk2_main()

            # log the current tour cost
            self.logger.debug(f"Current tour '{tour_count}' cost: {self.tour.cost:.3f}")

            # update tour count
            tour_count += 1

    def bf_improve(self):
        """
        The improve loop using Brute-Force computation, which converges for the global optimal tour. All permutations for nodes will be tested. It is recommended only for very small tours (< 10 nodes).
        """

        # assign values for the minimum cost and tour found
        min_cost = math.inf
        min_tour = None

        # create all possible permutations for the tour
        perms = permutations(self.nodes)

        # get the start node (to avoid repeated tours)
        start_node = self.nodes[0]

        # count number of tested tours
        tour_count = 0

        # loop through each permutation
        for perm in perms:

            # only permutation starting with the starting node is considered (so that repeated tours starting at a different node are not considered)
            if perm[0] == start_node:

                # loop through each node (starting from the last one)
                for i in range(-1, len(perm) - 1):

                    # get current and next nodes
                    curr_node = perm[i]
                    next_node = perm[i + 1]

                    # assign predecessor and successor nodes
                    curr_node.succ = next_node
                    next_node.pred = curr_node

                # update the cost value
                self.tour.set_cost(self.cost_matrix)

                # check if a best tour was found
                if self.tour.cost < min_cost:

                    # update best values
                    min_cost = self.tour.cost
                    min_tour = perm

                    # log the current  tour cost
                    self.logger.info(f"Current tour '{tour_count}'cost: {self.tour.cost:.3f}")

                # update and log count
                tour_count += 1
                if tour_count % 1000 == 0:
                    self.logger.info(f"Current tour '{tour_count}' cost: {min_cost:.3f}")

        # loop through each node of the best tour
        for i in range(-1, len(min_tour) - 1):

            # get current and next nodes
            curr_node = min_tour[i]
            next_node = min_tour[i + 1]

            # assign predecessor and successor nodes
            curr_node.succ = next_node
            next_node.pred = curr_node

        # update the cost value
        self.tour.set_cost(self.cost_matrix)

    def nn_improve(self):
        """
        The improve loop using Nearest-Neighbor computation.
        """

        # create a set of all tsp nodes
        nodes = set(self.nodes)

        # get the starting node
        curr_node = self.nodes[0]

        # start the set of visited nodes
        visited_nodes = set()
        visited_nodes.add(curr_node)

        # loop until all nodes are visited
        while (len(visited_nodes) < len(self.nodes)):

           # loop to find the nearest neighbor of current node
            cost = math.inf
            # loop through each node that were not visited yet
            for node in nodes - visited_nodes:
                if self.cost_matrix[(node.id, curr_node.id)] < cost:
                    cost = self.cost_matrix[(node.id, curr_node.id)]
                    next_node = node

            # update succ and pred for current and next nodes
            next_node.pred = curr_node
            curr_node.succ = next_node

            # add new node to visited nodes set
            visited_nodes.add(next_node)

            # update current node for next iteration
            curr_node = next_node

        # update the starting and ending nodes
        self.nodes[0].pred = curr_node
        curr_node.succ = self.nodes[0]

        # update the cost value
        self.tour.set_cost(self.cost_matrix)

    @classmethod
    def get_solution_methods(cls):
        """
        Get the list of solution methods for the tsp

        :return: list of solution method
        :rtype: list
        """
        return [cls.bf_improve, cls.nn_improve, cls.lk1_improve, cls.lk2_improve]
