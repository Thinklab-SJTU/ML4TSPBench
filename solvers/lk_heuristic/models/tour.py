from random import shuffle
from solvers.lk_heuristic.models.node import NodePivot
from solvers.lk_heuristic.models.edge import Edge


class Tour:
    """
    The tour class represents a sequence of edges that starts at one node, visits all tour nodes and ends at same starting node (for 'cycle' tours) or not (for 'path' tours). The nodes will be a list of nodes in the ordering of visit, while edges is a set of edges (so ordering is not considered).
    """

    def __init__(self, nodes, t="cycle"):
        """
        A tour is made by a sequence of edges. Since edges are defined by sequence of nodes, the node sequence is used as input. The tour type ('t') can be either "cycle" (for the classic hamiltonian cycle tsp) or "path" (for the hamiltonian path tsp).

        :param nodes: the list of Nodes that makes the tour
        :type nodes: list
        :param t: the type of the tour
        :type t: str
        """

        # initialize the tour type
        self.t = t

        # set the tour nodes and initialize tour node parameters
        self.nodes = nodes
        self.set_nodes()

        # set tour edges using defined tour nodes
        self.edges = set()
        self.set_edges()

        # Initialize the cost value, to be update when calling the function set_cost using a distance matrix.
        self.cost = 0

        # set the size of the tour
        self.size = len(nodes)

        # init the swap stack
        # the swap stack is the "memory" of swap functions executed in a specific tour (relevant when required to undo the swaps). it is a tuple on the form (n1, n2, n3, n4, swap_operation)
        self.swap_stack = []

    def set_nodes(self):
        """
        Update tour nodes with specific related tour properties
        """

        # for hamiltonian path tsp, create the pivot node and append it to nodes list
        if self.t == "path":
            self.nodes.append(NodePivot())

        # loop through each node and set the tour properties
        for i in range(len(self.nodes)):

            # set node's successor node
            if (i == len(self.nodes) - 1):
                self.nodes[i].succ = self.nodes[0]  # the successor of the last node is the first node
            else:
                self.nodes[i].succ = self.nodes[i + 1]

            # set node's predecessor node
            self.nodes[i].pred = self.nodes[i - 1]

            # set node position
            self.nodes[i].pos = i

            # set node id
            self.nodes[i].id = i

    def set_edges(self):
        """
        Generate the tour edges using current tour nodes
        """

        # initialize the set of edges
        tour_edges = set()

        # initialize the first node
        curr_node = self.nodes[0]

        # loop through each node and add the edge of current node and its neighbor
        while(curr_node.succ != self.nodes[0]):
            tour_edges.add(Edge(curr_node, curr_node.succ))
            curr_node = curr_node.succ

        # add the closing edge
        tour_edges.add(Edge(curr_node, curr_node.succ))

        # populate the edges
        self.edges = tour_edges

    def set_cost(self, cost_matrix):
        """
        Set the tour cost by summing all tour edge costs from a cost matrix

        :param cost_matrix: the matrix with cost(i,j) between node id "i" and node id "j"
        :type node: list

        """
        # initialize the cost
        tour_cost = 0

        # initialize the first node
        curr_node = self.nodes[0]

        # loop through each node and add the edge of current node and its neighbor
        while(curr_node.succ != self.nodes[0]):
            tour_cost += cost_matrix[(curr_node.id, curr_node.succ.id)]
            curr_node = curr_node.succ

        # add the cost of closing the loop (last node -> first node)
        tour_cost += cost_matrix[(curr_node.id, curr_node.succ.id)]

        # assign the cost value
        self.cost = tour_cost

    def set_pos(self):
        """
        Set the 'pos' attribute of the nodes from a feasible tour. This function is relevant after performing unfeasible swaps that only reorder nodes pred/succ attribute but leaves the pos attribute incorrect. After converging to a feasible tour, call this function to update the pos attribute.
        """

        # initialization with first node
        curr_node = self.nodes[0]
        curr_node.pos = 0

        # loop until all nodes have been seen
        while curr_node.succ != self.nodes[0]:

            # update current node
            curr_node = curr_node.succ
            curr_node.pos = curr_node.pred.pos + 1

    def get_nodes(self):
        """
        Get the nodes inside the tour sorted by their sequence and return the list of nodes

        :return: the list of node indexes
        :rtype: list
        """

        # a set of nodes to check if all nodes were analyzed
        visited_nodes = set(self.nodes)

        # start a list of sorted nodes
        tour_nodes = []

        # get the start index for first node
        # when tour is hamiltonian path, the first node will be the pivot node (so when exporting the tsp file, this node is removed)
        start_idx = 0
        if self.t == "path":
            for i, node in enumerate(self.nodes):
                if type(node) == NodePivot:
                    start_idx = i
                    break

        # initialization with first node
        curr_node = self.nodes[start_idx]
        visited_nodes.remove(curr_node)

        # loop until all nodes have been seen (this is necessary for unfeasible tours, like two separated subtours)
        while len(visited_nodes) > 0:

            # loop through each tour node checking if it was not visited yet
            while curr_node.succ in visited_nodes:

                # append the node index to list of indexes
                tour_nodes.append(curr_node)

                # update current node
                curr_node = curr_node.succ

                # remove current node from visited
                visited_nodes.remove(curr_node)

            # append last index to close the tour
            tour_nodes.append(curr_node)

            # check if there are still nodes in the visited nodes set
            # this will happen at unfeasible tours (like two subtours, instead of only one single tour)
            if len(visited_nodes) > 0:

                # get any unvisited node to continue the search
                curr_node = visited_nodes.pop()

        #  return the list of indexes
        return tour_nodes

    def shuffle(self):
        """
        Shuffle the tour nodes creating a random tour and re-initializing the tour edges 
        """

        # create random indexes between 1 and size-1
        indexes = [i for i in range(1, self.size)]
        shuffle(indexes)

        # add the 0 (first node) at the end of the indexes list
        indexes.append(0)

        # initilize the current node
        curr_node = self.nodes[0]

        # loop to change the predecessor, successor and position of each node
        for i in range(-1, self.size - 1):
            curr_node.succ = self.nodes[indexes[i + 1]]
            curr_node.pred = self.nodes[indexes[i - 1]]
            curr_node.pos = i + 1
            curr_node = curr_node.succ

        # rebuild the edges after shuffling the nodes
        self.set_edges()

    def restore(self, swaps=None):
        """
        Restore an initial tour that passed through some 2-opt swaps by doing a certain amount of reversed swaps using the swap_stack. If number of swaps is not defined, the entire swap stack will be undone

        :param swaps: the number of swaps to be undone
        :type swaps: int
        """

        # if swaps are not defined, the entire swap stack will be undone
        if swaps == None:
            swaps = len(self.swap_stack)

        # looping until number of swaps are reached
        for _ in range(swaps):

            # get last stack
            curr_stack = self.swap_stack[-1]

            # collect node values
            t1 = curr_stack[0]
            t2 = curr_stack[1]
            t3 = curr_stack[2]
            t4 = curr_stack[3]

            # collect the 2-Opt swap type made
            swap_type = curr_stack[-1]

            # execute the reversed swap based on the swap operation
            # swap is not recorded to the stack, since it is being undone
            if (swap_type == "swap_feasible"):
                self.swap_feasible(t4, t1, t2, t3, False, False)
            elif (swap_type == "swap_unfeasible"):
                self.swap_unfeasible(t4, t1, t2, t3, False, False)
            elif (swap_type == "swap_node_between_t2_t3"):
                self.swap_unfeasible(t4, t1, t2, t3, False, False)
            elif (swap_type == "swap_node_between_t2_t3_reversed"):
                self.swap_unfeasible(t4, t1, t2, t3, True, False)
            elif (swap_type == "swap_feasible_reversed"):
                self.swap_feasible(t4, t1, t2, t3, True, False)

            # remove last swap
            self.swap_stack.pop()

        # loop through each applied swap and check if there's any swap that don't recompute the pos attribute. If so, set the pos attribute
        for swap in self.swap_stack:
            if swap[-1] != "swap_feasible":
                self.set_pos()
                break

    def between(self, from_node, between_node, to_node, use_pos_attr=False):
        """
        Validate if a specific node is between two other nodes. There are two methods of search:

        1 - Using pos attribute: this is a fast search, but pos attribute must be correctly defined for the node segment being analyzed 
        2 - Using pred/succ attribute: a slow search (requires to traverse through the nodes), but doesn't require nodes to have pos attribute defined (this is relevant at unfeasible swaps)

        :param from_node: the starting node
        :type from_node: Node
        :param between_node: the node checked if between the other two nodes
        :type between_node: Node
        :param to_node: the ending node
        :type to_node: Node
        """

        # using pos attribute
        if use_pos_attr:

            # this is the check where the segment from-to does not contain the breaking pos value
            if from_node.pos <= to_node.pos:
                # segment does not pass through the breaking pos value
                return from_node.pos < between_node.pos and between_node.pos < to_node.pos
            else:
                # segment passes through the breaking pos value
                return from_node.pos < between_node.pos or between_node.pos < to_node.pos

        # using pred/succ attribute
        else:

            # get starting node
            node = from_node.succ

            # looping until to node is reached
            while node != to_node:
                # if between node is reached, return true
                if node == between_node:
                    return True
                else:
                    node = node.succ

            # if between node was not found, returns false
            return False

    def is_swap_feasible(self, t1, t2, t3, t4):
        """
        Validate if a 2Opt-Swap operation performed into a feasible tour/subtour results in another feasible tour/subtour using reference nodes. Edge (t1,t2) and (t3,t4) are broken so that a relink of (t2,t3) and (t1,t4) is made, as shown below:

          t4  t3           t4   t3
          ()--()           ()   ()
         /      \         /  \ /  \
        ()      ()  -->  ()   X   ()
         \      /         \  / \  /
          ()--()           ()   ()
          t2  t1           t2   t1

        For a feasible swap:
        *1: all nodes must be different from each other;
        *2: the order of the nodes in the tour must be (t1,t2,t4,t3) in any direction (clockwise or counter clockwise). Notice that in the scheme above, if we had (t1,t2,t3,t4), the relink would result in 2 closed loops, which are unfeasible. Notice that if we switch t1 with t2 and t3 with t4, it still feasible, the change was only made in the direction.

        * This function can be used into a subtour: when an unfeasible swap results in an unfeasible tour, there're two subtours that, isolated from each other, are valid subtours. This is relevant at the unfeasible search function (when node is between t1 and t4).

        :param t1: the tail node of the first broken edge
        :type t1: Node2D
        :param t2: the head node of the first broken edge
        :type t2: Node2D
        :param t3: the tail node of the second broken edge
        :type t3: Node2D
        :param t4: the head node of the second broken edge
        :type t4: Node2D
        :return: a boolean indicating if swap is valid and feasible
        :rtype: boolean
        """

        # for a feasible swap, all nodes must be different from each other (rule 1)
        if not (t1 != t2 and t1 != t3 and t1 != t4 and t2 != t3 and t2 != t4 and t3 != t4):
            return False

        # check t4 based on the orientation of the edge (t1,t2)
        # for a feasible swap, t4 can be validated using t1 and t2 orientation (rule 2)
        if t1.succ == t2:
            if t4 != t3.pred:
                return False
        elif t1.pred == t2:
            if t4 != t3.succ:
                return False

        # return true if swap is feasible
        return True

    def is_swap_unfeasible(self, t1, t2, t3, t4):
        """
        Validate if a 2Opt-Swap operation performed into a feasible tour results in an unfeasible tour using reference nodes. Edge (t1,t2) and (t3,t4) are broken so that a relink of (t2,t3) and (t1,t4) is made, as shown below:

          t3  t4           t3   t4
          ()--()           ()   ()
         /      \         / |   | \
        ()      ()  -->  () |   | ()
         \      /         \ |   | /
          ()--()           ()   ()
          t2  t1           t2   t1

        For an unfeasible swap:
        *1: all nodes must be different from each other;
        *2: the order of the nodes in the tour must be (t1,t2,t3,t4) in any direction (clockwise or counter clockwise), resulting in two separated subtours when relinking the edges;
        *3: each subtour must contain at least 3 nodes

        :param t1: the tail node of the first broken edge
        :type t1: Node2D
        :param t2: the head node of the first broken edge
        :type t2: Node2D
        :param t3: the tail node of the second broken edge
        :type t3: Node2D
        :param t4: the head node of the second broken edge
        :type t4: Node2D
        :return: a boolean indicating if swap is valid and unfeasible
        :rtype: boolean
        """

        # for an unfeasilbe swap, all nodes must be different from each other (rule 1)
        if not (t1 != t2 and t1 != t3 and t1 != t4 and t2 != t3 and t2 != t4 and t3 != t4):
            return False

        # check t4 based on the orientation of the edge (t1,t2)
        # for an unfeasible swap, t4 can be validated using t1 and t2 orientation (rule 2)
        if t1.succ == t2:
            if t4 == t3.pred:
                return False
        elif t1.pred == t2:
            if t4 == t3.succ:
                return False

        # t3 can't be a neighbor of t2 or t4 be a neighbor of t1: this result in a subtour segment with only 2 nodes, which is invalid (rule 3)
        if (t2.pred == t3 or t2.succ == t3 or t1.pred == t4 or t1.succ == t4):
            return False

        # return true if an unfeasible swap was found
        return True

    def is_swap_double_bridge(self, t1, t2, t3, t4, t5, t6, t7, t8):
        """
        Validate if a 4Opt-Swap double bridge operation performed into a feasible tour results in another feasible tour using reference nodes and, if a double bridge is found, returns the tuple of nodes in the sequence that the double bridge node shall be executed. Edges (t1,t2), (t3,t4), (t5,t6), (t7,t8) are broken so that a relink of nodes is made, as shown below:

            t3  t4             t3   t4
            ()--()             ()   ()
           /      \           / |   | \
        t8()      ()t5     t8()-------()t5 
          |        |   -->      |   |  
        t7()      ()t6     t7()-------()t6 
           \      /           \ |   | /
            ()--()             ()   ()
            t2  t1             t2   t1

        *It is possible to notice the double bridge move is a combination of two unfeasible swaps for segments t1-t4 and t5-t8. Both of this moves isolated results in two separated segments, while combined together results in a feasible tour.

        For an double bridge swap:
        *1: all nodes must be different from each other;
        *2: segment of 4 nodes t1-t4 and t5-t8 must result in an unfeasible swap 

        :param t1: the tail node of the first broken edge
        :type t1: Node2D
        :param t2: the head node of the first broken edge
        :type t2: Node2D
        :param t3: the tail node of the second broken edge
        :type t3: Node2D
        :param t4: the head node of the second broken edge
        :type t4: Node2D
        :param t5: the tail node of the third broken edge
        :type t5: Node2D
        :param t6: the head node of the third broken edge
        :type t6: Node2D
        :param t7: the tail node of the fourth broken edge
        :type t7: Node2D
        :param t8: the head node of the fourth broken edge
        :type t8: Node2D
        :return: a tuple of nodes for the double bridge swap (or empty if double bridge swap was not found)
        :rtype: tuple
        """

        # for a double bridge swap, all nodes must be different from each other (rule 1)
        if not (t1 != t3 and t1 != t4 and t1 != t5 and t1 != t6 and t1 != t7 and t1 != t8 and t2 != t3 and t2 != t4 and t2 != t5 and t2 != t6 and t2 != t7 and t2 != t8 and t3 != t5 and t3 != t6 and t3 != t7 and t3 != t8 and t4 != t5 and t4 != t6 and t4 != t7 and t4 != t8 and t5 != t7 and t5 != t8 and t6 != t7 and t6 != t8):
            return None

        # getting the nodes in succ sequence
        if t1.pred == t2:
            temp = t2
            t2 = t1
            t1 = temp
        if t3.pred == t4:
            temp = t4
            t4 = t3
            t3 = temp
        if t5.pred == t6:
            temp = t6
            t6 = t5
            t5 = temp
        if t7.pred == t8:
            temp = t8
            t8 = t7
            t7 = temp

        # get nodes in a sequence based on their position
        nodes = sorted((t1, t3, t5, t7), key=lambda el: el.pos)

        # return the nodes in the sequence for the double bridge swap
        return (nodes[0], nodes[0].succ, nodes[2], nodes[2].succ, nodes[1], nodes[1].succ, nodes[3], nodes[3].succ)

    def swap_feasible(self, t1, t2, t3, t4, is_subtour=False, record=True):
        """
        Execute a 2Opt-Swap operation in a feasible tour/subtour resulting in another feasible tour/subtour. Edge (t1,t2) and (t3,t4) are broken so that a relink of (t2,t3) and (t1,t4) is made, as shown below:

          t4  t3           t4   t3
          ()--()           ()   ()
         /      \         /  \ /  \
        ()      ()  -->  ()   X   ()
         \      /         \  / \  /
          ()--()           ()   ()
          t2  t1           t2   t1

        * This function can be used into a subtour: when an unfeasible swap results in an unfeasible tour, there're two subtours that, isolated from each other, are valid subtours. This is relevant at the unfeasible search function (when node is between t1 and t4). When this function is applied to subtours, the pos attribute must not be recomputed.

        :param t1: the tail node of the first broken edge
        :type t1: Node2D
        :param t2: the head node of the first broken edge
        :type t2: Node2D
        :param t3: the tail node of the second broken edge
        :type t3: Node2D
        :param t4: the head node of the second broken edge
        :type t4: Node2D
        :param is_subtour: a boolean indicating if swap is done into a subtour (from an unfeasible tour)
        :type is_subtour: boolean
        :param record: a boolean indicating if swap must be record into the swap stack
        :type record: boolean
        """

        # if t2 is not the successor of t1, invert t1 with t2 and t3 with t4
        # since the node reordering is always applied to t3 -> t1 segment, this is done so that the reordering loop will be done correctly.
        if (t1.succ != t2):
            temp = t1
            t1 = t2
            t2 = temp
            temp = t3
            t3 = t4
            t4 = temp

        # compute the segment size between t2 and t3 and compare it with the problem size
        # this validates if the segment between t2-t3 has more/less nodes than the segment t3-t1
        # if t2-t3 is shorter, we can switch t3 with t2 and t4 with t1 so that the swap result will be the same, but the amount of nodes to be reordered is smaller
        seg_size = t2.pos - t3.pos
        if seg_size < 0:
            seg_size += self.size  # the amount of nodes in t3-t1
        if (2 * seg_size > self.size):  # checking if there are more nodes in t3-t1 than t2-t3
            temp = t3
            t3 = t2
            t2 = temp
            temp = t4
            t4 = t1
            t1 = temp

        # initialize the position value from t1 (the value of t1 pos will be assigned to t3, since after the swap t3 is the predecessor of t2)
        pos = t1.pos

        # the starting node for the reversed segment is t3 (until t1)
        node = t3

        # the loop is broken at t1 successor, (since t1 is the last node)
        end_node = t1.succ

        # loop to reorder the nodes between t3-t1 segment (including t3 and t1)
        while (node != end_node):

            # invert the node ordering
            temp = node.succ
            node.succ = node.pred
            node.pred = temp

            # update node position only if is not a subtour
            if not is_subtour:
                node.pos = pos
                pos -= 1

            # update node for next loop with the last successor node
            node = temp

        # reassign the successor/predecessor values at each of the 4 reconnected nodes
        t3.succ = t2
        t2.pred = t3
        t1.pred = t4
        t4.succ = t1

        # update the swap stack
        if record:
            # the name is defined based on subtour parameter
            if not is_subtour:
                self.swap_stack.append((t1, t2, t3, t4, "swap_feasible"))
            else:
                self.swap_stack.append((t1, t2, t3, t4, "swap_feasible_reversed"))

    def swap_unfeasible(self, t1, t2, t3, t4, reverse_subtour=False, record=True):
        """
        Execute a 2Opt-Swap operation in a feasible tour resulting in an unfeasible tour. Edge (t1,t2) and (t3,t4) are broken so that a relink of (t2,t3) and (t1,t4) is made, as shown below:

          t3  t4           t3   t4
          ()--()           ()   ()
         /      \         / |   | \
        ()      ()  -->  () |   | ()
         \      /         \ |   | /
          ()--()           ()   ()
          t2  t1           t2   t1


        * Because some unfeasible swaps may reverse the subtour direction, the reverse_subtour parameter is used to reverse back the segment, useful when undoing those swaps.

        :param t1: the tail node of the first broken edge
        :type t1: Node2D
        :param t2: the head node of the first broken edge
        :type t2: Node2D
        :param t3: the tail node of the second broken edge
        :type t3: Node2D
        :param t4: the head node of the second broken edge
        :type t4: Node2D
        :param reverse_subtour: a boolean indicating if one sub-tour segment shall be reversed
        :type reverse_subtour: boolean
        :param record: a boolean indicating if swap must be record into the swap stack
        :type record: boolean
        """

        # reassign is done based on the direction of t1-t2 nodes
        if (t1.succ == t2):
            temp = t3
            t3 = t2
            t2 = temp
            temp = t4
            t4 = t1
            t1 = temp

        # reassign the successor/predecessor values at each of the 4 reconnected nodes
        t3.pred = t2
        t2.succ = t3
        t1.pred = t4
        t4.succ = t1

        # reverse the t1-t4 segment
        # relevant when undoing unfeasible swap of a node between t2 and t3, when the swap reverses one of the segments
        if reverse_subtour:

            # start from node t4
            node = t4

            # loop until t1
            while node.pred != t4:

                # reverse the node pred/succ attribute
                temp = node.pred
                node.pred = node.succ
                node.succ = temp

                # update for next node
                node = temp

            # reverse t1
            t1.pred = t1.succ
            t1.succ = t4

        # update the swap stack
        if record:
            self.swap_stack.append((t1, t2, t3, t4, "swap_unfeasible"))

    def swap_node_between_t2_t3(self, t1, t4, t5, t6, record=True):
        """
        Execute a 2Opt-Swap operation from an unfeasible tour with node t5 (and thus also t6) located between nodes t2 and t3. The break of edge (t1,t4) opens one of the closed tour and  connecting t1-t6 and t4-t5 results into a new valid tour.

        This function is written with reference nodes t5 and t6, but it works at any swap level if starting from an unfeasible tour. As an example, this function is used in LK algorithm to perform the swap of t7 (and thus also t8) between t2-t3.

        :param t1: the tail node of the first broken edge
        :type t1: Node2D
        :param t4: the head node of the first broken edge
        :type t4: Node2D
        :param t5: the tail node of the second broken edge
        :type t5: Node2D
        :param t6: the head node of the second broken edge
        :type t6: Node2D
        :param record: a boolean indicating if swap must be record into the swap stack
        :type record: boolean
        """

        # checking if t1-t4 is reversed (t4 = t1.pred) or not (t4 = t1.succ)
        # this is required to check succ/pred sequences
        t4_after_t1 = t1.succ == t4

        # checking if t5-t6 is reversed (t6 = t5.pred) or not (t6 = t5.succ)
        # this is required to check succ/pred sequences
        t6_after_t5 = t5.succ == t6

        # a boolean checking if segment must be reversed
        # requires to reverse t5-t6 segment when (t1-t4-t5-t6) or (t4-t1-t6-t5)
        # otherwise the tour (luckly) is in the correct sequence
        reverse_subtour = t4_after_t1 != t6_after_t5

        # apply the reverse loop
        if (reverse_subtour):

            # arranging the nodes for the reorder loop (t5-t6 segment)
            # reorder is always going backwards (using pred attribute)
            from_node = t6
            to_node = t5
            if t6_after_t5:
                from_node = t5
                to_node = t6

            # loop reversing the segment
            while (from_node != to_node):

                # reverse the node pred/succ attribute and update for next node
                temp = from_node.pred
                from_node.pred = from_node.succ
                from_node.succ = temp
                from_node = temp

            # reverse the last node
            temp = to_node.pred
            to_node.pred = to_node.succ
            to_node.succ = temp

        # reassign the swap nodes based on tour direction
        if (t4_after_t1):
            t1.succ = t6
            t6.pred = t1
            t5.succ = t4
            t4.pred = t5
        else:
            t1.pred = t6
            t6.succ = t1
            t5.pred = t4
            t4.succ = t5

        # update the swap stack
        if record:
            # record also in the swap name if the reversed loop was applied or not
            # this is relevant when undoing the swap
            if reverse_subtour:
                self.swap_stack.append((t1, t4, t5, t6, "swap_node_between_t2_t3_reversed"))
            else:
                self.swap_stack.append((t1, t4, t5, t6, "swap_node_between_t2_t3"))

    def swap_double_bridge(self, t1, t2, t3, t4, t5, t6, t7, t8, record=True):
        """
        Execute a 4Opt-Swap double bridge operation performed into a feasible tour resulting in another feasible tour using reference nodes. Edges (t1,t2), (t3,t4), (t5,t6), (t7,t8) are broken so that a relink of nodes is made, as shown below:

            t3  t4             t3   t4
            ()--()             ()   ()
           /      \           / |   | \
        t8()      ()t5     t8()-------()t5 
          |        |   -->      |   |  
        t7()      ()t6     t7()-------()t6 
           \      /           \ |   | /
            ()--()             ()   ()
            t2  t1             t2   t1

        :param t1: the tail node of the first broken edge
        :type t1: Node2D
        :param t2: the head node of the first broken edge
        :type t2: Node2D
        :param t3: the tail node of the second broken edge
        :type t3: Node2D
        :param t4: the head node of the second broken edge
        :type t4: Node2D
        :param t5: the tail node of the third broken edge
        :type t5: Node2D
        :param t6: the head node of the third broken edge
        :type t6: Node2D
        :param t7: the tail node of the fourth broken edge
        :type t7: Node2D
        :param t8: the head node of the fourth broken edge
        :type t8: Node2D
        :param record: a boolean indicating if swap must be record into the swap stack
        :type record: boolean
        """

        # execute the first unfeasible swap
        self.swap_unfeasible(t1, t2, t3, t4, False, False)

        # the second unfeasible swap is done in such way that new tour does not require segment reordering
        # checking direction of initial tour segment t1-t4
        from_node = t4
        to_node = t1
        if t1.pred == t2:
            from_node = t1
            to_node = t4

        # check if t5 is between t4 and t1 segment
        # if not, reorder the nodes so that t5 is between t1-t4
        if not self.between(from_node, t5, to_node):
            temp = t5
            t5 = t8
            t8 = temp
            temp = t6
            t6 = t7
            t7 = temp

        # checking if t5-t6 and t7-t8 needs to be switched to match tour orientation
        if (t1.succ == t2 and t5.pred == t6) or (t1.pred == t2 and t5.succ == t6):
            temp = t5
            t5 = t6
            t6 = t5
            temp = t7
            t7 = t8
            t8 = temp

        # after reordering the nodes, the second unfeasible swap is performed
        self.swap_unfeasible(t5, t6, t7, t8, False, False)

        # update pos attribute
        self.set_pos()

        # update the swap stack
        if record:
            self.swap_stack.append((t1, t2, t3, t4, t5, t6, t7, t8, "swap_double_bridge"))

    def __str__(self):
        """
        The display string when printing the object

        :return: the display string
        :rtype: str
        """

        # get the starting node
        curr_node = self.nodes[0]

        # initialize the string
        node_seq = str(curr_node.id)

        # loop until tour is closed
        while curr_node.succ != self.nodes[0]:

            # get next node
            curr_node = curr_node.succ

            # update the string
            node_seq += f",{curr_node.id}"

        # return the string sequence
        return f"({node_seq})"
