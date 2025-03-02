class Edge:
    """
    The edge class represent an edge in space. The edge is a connection between two nodes. The connection is simmetric, i.e, the Edge(AB) == Edge(BA). This simmetric property is implemented in initialization method.
    """

    def __init__(self, n1, n2):
        """
        Initialize an edge with its nodes. The node order (n1/n2) is switched based on node values to guarantee the simmetric property of TSP edges, i.e, {n1,n2} == {n2,n1}. By comparing the nodes, the order will always be the same, so is not necessary to check node ordering later.

        :param n1: the first node
        :type n1: Node
        :param n2: the second node
        :type n2: Node
        """

        # asserting that a valid edge can't connect to same node
        assert(n1 != n2)

        if n1 < n2:
            self.n1 = n1
            self.n2 = n2
        # switch the ordering
        else:
            self.n1 = n2
            self.n2 = n1

    def __eq__(self, other):
        """
        Comparison method between two edges. The edge is equal if its nodes are equal. The edge {n1,n2} will be equal to {n2,n1}. It is required to check the inverted conditions since node comparison will also compare node id (to be able to create two equal coordinates nodes - relevant for a280.tsp) and in this situation, node comparison would be false.

        :param other: the other edge to compare
        :type other: Edge
        :return: a boolean indicating if both edges are the same
        :rtype: boolean
        """
        return (self.n1 == other.n1 and self.n2 == other.n2) or (self.n1 == other.n2 and self.n2 == other.n1)

    def __hash__(self):
        """
        Hashing Edge object is required to allow building sets of edges, which are "fast" objects used in LK joined and broken edges sets. The hash is performed in the tuple of node 1 and node 2 id values.
        """
        return hash((self.n1.id, self.n2.id))

    def __str__(self):
        """
        The display string when printing the object

        :return: the display string
        :rtype: str
        """
        return f"({self.n1},{self.n2})"

    def __repr__(self):
        """
        The display string when printing the object

        :return: the display string
        :rtype: str
        """
        return f"({self.n1},{self.n2})"
