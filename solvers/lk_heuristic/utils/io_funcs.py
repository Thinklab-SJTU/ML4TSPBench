import logging
from solvers.lk_heuristic.models.node import Node2D, Node3D

# create a logger object
logger = logging.getLogger(__name__)


def import_tsp_file(tsp_file):
    """
    Import a .tsp file containing tsp definition and parse tsp header to a dictionary and tsp nodes to a list. Returns both objects in a tuple.

    *This function only works for .tsp instances of "TYPE" = TSP (simmetric tsp) and for "EDGE_WEIGH_TYPE" = EUC_2D or EUC_3D, which are the tsp problems that solver can handle currently.

    :param tsp_file: the .tsp file path
    :type tsp_file: str
    :return: a tuple with the header dict definition and a list of nodes
    :rtype: dict
    """

    # set the allowed type and edge weight for tsp
    allowed_types = ["TSP"]
    allowed_edge_weights = ["EUC_2D", "EUC_3D"]

    # return objects
    tsp_header = {}  # the .tsp header dict to be returned by the function
    nodes = []  # the list of nodes to be parsed from node coord section

    # a boolean value to be changed when node coord section is reached
    is_node_section = False

    # try to execute the import
    try:

        # open tsp file
        with open(tsp_file, "r+") as f:

            # read all tsp file lines
            tsp_lines = f.readlines()

            # loop through each tsp line
            for line in tsp_lines:

                # checking if coord section was already reached
                if is_node_section:

                    # check if problem is tsp (simmetric tsp) and edge_weight is either euc_2d or euc_3d
                    if tsp_header["TYPE"] in allowed_types and tsp_header["EDGE_WEIGHT_TYPE"] in allowed_edge_weights:

                        # check if line is not last end-of-file line
                        if line.strip() != "EOF":

                            # split the line using space separator
                            coords = line.replace("\n", "").split()

                            # create a node using node coord values and append it to list
                            if tsp_header["EDGE_WEIGHT_TYPE"] == "EUC_2D":
                                nodes.append(Node2D(float(coords[1]), float(coords[2])))
                            elif tsp_header["EDGE_WEIGHT_TYPE"] == "EUC_3D":
                                nodes.append(Node3D(float(coords[1]), float(coords[2]), float(coords[3])))

                # try to collect .tsp attributes
                else:

                    # add node coord section into dict
                    if line.strip() == "NODE_COORD_SECTION":
                        is_node_section = True

                    # split the line using ":" to get header attributes
                    line_split = line.split(":")

                    # check if split was made
                    if len(line_split) == 2:

                        # assign dict values
                        tsp_header[line_split[0].strip()] = line_split[1].strip()

    # print error to user
    except Exception as e:
        logger.error(f"Error during import of .tsp file: '{e}'")

    # return the .tsp dict and node values
    return tsp_header, nodes


def export_tsp_file(tsp_file_path, tsp_header, nodes, tour_type="cycle"):
    """
    Export a .tsp file using tsp_header header elements and list of nodes. Nodes are parsed into "NODES_COORD_SECTION" format.

    :param tsp_file_path: the name of tsp file to be exported
    :type tsp_file_path: str
    :param tsp_header: a dictionary containing header values for the .tsp file
    :type tsp_header: dict
    :param nodes: a list containing nodes to be parsed into tsp file
    :type nodes: list
    :param tour_type: the type of the tour (either 'path' or 'cycle')
    :type tour_type: str
    """

    # try to execute the export
    try:

        # create tsp file to write data
        with open(tsp_file_path, "w+") as f:

            # loop through each header value and write it to file
            for k, v in tsp_header.items():
                f.write(f"{k} : {v}\n")

            # write the node section
            f.write("NODE_COORD_SECTION\n")

            # removes pivot node when using hamiltonian path tour type
            # pivot node shall be the first node (from Tour get_nodes function)
            if tour_type == "path":
                nodes.pop(0)

            # loop through each node and write its values to file
            for i, node in enumerate(nodes):
                if type(node) == Node2D:
                    f.write(f"{i} {node.x} {node.y}\n")
                elif type(node) == Node3D:
                    f.write(f"{i} {node.x} {node.y} {node.z}\n")

            # write end of file line
            f.write("EOF\n")

    # print error to user
    except Exception as e:
        logger.error(f"Error during export of .tsp file: '{e}'")
