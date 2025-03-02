#################################################
#                Noraml Decoding                #
#################################################

from .greedy_search import tsp_greedy as tsp_greedy
from .beam_search import tsp_beam as tsp_beam


#################################################
#             MCTS Decoding Solver              #
#################################################

from .mcts_solver import tsp_mcts_solver as tsp_mcts_solver
from .rg_mcts import tsp_rg_mcts as tsp_rg_mcts
from .beam_mcts import tsp_beam_mcts as tsp_beam_mcts
from .rmcts import tsp_rmcts as tsp_rmcts


#################################################
#                 Local Search                  #
#################################################

from .two_opt import tsp_2opt as tsp_2opt
from .mcts import tsp_mcts as tsp_mcts
from .local_search import tsp_ls as tsp_ls
from .local_search import tsp_gls as tsp_gls
from .relocate import tsp_relocate as tsp_relocate


#################################################
#                 Class Define                  #
#################################################


DECODING_CLASS = {
    ('tsp', 'greedy'): tsp_greedy,
    ('tsp', 'beam'): tsp_beam,
    ('tsp', 'mcts_solver'): tsp_mcts_solver,
    ('tsp', 'rg_mcts'): tsp_rg_mcts,
    ('tsp', 'beam_mcts'): tsp_beam_mcts,
    ('tsp', 'rmcts'): tsp_rmcts,
}


LOCAL_SEARCH_CLASS = {
    ('tsp', '2opt'): tsp_2opt,
    ('tsp', 'relocate'): tsp_relocate,
    ('tsp', 'mcts'): tsp_mcts,
    ('tsp', 'ls'): tsp_ls,
    ('tsp', 'gls'): tsp_gls,
    ('tsp', None): None,
}


def get_decoding_func(task="tsp", name="greedy"):
    return DECODING_CLASS[(task, name)]


def get_local_search_func(task="tsp", name="greedy"):
    return LOCAL_SEARCH_CLASS[(task, name)]