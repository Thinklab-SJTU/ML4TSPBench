#include "tsp.h"
#include "2opt.h"
#include "init.h"
#include "utils.h"
#include "mdp.h"
#include "mcts.h"
#include <iostream>

extern "C" {
	int* mcts_solver(
		float *heatmap, 
		float *nodes_coords, 
		int nodes_num, 
		int max_depth, 
		float mcts_param_t,
		int max_iterations_2opt
	){	
		srand(RANDOM_SEED);
		city_num = nodes_num;
		max_depth = max_depth;
		param_t = mcts_param_t;
		max_iterations_2opt = max_iterations_2opt;
		begin_time = (double)clock();  
		best_distance = INF;   
		read_heatmap(heatmap); 
		read_nodes_coords(nodes_coords);
		calculate_all_pair_distance();	
		identify_candidate_set(); 	 		  		    
		mdp();
		convert_all_node_to_solution();
		release_memory(city_num);
		return solution;	
	}
}