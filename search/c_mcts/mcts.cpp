#include "tsp.h"
#include "2opt.h"
#include "init.h"
#include "utils.h"
#include "mcts.h"
#include <iostream>

extern "C" {
	int* mcts_local_search(
		short* tour, 
		float *heatmap, 
		float *nodes_coords, 
		int nodes_num, 
		int depth, 
		float mcts_param_t,
		int max_iterations_2opt
	){
		srand(RANDOM_SEED);
		city_num = nodes_num;
		max_depth = depth;
		param_t = mcts_param_t;
		max_iterations_2opt = max_iterations_2opt;
		begin_time = (double)clock();  
		best_distance = INF;   
		allocate_memory(city_num);
		read_heatmap(heatmap);
		read_nodes_coords(nodes_coords);
		read_initial_solution(tour);
		calculate_all_pair_distance();	
		identify_candidate_set(); 	  		    
		mdp();
		convert_all_node_to_solution();
		release_memory(city_num);
		return solution;	
	}
}
