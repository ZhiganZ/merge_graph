#include "../../include/hnswlib.h"
#include "assert.h"
#include "include/bruteforce.h"
#include "include/hnswalg.h"
#include "../../include/tool.h"

#include <chrono>
#include <cstdint>
#include <queue>
#include <random>
#include <cstdlib>
#include <stdexcept>
#include <set>
#include <string>
#include <unordered_set>
#include <fstream>
#include <iostream>
#include <vector>
#include <thread>
#include <atomic>
#include <mutex>
#include <algorithm>
#include <unistd.h>
#include <string.h>
#include <cstring>
#include <omp.h>

std::vector<std::unordered_set<mergegraph::labeltype>> get_groundtruth(mergegraph::BruteforceSearch<float>* alg_brute,std::vector<float*> queries,int k=5);
template<class Function>
inline void ParallelFor(size_t start, size_t end, size_t numThreads, Function fn) {
    if (numThreads <= 0) {
        numThreads = std::thread::hardware_concurrency();
    }

    if (numThreads == 1) {
        for (size_t id = start; id < end; id++) {
            fn(id, 0);
        }
    } else {
        std::vector<std::thread> threads;
        std::atomic<size_t> current(start);

        std::exception_ptr lastException = nullptr;
        std::mutex lastExceptMutex;

        for (size_t threadId = 0; threadId < numThreads; ++threadId) {
            threads.emplace_back([&, threadId] {
                while (true) {
                    size_t id = current.fetch_add(1);

                    if (id >= end) {
                        break;
                    }

                    try {
                        fn(id, threadId);
                    } catch (...) {
                        std::unique_lock<std::mutex> lastExcepLock(lastExceptMutex);
                        lastException = std::current_exception();
                        current = end;
                        break;
                    }
                }
            });
        }
        for (auto &thread : threads) {
            thread.join();
        }
        if (lastException) {
            std::rethrow_exception(lastException);
        }
    }
}

int vamana_l = 70;

float merge_alpha = 1.2;

int main(int argc,char* argv[]) {
    std::cout.precision(6);

    std::string dataset_name ;
    std::string dataset_file ;
    std::string query_file;

    bool read_bvecs_ = false;
    for(int i = 1 ; i < argc ; i ++ ){
        if(argv[i] == "bvecs")read_bvecs_ = true;
        std::cout<<argv[i]<<std::endl;
    }

    int M = 24;                  // Internal dimensionality parameter
    int ef_construction = 400;   // Controls index search speed/build speed tradeoff

    if(argc == 4){
        dataset_name = argv[1];
        dataset_file = argv[2];
        query_file = argv[3];
    }
    if(argc == 6){
        dataset_name = argv[1];
        dataset_file = argv[2];
        query_file = argv[3];
        M = std::stoi(argv[4]);
        ef_construction = std::stoi(argv[5]);
    }
    int opt_L  = 1;
    if(argc == 7){
        dataset_name = argv[1];
        dataset_file = argv[2];
        query_file = argv[3];
        M = std::stoi(argv[4]);
        ef_construction = std::stoi(argv[5]);
        opt_L = std::stoi(argv[6]);
    }

    // Initialize random number generator
    std::random_device rd;
    std::mt19937 gen(rd());

    // Read dataset
    int dim_dataset = 0;
    std::vector<float*> dataset;
    try {
        if(!read_bvecs_)
        dataset = my_tool::read_fvecs(dataset_file, dim_dataset);
        else
        dataset = my_tool::read_bvecs(dataset_file, dim_dataset);
    } catch (const std::exception& e) {
        std::cerr << "Error reading dataset: " << e.what() << std::endl;
        return -1;
    }

    // Read queries
    int dim_queries = 0;
    std::vector<float*> queries;
    try {
        if(!read_bvecs_)
        queries = my_tool::read_fvecs(query_file, dim_queries);
        else queries = my_tool::read_bvecs(query_file, dim_queries);
    } catch (const std::exception& e) {
        std::cerr << "Error reading queries: " << e.what() << std::endl;
        return -1;
    }

    std::cout<<dim_dataset<<std::endl;
    std::cout<<dim_queries<<std::endl;

    std::cout << "Dataset size is " << dataset.size() << std::endl;
    std::cout << "Query size is " << queries.size() << std::endl;

    int dim = dim_queries;               // Dimension of the elements
    int max_elements = dataset.size();    // Maximum number of elements
    // queries.resize(std::min(1000ul,queries.size()));
    // queries.shrink_to_fit();
    int num_queries = queries.size();
    int ef = 150;
    int num_threads = 0;

    int ef_1 = ef_construction ;
    vamana_l = ef_construction ;

    int seed = 100;
    mergegraph::L2Space space(dim);
    // mergegraph::L2SpaceEE space1(dim);
    mergegraph::HierarchicalNSW<float>* alg_mergegraph = new mergegraph::HierarchicalNSW<float>(&space, max_elements, M, ef_construction, seed, true);

    mergegraph::HierarchicalNSW<float>* alg_mergegraph_merge = new mergegraph::HierarchicalNSW<float>(&space, max_elements, M, ef_1, seed, true);

    mergegraph::HierarchicalNSW<float>* alg_mergegraph_merge_big_final = new mergegraph::HierarchicalNSW<float>(&space, max_elements, M, ef_1, seed, true);

    mergegraph::HierarchicalNSW<float>* alg_mergegraph_first_half = new mergegraph::HierarchicalNSW<float>(&space, max_elements/2, M, ef_1, seed, true);
    mergegraph::HierarchicalNSW<float>* alg_mergegraph_second_half = new mergegraph::HierarchicalNSW<float>(&space, max_elements - max_elements/2, M, ef_1, seed, true);
    
    alg_mergegraph->setEf(ef);
    alg_mergegraph->allow_replace_deleted_ = true;

    mergegraph::BruteforceSearch<float>* alg_brute = new mergegraph::BruteforceSearch<float>(&space, max_elements);

    int k = 10;

    std::vector<std::unordered_set<mergegraph::labeltype>> gt;
    for(int i=0;i<dataset.size();i++){
        alg_brute->addPoint(dataset[i],i);
    }
    gt = get_groundtruth(alg_brute,queries,k);

    auto start1 = std::chrono::high_resolution_clock::now();
    auto end1 = std::chrono::high_resolution_clock::now();
    start1 = std::chrono::high_resolution_clock::now();
    my_tool::build_vamana(alg_mergegraph,dataset,0,max_elements);
    end1 = std::chrono::high_resolution_clock::now();
    std::cout << "HNSW initial insertion time: " 
            << std::chrono::duration_cast<std::chrono::microseconds>(end1 - start1).count() / 1e6 
            << " seconds" << std::endl;

    start1 = std::chrono::high_resolution_clock::now();

    ParallelFor(0, max_elements/2, num_threads, [&](size_t row, size_t threadId) {
        alg_mergegraph_first_half->addPoint(dataset[row], row);
    });
    my_tool::build_vamana(alg_mergegraph_first_half,dataset,0,max_elements/2);

    end1 = std::chrono::high_resolution_clock::now();
    std::cout << "first half initial insertion time: " 
            << std::chrono::duration_cast<std::chrono::microseconds>(end1 - start1).count() / 1e6 
            << " seconds" << std::endl;


    start1 = std::chrono::high_resolution_clock::now();
    ParallelFor(max_elements/2, max_elements, num_threads, [&](size_t row, size_t threadId) {
        alg_mergegraph_second_half->addPoint(dataset[row], row);
    });
    my_tool::build_vamana(alg_mergegraph_second_half,dataset,max_elements/2,max_elements);

    end1 = std::chrono::high_resolution_clock::now();
    std::cout << "second half initial insertion time: " 
            << std::chrono::duration_cast<std::chrono::microseconds>(end1 - start1).count() / 1e6 
            << " seconds" << std::endl;
    std::vector<mergegraph::labeltype> delete_ids(max_elements);
    for(mergegraph::labeltype i = 0 ; i < max_elements ; ++i){
        delete_ids[i]=i;
    }

    std::shuffle(delete_ids.begin(),delete_ids.end(),gen);
    delete_ids.resize(0);
    for(mergegraph::labeltype i = 0 ; i < delete_ids.size() ; ++i){
        alg_mergegraph->markDelete(delete_ids[i]);
        if(delete_ids[i]<max_elements/2)
            alg_mergegraph_first_half->markDelete(delete_ids[i]);
        else alg_mergegraph_second_half->markDelete(delete_ids[i]);
    }
    std::vector<std::vector<size_t>> labels;
    std::vector<mergegraph::HierarchicalNSW<float>*> vamana_merge_vector ;
    mergegraph::HierarchicalNSW<float>* vamana_merged = new mergegraph::HierarchicalNSW<float>(&space, dataset.size(), M, ef_1, seed, true);

    if(0)
    {
        std::vector<mergegraph::HierarchicalNSW<float>*> alg_mergegraph_vector_16 ;
        std::vector<mergegraph::HierarchicalNSW<float>*> alg_mergegraph_vector_8 ;
        std::vector<mergegraph::HierarchicalNSW<float>*> alg_mergegraph_vector_4 ;
        std::vector<mergegraph::HierarchicalNSW<float>*> alg_mergegraph_vector_2 ;

        for(int i = 0 ; i < 16 ; ++i){
            alg_mergegraph_vector_16.emplace_back(new mergegraph::HierarchicalNSW<float>(&space, max_elements/16+1, M, ef_1, seed, true));
        }
        for(int i = 0 ; i < 16 ; ++i){
            my_tool::build_vamana(alg_mergegraph_vector_16[i],dataset,i*((max_elements+15)/16),(i+1)*((max_elements+15)/16)<max_elements?(i+1)*((max_elements+15)/16):max_elements);
        }
        // for(int i = 0 ; i < 4 ; ++i){
        //     alg_mergegraph_vector_4.emplace_back(new mergegraph::HierarchicalNSW<float>(&space, max_elements/4+1, M, ef_1, seed, true));
        // }
        // for(int i = 0 ; i < 4 ; ++i){
        //     // my_tool::build_vamana(alg_mergegraph_vector_4[i],dataset,i*((max_elements+3)/4),(i+1)*((max_elements+3)/4)<max_elements?(i+1)*((max_elements+3)/4):max_elements);
        //     my_tool::build_hnsw(alg_mergegraph_vector_4[i],dataset,i*((max_elements+3)/4),(i+1)*((max_elements+3)/4)<max_elements?(i+1)*((max_elements+3)/4):max_elements);
        // }
        for(int i = 0 ; i < 8 ; ++i){
            alg_mergegraph_vector_8.emplace_back(new mergegraph::HierarchicalNSW<float>(&space, max_elements/8+2, M, ef_1, seed, true));
            alg_mergegraph_vector_8[i]->set_alpha(merge_alpha);
        }
        for(int i = 0 ; i < 8 ; ++i){
            auto start = std::chrono::high_resolution_clock::now();
            my_tool::hnsw_merge_Skeleton_bridged_Local_Repair(alg_mergegraph_vector_8[i],alg_mergegraph_vector_16[i*2],alg_mergegraph_vector_16[i*2+1],delete_ids);
            // alg_mergegraph_vector_8[i]->Global_Repair();
            // alg_mergegraph_vector_8[i]->reprune();

            alg_mergegraph_vector_8[i]->fgim();
            alg_mergegraph_vector_8[i]->symmetrizeAndPruneAllLevelsMT();
            std::cout << "16->8 merge time: "
                << std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::high_resolution_clock::now() - start).count() / 1e6 
                << " seconds" << std::endl;
            delete alg_mergegraph_vector_16[i*2];
            delete alg_mergegraph_vector_16[i*2+1];
        }
        for(int i = 0 ; i < 4 ; ++i){
            alg_mergegraph_vector_4.emplace_back(new mergegraph::HierarchicalNSW<float>(&space, max_elements/4+4, M, ef_1, seed, true));
            alg_mergegraph_vector_4[i]->set_alpha(merge_alpha);
        }
        for(int i = 0 ; i < 4 ; ++i){
            auto start = std::chrono::high_resolution_clock::now();
            my_tool::hnsw_merge_Skeleton_bridged_Local_Repair(alg_mergegraph_vector_4[i],alg_mergegraph_vector_8[i*2],alg_mergegraph_vector_8[i*2+1],delete_ids);

            alg_mergegraph_vector_4[i]->fgim();
            alg_mergegraph_vector_4[i]->symmetrizeAndPruneAllLevelsMT();
            std::cout << "8->4 merge time: "
                << std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::high_resolution_clock::now() - start).count() / 1e6 
                << " seconds" << std::endl;
            delete alg_mergegraph_vector_8[i*2];
            delete alg_mergegraph_vector_8[i*2+1];
        }
        for(int i = 0 ; i < 2 ; ++i){
            alg_mergegraph_vector_2.emplace_back(new mergegraph::HierarchicalNSW<float>(&space, max_elements/2+8, M, ef_1, seed, true));
            alg_mergegraph_vector_2[i]->set_alpha(merge_alpha);
        }
        for(int i = 0 ; i < 2 ; ++i){
            auto start = std::chrono::high_resolution_clock::now();
            my_tool::hnsw_merge_Skeleton_bridged_Local_Repair(alg_mergegraph_vector_2[i],alg_mergegraph_vector_4[i*2],alg_mergegraph_vector_4[i*2+1],delete_ids);

            alg_mergegraph_vector_2[i]->fgim();
            alg_mergegraph_vector_2[i]->symmetrizeAndPruneAllLevelsMT();
            std::cout << "4->2 merge time: "
                << std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::high_resolution_clock::now() - start).count() / 1e6 
                << " seconds" << std::endl;
            delete alg_mergegraph_vector_4[i*2];
            delete alg_mergegraph_vector_4[i*2+1];
        }

        alg_mergegraph_merge_big_final->set_alpha(merge_alpha);
        auto start = std::chrono::high_resolution_clock::now();
        my_tool::hnsw_merge_Skeleton_bridged_Local_Repair(alg_mergegraph_merge_big_final,alg_mergegraph_vector_2[0],alg_mergegraph_vector_2[1],delete_ids);

        alg_mergegraph_merge_big_final->fgim(1.0f,1.5f);
        auto bi_add_start = mergegraph::get_time_now();
        alg_mergegraph_merge_big_final->symmetrizeAndPruneAllLevelsMT();
        mergegraph::print_using_time("bi add",bi_add_start);
        auto start2 = mergegraph::get_time_now();
        alg_mergegraph_merge_big_final->random_high_level();
        mergegraph::print_using_time("rebuild high level",start2);
        alg_mergegraph_merge_big_final->rebuild_upper_layers_mt();
        std::cout << "2->1 merge time: "
            << std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::high_resolution_clock::now() - start).count() / 1e6 
            << " seconds" << std::endl;

        my_tool::draw_recall_qps_graph(alg_mergegraph_merge_big_final,gt,queries,k,"4 -> 1 HNSW");
    }

    if(1){
        alg_mergegraph_merge_big_final->set_alpha(merge_alpha);
        alg_mergegraph_merge_big_final->flatten_to_level0();
        for(int i = 2 ; i<= 18 ; i+=4){
            int num_sub_graphs = i;
            int point_in_sub_graph = max_elements/num_sub_graphs;
            std::vector<int> offset_of_graph;
            for(int i = 0 ; i < num_sub_graphs ; ++i){
                offset_of_graph.emplace_back(i*point_in_sub_graph);
            }
            offset_of_graph.emplace_back(max_elements);
            for(int i = 0 ; i < offset_of_graph.size() ; ++i ){
                std::cout<<"offset_of_graph[i] is: "<<offset_of_graph[i]<<std::endl;
            }
            std::vector<mergegraph::HierarchicalNSW<float>*> sub_graphs ;

            for(int i = 0 ; i < num_sub_graphs ; ++i){
                sub_graphs.emplace_back(new mergegraph::HierarchicalNSW<float>(&space, offset_of_graph[i+1]-offset_of_graph[i-1], M, ef_1, seed, true));
            }
            for(int i = 0 ; i < num_sub_graphs ; ++i){
                my_tool::build_vamana(sub_graphs[i],dataset,offset_of_graph[i],offset_of_graph[i+1]);
            }

            my_tool::hnsw_merge_cross_all_sub_graph(alg_mergegraph_merge_big_final,sub_graphs,delete_ids);
            // alg_mergegraph_merge_big_final->flatten_to_level0();
            // alg_mergegraph_merge_big_final->random_high_level();
            // alg_mergegraph_merge_big_final->rebuild_upper_layers_mt();
            alg_mergegraph->enterpoint_node_ = alg_mergegraph->get_navi_point();
            my_tool::draw_recall_qps_graph(alg_mergegraph_merge_big_final,gt,queries,k,std::to_string(num_sub_graphs)+" -> 1 HNSW");

        }

    }

    std::vector<int> sample_rates{1,10,20,50,100};
    alg_mergegraph_merge->alpha = 1.2f;
    alg_mergegraph_first_half->flatten_to_level0();
    alg_mergegraph_second_half->flatten_to_level0();
    // for(auto sample_rate:sample_rates){
    //     my_tool::hnsw_merge_Skeleton_bridged_Local_Repair(alg_mergegraph_merge,alg_mergegraph_first_half,alg_mergegraph_second_half,delete_ids,sample_rate);
    //     alg_mergegraph_merge->enterpoint_node_ = alg_mergegraph_merge->get_navi_point(true);
    //     my_tool::draw_recall_qps_graph(alg_mergegraph_merge,gt,queries,k,"sample "+std::to_string(sample_rate)+"% graph");
    // }

    // my_tool::draw_recall_qps_graph(alg_mergegraph_first_half,alg_mergegraph_second_half,gt,queries,k,"two HNSW");
    alg_mergegraph->random_high_level();
    alg_mergegraph->rebuild_upper_layers_mt();
    my_tool::draw_recall_qps_graph(alg_mergegraph,gt,queries,k,"vamana");

    
    alg_mergegraph_merge->alpha = 1.2f;
    start1 = std::chrono::high_resolution_clock::now();
    my_tool::hnsw_merge_Skeleton_bridged_Local_Repair(alg_mergegraph_merge,alg_mergegraph_first_half,alg_mergegraph_second_half,delete_ids,1);

    alg_mergegraph_merge->fgim();
    auto bi_add_start = mergegraph::get_time_now();
    alg_mergegraph_merge->symmetrizeAndPruneAllLevelsMT();
    mergegraph::print_using_time("bi add",bi_add_start);
    auto start2 = mergegraph::get_time_now();
    alg_mergegraph_merge->random_high_level();
    alg_mergegraph_merge->rebuild_upper_layers_mt();
    mergegraph::print_using_time("rebuild high level",start2);
    end1 = std::chrono::high_resolution_clock::now();
    std::cout << "reprune time: " 
            << std::chrono::duration_cast<std::chrono::microseconds>(end1 - start1).count() / 1e6 
            << " seconds" << std::endl;
    my_tool::draw_recall_qps_graph(alg_mergegraph_merge,gt,queries,k,"merge HNSW");


    std::cout.precision(6);
    // my_tool::draw_recall_qps_graph(alg_mergegraph_first_half,alg_mergegraph_second_half,gt,queries,k,"two HNSW");
    // my_tool::draw_recall_distance_compuatation_graph(alg_mergegraph_first_half,alg_mergegraph_second_half,gt,queries,k,"two HNSW");
    if(alg_mergegraph_merge_big_final->cur_element_count>10000){
        // alg_mergegraph_merge_big_final->OptimizeGraph();
        my_tool::draw_recall_qps_graph(alg_mergegraph_merge_big_final,gt,queries,k,"4 -> 1 HNSW");
    }

    my_tool::draw_recall_qps_graph(alg_mergegraph_merge,gt,queries,k,"merge HNSW");
    my_tool::draw_recall_qps_graph(alg_mergegraph,gt,queries,k,"vamana");
    if(alg_mergegraph->maxlevel_>0){
        return 0;
    }
    alg_mergegraph->random_high_level();
    alg_mergegraph->rebuild_upper_layers_mt();

    my_tool::draw_recall_qps_graph(alg_mergegraph,gt,queries,k,"vamana with multi layer");
    // Clean up
    for (auto vec : dataset) {
        free(vec);
    }
    for (auto vec : queries) {
        free(vec);
    }

    delete alg_mergegraph;
    delete alg_brute;
    return 0;
}


std::vector<std::unordered_set<mergegraph::labeltype>> get_groundtruth(mergegraph::BruteforceSearch<float>* alg_brute,std::vector<float*> queries,int k){
    std::vector<std::unordered_set<mergegraph::labeltype>> ground_truth(queries.size());
    #pragma omp parallel for num_threads(96) schedule(dynamic)
    for (size_t i = 0; i < queries.size(); ++i) {
        auto result_brute = alg_brute->searchKnn(queries[i], k);
        std::unordered_set<mergegraph::labeltype> labels;

        while (!result_brute.empty()) {
            labels.insert(result_brute.top().second);
            result_brute.pop();
        }

        ground_truth[i] = std::move(labels);
    }

    return ground_truth;
}
