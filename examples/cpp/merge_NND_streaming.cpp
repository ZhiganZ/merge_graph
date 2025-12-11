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

// ParallelFor function remains the same
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


void merge_two_graph(mergegraph::HierarchicalNSW<float>* first_half,mergegraph::HierarchicalNSW<float>* second_half,mergegraph::HierarchicalNSW<float>* hnsw_merge,bool nnd = true){
    auto start =  std::chrono::high_resolution_clock::now();
    std::vector<size_t> empty_delete_ids;
    my_tool::hnsw_merge_Skeleton_bridged_Local_Repair(hnsw_merge,first_half,second_half,empty_delete_ids);
    if(nnd)
        hnsw_merge->fgim();
    hnsw_merge->symmetrizeAndPruneAllLevelsMT();
    std::cout << "merge time: " 
        << std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::high_resolution_clock::now() - start).count() / 1e6 
        << " seconds" << std::endl;
}

int main(int argc,char* argv[]) {
    std::cout.precision(6);

    //deep1m
    std::string dataset_name;
    std::string dataset_file;
    std::string query_file;

    bool read_bvecs_ = false;
    for(int i = 0 ; i < argc ; i ++ ){
        if (std::string(argv[i]) == "bvecs"){ 
            std::cout<<"read from bvecs"<<std::endl;
            read_bvecs_ = true;
        }
        std::cout<<argv[i]<<std::endl;
    }

    int M = 24;                  // Internal dimensionality parameter
    int ef_construction = 400;   // Controls index search speed/build speed tradeoff

    if(argc == 4){
        dataset_name = argv[1];
        dataset_file = argv[2];
        query_file = argv[3];
    }
    if(argc >= 6){
        dataset_name = argv[1];
        dataset_file = argv[2];
        query_file = argv[3];
        M = std::stoi(argv[4]);
        ef_construction = std::stoi(argv[5]);
    }

    std::random_device rd;
    std::mt19937 gen(rd());

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
    queries.resize(std::min(1000ul,queries.size()));
    queries.shrink_to_fit();
    int num_queries = queries.size();
    int ef = 150;
    int num_threads = 0;

    int ef_1 = ef_construction ;
    my_tool::vamana_l = ef_construction ;

    int seed = 100;
    mergegraph::L2Space space(dim);

    mergegraph::HierarchicalNSW<float>* alg_mergegraph_merge_big = nullptr;

    int k = 10;

    auto start1 = std::chrono::high_resolution_clock::now();
    auto end1 = std::chrono::high_resolution_clock::now();

    // modify here to control nn descent
    std::vector<int> nnd{1,2,3,5,8,12,24,36,54,63};

    // std::vector<int> nnd;
    // for(int i = 0 ; i < 64; ++i){
    //     nnd.emplace_back(i);
    // }

    std::vector<mergegraph::labeltype> delete_ids(max_elements);
    for(mergegraph::labeltype i = 0 ; i < max_elements ; ++i){
        delete_ids[i]=i;
    }

    std::shuffle(delete_ids.begin(),delete_ids.end(),gen);
    delete_ids.resize(0);

    std::vector<std::vector<size_t>> labels;

    auto start_all = std::chrono::high_resolution_clock::now();
    if(1){
        // Modify here part to control how many subgraphs the dataset is split into.
        int num_sub_graphs = 64;
        int point_in_sub_graph = max_elements/num_sub_graphs;
        std::vector<int> offset_of_graph;
        // Modify here part to control how many subgraphs are used in the merging process.
        int used_num_sub_graphs = num_sub_graphs/2;
        for(int i = 0 ; i < num_sub_graphs ; ++i){
            offset_of_graph.emplace_back(i*point_in_sub_graph);
        }
        offset_of_graph.emplace_back(max_elements);
        for(int i = 0 ; i < offset_of_graph.size() ; ++i ){
            std::cout<<"offset_of_graph[i] is: "<<offset_of_graph[i]<<std::endl;
        }
        std::vector<mergegraph::HierarchicalNSW<float>*> sub_graphs ;

        for(int i = 0 ; i < used_num_sub_graphs ; ++i){
            sub_graphs.emplace_back(new mergegraph::HierarchicalNSW<float>(&space, offset_of_graph[i+1]-offset_of_graph[i], M, ef_1, seed, true));
            sub_graphs[i]->set_alpha(1.2);
        }
        for(int i = 0 ; i < used_num_sub_graphs ; ++i){
            std::cout<<"build initial sub graph "<<i<<"\n"<<std::endl;
            my_tool::build_vamana(sub_graphs[i],dataset,offset_of_graph[i],offset_of_graph[i+1]);
            std::cout<<"\n\n";
        }
        mergegraph::HierarchicalNSW<float>* merged_graph = sub_graphs[0];

        for (int i = 1; i < used_num_sub_graphs; ++i) {
            std::cout << "\nmerge sub graph " << i << " into merged_graph\n" << std::endl;

            auto* new_merged = new mergegraph::HierarchicalNSW<float>(
                &space,
                merged_graph->cur_element_count + sub_graphs[i]->cur_element_count,
                M, ef_1, seed, true
            );
            new_merged->set_alpha(1.2);

            if(std::find(nnd.begin(),nnd.end(),i)!=nnd.end())
                merge_two_graph(merged_graph, sub_graphs[i], new_merged,true);
            else
                merge_two_graph(merged_graph, sub_graphs[i], new_merged,false);

            if (i > 1) {
                delete merged_graph;
            }
            delete sub_graphs[i];

            merged_graph = new_merged;
        }
        alg_mergegraph_merge_big = merged_graph;

        std::cout<<"start to compute gt"<<std::endl;
        mergegraph::BruteforceSearch<float>* alg_brute = new mergegraph::BruteforceSearch<float>(&space, alg_mergegraph_merge_big->cur_element_count);
        for(int i=0;i<alg_mergegraph_merge_big->cur_element_count;i++){
            alg_brute->addPoint(alg_mergegraph_merge_big->getDataByInternalId(alg_mergegraph_merge_big->label_lookup_[i]),i);
        }
        auto gt = get_groundtruth(alg_brute,queries,k);
        delete alg_brute;
        my_tool::draw_recall_qps_graph(alg_mergegraph_merge_big,gt,queries,k,"merged graph");
        alg_mergegraph_merge_big->alpha = 1.0f;
        auto alg_mergegraph = new mergegraph::HierarchicalNSW<float>(&space, offset_of_graph[num_sub_graphs/2], M, ef_1, seed, true);
        my_tool::build_hnsw(alg_mergegraph,dataset,0,offset_of_graph[num_sub_graphs/2]);
        my_tool::draw_recall_qps_graph(alg_mergegraph,gt,queries,k,"HNSW");
        delete alg_mergegraph;
        alg_mergegraph_merge_big->flatten_to_level0();
        alg_mergegraph_merge_big->alpha = 1.2f;
        my_tool::build_vamana(alg_mergegraph_merge_big,dataset,0,offset_of_graph[num_sub_graphs/2]);
        my_tool::draw_recall_qps_graph(alg_mergegraph_merge_big,gt,queries,k,"vamana");
    }

    delete alg_mergegraph_merge_big;
    for (auto vec : dataset) {
        free(vec);
    }
    for (auto vec : queries) {
        free(vec);
    }

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
