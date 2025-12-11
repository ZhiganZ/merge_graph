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

double calculate_different_ef(mergegraph::HierarchicalNSW<float>* alg_mergegraph,std::vector<std::unordered_set<mergegraph::labeltype>> groundtruth,std::vector<float*> queries,int ef,int M,int ef_construction,std::string name,int k=5);
std::vector<std::priority_queue<float>> get_groundtruth_with_distance(mergegraph::BruteforceSearch<float>* alg_brute,std::vector<float*> queries,int k);

std::vector<std::unordered_set<mergegraph::labeltype>> get_groundtruth(mergegraph::BruteforceSearch<float>* alg_brute,std::vector<float*> queries,int k=5);
std::vector<std::unordered_set<mergegraph::labeltype>> brute_single_thread(mergegraph::BruteforceSearch<float>* alg_brute,std::vector<float*> queries,int k){
    std::vector<std::unordered_set<mergegraph::labeltype>> ground_truth(queries.size());
    uint64_t search_time_st;
    // #pragma omp parallel for num_threads(80) schedule(dynamic)
    for (size_t i = 0; i < queries.size(); ++i) {
        auto temp1 = std::chrono::high_resolution_clock::now();
        auto result_brute = alg_brute->searchKnn(queries[i], k);
        auto temp2 = std::chrono::high_resolution_clock::now();
        std::unordered_set<mergegraph::labeltype> labels;

        while (!result_brute.empty()) {
            labels.insert(result_brute.top().second);
            result_brute.pop();
        }

        ground_truth[i] = std::move(labels);
        search_time_st += std::chrono::duration_cast<std::chrono::microseconds>(temp2 - temp1).count();
    }
    std::cout<<search_time_st<<std::endl;
    std::cout<<"brute search qps: "<<static_cast<double>((double)queries.size()/((search_time_st) / 1e6))<<std::endl;
    // static_cast<double>((double)num_queries/((hnsw_search_time_st_min) / 1e6))
    return ground_truth;
}

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

std::vector<float*> read_fvecs(const std::string& filename, int& dimension) {
    std::ifstream ifs(filename, std::ios::binary);
    if (!ifs.is_open()) {
        throw std::runtime_error("Cannot open file: " + filename);
    }

    std::vector<float*> data;
    while (ifs.peek() != EOF) {
        int32_t d;
        ifs.read(reinterpret_cast<char*>(&d), sizeof(int32_t));
        if (ifs.eof()) break; // Handle possible EOF after reading d

        dimension = d; // Assuming all vectors have the same dimension
        float* vec = (float*)malloc(d * sizeof(float));
        ifs.read(reinterpret_cast<char*>(vec), sizeof(float) * d);
        if (ifs.eof()) {
            free(vec); // Free allocated memory if EOF reached unexpectedly
            throw std::runtime_error("Unexpected end of file while reading vector data.");
        }
        data.emplace_back(vec);
    }

    ifs.close();
    return data;
}

// this function read bvecs like vector number, vector dim, vector,vector......
std::vector<float*> read_bvecs(const std::string& filename, int& dimension) {
    std::ifstream ifs(filename, std::ios::binary);
    if (!ifs.is_open()) {
        throw std::runtime_error("Cannot open file: " + filename);
    }
    int32_t d;
    ifs.read(reinterpret_cast<char*>(&d), sizeof(int32_t));
    ifs.read(reinterpret_cast<char*>(&d), sizeof(int32_t));

    std::vector<float*> data;
    while (ifs.peek() != EOF) {

        dimension = d;

        std::vector<uint8_t> temp(d);
        ifs.read(reinterpret_cast<char*>(temp.data()), d * sizeof(uint8_t));
        if (ifs.eof()) {
            throw std::runtime_error("Unexpected end of file while reading vector data.");
        }

        float* vec = static_cast<float*>(malloc(d * sizeof(float)));
        // 将每个uint8数据转换为float
        for (int i = 0; i < d; ++i) {
            vec[i] = static_cast<float>(temp[i]);
        }

        data.emplace_back(vec);
    }

    ifs.close();
    return data;
}


int main(int argc,char* argv[]) {
    std::cout.precision(6);


    //deep1m
    std::string dataset_name ;
    std::string dataset_file ;
    std::string query_file;
    
    for(int i = 1 ; i < argc ; i ++ ){
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

    if(argc == 7){
        dataset_name = argv[1];
        dataset_file = argv[2];
        query_file = argv[3];
        M = std::stoi(argv[4]);
        ef_construction = std::stoi(argv[5]);
    }

    // Initialize random number generator
    std::random_device rd;
    std::mt19937 gen(rd());

    // Read dataset
    int dim_dataset = 0;
    std::vector<float*> dataset;
    try {
        if(argc<=6)
        dataset = read_fvecs(dataset_file, dim_dataset);
        else
        dataset = read_bvecs(dataset_file, dim_dataset);
    } catch (const std::exception& e) {
        std::cerr << "Error reading dataset: " << e.what() << std::endl;
        return -1;
    }

    // Read queries
    int dim_queries = 0;
    std::vector<float*> queries;
    try {
        if(argc<=6)queries = read_fvecs(query_file, dim_queries);
        else queries = read_bvecs(query_file, dim_queries);
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


    // modify this to save
    std::string save_path = "";

    int seed = 100;
    mergegraph::L2Space space(dim);
    mergegraph::HierarchicalNSW<float>* alg_mergegraph = new mergegraph::HierarchicalNSW<float>(&space, max_elements, M, ef_construction, seed, true);

    mergegraph::HierarchicalNSW<float>* alg_mergegraph_vamana = new mergegraph::HierarchicalNSW<float>(&space, max_elements, M, ef_construction, seed, true);
    
    alg_mergegraph->setEf(ef);
    alg_mergegraph->allow_replace_deleted_ = true;

    mergegraph::BruteforceSearch<float>* alg_brute = new mergegraph::BruteforceSearch<float>(&space, max_elements);


    auto start1 = std::chrono::high_resolution_clock::now();
    auto end1 = std::chrono::high_resolution_clock::now();
    if (my_tool::fileExists(save_path + dataset_name + std::to_string(M)+"_"+ std::to_string(ef_construction) +"original.bin")) {
        std::cout<<"load index hnsw"<<std::endl;
        alg_mergegraph->loadIndex(save_path + dataset_name+ std::to_string(M)+"_"+ std::to_string(ef_construction) +"original.bin",&space);
    }else{
        start1 = std::chrono::high_resolution_clock::now();
        ParallelFor(0, max_elements, num_threads, [&](size_t row, size_t threadId) {
            
            if(row*10%((max_elements/10)*10)==0){
                auto tmp = std::chrono::high_resolution_clock::now();
                std::cout<<"complete "<<row*100/((max_elements/100)*100)<<"%"<<std::endl;
                std::cout << "Now time: " 
                << std::chrono::duration_cast<std::chrono::microseconds>(tmp - start1).count() / 1e6 
                << " seconds" << std::endl;
            }
            alg_mergegraph->addPoint(dataset[row], row);
        });
        end1 = std::chrono::high_resolution_clock::now();
        std::cout << "HNSW initial insertion time: " 
                << std::chrono::duration_cast<std::chrono::microseconds>(end1 - start1).count() / 1e6 
                << " seconds" << std::endl;
        alg_mergegraph->saveIndex(save_path + dataset_name+ std::to_string(M)+"_"+ std::to_string(ef_construction) +"original.bin");
    }

    if (my_tool::fileExists(save_path + dataset_name + std::to_string(M)+"_"+ std::to_string(ef_construction) +"vamana.bin")) {
        std::cout<<"load index vamana build"<<std::endl;
        alg_mergegraph_vamana->loadIndex(save_path + dataset_name+ std::to_string(M)+"_"+ std::to_string(ef_construction) +"vamana.bin",&space);
    }else{
        start1 = std::chrono::high_resolution_clock::now();
        alg_mergegraph_vamana->maxlevel_ = 0;
        auto data_s = std::chrono::high_resolution_clock::now();
        my_tool::set_data(alg_mergegraph_vamana,dataset,0,dataset.size());
        auto data_e = std::chrono::high_resolution_clock::now();
        std::cout << "data copy time: " 
                << std::chrono::duration_cast<std::chrono::microseconds>(data_e - data_s).count() / 1e6 
                << " seconds" << std::endl;
        alg_mergegraph_vamana->vamana_build(50);
        alg_mergegraph_vamana->saveIndex(save_path + dataset_name+ std::to_string(M)+"_"+ std::to_string(ef_construction) +"vamana.bin");
    }
    end1 = std::chrono::high_resolution_clock::now();
    std::cout << "HNSW Vamana rebuild time: " 
            << std::chrono::duration_cast<std::chrono::microseconds>(end1 - start1).count() / 1e6 
            << " seconds" << std::endl;

    std::cout<<"max level is: "<< alg_mergegraph_vamana->maxlevel_<<", ep is:"<<alg_mergegraph_vamana->enterpoint_node_<<std::endl;

    // modify this to save
    std::string gt_save_path = "";

    int k = 100;

    std::vector<std::unordered_set<mergegraph::labeltype>> gt;
    if(my_tool::fileExists(gt_save_path+dataset_name+"gt"+std::to_string(k))){
        std::cout<<"load groundtruth"<<std::endl;
        my_tool::load_groundtruth(gt, gt_save_path+dataset_name+"gt"+std::to_string(k));
    }
    else {
        for(int i=0;i<alg_mergegraph->cur_element_count;i++){
            alg_brute->addPoint((const void *)alg_mergegraph->getDataByInternalId(i),alg_mergegraph->getExternalLabel(i));
        }
        gt = get_groundtruth(alg_brute,queries,k);
        my_tool::save_groundtruth(gt, gt_save_path+dataset_name+"gt"+std::to_string(k));
    }

    alg_mergegraph_vamana->loadIndex(save_path + dataset_name + std::to_string(M)+"_"+ std::to_string(ef_construction) +"vamana.bin",&space);
    my_tool::draw_recall_qps_graph(alg_mergegraph_vamana,gt,queries,k,"vamana HNSW refine twice");

    my_tool::draw_recall_qps_graph(alg_mergegraph,gt,queries,k,"HNSW");

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


std::vector<std::priority_queue<float>> get_groundtruth_with_distance(mergegraph::BruteforceSearch<float>* alg_brute,std::vector<float*> queries,int k){
    std::vector<std::priority_queue<float>> ground_truth(queries.size());
    #pragma omp parallel for num_threads(96) schedule(dynamic)
    for (size_t i = 0; i < queries.size(); ++i) {
        auto result_brute = alg_brute->searchKnn(queries[i], k);
        std::priority_queue<float> distance;

        while (!result_brute.empty()) {
            distance.emplace(result_brute.top().first);
            result_brute.pop();
        }

        ground_truth[i] = std::move(distance);
    }

    return ground_truth;
}