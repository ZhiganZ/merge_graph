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





double calculate_different_ef(mergegraph::HierarchicalNSW<float>* alg_hnsw,std::vector<std::unordered_set<mergegraph::labeltype>> groundtruth,std::vector<float*> queries,int ef,int M,int ef_construction,std::string name,int k=5);
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

int compare_times = 3;
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

// Function to read fvecs files remains the same
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

        dimension = d; // 假设所有向量维度一致

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

void load_diskann_graph_to_hnsw_level0(
    mergegraph::HierarchicalNSW<float>* hnsw_index,
    const std::string& diskann_index_prefix) {

    std::cout << "开始从 DiskANN 图文件加载图结构到 HNSW level 0..." << std::endl;
    
    // --- 步骤 1: 读取 DiskANN 图文件到内存 ---
    std::ifstream in(diskann_index_prefix, std::ios::binary);
    if (!in.is_open()) {
        throw std::runtime_error("无法打开 DiskANN 图文件: " + diskann_index_prefix);
    }

    // 读取 DiskANN 图文件的头部元数据
    size_t file_size;
    unsigned int max_degree, start_node_loc;
    size_t num_frozen_points;
    
    in.read(reinterpret_cast<char*>(&file_size), sizeof(size_t));
    in.read(reinterpret_cast<char*>(&max_degree), sizeof(unsigned int));
    in.read(reinterpret_cast<char*>(&start_node_loc), sizeof(unsigned int));
    in.read(reinterpret_cast<char*>(&num_frozen_points), sizeof(size_t));

    std::cout << "从 DiskANN 文件读取到元数据 -> MaxDegree: " << max_degree 
              << ", StartNode: " << start_node_loc 
              << ", FrozenPoints: " << num_frozen_points << std::endl;
    std::cout << "注意: HNSW 的入口点和更高层级连接将不会被此操作修改。" << std::endl;
    hnsw_index->enterpoint_node_ = start_node_loc;
    
    // 读取整个 DiskANN 图到内存中的邻接表
    std::vector<std::vector<unsigned int>> diskann_graph;
    // 预分配内存，DiskANN图包含数据点和冻结节点
    size_t total_points_in_diskann = hnsw_index->cur_element_count + num_frozen_points;
    diskann_graph.reserve(total_points_in_diskann);

    while (in.peek() != EOF) {
        unsigned int k;
        in.read(reinterpret_cast<char*>(&k), sizeof(unsigned int));
        if (in.gcount() == 0) break;

        std::vector<unsigned int> neighbors(k);
        if (k > 0) {
            in.read(reinterpret_cast<char*>(neighbors.data()), k * sizeof(unsigned int));
        }
        diskann_graph.push_back(std::move(neighbors));
    }
    in.close();
    std::cout << "从 DiskANN 文件中成功加载了 " << diskann_graph.size() << " 个节点的邻接关系。" << std::endl;
    
    // 验证节点数量是否匹配
    size_t num_elements_in_hnsw = hnsw_index->cur_element_count;
    if (diskann_graph.size() != num_elements_in_hnsw + num_frozen_points) {
        std::string error_msg = "错误: DiskANN 图中的节点数 (" + std::to_string(diskann_graph.size()) 
                              + ") 与 HNSW 元素数 (" + std::to_string(num_elements_in_hnsw) 
                              + ") + 冻结节点数 (" + std::to_string(num_frozen_points)
                              + ") 不匹配。无法继续操作。";
        throw std::runtime_error(error_msg);
    }
    
    // --- 步骤 2: 准备 HNSW 内部ID映射 ---
    // HNSW 内部使用 tableint ID，我们需要一个从外部标签到内部 ID 的映射。
    std::vector<mergegraph::tableint> external_to_internal_map(num_elements_in_hnsw);
    for (size_t i = 0; i < num_elements_in_hnsw; ++i) {
        external_to_internal_map[hnsw_index->getExternalLabel(i)] = i;
    }

    // --- 步骤 3: 遍历所有节点并重写 HNSW level 0 的连接 ---
    std::cout << "开始重写 HNSW level 0 的连接..." << std::endl;
    
    // 我们只关心存在于HNSW中的数据点，即DiskANN中的[0, num_elements_in_hnsw - 1]部分
    for (mergegraph::labeltype hnsw_label = 0; hnsw_label < num_elements_in_hnsw; ++hnsw_label) {
        // 在我们的假设中, HNSW label == DiskANN location
        unsigned int diskann_location = hnsw_label;

        // 获取当前标签对应的 HNSW 内部 ID
        mergegraph::tableint internal_id_to_update = external_to_internal_map[hnsw_label];
        
        // 从已加载的 DiskANN 图中获取该 location 的邻居列表
        const std::vector<unsigned int>& diskann_neighbor_locations = diskann_graph[diskann_location];
        
        // 将 DiskANN 邻居的 location 转换为 HNSW 的内部 ID
        std::vector<mergegraph::tableint> hnsw_neighbors_internal_ids;
        hnsw_neighbors_internal_ids.reserve(diskann_neighbor_locations.size());
        
        for (unsigned int neighbor_location : diskann_neighbor_locations) {
            // **关键步骤**: 忽略冻结节点，因为它们不在HNSW索引中
            if (neighbor_location < num_elements_in_hnsw) {
                // neighbor_location 同样对应一个 HNSW 的外部标签
                hnsw_neighbors_internal_ids.push_back(external_to_internal_map[neighbor_location]);
            }
        }
        
        // 获取 HNSW level 0 链表的指针
        mergegraph::linklistsizeint* ll_cur = hnsw_index->get_linklist0(internal_id_to_update);
        
        // 检查邻居数量是否超过 HNSW level 0 的容量 (maxM0_)
        if (hnsw_neighbors_internal_ids.size() > hnsw_index->maxM0_) {
            // 可以选择性地打印警告
            // std::cerr << "警告: 节点 " << hnsw_label << " 的邻居数将被截断。" << std::endl;
            hnsw_neighbors_internal_ids.resize(hnsw_index->maxM0_);
        }
        
        // 加锁以安全地修改链表
        std::unique_lock<std::mutex> lock(hnsw_index->link_list_locks_[internal_id_to_update]);
        
        // 设置新的邻居数量
        hnsw_index->setListCount(ll_cur, hnsw_neighbors_internal_ids.size());
        
        // 获取数据指针并复制新的邻居列表（完成覆盖）
        mergegraph::tableint* data_ptr = (mergegraph::tableint*)(ll_cur + 1);
        if (!hnsw_neighbors_internal_ids.empty()) {
            std::copy(hnsw_neighbors_internal_ids.begin(), hnsw_neighbors_internal_ids.end(), data_ptr);
        }
    }
    
    std::cout << "HNSW level 0 的连接已成功从 DiskANN 图文件更新。" << std::endl;
}

void load_nsg_graph_to_hnsw_level0(
    mergegraph::HierarchicalNSW<float>* hnsw_index,
    const std::string& nsg_graph_filename) {

    std::cout << "开始从 NSG 图文件加载图结构到 HNSW level 0..." << std::endl;
    
    // --- 步骤 1: 读取 NSG 图文件到内存 ---
    std::ifstream in(nsg_graph_filename, std::ios::binary);
    if (!in.is_open()) {
        throw std::runtime_error("无法打开 NSG 图文件: " + nsg_graph_filename);
    }

    unsigned int nsg_width, nsg_ep;
    in.read(reinterpret_cast<char*>(&nsg_width), sizeof(unsigned int));
    in.read(reinterpret_cast<char*>(&nsg_ep), sizeof(unsigned int));
    // hnsw_index->enterpoint_node_ = hnsw_index->label_lookup_[nsg_ep];
    std::cout << "从 NSG 文件读取到元数据 -> Width: " << nsg_width << ", EntryPoint: " << nsg_ep << std::endl;
    std::cout << "注意: HNSW 的入口点和更高层级连接将不会被此操作修改。" << std::endl;
    
    // 读取整个 NSG 图到内存中的邻接表
    std::vector<std::vector<unsigned int>> nsg_graph;
    // 预分配内存可以提高效率，如果我们能提前知道节点数
    nsg_graph.reserve(hnsw_index->cur_element_count); 

    while (in.peek() != EOF) {
        unsigned int k;
        in.read(reinterpret_cast<char*>(&k), sizeof(unsigned int));
        if (in.gcount() == 0) break; // 检查是否真的读到了数据

        std::vector<unsigned int> neighbors(k);
        if (k > 0) {
            in.read(reinterpret_cast<char*>(neighbors.data()), k * sizeof(unsigned int));
        }
        nsg_graph.push_back(std::move(neighbors));
    }
    in.close();
    std::cout << "从 NSG 文件中成功加载了 " << nsg_graph.size() << " 个节点的邻接关系。" << std::endl;

    size_t num_elements = hnsw_index->cur_element_count;
    if (nsg_graph.size() != num_elements) {
        std::string error_msg = "错误: NSG 图中的节点数 (" + std::to_string(nsg_graph.size()) 
                              + ") 与 HNSW 中的元素数 (" + std::to_string(num_elements) 
                              + ") 不匹配。无法继续操作。";
        throw std::runtime_error(error_msg);
    }
    
    // --- 步骤 2: 准备 HNSW 内部ID映射 ---
    // 由于 HNSW 内部使用 tableint ID，我们需要一个从外部标签到内部 ID 的映射
    // 即使标签是连续的，HNSW 内部ID的顺序也可能因多线程添加而与标签顺序不同
    std::vector<mergegraph::tableint> external_to_internal_map(num_elements);
    for (size_t i = 0; i < num_elements; ++i) {
        external_to_internal_map[hnsw_index->getExternalLabel(i)] = i;
    }

    // --- 步骤 3: 遍历所有节点并重写 HNSW level 0 的连接 ---
    std::cout << "开始重写 HNSW level 0 的连接..." << std::endl;
    for (mergegraph::labeltype current_label = 0; current_label < num_elements; ++current_label) {
        // 获取当前标签对应的 HNSW 内部 ID
        mergegraph::tableint internal_id_to_update = external_to_internal_map[current_label];
        
        // 从已加载的 NSG 图中获取该标签的邻居列表
        const std::vector<unsigned int>& nsg_neighbors_labels = nsg_graph[current_label];
        
        // 将 NSG 邻居的外部标签转换为 HNSW 的内部 ID
        std::vector<mergegraph::tableint> hnsw_neighbors_internal_ids;
        hnsw_neighbors_internal_ids.reserve(nsg_neighbors_labels.size());
        
        for (unsigned int neighbor_label : nsg_neighbors_labels) {
             // 因为标签是连续且匹配的，所以可以直接查找
            hnsw_neighbors_internal_ids.push_back(external_to_internal_map[neighbor_label]);
        }
        
        // 获取 HNSW level 0 链表的指针
        mergegraph::linklistsizeint* ll_cur = hnsw_index->get_linklist0(internal_id_to_update);
        
        // 检查邻居数量是否超过 HNSW level 0 的容量 (maxM0_)
        if (hnsw_neighbors_internal_ids.size() > hnsw_index->maxM0_) {
            //  std::cerr << "警告: 节点 " << current_label << " 的 NSG 邻居数 (" 
                    //    << hnsw_neighbors_internal_ids.size() 
                    //    << ") 超过 HNSW level 0 的最大容量 (" << hnsw_index->maxM0_
                    //    << ")。邻居列表将被截断。" << std::endl;
             hnsw_neighbors_internal_ids.resize(hnsw_index->maxM0_);
        }
        
        // 加锁以安全地修改链表
        std::unique_lock<std::mutex> lock(hnsw_index->link_list_locks_[internal_id_to_update]);
        
        // 设置新的邻居数量
        hnsw_index->setListCount(ll_cur, hnsw_neighbors_internal_ids.size());
        
        // 获取数据指针并复制新的邻居列表（完成覆盖）
        mergegraph::tableint* data_ptr = (mergegraph::tableint*)(ll_cur + 1);
        if (!hnsw_neighbors_internal_ids.empty()) {
            std::copy(hnsw_neighbors_internal_ids.begin(), hnsw_neighbors_internal_ids.end(), data_ptr);
        }
    }
    
    std::cout << "HNSW level 0 的连接已成功从 NSG 图文件更新。" << std::endl;
}

int vamana_l = 70;

float merge_alpha = 1.2;

void build(mergegraph::HierarchicalNSW<float>* hnsw_index,std::vector<float *> dataset,int begin,int end){
    auto start = std::chrono::high_resolution_clock::now();
    my_tool::set_data(hnsw_index,dataset,begin,end);
    hnsw_index->vamana_build(vamana_l,merge_alpha);
    std::cout << "build time: " 
        << std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::high_resolution_clock::now() - start).count() / 1e6 
        << " seconds" << std::endl;
}

void build_hnsw(mergegraph::HierarchicalNSW<float>* hnsw_index,std::vector<float *> dataset,int begin,int end){
    auto start = std::chrono::high_resolution_clock::now();
    #pragma omp parallel for
    for(int i = begin ; i < end ; ++i){
        hnsw_index->addPoint(dataset[i],i);
    }
    std::cout << "build time: " 
        << std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::high_resolution_clock::now() - start).count() / 1e6 
        << " seconds" << std::endl;
    hnsw_index->flatten_to_level0();
}



int main(int argc,char* argv[]) {
    std::cout.precision(6);
    // File paths
    // sift1m
    // std::string dataset_name = "sift1m";
    // std::string dataset_file = "/home/zhouzhigan/dynamic_knn/dataset/sift/sift_base.fvecs";   // 数据集文件路径
    // std::string query_file = "/home/zhouzhigan/dynamic_knn/dataset/sift/sift_query.fvecs";    // 查询集文件路径

    // crawl
    // std::string dataset_file = "/home/zhouzhigan/dynamic_knn/dataset/crawl/crawl_base.fvecs";   // 数据集文件路径
    // std::string query_file = "/home/zhouzhigan/dynamic_knn/dataset/crawl/crawl_query.fvecs";    // 查询集文件路径

    //fasion-mnist
    // std::string dataset_file = "/home/zhouzhigan/dataset/fashion_mnist_784/fasion_mnist_784_base.fvecs";   // 数据集文件路径
    // std::string query_file = "/home/zhouzhigan/dataset/fashion_mnist_784/fasion_mnist_784_query.fvecs";    // 查询集文件路径

    // gist
    // std::string dataset_name = "gist";
    // std::string dataset_file = "/home/zhouzhigan/dataset/gist/gist_base.fvecs";   // 数据集文件路径
    // std::string query_file = "/home/zhouzhigan/dataset/gist/gist_query.fvecs";    // 查询集文件路径

    //glove-1.2m
    // std::string dataset_file = "/home/zhouzhigan/dynamic_knn/dataset/glove1.2m/glove1.2m_base.fvecs";   // 数据集文件路径
    // std::string query_file = "/home/zhouzhigan/dynamic_knn/dataset/glove1.2m/glove1.2m_query.fvecs";    // 查询集文件路径

    //glove-100
    // std::string dataset_file = "/home/zhouzhigan/dynamic_knn/dataset/glove-100/glove-100_base.fvecs";   // 数据集文件路径
    // std::string query_file = "/home/zhouzhigan/dynamic_knn/dataset/glove-100/glove-100_query.fvecs";    // 查询集文件路径

    //msong
    // std::string dataset_name = "msong";
    // std::string dataset_file = "/home/zhouzhigan/dataset/msong/msong_base.fvecs";   // 数据集文件路径
    // std::string query_file = "/home/zhouzhigan/dataset/msong/msong_query.fvecs";    // 查询集文件路径

    //sift-10m
    // std::string dataset_file = "/home/zhouzhigan/dataset/sift10m/sift10m_base.fvecs";   // 数据集文件路径
    // std::string query_file = "/home/zhouzhigan/dataset/sift10m/sift10m_query.fvecs";    // 查询集文件路径

    //tiny5m
    // std::string dataset_file = "/home/zhouzhigan/dataset/tiny5m/tiny5m_base.fvecs";   // 数据集文件路径
    // std::string query_file = "/home/zhouzhigan/dataset/tiny5m/tiny5m_query.fvecs";    // 查询集文件路径

    //word2vec using alpha=1.2 reprune_ 
    // std::string dataset_name = "word2vec";
    // std::string dataset_file = "/home/zhouzhigan/dataset/word2vec/word2vec_base.fvecs";   // 数据集文件路径
    // std::string query_file = "/home/zhouzhigan/dataset/word2vec/word2vec_query.fvecs";    // 查询集文件路径

    //imagenet
    // std::string dataset_name = "imagenet";
    // std::string dataset_file = "/home/zhouzhigan/dynamic_knn/dataset/imagenet/imagenet_base.fvecs";   // 数据集文件路径
    // std::string query_file = "/home/zhouzhigan/dynamic_knn/dataset/imagenet/imagenet_query.fvecs";    // 查询集文件路径

    //deep1m
    std::string dataset_name ;
    std::string dataset_file ;
    std::string query_file;

    //deep 10m
    // std::string dataset_name = "deep10m";
    // std::string dataset_file = "/home/zhouzhigan/dynamic_knn/dataset/deep1M/deep10M_base.fvecs";   // 数据集文件路径
    // std::string query_file = "/home/zhouzhigan/dynamic_knn/dataset/deep1M/deep10M_query.fvecs";   // 数据集文件路径

    // random
    // std::string dataset_name = "random";
    // std::string dataset_file = "/home/zhouzhigan/script/tmp/base.fvecs";
    // std::string query_file = "/home/zhouzhigan/script/tmp/query.fvecs";
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
        if(!read_bvecs_)
        queries = read_fvecs(query_file, dim_queries);
        else queries = read_bvecs(query_file, dim_queries);
    } catch (const std::exception& e) {
        std::cerr << "Error reading queries: " << e.what() << std::endl;
        return -1;
    }

    // // Check that dataset and queries have the same dimension
    // if (dim_dataset != dim_queries) {
    //     std::cerr << "Dataset and queries have different dimensions." << std::endl;
    //     return -1;
    // }

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
    // omp_set_num_threads();

    std::string save_path = "/home/zhouzhigan/dynamic_knn/tmp/index/";

    int ef_1 = ef_construction ;
    vamana_l = ef_construction ;

    int seed = 100;
    mergegraph::L2Space space(dim);
    // mergegraph::L2SpaceEE space1(dim);
    mergegraph::HierarchicalNSW<float>* alg_hnsw = new mergegraph::HierarchicalNSW<float>(&space, max_elements, M, ef_construction, seed, true);

    mergegraph::HierarchicalNSW<float>* alg_hnsw_merge = new mergegraph::HierarchicalNSW<float>(&space, max_elements, M, ef_1, seed, true);

    mergegraph::HierarchicalNSW<float>* alg_hnsw_merge_big = new mergegraph::HierarchicalNSW<float>(&space, max_elements, M, ef_1, seed, true);

    mergegraph::HierarchicalNSW<float>* alg_hnsw_first_half = new mergegraph::HierarchicalNSW<float>(&space, max_elements/2, M, ef_1, seed, true);
    mergegraph::HierarchicalNSW<float>* alg_hnsw_second_half = new mergegraph::HierarchicalNSW<float>(&space, max_elements - max_elements/2, M, ef_1, seed, true);
    
    alg_hnsw->setEf(ef);
    alg_hnsw->allow_replace_deleted_ = true;

    mergegraph::BruteforceSearch<float>* alg_brute = new mergegraph::BruteforceSearch<float>(&space, max_elements);

    std::string gt_save_path = "/home/zhouzhigan/dynamic_knn/tmp/gt/prune_gt/";

    int k = 10;
    // alg_brute->fstdistfunc_ = mergegraph::InnerProduct;

    // auto groundtruth =my_tool::get_groundtruth(alg_brute, queries, 100);

    // auto gt = get_groundtruth(alg_brute,queries,k);

    std::vector<std::unordered_set<mergegraph::labeltype>> gt;
    if(my_tool::fileExists(gt_save_path+dataset_name+"gt"+std::to_string(k))){
        std::cout<<"load groundtruth"<<std::endl;
        my_tool::load_groundtruth(gt, gt_save_path+dataset_name+"gt"+std::to_string(k));
    }
    else {
        for(int i=0;i<dataset.size();i++){
            alg_brute->addPoint(dataset[i],i);
        }
        gt = get_groundtruth(alg_brute,queries,k);
        my_tool::save_groundtruth(gt, gt_save_path+dataset_name+"gt"+std::to_string(k));
        delete alg_brute;
    }


    auto start1 = std::chrono::high_resolution_clock::now();
    auto end1 = std::chrono::high_resolution_clock::now();
    if (my_tool::fileExists(save_path + dataset_name + std::to_string(M)+"_"+ std::to_string(ef_construction) +"original.bin")) {
        std::cout<<"load index hnsw"<<std::endl;
        alg_hnsw->loadIndex(save_path + dataset_name+ std::to_string(M)+"_"+ std::to_string(ef_construction) +"original.bin",&space);
    }else{
        // std::cout<<"start compute gt"<<std::endl;
        start1 = std::chrono::high_resolution_clock::now();
        // alg_hnsw->addPoint(dataset[123742], 123742);
        ParallelFor(0, max_elements, num_threads, [&](size_t row, size_t threadId) {
            
            if(row*10%((max_elements/10)*10)==0){
                auto tmp = std::chrono::high_resolution_clock::now();
                std::cout<<"complete "<<row*100/((max_elements/100)*100)<<"%"<<std::endl;
                std::cout << "Now time: " 
                << std::chrono::duration_cast<std::chrono::microseconds>(tmp - start1).count() / 1e6 
                << " seconds" << std::endl;
            }
            // if(row!=123742)
            alg_hnsw->addPoint(dataset[row], row);
        });

        // alg_hnsw->set_alpha(1.2);
        // alg_hnsw->resize_M(M*2.6);
        // alg_hnsw->maxlevel_ = 0 ;
        // my_tool::set_data(alg_hnsw,dataset,0,max_elements);
        // alg_hnsw->vamana_build(vamana_l);
        // alg_hnsw->quick_check();
        // alg_hnsw->clean_after_build(M*2);
        // alg_hnsw->quick_check();
        // alg_hnsw->resize_M(M*2);

        end1 = std::chrono::high_resolution_clock::now();
        std::cout << "HNSW initial insertion time: " 
                << std::chrono::duration_cast<std::chrono::microseconds>(end1 - start1).count() / 1e6 
                << " seconds" << std::endl;
        alg_hnsw->saveIndex(save_path + dataset_name+ std::to_string(M)+"_"+ std::to_string(ef_construction) +"original.bin");
    }
    // auto lids = alg_hnsw->estimateLIDsMLEAll(100);
    // std::sort(lids.begin(),lids.end(),[&](std::pair<float,mergegraph::labeltype>a,std::pair<float,mergegraph::labeltype>b){return a.first<b.first;});
    // std::cout<<"lid[0] is: "<<lids[0].first<<"lids[N/10] is: "<<lids[alg_hnsw->cur_element_count/10].first<<", lids[N] is: "<<lids[alg_hnsw->cur_element_count-1].first <<std::endl;


    if (my_tool::fileExists(save_path + dataset_name + std::to_string(M)+"_"+ std::to_string(ef_construction) +"first_half.bin")) {
        std::cout<<"load index hnsw"<<std::endl;
        alg_hnsw_first_half->loadIndex(save_path + dataset_name+ std::to_string(M)+"_"+ std::to_string(ef_construction) +"first_half.bin",&space);
    }else{
        start1 = std::chrono::high_resolution_clock::now();

        ParallelFor(0, max_elements/2, num_threads, [&](size_t row, size_t threadId) {
            alg_hnsw_first_half->addPoint(dataset[row], row);
        });
        // alg_hnsw_first_half->maxlevel_ = 0 ;
        // my_tool::set_data(alg_hnsw_first_half,dataset,0,max_elements/2);
        // alg_hnsw_first_half->vamana_build(vamana_l);


        end1 = std::chrono::high_resolution_clock::now();
        std::cout << "first half initial insertion time: " 
                << std::chrono::duration_cast<std::chrono::microseconds>(end1 - start1).count() / 1e6 
                << " seconds" << std::endl;
        alg_hnsw_first_half->saveIndex(save_path + dataset_name+ std::to_string(M)+"_"+ std::to_string(ef_construction) +"first_half.bin");
    }


    if (my_tool::fileExists(save_path + dataset_name + std::to_string(M)+"_"+ std::to_string(ef_construction) +"second_half.bin")) {
        std::cout<<"load index hnsw"<<std::endl;
        alg_hnsw_second_half->loadIndex(save_path + dataset_name+ std::to_string(M)+"_"+ std::to_string(ef_construction) +"second_half.bin",&space);
    }else{
        start1 = std::chrono::high_resolution_clock::now();
        ParallelFor(max_elements/2, max_elements, num_threads, [&](size_t row, size_t threadId) {
            alg_hnsw_second_half->addPoint(dataset[row], row);
        });
        // alg_hnsw_second_half->maxlevel_ = 0 ;
        // my_tool::set_data(alg_hnsw_second_half,dataset,max_elements/2,dataset.size());
        // alg_hnsw_second_half->vamana_build(vamana_l);


        end1 = std::chrono::high_resolution_clock::now();
        std::cout << "second half initial insertion time: " 
                << std::chrono::duration_cast<std::chrono::microseconds>(end1 - start1).count() / 1e6 
                << " seconds" << std::endl;
        alg_hnsw_second_half->saveIndex(save_path + dataset_name+ std::to_string(M)+"_"+ std::to_string(ef_construction) +"second_half.bin");
    }


    std::vector<mergegraph::labeltype> delete_ids(max_elements);
    for(mergegraph::labeltype i = 0 ; i < max_elements ; ++i){
        delete_ids[i]=i;
    }

    std::shuffle(delete_ids.begin(),delete_ids.end(),gen);
    delete_ids.resize(0);
    for(mergegraph::labeltype i = 0 ; i < delete_ids.size() ; ++i){
        alg_hnsw->markDelete(delete_ids[i]);
        if(delete_ids[i]<max_elements/2)
            alg_hnsw_first_half->markDelete(delete_ids[i]);
        else alg_hnsw_second_half->markDelete(delete_ids[i]);
    }
    std::vector<std::vector<size_t>> labels;
    std::vector<mergegraph::HierarchicalNSW<float>*> vamana_merge_vector ;
    mergegraph::HierarchicalNSW<float>* vamana_merged = new mergegraph::HierarchicalNSW<float>(&space, dataset.size(), M, ef_1, seed, true);

    if(0)
    {
        std::vector<mergegraph::HierarchicalNSW<float>*> alg_hnsw_vector_16 ;
        std::vector<mergegraph::HierarchicalNSW<float>*> alg_hnsw_vector_8 ;
        std::vector<mergegraph::HierarchicalNSW<float>*> alg_hnsw_vector_4 ;
        std::vector<mergegraph::HierarchicalNSW<float>*> alg_hnsw_vector_2 ;

        for(int i = 0 ; i < 16 ; ++i){
            alg_hnsw_vector_16.emplace_back(new mergegraph::HierarchicalNSW<float>(&space, max_elements/16+1, M, ef_1, seed, true));
        }
        for(int i = 0 ; i < 16 ; ++i){
            build(alg_hnsw_vector_16[i],dataset,i*((max_elements+15)/16),(i+1)*((max_elements+15)/16)<max_elements?(i+1)*((max_elements+15)/16):max_elements);
        }
        // for(int i = 0 ; i < 4 ; ++i){
        //     alg_hnsw_vector_4.emplace_back(new mergegraph::HierarchicalNSW<float>(&space, max_elements/4+1, M, ef_1, seed, true));
        // }
        // for(int i = 0 ; i < 4 ; ++i){
        //     // build(alg_hnsw_vector_4[i],dataset,i*((max_elements+3)/4),(i+1)*((max_elements+3)/4)<max_elements?(i+1)*((max_elements+3)/4):max_elements);
        //     build_hnsw(alg_hnsw_vector_4[i],dataset,i*((max_elements+3)/4),(i+1)*((max_elements+3)/4)<max_elements?(i+1)*((max_elements+3)/4):max_elements);
        // }
        for(int i = 0 ; i < 8 ; ++i){
            alg_hnsw_vector_8.emplace_back(new mergegraph::HierarchicalNSW<float>(&space, max_elements/8+2, M, ef_1, seed, true));
            alg_hnsw_vector_8[i]->set_alpha(merge_alpha);
        }
        for(int i = 0 ; i < 8 ; ++i){
            auto start = std::chrono::high_resolution_clock::now();
            my_tool::hnsw_merge_Skeleton_bridged_Local_Repair(alg_hnsw_vector_8[i],alg_hnsw_vector_16[i*2],alg_hnsw_vector_16[i*2+1],delete_ids);
            // alg_hnsw_vector_8[i]->Global_Repair();
            // alg_hnsw_vector_8[i]->reprune();

            alg_hnsw_vector_8[i]->fgim();
            alg_hnsw_vector_8[i]->symmetrizeAndPruneAllLevelsMT();
            std::cout << "16->8 merge time: "
                << std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::high_resolution_clock::now() - start).count() / 1e6 
                << " seconds" << std::endl;
            delete alg_hnsw_vector_16[i*2];
            delete alg_hnsw_vector_16[i*2+1];
        }
        for(int i = 0 ; i < 4 ; ++i){
            alg_hnsw_vector_4.emplace_back(new mergegraph::HierarchicalNSW<float>(&space, max_elements/4+4, M, ef_1, seed, true));
            alg_hnsw_vector_4[i]->set_alpha(merge_alpha);
        }
        for(int i = 0 ; i < 4 ; ++i){
            auto start = std::chrono::high_resolution_clock::now();
            my_tool::hnsw_merge_Skeleton_bridged_Local_Repair(alg_hnsw_vector_4[i],alg_hnsw_vector_8[i*2],alg_hnsw_vector_8[i*2+1],delete_ids);
            // alg_hnsw_vector_4[i]->Global_Repair();
            // alg_hnsw_vector_4[i]->reprune();

            alg_hnsw_vector_4[i]->fgim();
            alg_hnsw_vector_4[i]->symmetrizeAndPruneAllLevelsMT();
            std::cout << "8->4 merge time: "
                << std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::high_resolution_clock::now() - start).count() / 1e6 
                << " seconds" << std::endl;
            delete alg_hnsw_vector_8[i*2];
            delete alg_hnsw_vector_8[i*2+1];
        }
        for(int i = 0 ; i < 2 ; ++i){
            alg_hnsw_vector_2.emplace_back(new mergegraph::HierarchicalNSW<float>(&space, max_elements/2+8, M, ef_1, seed, true));
            alg_hnsw_vector_2[i]->set_alpha(merge_alpha);
        }
        for(int i = 0 ; i < 2 ; ++i){
            auto start = std::chrono::high_resolution_clock::now();
            my_tool::hnsw_merge_Skeleton_bridged_Local_Repair(alg_hnsw_vector_2[i],alg_hnsw_vector_4[i*2],alg_hnsw_vector_4[i*2+1],delete_ids);
            // alg_hnsw_vector_2[i]->Global_Repair();
            // alg_hnsw_vector_2[i]->reprune();  

            alg_hnsw_vector_2[i]->fgim();
            alg_hnsw_vector_2[i]->symmetrizeAndPruneAllLevelsMT();
            std::cout << "4->2 merge time: "
                << std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::high_resolution_clock::now() - start).count() / 1e6 
                << " seconds" << std::endl;
            delete alg_hnsw_vector_4[i*2];
            delete alg_hnsw_vector_4[i*2+1];
        }

        alg_hnsw_merge_big->set_alpha(merge_alpha);
        auto start = std::chrono::high_resolution_clock::now();
        my_tool::hnsw_merge_Skeleton_bridged_Local_Repair(alg_hnsw_merge_big,alg_hnsw_vector_2[0],alg_hnsw_vector_2[1],delete_ids);
        // my_tool::hnsw_merge_Skeleton_bridged_Local_Repair(alg_hnsw_merge_big,alg_hnsw_vector_16,delete_ids);
        // my_tool::hnsw_merge_Skeleton_bridged_Local_Repair(alg_hnsw_merge_big,alg_hnsw_vector_4,delete_ids);
        // alg_hnsw_merge_big->Global_Repair();
        // alg_hnsw_merge_big->reprune();  

        alg_hnsw_merge_big->fgim(1.0f,1.5f);
        auto bi_add_start = mergegraph::get_time_now();
        alg_hnsw_merge_big->symmetrizeAndPruneAllLevelsMT();
        mergegraph::print_using_time("bi add",bi_add_start);
        auto start2 = mergegraph::get_time_now();
        alg_hnsw_merge_big->random_high_level();
        mergegraph::print_using_time("rebuild high level",start2);
        alg_hnsw_merge_big->rebuild_upper_layers_mt();
        std::cout << "2->1 merge time: "
            << std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::high_resolution_clock::now() - start).count() / 1e6 
            << " seconds" << std::endl;

        my_tool::draw_recall_qps_graph(alg_hnsw_merge_big,gt,queries,k,"4 -> 1 HNSW");
    }

    if(1){
        alg_hnsw_merge_big->set_alpha(merge_alpha);
        alg_hnsw_merge_big->flatten_to_level0();
        for(int i = 2 ; i<= 18 ; i+=4){
            int num_sub_graphs = i;
            int point_in_sub_graph = max_elements/num_sub_graphs;
            std::vector<int> offset_of_graph;
            // offset_of_graph.emplace_back(0);
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
                build(sub_graphs[i],dataset,offset_of_graph[i],offset_of_graph[i+1]);
            }

            // my_tool::hnsw_merge_cross_all_sub_graph(alg_hnsw_merge_big,sub_graphs,delete_ids);
            my_tool::hnsw_merge_cross_all_sub_graph(alg_hnsw_merge_big,sub_graphs,delete_ids);
            // alg_hnsw_merge_big->flatten_to_level0();
            // alg_hnsw_merge_big->random_high_level();
            // alg_hnsw_merge_big->rebuild_upper_layers_mt();
            alg_hnsw->enterpoint_node_ = alg_hnsw->get_navi_point();
            my_tool::draw_recall_qps_graph(alg_hnsw_merge_big,gt,queries,k,std::to_string(num_sub_graphs)+" -> 1 HNSW");

        }

    }
    return 0;



    // start1 = mergegraph::get_time_now();
    // my_tool::set_data(alg_hnsw_merge_big,dataset,0,max_elements);
    // alg_hnsw_merge_big->enterpoint_node_ = alg_hnsw->get_navi_point(false);
    // alg_hnsw_merge_big->random_edge();
    // alg_hnsw_merge_big->fgim();
    // alg_hnsw_merge_big->symmetrizeAndPruneAllLevelsMT();
    // mergegraph::print_using_time("random build",start1);


    // for(mergegraph::labeltype i = 0 ; i < delete_ids.size() ; ++i){
    //     alg_hnsw_merge->markDelete(delete_ids[i]);
    // }


    std::vector<int> sample_rates{1,10,20,50,100};
    alg_hnsw_merge->alpha = 1.2f;
    alg_hnsw_first_half->flatten_to_level0();
    alg_hnsw_second_half->flatten_to_level0();
    for(auto sample_rate:sample_rates){
        my_tool::hnsw_merge_Skeleton_bridged_Local_Repair(alg_hnsw_merge,alg_hnsw_first_half,alg_hnsw_second_half,delete_ids,sample_rate);
        alg_hnsw_merge->enterpoint_node_ = alg_hnsw_merge->get_navi_point(true);
        my_tool::draw_recall_qps_graph(alg_hnsw_merge,gt,queries,k,"sample "+std::to_string(sample_rate)+"% graph");
    }

    my_tool::draw_recall_qps_graph(alg_hnsw_first_half,alg_hnsw_second_half,gt,queries,k,"two HNSW");
    alg_hnsw->random_high_level();
    alg_hnsw->rebuild_upper_layers_mt();
    my_tool::draw_recall_qps_graph(alg_hnsw,gt,queries,k,"vamana");

    
    alg_hnsw_merge->alpha = 1.2f;
    start1 = std::chrono::high_resolution_clock::now();
    my_tool::hnsw_merge_Skeleton_bridged_Local_Repair(alg_hnsw_merge,alg_hnsw_first_half,alg_hnsw_second_half,delete_ids,1);
    // my_tool::cal_best_recall(alg_hnsw_merge,gt,queries,k,"two HNSW");

    alg_hnsw_merge->fgim();
    auto bi_add_start = mergegraph::get_time_now();
    alg_hnsw_merge->symmetrizeAndPruneAllLevelsMT();
    mergegraph::print_using_time("bi add",bi_add_start);
    auto start2 = mergegraph::get_time_now();
    alg_hnsw_merge->random_high_level();
    alg_hnsw_merge->rebuild_upper_layers_mt();
    mergegraph::print_using_time("rebuild high level",start2);
    end1 = std::chrono::high_resolution_clock::now();
    std::cout << "reprune time: " 
            << std::chrono::duration_cast<std::chrono::microseconds>(end1 - start1).count() / 1e6 
            << " seconds" << std::endl;
    my_tool::draw_recall_qps_graph(alg_hnsw_merge,gt,queries,k,"merge HNSW");


    // alg_hnsw_merge->quick_check();
    // alg_hnsw->quick_check();

    // if(alg_hnsw_merge_big->cur_element_count>10000)
    //     alg_hnsw_merge_big->analyzeIndex();
    // alg_hnsw_merge->analyzeIndex();
    // alg_hnsw->analyzeIndex();

    std::cout.precision(6);
    // my_tool::draw_recall_qps_graph(alg_hnsw_first_half,alg_hnsw_second_half,gt,queries,k,"two HNSW");
    // my_tool::draw_recall_distance_compuatation_graph(alg_hnsw_first_half,alg_hnsw_second_half,gt,queries,k,"two HNSW");
    if(alg_hnsw_merge_big->cur_element_count>10000){
        // alg_hnsw_merge_big->OptimizeGraph();
        my_tool::draw_recall_qps_graph(alg_hnsw_merge_big,gt,queries,k,"4 -> 1 HNSW");
    }


    // my_tool::draw_recall_qps_graph(alg_hnsw_merge,gt,queries,k,"H merge HNSW");
    // alg_hnsw_merge->flatten_to_level0();
    // alg_hnsw_merge->OptimizeGraph();
    my_tool::draw_recall_qps_graph(alg_hnsw_merge,gt,queries,k,"merge HNSW");
    // load_diskann_graph_to_hnsw_level0(alg_hnsw,"/home/zhouzhigan/k-NN/jni/external/DiskANN/build/data/sift/index_gist_learn_R48_L400_A1.2");

    // alg_hnsw->quick_check();
    // my_tool::draw_recall_distance_compuatation_graph(alg_hnsw_merge,gt,queries,k,"merge HNSW");
    // if(alg_hnsw->maxlevel_==0){
    //     alg_hnsw->flatten_to_level0();
    //     alg_hnsw->random_high_level();
    //     alg_hnsw->rebuild_upper_layers_mt();
    // }
    // alg_hnsw->OptimizeGraph();
    // alg_hnsw->OptimizeGraph();
    my_tool::draw_recall_qps_graph(alg_hnsw,gt,queries,k,"my vamana");
    if(alg_hnsw->maxlevel_>0){
        return 0;
    }
    alg_hnsw->random_high_level();
    alg_hnsw->rebuild_upper_layers_mt();

    my_tool::draw_recall_qps_graph(alg_hnsw,gt,queries,k,"my vamana with multi layer");
    // Clean up
    for (auto vec : dataset) {
        free(vec);
    }
    for (auto vec : queries) {
        free(vec);
    }

    delete alg_hnsw;
    // delete alg_hnsw_1;
    // delete alg_hnsw_2;
    // delete alg_hnsw_3;
    // delete alg_hnsw_4;
    // delete alg_hnsw_5;
    // delete alg_hnsw_6;
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