#include "hnswlib.h"
#include "assert.h"

#include <chrono>
#include <cstddef>
#include <random>
#include <cstdlib>
#include <stdexcept>
#include <set>
#include <string>
#include <unistd.h>
#include <unordered_map>
#include <unordered_set>
#include <fstream>
#include <iostream>
#include <utility>
#include <vector>
#include <thread>
#include <atomic>
#include <mutex>
#include <algorithm>
#include <sys/stat.h>

namespace my_tool {

    constexpr int times_ = 4;

    int vamana_l = 400;

    float merge_alpha = 1.2;

    void set_data(mergegraph::HierarchicalNSW<float>* alg_mergegraph,std::vector<float*> vectors,int start , int end){
        alg_mergegraph->cur_element_count = end-start;
        int max_l = 0 ;
        for(size_t i = start ; i < end ; ++i){
            alg_mergegraph->label_lookup_[i] = (mergegraph::tableint)(i-start);

            memcpy(alg_mergegraph->getExternalLabeLp((mergegraph::labeltype)(i-start)), &i, sizeof(mergegraph::labeltype));

            alg_mergegraph->element_levels_[i-start] = 0;

            memcpy(alg_mergegraph->getDataByInternalId(i-start), vectors[i], alg_mergegraph->data_size_);

        }
    }

    // TODO:
    void set_data_with_label(mergegraph::HierarchicalNSW<float>* alg_mergegraph,std::vector<float*> vectors,std::vector<mergegraph::labeltype>& label){
        alg_mergegraph->cur_element_count = label.size();
        int max_l = 0 ;
        for(size_t i = 0 ; i < label.size() ; ++i){
            alg_mergegraph->label_lookup_[label[i]] = (mergegraph::tableint)(i);

            memcpy(alg_mergegraph->getExternalLabeLp((mergegraph::labeltype)(i)), &label[i], sizeof(mergegraph::labeltype));

            alg_mergegraph->element_levels_[i] = 0;

            memcpy(alg_mergegraph->getDataByInternalId(i), vectors[label[i]], alg_mergegraph->data_size_);

        }
    }
    void build_vamana(mergegraph::HierarchicalNSW<float>* hnsw_index,std::vector<float *> dataset,int begin,int end){
        auto start = std::chrono::high_resolution_clock::now();
        set_data(hnsw_index,dataset,begin,end);
        hnsw_index->vamana_build(vamana_l,merge_alpha);
        std::cout << "build time: " 
            << std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::high_resolution_clock::now() - start).count() / 1e6 
            << " seconds" << std::endl;
    }

    void build_hnsw(mergegraph::HierarchicalNSW<float>* hnsw_index,std::vector<float *> dataset,int begin,int end){
        auto start = std::chrono::high_resolution_clock::now();
        std::cout<<"start to build HNSW, "<<end-begin<<" points"<<std::endl;
        #pragma omp parallel for
        for(int i = begin ; i < end ; ++i){
            if((i-begin) % 10000 == 0)
                std::cout<<"build hnsw point "<<i<<"/"<<end-begin<<". Now use" << std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::high_resolution_clock::now() - start).count() / 1e6 << " seconds" << std::endl;
            hnsw_index->addPoint(dataset[i],i);
        }
        std::cout << "build time: " 
            << std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::high_resolution_clock::now() - start).count() / 1e6 
            << " seconds" << std::endl;
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
            if (ifs.eof()) break;

            dimension = d; 
            float* vec = (float*)malloc(d * sizeof(float));
            ifs.read(reinterpret_cast<char*>(vec), sizeof(float) * d);
            if (ifs.eof()) {
                free(vec); 
                throw std::runtime_error("Unexpected end of file while reading vector data.");
            }
            data.emplace_back(vec);
        }

        ifs.close();
        return data;
    }

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
            for (int i = 0; i < d; ++i) {
                vec[i] = static_cast<float>(temp[i]);
            }

            data.emplace_back(vec);
        }

        ifs.close();
        return data;
    }

    void copy_meta_data_from_multiple_with_overlap(
        mergegraph::HierarchicalNSW<float>* dst,
        const std::vector<mergegraph::HierarchicalNSW<float>*>& srcs) 
    {
        using HNSW = mergegraph::HierarchicalNSW<float>;
        using label_t = mergegraph::labeltype;
        using table_t = mergegraph::tableint;
        using llsize_t = mergegraph::linklistsizeint;

        if (srcs.empty()) {
            std::cout << "Warning: source_graphs vector is empty. Nothing to merge." << std::endl;
            return;
        }

        auto t0 = std::chrono::high_resolution_clock::now();

        for (auto* g : srcs) {
            if (!g) continue;
            g->setEf(dst->ef_construction_);
        }

        struct RepVec { HNSW* g; size_t internal_id; label_t label; };
        std::unordered_map<label_t, size_t> label2nid;  
        std::vector<RepVec> reps;                       
        std::vector<int> merged_levels;                 
        std::vector<std::vector<size_t>> id_maps;       
        id_maps.resize(srcs.size());

        size_t unique_cnt = 0;
        int global_max_level = 0;

        for (size_t si = 0; si < srcs.size(); ++si) {
            auto* sg = srcs[si];
            if (!sg) continue;
            size_t n = sg->cur_element_count;
            id_maps[si].resize(n);
            for (size_t i = 0; i < n; ++i) {
                label_t L = sg->getExternalLabel(i);
                auto it = label2nid.find(L);
                if (it == label2nid.end()) {
                    size_t nid = unique_cnt++;
                    label2nid.emplace(L, nid);
                    reps.push_back({sg, i, L});
                    int lvl = 0;
                    if (i < sg->element_levels_.size())
                        lvl = sg->element_levels_[i];
                    merged_levels.push_back(lvl);
                    global_max_level = std::max(global_max_level, lvl);
                    id_maps[si][i] = nid;
                } else {
                    size_t nid = it->second;
                    id_maps[si][i] = nid;
                    int lvl = 0;
                    if (i < sg->element_levels_.size())
                        lvl = sg->element_levels_[i];
                    if (lvl > merged_levels[nid]) {
                        merged_levels[nid] = lvl;
                        global_max_level = std::max(global_max_level, lvl);
                    }
                }
            }
        }

        if (unique_cnt > dst->max_elements_) {
            throw std::runtime_error("The merged graph unique element count exceeds destination max_elements_.");
        }

        dst->cur_element_count = unique_cnt;
        dst->maxlevel_ = global_max_level;

        {
            auto* g0 = srcs[0];
            size_t ep_old = (g0 && g0->cur_element_count > 0) ? (size_t)g0->enterpoint_node_ : 0;
            if (g0 && ep_old < id_maps[0].size()) {
                dst->enterpoint_node_ = static_cast<table_t>(id_maps[0][ep_old]);
            } else {
                dst->enterpoint_node_ = 0;
            }
        }

        std::cout << "Unique elements after merge (with overlap handled): "
                << dst->cur_element_count << " from " << srcs.size() << " graphs.\n";

        std::vector<label_t> new_labels(unique_cnt);
        for (size_t nid = 0; nid < unique_cnt; ++nid) new_labels[nid] = reps[nid].label;

        #pragma omp parallel for
        for (ptrdiff_t nid = 0; nid < static_cast<ptrdiff_t>(unique_cnt); ++nid) {
            label_t lab = new_labels[nid];
            std::memcpy(dst->getExternalLabeLp(nid), &lab, sizeof(label_t));
            dst->element_levels_[nid] = merged_levels[nid];
            auto rep = reps[nid];
            std::memcpy(dst->getDataByInternalId(nid),
                        rep.g->getDataByInternalId(rep.internal_id),
                        dst->data_size_);
        }

        for (size_t nid = 0; nid < unique_cnt; ++nid) {
            dst->label_lookup_[new_labels[nid]] = static_cast<table_t>(nid);
        }

        for (size_t nid = 0; nid < unique_cnt; ++nid) {
            int lvl = merged_levels[nid];
            if (lvl > 0) {
                if (dst->linkLists_[nid] != nullptr) {
                    free(dst->linkLists_[nid]);
                    dst->linkLists_[nid] = nullptr;
                }
                size_t bytes = dst->size_links_per_element_ * static_cast<size_t>(lvl);
                dst->linkLists_[nid] = (char*)std::malloc(bytes);
                if (!dst->linkLists_[nid]) {
                    throw std::runtime_error("Not enough memory: allocate linkLists_ failed.");
                }
                for (int level = 1; level <= lvl; ++level) {
                    llsize_t* ll = dst->get_linklist(nid, level);
                    if (ll) dst->setListCount(ll, 0);
                }
            }
        }

        std::vector<std::vector<std::unordered_set<table_t>>> neigh_sets(unique_cnt);
        neigh_sets.reserve(unique_cnt);
        for (size_t nid = 0; nid < unique_cnt; ++nid) {
            int lvl = std::max(0, merged_levels[nid]); 
            neigh_sets[nid].resize(lvl + 1);
            for (int lv = 0; lv <= lvl; ++lv) {
                neigh_sets[nid][lv].reserve(8);
            }
        }

        for (size_t si = 0; si < srcs.size(); ++si) {
            auto* sg = srcs[si];
            if (!sg) continue;
            size_t n = sg->cur_element_count;
            for (size_t old_id = 0; old_id < n; ++old_id) {
                size_t nid = id_maps[si][old_id];
                int src_lvl = 0;
                if (old_id < sg->element_levels_.size())
                    src_lvl = sg->element_levels_[old_id];

                llsize_t* sll0 = sg->get_linklist0(old_id);
                size_t cnt0 = sll0 ? sg->getListCount(sll0) : 0;
                table_t* snei0 = sll0 ? (table_t*)(sll0 + 1) : nullptr;
                for (size_t j = 0; j < cnt0; ++j) {
                    size_t nb_old = (size_t)snei0[j];
                    if (nb_old >= id_maps[si].size()) continue; 
                    table_t nb_new = static_cast<table_t>(id_maps[si][nb_old]);
                    if (nb_new == (table_t)nid) continue; 
                    neigh_sets[nid][0].insert(nb_new);
                }

                for (int lv = 1; lv <= src_lvl; ++lv) {
                    llsize_t* sll = sg->get_linklist(old_id, lv);
                    if (!sll) continue; 
                    size_t cnt = sg->getListCount(sll);
                    table_t* snei = (table_t*)(sll + 1);
                    for (size_t j = 0; j < cnt; ++j) {
                        size_t nb_old = (size_t)snei[j];
                        if (nb_old >= id_maps[si].size()) continue; 
                        table_t nb_new = static_cast<table_t>(id_maps[si][nb_old]);
                        if (nb_new == (table_t)nid) continue; 
                        if (lv < (int)neigh_sets[nid].size())
                            neigh_sets[nid][lv].insert(nb_new);
                    }
                }
            }
        }

        const size_t capL0 = static_cast<size_t>(dst->maxM0_);
        const size_t capHi = static_cast<size_t>(dst->maxM_);

        #pragma omp parallel for
        for (ptrdiff_t nid = 0; nid < static_cast<ptrdiff_t>(unique_cnt); ++nid) {
            llsize_t* dll0 = dst->get_linklist0((size_t)nid);
            if (!dll0) continue; 
            size_t sz = neigh_sets[(size_t)nid][0].size();
            size_t use = std::min(sz, capL0);
            dst->setListCount(dll0, (llsize_t)use);
            table_t* dnb0 = (table_t*)(dll0 + 1);
            size_t k = 0;
            for (auto it = neigh_sets[(size_t)nid][0].begin(); it != neigh_sets[(size_t)nid][0].end() && k < use; ++it, ++k) {
                dnb0[k] = *it;
            }
        }

        #pragma omp parallel for
        for (ptrdiff_t nid = 0; nid < static_cast<ptrdiff_t>(unique_cnt); ++nid) {
            int lvl = merged_levels[(size_t)nid];
            for (int lv = 1; lv <= lvl; ++lv) {
                llsize_t* dll = dst->get_linklist((size_t)nid, lv);
                if (!dll) continue; 
                size_t sz = neigh_sets[(size_t)nid][lv].size();
                size_t use = std::min(sz, capHi);
                dst->setListCount(dll, (llsize_t)use);
                table_t* dnb = (table_t*)(dll + 1);
                size_t k = 0;
                for (auto it = neigh_sets[(size_t)nid][lv].begin(); it != neigh_sets[(size_t)nid][lv].end() && k < use; ++it, ++k) {
                    dnb[k] = *it;
                }
            }
        }

        auto t1 = std::chrono::high_resolution_clock::now();
        std::cout << "Merged (with overlaps) metadata, vectors, and deduped links in "
                << std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count()
                << " ms." << std::endl;
    }

    // no overlap
    void copy_meta_data_from_multiple(mergegraph::HierarchicalNSW<float>* alg_mergegraph_merge, 
                                  const std::vector<mergegraph::HierarchicalNSW<float>*>& source_graphs) {
        if (source_graphs.empty()) {
            std::cout << "Warning: source_graphs vector is empty. Nothing to merge." << std::endl;
            return;
        }

        auto metadata_start = std::chrono::high_resolution_clock::now();

        size_t total_elements = 0;
        for (const auto& graph : source_graphs) {
            total_elements += graph->cur_element_count;
        }

        int max_level = 0;
        for (const auto& graph : source_graphs) {
            if (graph->maxlevel_ > max_level) {
                max_level = graph->maxlevel_;
            }
        }

        for (const auto& graph : source_graphs) {
            graph->setEf(alg_mergegraph_merge->ef_construction_);
        }

        if (total_elements > alg_mergegraph_merge->max_elements_) {
            throw std::runtime_error("The merged graph size exceeds the max_elements capacity of the destination graph.");
        }

        alg_mergegraph_merge->cur_element_count = total_elements;
        std::cout<<"element number after merge "<<alg_mergegraph_merge->cur_element_count<<std::endl;
        alg_mergegraph_merge->maxlevel_ = max_level;
        alg_mergegraph_merge->enterpoint_node_ = source_graphs[0]->enterpoint_node_; 

        std::cout << "Total elements to merge from " << source_graphs.size() << " graphs: " << total_elements << std::endl;

        size_t cumulative_offset = 0;
        for (const auto& source_graph : source_graphs) {
            size_t source_size = source_graph->cur_element_count;

            #pragma omp parallel for
            for (size_t i = 0; i < source_size; ++i) {
                size_t new_id = i + cumulative_offset;

                memcpy(alg_mergegraph_merge->getExternalLabeLp(new_id), source_graph->getExternalLabeLp(i), sizeof(mergegraph::labeltype));

                alg_mergegraph_merge->element_levels_[new_id] = source_graph->element_levels_[i];

                memcpy(alg_mergegraph_merge->getDataByInternalId(new_id), source_graph->getDataByInternalId(i), alg_mergegraph_merge->data_size_);
            }

            for (size_t i = 0; i < source_size; ++i) {
                size_t new_id = i + cumulative_offset;
                alg_mergegraph_merge->label_lookup_[source_graph->getExternalLabel(i)] = new_id;
            }

            cumulative_offset += source_size;
        }



        cumulative_offset = 0; 
        for (const auto& source_graph : source_graphs) {
            size_t source_size = source_graph->cur_element_count;
            size_t neighbor_id_offset = cumulative_offset; 

            #pragma omp parallel for
            for (size_t old_id = 0; old_id < source_size; ++old_id) {
                size_t new_id = old_id + cumulative_offset;

                mergegraph::linklistsizeint* source_ll_0 = source_graph->get_linklist0(old_id);
                size_t neighbor_count_0 = source_graph->getListCount(source_ll_0);
                mergegraph::tableint* source_neighbors_0 = (mergegraph::tableint*)(source_ll_0 + 1);

                mergegraph::linklistsizeint* dest_ll_0 = alg_mergegraph_merge->get_linklist0(new_id);
                alg_mergegraph_merge->setListCount(dest_ll_0, neighbor_count_0);
                mergegraph::tableint* dest_neighbors_0 = (mergegraph::tableint*)(dest_ll_0 + 1);

                for (size_t j = 0; j < neighbor_count_0; ++j) {
                    dest_neighbors_0[j] = source_neighbors_0[j] + neighbor_id_offset; 
                }

                int element_level = alg_mergegraph_merge->element_levels_[new_id]; 
                for (int level = 1; level <= element_level; ++level) {
                    mergegraph::linklistsizeint* source_ll = source_graph->get_linklist(old_id, level);
                    size_t neighbor_count = source_graph->getListCount(source_ll);
                    mergegraph::tableint* source_neighbors = (mergegraph::tableint*)(source_ll + 1);

                    mergegraph::linklistsizeint* dest_ll = alg_mergegraph_merge->get_linklist(new_id, level);
                    alg_mergegraph_merge->setListCount(dest_ll, neighbor_count);
                    mergegraph::tableint* dest_neighbors = (mergegraph::tableint*)(dest_ll + 1);

                    for (size_t j = 0; j < neighbor_count; ++j) {
                        dest_neighbors[j] = source_neighbors[j] + neighbor_id_offset; 
                    }
                }
            }
            cumulative_offset += source_size;
        }

        auto metadata_end = std::chrono::high_resolution_clock::now();
        std::cout << "Metadata, vector, and link copy finished in " 
                << std::chrono::duration_cast<std::chrono::milliseconds>(metadata_end - metadata_start).count() << " ms." << std::endl;
    }

    // copy meta data from two graph
    void copy_meta_data(mergegraph::HierarchicalNSW<float>* alg_mergegraph_merge,mergegraph::HierarchicalNSW<float>* first_half,mergegraph::HierarchicalNSW<float>* second_half){
        size_t first_half_size = first_half->cur_element_count;
        size_t second_half_size = second_half->cur_element_count;
        size_t total_elements = first_half_size + second_half_size;

        first_half->setEf(alg_mergegraph_merge->ef_construction_);
        second_half->setEf(alg_mergegraph_merge->ef_construction_);


        if (total_elements > alg_mergegraph_merge->max_elements_) {
            throw std::runtime_error("The merged graph size exceeds the max_elements capacity of the destination graph.");
        }

        alg_mergegraph_merge->cur_element_count = total_elements;
        alg_mergegraph_merge->maxlevel_ = std::max(first_half->maxlevel_, second_half->maxlevel_);
        alg_mergegraph_merge->enterpoint_node_ = first_half->enterpoint_node_; 


        auto metadata_start = std::chrono::high_resolution_clock::now();

        #pragma omp parallel for
        for (size_t i = 0; i < first_half_size; ++i) {

            memcpy(alg_mergegraph_merge->getExternalLabeLp(i), first_half->getExternalLabeLp(i), sizeof(mergegraph::labeltype));

            alg_mergegraph_merge->element_levels_[i] = first_half->element_levels_[i];

            memcpy(alg_mergegraph_merge->getDataByInternalId(i), first_half->getDataByInternalId(i), alg_mergegraph_merge->data_size_);
        }

        for (size_t i = 0; i < first_half_size; ++i) {
            alg_mergegraph_merge->label_lookup_[first_half->getExternalLabel(i)] = i;
        }

        #pragma omp parallel for
        for (size_t i = 0; i < second_half_size; ++i) {
            size_t new_id = i + first_half_size; 


            memcpy(alg_mergegraph_merge->getExternalLabeLp(new_id), second_half->getExternalLabeLp(i), sizeof(mergegraph::labeltype));

            alg_mergegraph_merge->element_levels_[new_id] = second_half->element_levels_[i];

            memcpy(alg_mergegraph_merge->getDataByInternalId(new_id), second_half->getDataByInternalId(i), alg_mergegraph_merge->data_size_);
        }
        for (size_t i = 0; i < second_half_size; ++i) {
            size_t new_id = i + first_half_size; 

            alg_mergegraph_merge->label_lookup_[second_half->getExternalLabel(i)] = new_id;
        }


        #pragma omp parallel for
        for (size_t i = 0; i < total_elements; ++i) {
            if (alg_mergegraph_merge->element_levels_[i] > 0) {
                if (alg_mergegraph_merge->linkLists_[i] != nullptr) {
                    free(alg_mergegraph_merge->linkLists_[i]);
                }
                alg_mergegraph_merge->linkLists_[i] = (char*)malloc(alg_mergegraph_merge->size_links_per_element_ * alg_mergegraph_merge->element_levels_[i]);
                if (alg_mergegraph_merge->linkLists_[i] == nullptr) {
                    throw std::runtime_error("Not enough memory: failed to allocate linklist for high-level elements.");
                }
            }
        }


        #pragma omp parallel for
        for (size_t i = 0; i < total_elements; ++i) {
            mergegraph::HierarchicalNSW<float>* source_graph;
            size_t old_id;
            size_t id_offset;

            if (i < first_half_size) {
                source_graph = first_half;
                old_id = i;
                id_offset = 0; 
            } else {
                source_graph = second_half;
                old_id = i - first_half_size;
                id_offset = first_half_size; 
            }

            mergegraph::linklistsizeint* source_ll_0 = source_graph->get_linklist0(old_id);
            size_t neighbor_count_0 = source_graph->getListCount(source_ll_0);
            mergegraph::tableint* source_neighbors_0 = (mergegraph::tableint*)(source_ll_0 + 1);

            mergegraph::linklistsizeint* dest_ll_0 = alg_mergegraph_merge->get_linklist0(i);
            alg_mergegraph_merge->setListCount(dest_ll_0, neighbor_count_0);
            mergegraph::tableint* dest_neighbors_0 = (mergegraph::tableint*)(dest_ll_0 + 1);

            for (size_t j = 0; j < neighbor_count_0; ++j) {
                dest_neighbors_0[j] = source_neighbors_0[j] + id_offset; 
            }

            int element_level = alg_mergegraph_merge->element_levels_[i];
            for (int level = 1; level <= element_level; ++level) {
                mergegraph::linklistsizeint* source_ll = source_graph->get_linklist(old_id, level);
                size_t neighbor_count = source_graph->getListCount(source_ll);
                mergegraph::tableint* source_neighbors = (mergegraph::tableint*)(source_ll + 1);

                mergegraph::linklistsizeint* dest_ll = alg_mergegraph_merge->get_linklist(i, level);
                alg_mergegraph_merge->setListCount(dest_ll, neighbor_count);
                mergegraph::tableint* dest_neighbors = (mergegraph::tableint*)(dest_ll + 1);

                for (size_t j = 0; j < neighbor_count; ++j) {
                    dest_neighbors[j] = source_neighbors[j] + id_offset; 
                }
            }
        }

        auto metadata_end = std::chrono::high_resolution_clock::now();
        std::cout << "Metadata and vector copy finished in " 
                << std::chrono::duration_cast<std::chrono::milliseconds>(metadata_end - metadata_start).count() << " ms." << std::endl;
    }

    // merge from two graph
    void hnsw_merge_Skeleton_bridged_Local_Repair(mergegraph::HierarchicalNSW<float>* alg_mergegraph_merge,mergegraph::HierarchicalNSW<float>* first_half,mergegraph::HierarchicalNSW<float>* second_half,std::vector<mergegraph::labeltype> & delete_ids,int random_part=10){

        std::cout << "Starting HNSW merge..." << std::endl;
        auto start = std::chrono::high_resolution_clock::now();


        size_t first_half_size = first_half->cur_element_count;
        size_t second_half_size = second_half->cur_element_count;
        size_t total_elements = first_half_size + second_half_size;

        copy_meta_data(alg_mergegraph_merge,first_half,second_half);

        boost::dynamic_bitset<> flags{total_elements, 0};

        auto links_start = std::chrono::high_resolution_clock::now();

        for(mergegraph::tableint i = 0 ; i < delete_ids.size() ; ++i){
            alg_mergegraph_merge->markDelete(delete_ids[i]);
        }

        int layer = 0;

        float percent = 0.8;

        size_t search_para = alg_mergegraph_merge->M_;
        first_half->setEf(search_para);
        second_half->setEf(search_para);
        alg_mergegraph_merge->setEf(search_para);

        std::vector<size_t>first_ske(first_half_size);
        std::vector<size_t>second_ske(second_half_size);

        std::random_device rd;
        std::mt19937 gen(rd());
        for(int i = 0 ; i < first_ske.size();++i){
            first_ske[i]=i;
        }

        for(int i = 0; i < second_ske.size();++i){
            second_ske[i]=i+first_half_size;
        }

        alg_mergegraph_merge->enterpoint_node_ = second_half->enterpoint_node_ + first_half_size;
        #pragma omp parallel for
        for(size_t k = 0 ; k < first_half_size;++k){
            if(random()%100>=random_part)continue;

            mergegraph::tableint skeleton = k;

            std::unordered_set<mergegraph::tableint> visited_set;
            std::vector<std::pair<mergegraph::labeltype,float>> visited_pair_1;
            std::vector<std::pair<mergegraph::labeltype,float>> visited_pair_2;

            std::priority_queue<std::pair<float, mergegraph::tableint>, std::vector<std::pair<float, mergegraph::tableint>>, typename mergegraph::HierarchicalNSW<float>::CompareByFirst> top_candidates;

            std::unordered_set<mergegraph::tableint> two_hop_neighs;
            alg_mergegraph_merge->get_one_hop_neigh(skeleton,layer,two_hop_neighs);
            for(auto && neigh:two_hop_neighs){
                visited_set.insert(neigh);
                if(neigh!=skeleton&&!alg_mergegraph_merge->isMarkedDeleted(neigh))
                    visited_pair_1.emplace_back(neigh,alg_mergegraph_merge->fstdistfunc_(alg_mergegraph_merge->getDataByInternalId(skeleton),alg_mergegraph_merge->getDataByInternalId(neigh),alg_mergegraph_merge->dist_func_param_));
            }

            for(auto && node:visited_pair_1){
                if(top_candidates.size()<alg_mergegraph_merge->maxM0_){
                    top_candidates.emplace(node.second,node.first);
                }
                else if(node.second < top_candidates.top().first){
                    top_candidates.pop();
                    top_candidates.emplace(node.second,node.first);
                }
            }

            alg_mergegraph_merge->searchKnn_with_visited_lists(alg_mergegraph_merge->getDataByInternalId(skeleton),search_para,visited_pair_2);

            for(auto && node:visited_pair_2){
                if(alg_mergegraph_merge->label_lookup_[node.first]==skeleton||visited_set.find(alg_mergegraph_merge->label_lookup_[node.first])!=visited_set.end())continue;
                if(top_candidates.size()<alg_mergegraph_merge->maxM0_){
                    top_candidates.emplace(node.second,alg_mergegraph_merge->label_lookup_[node.first]);
                }
                else if(node.second < top_candidates.top().first){
                    top_candidates.pop();
                    top_candidates.emplace(node.second,alg_mergegraph_merge->label_lookup_[node.first]);
                }
            }

            while(top_candidates.size()>alg_mergegraph_merge->maxM0_)top_candidates.pop();
            alg_mergegraph_merge->mutuallyConnectNewElement_(alg_mergegraph_merge->getDataByInternalId(skeleton),skeleton,top_candidates,0,true,true);
        }

        alg_mergegraph_merge->enterpoint_node_ = first_half->enterpoint_node_; 

        #pragma omp parallel for
        for(size_t k = first_half_size; k < alg_mergegraph_merge->cur_element_count;++k){
            if(random()%100>=random_part)continue;

            mergegraph::tableint skeleton = k;

            std::unordered_set<mergegraph::tableint> visited_set;
            std::vector<std::pair<mergegraph::labeltype,float>> visited_pair_1;
            std::vector<std::pair<mergegraph::labeltype,float>> visited_pair_2;

            std::priority_queue<std::pair<float, mergegraph::tableint>, std::vector<std::pair<float, mergegraph::tableint>>, typename mergegraph::HierarchicalNSW<float>::CompareByFirst> top_candidates;

            std::unordered_set<mergegraph::tableint> two_hop_neighs;
            alg_mergegraph_merge->get_one_hop_neigh(skeleton,layer,two_hop_neighs);
            for(auto && neigh:two_hop_neighs){
                visited_set.insert(neigh);
                if(neigh!=skeleton&&!alg_mergegraph_merge->isMarkedDeleted(neigh))
                    visited_pair_1.emplace_back(neigh,alg_mergegraph_merge->fstdistfunc_(alg_mergegraph_merge->getDataByInternalId(skeleton),alg_mergegraph_merge->getDataByInternalId(neigh),alg_mergegraph_merge->dist_func_param_));
            }


            for(auto && node:visited_pair_1){
                if(top_candidates.size()<alg_mergegraph_merge->ef_construction_){
                    top_candidates.emplace(node.second,node.first);
                }
                else if(node.second < top_candidates.top().first){
                    top_candidates.pop();
                    top_candidates.emplace(node.second,node.first);
                }
            }


            alg_mergegraph_merge->searchKnn_with_visited_lists(alg_mergegraph_merge->getDataByInternalId(skeleton),search_para,visited_pair_2);


            for(auto && node:visited_pair_2){
                if(alg_mergegraph_merge->label_lookup_[node.first]==skeleton||visited_set.find(alg_mergegraph_merge->label_lookup_[node.first])!=visited_set.end())continue;
                if(top_candidates.size()<alg_mergegraph_merge->maxM0_){
                    top_candidates.emplace(node.second,alg_mergegraph_merge->label_lookup_[node.first]);
                }
                else if(node.second < top_candidates.top().first){
                    top_candidates.pop();
                    top_candidates.emplace(node.second,alg_mergegraph_merge->label_lookup_[node.first]);
                }
            }
            while(top_candidates.size()>alg_mergegraph_merge->maxM0_)top_candidates.pop();
            alg_mergegraph_merge->mutuallyConnectNewElement_(alg_mergegraph_merge->getDataByInternalId(skeleton),skeleton,top_candidates,0,true,true);
        }

        std::cout<<"link Skeleton now time use: "<<std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::high_resolution_clock::now() - start).count()/1e9<<" s"<<std::endl;

        alg_mergegraph_merge->enterpoint_node_ = alg_mergegraph_merge->get_navi_point();

        alg_mergegraph_merge->maxlevel_ = 0 ;
        auto end = std::chrono::high_resolution_clock::now();
        std::cout<<"hnsw merge time: "<<std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count()/1e9<<" s"<<std::endl;
    }

    // merge from multi graph
    void hnsw_merge_Skeleton_bridged_Local_Repair(mergegraph::HierarchicalNSW<float>* alg_mergegraph_merge,std::vector<mergegraph::HierarchicalNSW<float>*> &source_graph,std::vector<mergegraph::labeltype> & delete_ids){

        std::cout << "Starting HNSW merge..." << std::endl;
        auto start = std::chrono::high_resolution_clock::now();

        copy_meta_data_from_multiple(alg_mergegraph_merge,source_graph);


        int num_graph = source_graph.size();

        std::vector<size_t> offests_(num_graph);
        offests_[0] = 0;
        for(int i = 1 ; i < num_graph ; ++ i){
            offests_[i] = offests_[i-1]+source_graph[i-1]->cur_element_count;
        }
        int layer = 0 ;


        auto links_start = std::chrono::high_resolution_clock::now();

        size_t search_para = alg_mergegraph_merge->M_;

        for(mergegraph::tableint i = 0 ; i < delete_ids.size() ; ++i){
            alg_mergegraph_merge->markDelete(delete_ids[i]);
        }

        std::mt19937 rng(time(0)); 
        std::uniform_int_distribution<mergegraph::tableint> distrib(0, alg_mergegraph_merge->cur_element_count - 1);
        for(int i = num_graph ; i < num_graph * 2 ; ++i){

            alg_mergegraph_merge->enterpoint_node_ = offests_[(i-1)%num_graph] + source_graph[(i-1)%num_graph]->enterpoint_node_;

            int next_offset = i%num_graph < num_graph-1 ? offests_[(i+1)%num_graph]:(unsigned long)alg_mergegraph_merge->cur_element_count;
            #pragma omp parallel for
            for(int k = offests_[i%num_graph] ; k < next_offset ; ++k){
                if(random()%10>0)continue;

                mergegraph::tableint skeleton = k;

                std::unordered_set<mergegraph::tableint> visited_set;
                std::vector<std::pair<mergegraph::labeltype,float>> visited_pair_1;
                std::vector<std::pair<mergegraph::labeltype,float>> visited_pair_2;

                std::priority_queue<std::pair<float, mergegraph::tableint>, std::vector<std::pair<float, mergegraph::tableint>>, typename mergegraph::HierarchicalNSW<float>::CompareByFirst> top_candidates;

                std::unordered_set<mergegraph::tableint> two_hop_neighs;
                alg_mergegraph_merge->get_one_hop_neigh(skeleton,layer,two_hop_neighs);
                for(auto && neigh:two_hop_neighs){
                    visited_set.insert(neigh);
                    if(neigh!=skeleton&&!alg_mergegraph_merge->isMarkedDeleted(neigh))
                        visited_pair_1.emplace_back(neigh,alg_mergegraph_merge->fstdistfunc_(alg_mergegraph_merge->getDataByInternalId(skeleton),alg_mergegraph_merge->getDataByInternalId(neigh),alg_mergegraph_merge->dist_func_param_));
                }



                for(auto && node:visited_pair_1){
                    if(top_candidates.size()<alg_mergegraph_merge->maxM0_){
                        top_candidates.emplace(node.second,node.first);
                    }
                    else if(node.second < top_candidates.top().first){
                        top_candidates.pop();
                        top_candidates.emplace(node.second,node.first);
                    }
                }

                alg_mergegraph_merge->searchKnn_with_visited_lists(alg_mergegraph_merge->getDataByInternalId(skeleton),search_para,visited_pair_2);

                for(auto && node:visited_pair_2){
                    if(alg_mergegraph_merge->label_lookup_[node.first]==skeleton||visited_set.find(alg_mergegraph_merge->label_lookup_[node.first])!=visited_set.end())continue;
                    visited_pair_1.emplace_back(node);
                    if(top_candidates.size()<alg_mergegraph_merge->maxM0_){
                        top_candidates.emplace(node.second,alg_mergegraph_merge->label_lookup_[node.first]);
                    }
                    else if(node.second < top_candidates.top().first){
                        top_candidates.pop();
                        top_candidates.emplace(node.second,alg_mergegraph_merge->label_lookup_[node.first]);
                    }
                }
                while(top_candidates.size()>alg_mergegraph_merge->maxM0_)top_candidates.pop();
                alg_mergegraph_merge->mutuallyConnectNewElement_(alg_mergegraph_merge->getDataByInternalId(skeleton),skeleton,top_candidates,0,true);

            }
        }

        std::cout<<"link Skeleton now time use: "<<std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::high_resolution_clock::now() - start).count()/1e9<<" s"<<std::endl;

        alg_mergegraph_merge->maxlevel_ = 0 ;
        auto end = std::chrono::high_resolution_clock::now();
        std::cout<<"hnsw merge time: "<<std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count()/1e9<<" s"<<std::endl;
    }


    void hnsw_merge_cross_all_sub_graph(mergegraph::HierarchicalNSW<float>* alg_mergegraph_merge,std::vector<mergegraph::HierarchicalNSW<float>*> &source_graph,std::vector<mergegraph::labeltype> & delete_ids){

        std::cout << "Starting HNSW merge..." << std::endl;
        auto start = std::chrono::high_resolution_clock::now();

        copy_meta_data_from_multiple(alg_mergegraph_merge,source_graph);

        int num_graph = source_graph.size();

        std::vector<size_t> offests_(num_graph);
        offests_[0] = 0;
        for(int i = 1 ; i < num_graph ; ++ i){
            offests_[i] = offests_[i-1]+source_graph[i-1]->cur_element_count;
        }

        int layer = 0 ;

        auto links_start = std::chrono::high_resolution_clock::now();

        size_t search_para = alg_mergegraph_merge->M_;

        for(mergegraph::tableint i = 0 ; i < delete_ids.size() ; ++i){
            alg_mergegraph_merge->markDelete(delete_ids[i]);
        }

        std::mt19937 rng(time(0)); 
        std::uniform_int_distribution<mergegraph::tableint> distrib(0, alg_mergegraph_merge->cur_element_count - 1);
        for(int i = 0 ; i < num_graph ; ++i){

            for(int j = 0 ; j < num_graph ; ++j)
            {
                if(j==i)continue;
                alg_mergegraph_merge->enterpoint_node_ = offests_[j] + source_graph[j]->enterpoint_node_;

                int next_offset = i%num_graph < num_graph-1 ? offests_[(i+1)%num_graph]:(unsigned long)alg_mergegraph_merge->cur_element_count;
                #pragma omp parallel for
                for(int k = offests_[i%num_graph] ; k < next_offset ; ++k){
                    if(random()%10>4)continue;

                    mergegraph::tableint skeleton = k;

                    std::unordered_set<mergegraph::tableint> visited_set;
                    std::vector<std::pair<mergegraph::labeltype,float>> visited_pair_1;
                    std::vector<std::pair<mergegraph::labeltype,float>> visited_pair_2;

                    std::priority_queue<std::pair<float, mergegraph::tableint>, std::vector<std::pair<float, mergegraph::tableint>>, typename mergegraph::HierarchicalNSW<float>::CompareByFirst> top_candidates;

                    std::unordered_set<mergegraph::tableint> two_hop_neighs;
                    alg_mergegraph_merge->get_one_hop_neigh(skeleton,layer,two_hop_neighs);
                    for(auto && neigh:two_hop_neighs){
                        visited_set.insert(neigh);
                        if(neigh!=skeleton&&!alg_mergegraph_merge->isMarkedDeleted(neigh))
                            visited_pair_1.emplace_back(neigh,alg_mergegraph_merge->fstdistfunc_(alg_mergegraph_merge->getDataByInternalId(skeleton),alg_mergegraph_merge->getDataByInternalId(neigh),alg_mergegraph_merge->dist_func_param_));
                    }

                    for(auto && node:visited_pair_1){
                        if(top_candidates.size()<alg_mergegraph_merge->maxM0_){
                            top_candidates.emplace(node.second,node.first);
                        }
                        else if(node.second < top_candidates.top().first){
                            top_candidates.pop();
                            top_candidates.emplace(node.second,node.first);
                        }
                    }

                    source_graph[j]->searchKnn_with_visited_lists(alg_mergegraph_merge->getDataByInternalId(skeleton),search_para,visited_pair_2);

                    for(auto && node:visited_pair_2){
                        if(alg_mergegraph_merge->label_lookup_[node.first]==skeleton||visited_set.find(alg_mergegraph_merge->label_lookup_[node.first])!=visited_set.end())continue;
                        visited_pair_1.emplace_back(node);
                        if(top_candidates.size()<alg_mergegraph_merge->maxM0_){
                            top_candidates.emplace(node.second,alg_mergegraph_merge->label_lookup_[node.first]);
                        }
                        else if(node.second < top_candidates.top().first){
                            top_candidates.pop();
                            top_candidates.emplace(node.second,alg_mergegraph_merge->label_lookup_[node.first]);
                        }
                    }
                    while(top_candidates.size()>alg_mergegraph_merge->maxM0_)top_candidates.pop();
                    alg_mergegraph_merge->mutuallyConnectNewElement_(alg_mergegraph_merge->getDataByInternalId(skeleton),skeleton,top_candidates,0,true);
                }
            }
        }

        std::cout<<"link Skeleton now time use: "<<std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::high_resolution_clock::now() - start).count()/1e9<<" s"<<std::endl;


        auto end = std::chrono::high_resolution_clock::now();
        std::cout<<"hnsw merge time: "<<std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count()/1e9<<" s"<<std::endl;
    }

    inline void save_groundtruth(
        const std::vector<std::unordered_set<mergegraph::labeltype>>& groundtruth,
        const std::string& filename)
    {
        std::ofstream ofs(filename);
        if (!ofs) throw std::runtime_error("can't open file " + filename);
        ofs << groundtruth.size() << '\n';
        for (auto const& s : groundtruth) {
            ofs << s.size();
            for (int x : s) {
                ofs << ' ' << x;
            }
            ofs << '\n';
        }
    }

    inline void load_groundtruth(
        std::vector<std::unordered_set<mergegraph::labeltype>>& groundtruth,
        const std::string& filename)
    {
        std::ifstream ifs(filename);
        if (!ifs) throw std::runtime_error("can't open file " + filename);
        size_t num_sets;
        ifs >> num_sets;
        groundtruth.clear();
        groundtruth.reserve(num_sets);

        for (size_t i = 0; i < num_sets; ++i) {
            size_t set_size;
            ifs >> set_size;
            std::unordered_set<mergegraph::labeltype> s;
            for (size_t j = 0; j < set_size; ++j) {
                int x;
                ifs >> x;
                s.insert(x);
            }
            groundtruth.push_back(std::move(s));
        }
    }

    inline double calculate_recall(mergegraph::HierarchicalNSW<float>* alg_mergegraph,std::vector<std::unordered_set<mergegraph::labeltype>> groundtruth,std::vector<float*> queries,int ef,int k){
        alg_mergegraph->setEf(ef);
        int num_queries = queries.size();
        double recall_hnsw_total = 0.0;
        #pragma omp parallel for reduction(+:recall_hnsw_total) \
        schedule(dynamic)
        for (int i = 0; i < num_queries; i++) {
            float* query_data = queries[i];
            auto result_hnsw = alg_mergegraph->searchKnn(query_data, k);
            std::unordered_set<mergegraph::labeltype> hnsw_labels;

            while (!result_hnsw.empty()) {
                hnsw_labels.insert(result_hnsw.top().second);
                result_hnsw.pop();
            }

            std::unordered_set<mergegraph::labeltype> gt_labels;
            gt_labels = groundtruth[i];

            float correct_hnsw = 0;
            for (const auto& label : hnsw_labels) {
                if (gt_labels.find(label) != gt_labels.end()) {
                    correct_hnsw += 1.0f;
                }
            }
            float recall_hnsw = correct_hnsw / gt_labels.size();
            recall_hnsw_total += recall_hnsw;
        }
        return recall_hnsw_total/num_queries;
    }

    inline double calculate_recall(mergegraph::HierarchicalNSW<float>* alg_mergegraph_first_half,mergegraph::HierarchicalNSW<float>* alg_mergegraph_second_half,std::vector<std::unordered_set<mergegraph::labeltype>> groundtruth,std::vector<float*> queries,int ef,int k){
        alg_mergegraph_first_half->setEf(ef);
        alg_mergegraph_second_half->setEf(ef);
        int num_queries = queries.size();
        double recall_hnsw_total = 0.0;
        #pragma omp parallel for reduction(+:recall_hnsw_total) \
        schedule(dynamic) 
        for (int i = 0; i < num_queries; i++) {
            float* query_data = queries[i];
            auto result_hnsw_first = alg_mergegraph_first_half->searchKnn(query_data, k);
            auto result_hnsw_second = alg_mergegraph_second_half->searchKnn(query_data, k);

            std::unordered_set<mergegraph::labeltype> hnsw_labels;

            while(!result_hnsw_second.empty()){
                result_hnsw_first.emplace(result_hnsw_second.top());
                result_hnsw_second.pop();
            }

            while (!result_hnsw_first.empty()) {
                hnsw_labels.insert(result_hnsw_first.top().second);
                result_hnsw_first.pop();
            }

            std::unordered_set<mergegraph::labeltype> gt_labels;
            gt_labels = groundtruth[i];

            float correct_hnsw = 0;
            for (const auto& label : hnsw_labels) {
                if (gt_labels.find(label) != gt_labels.end()) {
                    correct_hnsw += 1.0f;
                }
            }
            float recall_hnsw = correct_hnsw / gt_labels.size();
            recall_hnsw_total += recall_hnsw;
        }
        return recall_hnsw_total/num_queries;
    }

    inline std::pair<int,double> quick_get_ef(mergegraph::HierarchicalNSW<float>* alg_mergegraph,
                        std::vector<std::unordered_set<mergegraph::labeltype>> groundtruth,
                        std::vector<float*> queries, int k, float target_recall) {
        int left = k;
        int right = 100;
        float recall_right = calculate_recall(alg_mergegraph, groundtruth, queries, right, k);
        float recall_left_ = calculate_recall(alg_mergegraph, groundtruth, queries, left, k);

        if (recall_left_ >= target_recall+0.003){
            return std::make_pair(k, recall_left_);
        }

        while (recall_right < target_recall && right < 8000) {
            left = right;
            right *= 2;
            recall_right = calculate_recall(alg_mergegraph, groundtruth, queries, right, k);
        }
        if (right >= 8000 && recall_right < target_recall) {
            return std::make_pair(1,0.0);
        }

        while (right - left > 1) {
            int mid = (left + right) / 2;
            float recall_mid = calculate_recall(alg_mergegraph, groundtruth, queries, mid, k);
            if (recall_mid >= target_recall)
                right = mid;
            else
                left = mid;
        }

        float recall_left = calculate_recall(alg_mergegraph, groundtruth, queries, left, k);
        float recall_right_final = calculate_recall(alg_mergegraph, groundtruth, queries, right, k);
        if (std::fabs(recall_left - target_recall) <= std::fabs(recall_right_final - target_recall))
            return std::make_pair(left, recall_left);
        else
            return std::make_pair(right, recall_right_final);
    }


    inline std::pair<int,double> quick_get_ef(mergegraph::HierarchicalNSW<float>* alg_mergegraph_first_half,mergegraph::HierarchicalNSW<float>* alg_mergegraph_second_half,
                        std::vector<std::unordered_set<mergegraph::labeltype>> groundtruth,
                        std::vector<float*> queries, int k, float target_recall) {
        int left = k;
        int right = 100;
        float recall_right = calculate_recall(alg_mergegraph_first_half,alg_mergegraph_second_half, groundtruth, queries, right, k);
        float recall_left_ = calculate_recall(alg_mergegraph_first_half,alg_mergegraph_second_half, groundtruth, queries, left, k);

        if (recall_left_ >= target_recall+0.003){
            return std::make_pair(k, recall_left_);
        }

        while (recall_right < target_recall && right < 8000) {
            left = right;
            right *= 2;
            recall_right = calculate_recall(alg_mergegraph_first_half,alg_mergegraph_second_half, groundtruth, queries, right, k);
        }
        if (right >= 8000 && recall_right < target_recall) {
            return std::make_pair(1,0.0);
        }

        while (right - left > 1) {
            int mid = (left + right) / 2;
            float recall_mid = calculate_recall(alg_mergegraph_first_half,alg_mergegraph_second_half, groundtruth, queries, mid, k);
            if (recall_mid >= target_recall)
                right = mid;
            else
                left = mid;
        }

        float recall_left = calculate_recall(alg_mergegraph_first_half,alg_mergegraph_second_half, groundtruth, queries, left, k);
        float recall_right_final = calculate_recall(alg_mergegraph_first_half,alg_mergegraph_second_half, groundtruth, queries, right, k);
        if (std::fabs(recall_left - target_recall) <= std::fabs(recall_right_final - target_recall))
            return std::make_pair(left, recall_left);
        else
            return std::make_pair(right, recall_right_final);
    }

    inline std::pair<double, double> calculate_qps(mergegraph::HierarchicalNSW<float>* alg_mergegraph,std::vector<float*> queries,int ef){
        alg_mergegraph->setEf(ef);
        int num_queries = queries.size();
        size_t dim = *(size_t *)alg_mergegraph->dist_func_param_;
        char * q = (char *)malloc(num_queries*sizeof(float)* dim);
        for(int i = 0 ; i < num_queries;++i){
            memcpy(q+dim*sizeof(float)*i,queries[i],dim*sizeof(float));
        }

        uint64_t hnsw_search_time_st = 0;
        uint64_t hnsw_search_time_st_min = 0;
        uint64_t hnsw_search_time_st_all = 0;

        std::vector<std::priority_queue<std::pair<float, mergegraph::labeltype>>> result_hnsw(num_queries);

        int times = times_;
        for (int j = 0;j< times; j++){
                auto temp1 = std::chrono::high_resolution_clock::now();
            for (int i = 0; i < num_queries; i++) {
                result_hnsw[i] = alg_mergegraph->searchKnn((void *)(q+i*dim*sizeof(float)), ef);
            }
            auto temp2 = std::chrono::high_resolution_clock::now();
            hnsw_search_time_st += std::chrono::duration_cast<std::chrono::microseconds>(temp2 - temp1).count();
            if(j==0){
                hnsw_search_time_st_min = hnsw_search_time_st;
            }else{
                if(hnsw_search_time_st_min>hnsw_search_time_st){
                    hnsw_search_time_st_min = hnsw_search_time_st;}
            }


            hnsw_search_time_st_all += hnsw_search_time_st ;


            hnsw_search_time_st = 0;
        }
        free(q);
        return std::make_pair(static_cast<double>((double)num_queries/((hnsw_search_time_st_min) / 1e6)),static_cast<double>(static_cast<double>(num_queries)/((hnsw_search_time_st_all)/static_cast<double>(times) / 1e6)));
    }

    // use two subgraph search
    inline std::pair<double, double> calculate_qps(mergegraph::HierarchicalNSW<float>* alg_mergegraph_first_half,mergegraph::HierarchicalNSW<float>* alg_mergegraph_second_half,std::vector<float*> queries,int ef){
        alg_mergegraph_first_half->setEf(ef);
        alg_mergegraph_second_half->setEf(ef);
        int num_queries = queries.size();
        size_t dim = *(size_t *)alg_mergegraph_first_half->dist_func_param_;
        char * q = (char *)malloc(num_queries*sizeof(float)* dim);
        for(int i = 0 ; i < num_queries;++i){
            memcpy(q+dim*sizeof(float)*i,queries[i],dim*sizeof(float));
        }

        uint64_t hnsw_search_time_st = 0;
        uint64_t hnsw_search_time_st_min = 0;
        uint64_t hnsw_search_time_st_all = 0;

        std::vector<std::priority_queue<std::pair<float, mergegraph::labeltype>>> result_hnsw_first(num_queries);
        std::vector<std::priority_queue<std::pair<float, mergegraph::labeltype>>> result_hnsw_second(num_queries);

        int times = times_;
        for (int j = 0;j< times; j++){
            auto temp1 = std::chrono::high_resolution_clock::now();
            for (int i = 0; i < num_queries; i++) {
                result_hnsw_first[i] = alg_mergegraph_first_half->searchKnn((void *)(q+i*dim*sizeof(float)), ef);
                result_hnsw_second[i] = alg_mergegraph_second_half->searchKnn((void *)(q+i*dim*sizeof(float)), ef);
            }
            auto temp2 = std::chrono::high_resolution_clock::now();
            hnsw_search_time_st += std::chrono::duration_cast<std::chrono::microseconds>(temp2 - temp1).count();
            if(j==0){
                hnsw_search_time_st_min = hnsw_search_time_st;
            }else{
                if(hnsw_search_time_st_min>hnsw_search_time_st){
                    hnsw_search_time_st_min = hnsw_search_time_st;}
            }

            hnsw_search_time_st_all += hnsw_search_time_st ;

            hnsw_search_time_st = 0;
        }
        free(q);
        return std::make_pair(static_cast<double>((double)num_queries/((hnsw_search_time_st_min) / 1e6)),static_cast<double>(static_cast<double>(num_queries)/((hnsw_search_time_st_all)/static_cast<double>(times) / 1e6)));
    }

    inline void draw_recall_qps_graph(mergegraph::HierarchicalNSW<float> *alg_mergegraph,std::vector<std::unordered_set<mergegraph::labeltype>> groundtruth, std::vector<float *> queries, int k,std::string name){
        std::vector<float> target_recall = {0.9,0.91,0.92,0.93,0.94,0.95,0.96,0.97,0.98,0.99};
        for(auto recall:target_recall){
            auto ef = quick_get_ef(alg_mergegraph,groundtruth,queries,k,recall);
            auto time = calculate_qps(alg_mergegraph, queries, ef.first);
            if(ef.first!=0){
                std::cout<<name<<std::endl;
                std::cout<<"|"<<ef.first<<"|"<<ef.second<<"|"<<time.first<<"|"<<time.second<<"|"<<std::endl;
            }
        }
    }

    inline void draw_recall_qps_graph(mergegraph::HierarchicalNSW<float> *alg_mergegraph_firsrt_half,mergegraph::HierarchicalNSW<float> *alg_mergegraph_second_half,std::vector<std::unordered_set<mergegraph::labeltype>> groundtruth, std::vector<float *> queries, int k,std::string name){
        std::vector<float> target_recall = {0.9,0.91,0.92,0.93,0.94,0.95,0.96,0.97,0.98,0.99};
        for(auto recall:target_recall){
            auto ef = quick_get_ef(alg_mergegraph_firsrt_half,alg_mergegraph_second_half,groundtruth,queries,k,recall);
            auto time = calculate_qps(alg_mergegraph_firsrt_half,alg_mergegraph_second_half, queries, ef.first);
            if(ef.first!=0){
                std::cout<<name<<std::endl;
                std::cout<<"|"<<ef.first<<"|"<<ef.second<<"|"<<time.first<<"|"<<std::endl;
            }
        }
    }


    template <typename  dist_t>
    inline std::vector<std::priority_queue<std::pair<dist_t, mergegraph::labeltype >>> get_groundtruth(mergegraph::BruteforceSearch<dist_t>* alg_brute,std::vector<dist_t*> queries,int k){
        std::vector<std::priority_queue<std::pair<dist_t, mergegraph::labeltype >>> ground_truth(queries.size());
        #pragma omp parallel for schedule(dynamic)
        for (size_t i = 0; i < queries.size(); ++i) {
            auto result_brute = alg_brute->searchKnn(queries[i], k);
            ground_truth[i] = std::move(result_brute);
        }

        return ground_truth;
    }

    inline bool fileExists(const std::string& filename) {
        struct stat buffer;
        return (stat(filename.c_str(), &buffer) == 0);
    }
}