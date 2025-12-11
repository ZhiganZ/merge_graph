#pragma once

#include "visited_list_pool.h"
#include "hnswlib.h"
#include <atomic>
#include <climits>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <random>
#include <stdexcept>
#include <stdlib.h>
#include <assert.h>
#include <unordered_set>
#include <list>
#include <memory>
#include <stack>
#include <queue>
#include <utility>
#include <vector>
#include <functional>
#include <cassert>
#include <iostream>
#include <unordered_set>
#include <unordered_map>
#include <mutex>
#include <memory>
#include <algorithm>
#include <thread>
#include <set>
#include <chrono>
#include <iomanip>      
#include <map>          
#include <xmmintrin.h>
#include <boost/dynamic_bitset.hpp>
#include <omp.h>

namespace mergegraph {

typedef unsigned int tableint;
typedef unsigned int linklistsizeint;

using std::cout;
using std::endl;

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
            threads.push_back(std::thread([&, threadId] {
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
            }));
        }
        for (auto &thread : threads) {
            thread.join();
        }
        if (lastException) {
            std::rethrow_exception(lastException);
        }
    }
}

auto get_time_now(){
    return std::chrono::high_resolution_clock::now();
}

auto print_using_time(std::string s,decltype(std::chrono::high_resolution_clock::now()) start){
    std::cout<<s<<" use "<<std::chrono::duration_cast<std::chrono::nanoseconds>(get_time_now()-start).count()/1e9<<"s"<<std::endl;
}

template<typename dist_t>
class HierarchicalNSW : public AlgorithmInterface<dist_t> {
 public:
    static const tableint MAX_LABEL_OPERATION_LOCKS = 65536;
    static const unsigned char DELETE_MARK = 0x01;

    size_t max_elements_{0};
    mutable std::atomic<size_t> cur_element_count{0};  
    size_t size_data_per_element_{0};
    size_t size_links_per_element_{0};
    mutable std::atomic<size_t> num_deleted_{0};  
    size_t M_{0};
    size_t maxM_{0};
    size_t maxM0_{0};

    size_t ef_construction_{0};
    size_t ef_{ 0 };

    double mult_{0.0}, revSize_{0.0};
    int maxlevel_{0};

    std::unique_ptr<VisitedListPool> visited_list_pool_{nullptr};

    mutable std::vector<std::mutex> label_op_locks_;

    std::mutex global;
    std::vector<std::mutex> link_list_locks_;

    tableint enterpoint_node_{0};
    size_t size_links_level0_{0};
    size_t offsetData_{0}, offsetLevel0_{0}, label_offset_{ 0 };
    size_t offsetNorm_{0};
    size_t size_norm_per_element_{0};

    char *data_level0_memory_{nullptr};
    char *opt_graph_{nullptr};
    char **linkLists_{nullptr};
    std::vector<int> element_levels_;  

    size_t data_size_{0};

    DISTFUNC<dist_t> fstdistfunc_;
    void *dist_func_param_{nullptr};

    mutable std::mutex label_lookup_lock;  
    std::unordered_map<labeltype, tableint> label_lookup_;

    std::default_random_engine level_generator_;
    std::default_random_engine update_probability_generator_;

    mutable std::atomic<long> metric_distance_computations{0};
    mutable std::atomic<long> metric_hops{0};

    bool allow_replace_deleted_ = false;  

    std::mutex deleted_elements_lock;  
    std::set<tableint> deleted_elements;  

    HierarchicalNSW(SpaceInterface<dist_t> *s) {
    }

    HierarchicalNSW(
        SpaceInterface<dist_t> *s,
        const std::string &location,
        bool nmslib = false,
        size_t max_elements = 0,
        bool allow_replace_deleted = false)
        : allow_replace_deleted_(allow_replace_deleted) {
        loadIndex(location, s, max_elements);
    }

    HierarchicalNSW(
        SpaceInterface<dist_t> *s,
        size_t max_elements,
        size_t M = 16,
        size_t ef_construction = 200,
        size_t random_seed = 100,
        bool allow_replace_deleted = false)
        : label_op_locks_(MAX_LABEL_OPERATION_LOCKS),
            link_list_locks_(max_elements),
            element_levels_(max_elements),
            allow_replace_deleted_(allow_replace_deleted) {
        max_elements_ = max_elements;
        for(int i = 0;i<max_elements;i++){
            element_levels_[i]=-1;
        }
        num_deleted_ = 0;
        data_size_ = s->get_data_size();
        fstdistfunc_ = s->get_dist_func();


        dist_func_param_ = s->get_dist_func_param();
        if ( M <= 10000 ) {
            M_ = M;
        } else {
            HNSWERR << "warning: M parameter exceeds 10000 which may lead to adverse effects." << std::endl;
            HNSWERR << "         Cap to 10000 will be applied for the rest of the processing." << std::endl;
            M_ = 10000;
        }
        maxM_ = M_;
        maxM0_ = M_ * 2;
        ef_construction_ = std::max(ef_construction, M_);
        ef_ = 10;

        level_generator_.seed(random_seed);
        update_probability_generator_.seed(random_seed + 1);

        size_links_level0_ = maxM0_ * sizeof(tableint) + sizeof(linklistsizeint);
        size_data_per_element_ = size_links_level0_ + data_size_ + sizeof(labeltype);
        offsetData_ = size_links_level0_;
        label_offset_ = size_links_level0_ + data_size_;
        offsetLevel0_ = 0;

        data_level0_memory_ = (char *) malloc(max_elements_ * size_data_per_element_);
        if (data_level0_memory_ == nullptr)
            throw std::runtime_error("Not enough memory");

        cur_element_count = 0;

        visited_list_pool_ = std::unique_ptr<VisitedListPool>(new VisitedListPool(1, max_elements));

        enterpoint_node_ = -1;
        maxlevel_ = -1;

        linkLists_ = (char **) malloc(sizeof(void *) * max_elements_);
        if (linkLists_ == nullptr)
            throw std::runtime_error("Not enough memory: HierarchicalNSW failed to allocate linklists");
        size_links_per_element_ = maxM_ * sizeof(tableint) + sizeof(linklistsizeint);

        mult_ = 1 / log(1.0 * M_);
        revSize_ = 1.0 / mult_;
    }


    ~HierarchicalNSW() {
        clear();
    }

    void clear() {
        free(data_level0_memory_);
        data_level0_memory_ = nullptr;
        for (tableint i = 0; i < cur_element_count; i++) {
            if (element_levels_[i] > 0)
                free(linkLists_[i]);
        }
        free(linkLists_);
        linkLists_ = nullptr;
        cur_element_count = 0;
        visited_list_pool_.reset(nullptr);
        label_lookup_.clear();
    }

    struct CompareByFirst {
        constexpr bool operator()(std::pair<dist_t, tableint> const& a,
            std::pair<dist_t, tableint> const& b) const noexcept {
            return a.first < b.first;
        }
    };

    void setEf(size_t ef) {
        ef_ = ef;
    }

    inline std::mutex& getLabelOpMutex(labeltype label) const {
        size_t lock_id = label & (MAX_LABEL_OPERATION_LOCKS - 1);
        return label_op_locks_[lock_id];
    }

    inline labeltype getExternalLabel(tableint internal_id) const {
        labeltype return_label;
        memcpy(&return_label, (data_level0_memory_ + internal_id * size_data_per_element_ + label_offset_), sizeof(labeltype));
        return return_label;
    }

    inline void setExternalLabel(tableint internal_id, labeltype label) const {
        memcpy((data_level0_memory_ + internal_id * size_data_per_element_ + label_offset_), &label, sizeof(labeltype));
    }

    inline labeltype *getExternalLabeLp(tableint internal_id) const {
        return (labeltype *) (data_level0_memory_ + internal_id * size_data_per_element_ + label_offset_);
    }

    inline char *getDataByInternalId(tableint internal_id) const {
        return (data_level0_memory_ + internal_id * size_data_per_element_ + offsetData_);
    }

    int getRandomLevel(double reverse_size) {
        std::uniform_real_distribution<double> distribution(0.0, 1.0);
        double r = -log(distribution(level_generator_)) * reverse_size;
        return (int) r;
    }

    inline size_t getMaxElements() {
        return max_elements_;
    }

    inline size_t getCurrentElementCount() {
        return cur_element_count;
    }

    inline size_t getDeletedCount() {
        return num_deleted_;
    }

    std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst>
    searchBaseLayer(tableint ep_id, const void *data_point, int layer) {
        VisitedList *vl = visited_list_pool_->getFreeVisitedList();
        vl_type *visited_array = vl->mass;
        vl_type visited_array_tag = vl->curV;

        std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst> top_candidates;
        std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst> candidateSet;

        dist_t lowerBound;

        if (!isMarkedDeleted(ep_id)) {
            dist_t dist = fstdistfunc_(data_point, getDataByInternalId(ep_id), dist_func_param_);
            top_candidates.emplace(dist, ep_id);
            lowerBound = dist;
            candidateSet.emplace(-dist, ep_id);
        } else {        
            lowerBound = std::numeric_limits<dist_t>::max();
            candidateSet.emplace(-lowerBound, ep_id);
        }

        visited_array[ep_id] = visited_array_tag;

        while (!candidateSet.empty()) {
            std::pair<dist_t, tableint> curr_el_pair = candidateSet.top();
            if ((-curr_el_pair.first) > lowerBound && top_candidates.size() == ef_construction_) {
                break;
            }
            candidateSet.pop();

            tableint curNodeNum = curr_el_pair.second;

            std::unique_lock <std::mutex> lock(link_list_locks_[curNodeNum]);

            int *data;  

            if (layer == 0) {
                data = (int*)get_linklist0(curNodeNum);
            } else {
                data = (int*)get_linklist(curNodeNum, layer);
            }

            size_t size = getListCount((linklistsizeint*)data);
            tableint *datal = (tableint *) (data + 1);
#ifdef USE_SSE
            _mm_prefetch((char *) (visited_array + *(data + 1)), _MM_HINT_T0);
            _mm_prefetch((char *) (visited_array + *(data + 1) + 64), _MM_HINT_T0);
            _mm_prefetch(getDataByInternalId(*datal), _MM_HINT_T0);
            _mm_prefetch(getDataByInternalId(*(datal + 1)), _MM_HINT_T0);
#endif

            for (size_t j = 0; j < size; j++) {
                tableint candidate_id = *(datal + j);
#ifdef USE_SSE
                _mm_prefetch((char *) (visited_array + *(datal + j + 1)), _MM_HINT_T0);
                _mm_prefetch(getDataByInternalId(*(datal + j + 1)), _MM_HINT_T0);
#endif
                if (visited_array[candidate_id] == visited_array_tag) continue;
                visited_array[candidate_id] = visited_array_tag;
                char *currObj1 = (getDataByInternalId(candidate_id));

                dist_t dist1 = fstdistfunc_(data_point, currObj1, dist_func_param_);
                if (top_candidates.size() < ef_construction_ || lowerBound > dist1) {
                    candidateSet.emplace(-dist1, candidate_id);
#ifdef USE_SSE
                    _mm_prefetch(getDataByInternalId(candidateSet.top().second), _MM_HINT_T0);
#endif
                    if (!isMarkedDeleted(candidate_id))
                        top_candidates.emplace(dist1, candidate_id);
                    if (top_candidates.size() > ef_construction_)
                        top_candidates.pop();
                    if (!top_candidates.empty())
                        lowerBound = top_candidates.top().first;
                }
            }
        }
        visited_list_pool_->releaseVisitedList(vl);

        return top_candidates;
    }

    class SpinLock {
        std::atomic_flag flag = ATOMIC_FLAG_INIT;

    public:
        void lock() noexcept {
            while (flag.test_and_set(std::memory_order_acquire)) {
    #if defined(__cpp_lib_atomic_flag_test)
                while (flag.test(std::memory_order_relaxed)); 
    #else
                std::this_thread::yield(); 
    #endif
            }
        }
        void unlock() noexcept {
            flag.clear(std::memory_order_release);
        }
    };

    class SpinLockGuard {
        SpinLock& lock_;
    public:
        explicit SpinLockGuard(SpinLock& lock) : lock_(lock) { lock_.lock(); }
        ~SpinLockGuard() { lock_.unlock(); }
        SpinLockGuard(const SpinLockGuard&) = delete;
        SpinLockGuard& operator=(const SpinLockGuard&) = delete;
    };

    struct Neighbor {
        tableint id;
        dist_t distance;
        bool flag;

        Neighbor() = default;
        Neighbor(tableint i, dist_t d, bool f) : id(i), distance(d), flag(f) {}

        inline bool operator<(const Neighbor& n) const noexcept {
            return distance < n.distance;
        }
        inline bool operator==(const Neighbor& n) const noexcept {
            return id == n.id;
        }
    };

    struct alignas(64) Neighborhood {
        SpinLock lock_;
        std::vector<Neighbor> candidates_;

        Neighborhood() = default;

        inline unsigned pushHeap(tableint id, dist_t dist) {
            SpinLockGuard guard(lock_);
            for (const auto& c : candidates_) {
                if (c.id == id) return 0;
            }
            if (candidates_.size() >= candidates_.capacity() &&
                dist >= candidates_.front().distance) {
                return 0;
            }
            if (candidates_.size() < candidates_.capacity()) {
                candidates_.emplace_back(id, dist, true);
                std::push_heap(candidates_.begin(), candidates_.end());
            } else {
                std::pop_heap(candidates_.begin(), candidates_.end());
                candidates_.back() = Neighbor(id, dist, true);
                std::push_heap(candidates_.begin(), candidates_.end());
            }
            return 1;
        }

        inline dist_t worst_distance_nolock() const {
            return candidates_.empty() ? std::numeric_limits<dist_t>::max()
                                    : candidates_.front().distance;
        }
    };

    static inline int InsertIntoPool (Neighbor *addr, unsigned K, Neighbor nn) {
        int left=0,right=K-1;
        if(addr[left].distance>nn.distance){
            memmove((char *)&addr[left+1], &addr[left],K * sizeof(Neighbor));
            addr[left] = nn;
            return left;
        }
        if(addr[right].distance<nn.distance){
            addr[K] = nn;
            return K;
        }
        while(left<right-1){
            int mid=(left+right)/2;
            if(addr[mid].distance>nn.distance)right=mid;
            else left=mid;
        }

        while (left > 0){
            if (addr[left].distance < nn.distance) break;
            if (addr[left].id == nn.id) return K + 1;
            left--;
        }
        if(addr[left].id == nn.id||addr[right].id==nn.id)return K+1;
        memmove((char *)&addr[right+1], &addr[right],(K-right) * sizeof(Neighbor));
        addr[right]=nn;
        return right;
    }

    template <bool bare_bone_search = true, bool collect_metrics = true>
    std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst>
    searchBaseLayerST(
        tableint ep_id,
        const void *data_point,
        size_t ef,
        BaseFilterFunctor* isIdAllowed = nullptr,
        BaseSearchStopCondition<dist_t>* stop_condition = nullptr) const {
        VisitedList *vl = visited_list_pool_->getFreeVisitedList();
        vl_type *visited_array = vl->mass;
        vl_type visited_array_tag = vl->curV;

        std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst> top_candidates;
        std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst> candidate_set;

        dist_t lowerBound;
        if (bare_bone_search || 
            (!isMarkedDeleted(ep_id) && ((!isIdAllowed) || (*isIdAllowed)(getExternalLabel(ep_id))))) {
            char* ep_data = getDataByInternalId(ep_id);
            dist_t dist = fstdistfunc_(data_point, ep_data, dist_func_param_);
            lowerBound = dist;
            top_candidates.emplace(dist, ep_id);
            if (!bare_bone_search && stop_condition) {
                stop_condition->add_point_to_result(getExternalLabel(ep_id), ep_data, dist);
            }
            candidate_set.emplace(-dist, ep_id);
        } else {
            lowerBound = std::numeric_limits<dist_t>::max();
            candidate_set.emplace(-lowerBound, ep_id);
        }

        visited_array[ep_id] = visited_array_tag;

        while (!candidate_set.empty()) {
            std::pair<dist_t, tableint> current_node_pair = candidate_set.top();
            dist_t candidate_dist = -current_node_pair.first;

            bool flag_stop_search;
            if (bare_bone_search) {
                flag_stop_search = candidate_dist > lowerBound;
            } else {
                if (stop_condition) {
                    flag_stop_search = stop_condition->should_stop_search(candidate_dist, lowerBound);
                } else {
                    flag_stop_search = candidate_dist > lowerBound && top_candidates.size() == ef;
                }
            }
            if (flag_stop_search) {
                break;
            }
            candidate_set.pop();

            tableint current_node_id = current_node_pair.second;
            int *data = (int *) get_linklist0(current_node_id);
            size_t size = getListCount((linklistsizeint*)data);
            if (collect_metrics) {
                metric_hops++;
                metric_distance_computations+=size;
            }

#ifdef USE_SSE
            _mm_prefetch((char *) (visited_array + *(data + 1)), _MM_HINT_T0);
            _mm_prefetch((char *) (visited_array + *(data + 1) + 64), _MM_HINT_T0);
            _mm_prefetch(data_level0_memory_ + (*(data + 1)) * size_data_per_element_ + offsetData_, _MM_HINT_T0);
            _mm_prefetch((char *) (data + 2), _MM_HINT_T0);
#endif

            for (size_t j = 1; j <= size; j++) {
                int candidate_id = *(data + j);
#ifdef USE_SSE
                _mm_prefetch((char *) (visited_array + *(data + j + 1)), _MM_HINT_T0);
                _mm_prefetch(data_level0_memory_ + (*(data + j + 1)) * size_data_per_element_ + offsetData_,
                                _MM_HINT_T0);  
#endif
                if (!(visited_array[candidate_id] == visited_array_tag)) {
                    visited_array[candidate_id] = visited_array_tag;

                    char *currObj1 = (getDataByInternalId(candidate_id));
                    dist_t dist = fstdistfunc_(data_point, currObj1, dist_func_param_);

                    bool flag_consider_candidate;
                    if (!bare_bone_search && stop_condition) {
                        flag_consider_candidate = stop_condition->should_consider_candidate(dist, lowerBound);
                    } else {
                        flag_consider_candidate = top_candidates.size() < ef || lowerBound > dist;
                    }

                    if (flag_consider_candidate) {
                        candidate_set.emplace(-dist, candidate_id);
#ifdef USE_SSE
                        _mm_prefetch(data_level0_memory_ + candidate_set.top().second * size_data_per_element_ +
                                        offsetLevel0_,  
                                        _MM_HINT_T0);  
#endif

                        if (bare_bone_search || 
                            (!isMarkedDeleted(candidate_id) && ((!isIdAllowed) || (*isIdAllowed)(getExternalLabel(candidate_id))))) {
                            top_candidates.emplace(dist, candidate_id);
                            if (!bare_bone_search && stop_condition) {
                                stop_condition->add_point_to_result(getExternalLabel(candidate_id), currObj1, dist);
                            }
                        }

                        bool flag_remove_extra = false;
                        if (!bare_bone_search && stop_condition) {
                            flag_remove_extra = stop_condition->should_remove_extra();
                        } else {
                            flag_remove_extra = top_candidates.size() > ef;
                        }
                        while (flag_remove_extra) {
                            tableint id = top_candidates.top().second;
                            top_candidates.pop();
                            if (!bare_bone_search && stop_condition) {
                                stop_condition->remove_point_from_result(getExternalLabel(id), getDataByInternalId(id), dist);
                                flag_remove_extra = stop_condition->should_remove_extra();
                            } else {
                                flag_remove_extra = top_candidates.size() > ef;
                            }
                        }

                        if (!top_candidates.empty())
                            lowerBound = top_candidates.top().first;
                    }
                }
            }
        }

        visited_list_pool_->releaseVisitedList(vl);
        return top_candidates;
    }

    template <bool bare_bone_search = true, bool collect_metrics = false>
    std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst>
    searchBaseLayer_with_visited_node(
        tableint ep_id,
        const void *data_point,
        size_t ef,
        std::vector<std::pair<tableint,dist_t>>& visited_nodes ,
        BaseFilterFunctor* isIdAllowed = nullptr,
        BaseSearchStopCondition<dist_t>* stop_condition = nullptr) const noexcept {
        VisitedList *vl = visited_list_pool_->getFreeVisitedList();
        vl_type *visited_array = vl->mass;
        vl_type visited_array_tag = vl->curV;

        std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst> top_candidates;
        std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst> candidate_set;

        dist_t lowerBound;
        if (bare_bone_search || 
            (!isMarkedDeleted(ep_id) && ((!isIdAllowed) || (*isIdAllowed)(getExternalLabel(ep_id))))) {
            char* ep_data = getDataByInternalId(ep_id);
            dist_t dist = fstdistfunc_(data_point, ep_data, dist_func_param_);
            if(!isMarkedDeleted(ep_id))
                visited_nodes.emplace_back(ep_id,dist);
            lowerBound = dist;
            top_candidates.emplace(dist, ep_id);
            if (!bare_bone_search && stop_condition) {
                stop_condition->add_point_to_result(getExternalLabel(ep_id), ep_data, dist);
            }
            candidate_set.emplace(-dist, ep_id);
        } else {
            lowerBound = std::numeric_limits<dist_t>::max();
            candidate_set.emplace(-lowerBound, ep_id);
        }

        visited_array[ep_id] = visited_array_tag;

        while (!candidate_set.empty()) {
            std::pair<dist_t, tableint> current_node_pair = candidate_set.top();
            dist_t candidate_dist = -current_node_pair.first;
            if(!isMarkedDeleted(current_node_pair.second))
                visited_nodes.emplace_back(current_node_pair.second,candidate_dist);

            bool flag_stop_search;
            if (bare_bone_search) {
                flag_stop_search = candidate_dist > lowerBound;
            } else {
                if (stop_condition) {
                    flag_stop_search = stop_condition->should_stop_search(candidate_dist, lowerBound);
                } else {
                    flag_stop_search = candidate_dist > lowerBound && top_candidates.size() == ef;
                }
            }
            if (flag_stop_search) {
                break;
            }
            candidate_set.pop();

            tableint current_node_id = current_node_pair.second;
            int *data = (int *) get_linklist0(current_node_id);
            size_t size = getListCount((linklistsizeint*)data);
            if (collect_metrics) {
                metric_hops++;
                metric_distance_computations+=size;
            }

#ifdef USE_SSE
            _mm_prefetch((char *) (visited_array + *(data + 1)), _MM_HINT_T0);
            _mm_prefetch((char *) (visited_array + *(data + 1) + 64), _MM_HINT_T0);
            _mm_prefetch(data_level0_memory_ + (*(data + 1)) * size_data_per_element_ + offsetData_, _MM_HINT_T0);
            _mm_prefetch((char *) (data + 2), _MM_HINT_T0);
#endif

            for (size_t j = 1; j <= size; j++) {
                int candidate_id = *(data + j);
#ifdef USE_SSE
                _mm_prefetch((char *) (visited_array + *(data + j + 1)), _MM_HINT_T0);
                _mm_prefetch(data_level0_memory_ + (*(data + j + 1)) * size_data_per_element_ + offsetData_,
                                _MM_HINT_T0);  
#endif
                if (!(visited_array[candidate_id] == visited_array_tag)) {
                    visited_array[candidate_id] = visited_array_tag;

                    char *currObj1 = (getDataByInternalId(candidate_id));
                    dist_t dist = fstdistfunc_(data_point, currObj1, dist_func_param_);

                    bool flag_consider_candidate;
                    if (!bare_bone_search && stop_condition) {
                        flag_consider_candidate = stop_condition->should_consider_candidate(dist, lowerBound);
                    } else {
                        flag_consider_candidate = top_candidates.size() < ef || lowerBound > dist;
                    }

                    if (flag_consider_candidate) {
                        candidate_set.emplace(-dist, candidate_id);
#ifdef USE_SSE
                        _mm_prefetch(data_level0_memory_ + candidate_set.top().second * size_data_per_element_ +
                                        offsetLevel0_,  
                                        _MM_HINT_T0);  
#endif

                        if (bare_bone_search || 
                            (!isMarkedDeleted(candidate_id) && ((!isIdAllowed) || (*isIdAllowed)(getExternalLabel(candidate_id))))) {
                            top_candidates.emplace(dist, candidate_id);
                            if (!bare_bone_search && stop_condition) {
                                stop_condition->add_point_to_result(getExternalLabel(candidate_id), currObj1, dist);
                            }
                        }

                        bool flag_remove_extra = false;
                        if (!bare_bone_search && stop_condition) {
                            flag_remove_extra = stop_condition->should_remove_extra();
                        } else {
                            flag_remove_extra = top_candidates.size() > ef;
                        }
                        while (flag_remove_extra) {
                            tableint id = top_candidates.top().second;
                            top_candidates.pop();
                            if (!bare_bone_search && stop_condition) {
                                stop_condition->remove_point_from_result(getExternalLabel(id), getDataByInternalId(id), dist);
                                flag_remove_extra = stop_condition->should_remove_extra();
                            } else {
                                flag_remove_extra = top_candidates.size() > ef;
                            }
                        }
                        if (!top_candidates.empty())
                            lowerBound = top_candidates.top().first;
                    }
                }
            }
        }

        visited_list_pool_->releaseVisitedList(vl);
        return top_candidates;
    }

    float alpha = 1.0;
    void set_alpha(float a){
        alpha = a;
    }

    void getNeighborsByHeuristic2_VamanaStyle(
    std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst> &top_candidates,
    const size_t M,
    bool tag = true)
    {
        if (tag && top_candidates.size() <= M) {
            return;
        }

        std::vector<std::pair<dist_t, tableint>> candidate_pool;
        candidate_pool.reserve(top_candidates.size());
        while (!top_candidates.empty()) {
            candidate_pool.push_back(top_candidates.top());
            top_candidates.pop();
        }
        std::reverse(candidate_pool.begin(), candidate_pool.end());


        std::vector<std::pair<dist_t, tableint>> return_list;
        return_list.reserve(M);

        std::vector<float> occlusion_factors(candidate_pool.size(), 0.0f);


        float cur_alpha = 1.0f;
        while (cur_alpha <= alpha && return_list.size() < M) {

            for (size_t i = 0; i < candidate_pool.size() && return_list.size() < M; ++i) {

                if (occlusion_factors[i] > cur_alpha) {
                    continue;
                }

                occlusion_factors[i] = std::numeric_limits<float>::max();
                const auto& curent_pair = candidate_pool[i];
                return_list.push_back(curent_pair);

                for (size_t j = i + 1; j < candidate_pool.size(); ++j) {
                    if (occlusion_factors[j] > alpha) {
                        continue;
                    }

                    const auto& other_pair = candidate_pool[j];

                    dist_t dist_between_candidates =
                            fstdistfunc_(getDataByInternalId(curent_pair.second),
                                        getDataByInternalId(other_pair.second),
                                        dist_func_param_);

                    dist_t dist_to_query_of_other = other_pair.first;

                    if (dist_between_candidates < 1e-6) continue;

                    float current_occlusion_ratio = dist_to_query_of_other / dist_between_candidates;

                    if (current_occlusion_ratio > occlusion_factors[j]) {
                        occlusion_factors[j] = current_occlusion_ratio;
                    }
                }
            }
            cur_alpha *= 1.2f;
        }

        for (const auto& pair : return_list) {
            top_candidates.push(pair);
        }
    }

    void getNeighborsByHeuristic2(
        std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst> &top_candidates,
        const size_t M,bool tag=true) {
        if (tag && top_candidates.size() < M) {
            return;
        }

        std::priority_queue<std::pair<dist_t, tableint>> queue_closest;
        std::vector<std::pair<dist_t, tableint>> return_list;
        while (top_candidates.size() > 0) {
            queue_closest.emplace(-top_candidates.top().first, top_candidates.top().second);
            top_candidates.pop();
        }

        while (queue_closest.size()) {
            if (return_list.size() >= M)
                break;
            std::pair<dist_t, tableint> curent_pair = queue_closest.top();
            dist_t dist_to_query = -curent_pair.first;
            queue_closest.pop();
            bool good = true;

            for (std::pair<dist_t, tableint> second_pair : return_list) {
                dist_t curdist =
                        fstdistfunc_(getDataByInternalId(second_pair.second),
                                        getDataByInternalId(curent_pair.second),
                                        dist_func_param_);
                if(curdist < dist_to_query){
                    good = false;
                    break;
                }
            }
            if (good) {
                return_list.push_back(curent_pair);
            }
        }
        for (std::pair<dist_t, tableint> curent_pair : return_list) {
            top_candidates.emplace(-curent_pair.first, curent_pair.second);
        }
    }

    linklistsizeint *get_linklist0(tableint internal_id) const {
        return (linklistsizeint *) (data_level0_memory_ + internal_id * size_data_per_element_ + offsetLevel0_);
    }

    linklistsizeint *get_linklist0(tableint internal_id, char *data_level0_memory_) const {
        return (linklistsizeint *) (data_level0_memory_ + internal_id * size_data_per_element_ + offsetLevel0_);
    }

    linklistsizeint *get_linklist(tableint internal_id, int level) const {
        return (linklistsizeint *) (linkLists_[internal_id] + (level - 1) * size_links_per_element_);
    }

    linklistsizeint *get_linklist_at_level(tableint internal_id, int level) const {
        return level == 0 ? get_linklist0(internal_id) : get_linklist(internal_id, level);
    }

    // vamana style link
    tableint mutuallyConnectNewElement_(
        const void *data_point,
        tableint cur_c,
        std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst> &top_candidates,
        int level,
        bool isUpdate,bool mul_all = false) {
        size_t Mcurmax = level ? maxM_ : maxM0_;        
        if(mul_all)
            getNeighborsByHeuristic2_VamanaStyle(top_candidates, level==0?Mcurmax:maxM_);
        else
            getNeighborsByHeuristic2_VamanaStyle(top_candidates, level==0?floor(Mcurmax/1.3):maxM_);

        std::vector<tableint> selectedNeighbors;
        selectedNeighbors.reserve(Mcurmax);
        while (top_candidates.size() > 0) {
            selectedNeighbors.push_back(top_candidates.top().second);
            top_candidates.pop();
        }

        tableint next_closest_entry_point = selectedNeighbors.back();

        {
            std::unique_lock<std::mutex> lock(link_list_locks_[cur_c], std::try_to_lock);
            linklistsizeint *ll_cur;
            if (level == 0)
                ll_cur = get_linklist0(cur_c);
            else
                ll_cur = get_linklist(cur_c, level);

            setListCount(ll_cur, selectedNeighbors.size());
            tableint *data = (tableint *) (ll_cur + 1);
            for (size_t idx = 0; idx < selectedNeighbors.size(); idx++) {
                if (data[idx] && !isUpdate)
                    throw std::runtime_error("Possible memory corruption");
                data[idx] = selectedNeighbors[idx];
            }
        }

        for (size_t idx = 0; idx < selectedNeighbors.size(); idx++) {
            std::unique_lock <std::mutex> lock(link_list_locks_[selectedNeighbors[idx]]);

            linklistsizeint *ll_other;
            if (level == 0)
                ll_other = get_linklist0(selectedNeighbors[idx]);
            else
                ll_other = get_linklist(selectedNeighbors[idx], level);

            size_t sz_link_list_other = getListCount(ll_other);

            tableint *data = (tableint *) (ll_other + 1);

            bool is_cur_c_present = false;
            if (isUpdate) {
                for (size_t j = 0; j < sz_link_list_other; j++) {
                    if (data[j] == cur_c) {
                        is_cur_c_present = true;
                        break;
                    }
                }
            }

            if (!is_cur_c_present) {
                if (sz_link_list_other < Mcurmax) {
                    data[sz_link_list_other] = cur_c;
                    setListCount(ll_other, sz_link_list_other + 1);
                } else {

                    dist_t d_max = fstdistfunc_(getDataByInternalId(cur_c), getDataByInternalId(selectedNeighbors[idx]),
                                                dist_func_param_);
                    std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst> candidates;
                    candidates.emplace(d_max, cur_c);

                    for (size_t j = 0; j < sz_link_list_other; j++) {
                        if(isMarkedDeleted(data[j])){
                            continue;
                        }
                        candidates.emplace(
                                fstdistfunc_(getDataByInternalId(data[j]), getDataByInternalId(selectedNeighbors[idx]),
                                                dist_func_param_), data[j]);
                    }

                    getNeighborsByHeuristic2_VamanaStyle(candidates, Mcurmax);
                    int indx = 0;
                    while (candidates.size() > 0) {
                        data[indx] = candidates.top().second;
                        candidates.pop();
                        indx++;
                    }

                    setListCount(ll_other, indx);
                }
            }
        }

        return next_closest_entry_point;
    }

    tableint mutuallyConnectNewElement(
        const void *data_point,
        tableint cur_c,
        std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst> &top_candidates,
        int level,
        bool isUpdate) {
        size_t Mcurmax = level ? maxM_ : maxM0_;        
        getNeighborsByHeuristic2(top_candidates,M_);

        std::vector<tableint> selectedNeighbors;
        selectedNeighbors.reserve(M_);
        while (top_candidates.size() > 0) {
            selectedNeighbors.push_back(top_candidates.top().second);
            top_candidates.pop();
        }

        tableint next_closest_entry_point = selectedNeighbors.back();

        {
            std::unique_lock <std::mutex> lock(link_list_locks_[cur_c], std::defer_lock);
            if (isUpdate) {
                lock.lock();
            }
            linklistsizeint *ll_cur;
            if (level == 0)
                ll_cur = get_linklist0(cur_c);
            else
                ll_cur = get_linklist(cur_c, level);

            if (*ll_cur && !isUpdate) {
                throw std::runtime_error("The newly inserted element should have blank link list");
            }

            setListCount(ll_cur, selectedNeighbors.size());
            tableint *data = (tableint *) (ll_cur + 1);
            for (size_t idx = 0; idx < selectedNeighbors.size(); idx++) {
                if (data[idx] && !isUpdate)
                    throw std::runtime_error("Possible memory corruption");

                data[idx] = selectedNeighbors[idx];
            }
        }

        for (size_t idx = 0; idx < selectedNeighbors.size(); idx++) {
            std::unique_lock <std::mutex> lock(link_list_locks_[selectedNeighbors[idx]]);

            linklistsizeint *ll_other;
            if (level == 0)
                ll_other = get_linklist0(selectedNeighbors[idx]);
            else
                ll_other = get_linklist(selectedNeighbors[idx], level);

            size_t sz_link_list_other = getListCount(ll_other);

            tableint *data = (tableint *) (ll_other + 1);

            bool is_cur_c_present = false;
            if (isUpdate) {
                for (size_t j = 0; j < sz_link_list_other; j++) {
                    if (data[j] == cur_c) {
                        is_cur_c_present = true;
                        break;
                    }
                }
            }

            if (!is_cur_c_present) {
                if (sz_link_list_other < Mcurmax) {
                    data[sz_link_list_other] = cur_c;
                    setListCount(ll_other, sz_link_list_other + 1);
                } else {

                    dist_t d_max = fstdistfunc_(getDataByInternalId(cur_c), getDataByInternalId(selectedNeighbors[idx]),
                                                dist_func_param_);
                    std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst> candidates;
                    candidates.emplace(d_max, cur_c);

                    for (size_t j = 0; j < sz_link_list_other; j++) {
                        candidates.emplace(
                                fstdistfunc_(getDataByInternalId(data[j]), getDataByInternalId(selectedNeighbors[idx]),
                                                dist_func_param_), data[j]);
                    }

                    getNeighborsByHeuristic2(candidates, Mcurmax);

                    int indx = 0;
                    while (candidates.size() > 0) {
                        data[indx] = candidates.top().second;
                        candidates.pop();
                        indx++;
                    }

                    setListCount(ll_other, indx);
                }
            }
        }

        return next_closest_entry_point;
    }

    void resizeIndex(size_t new_max_elements) {
        if (new_max_elements < cur_element_count)
            throw std::runtime_error("Cannot resize, max element is less than the current number of elements");

        visited_list_pool_.reset(new VisitedListPool(1, new_max_elements));

        element_levels_.resize(new_max_elements);

        std::vector<std::mutex>(new_max_elements).swap(link_list_locks_);

        char * data_level0_memory_new = (char *) realloc(data_level0_memory_, new_max_elements * size_data_per_element_);
        if (data_level0_memory_new == nullptr)
            throw std::runtime_error("Not enough memory: resizeIndex failed to allocate base layer");
        data_level0_memory_ = data_level0_memory_new;

        char ** linkLists_new = (char **) realloc(linkLists_, sizeof(void *) * new_max_elements);
        if (linkLists_new == nullptr)
            throw std::runtime_error("Not enough memory: resizeIndex failed to allocate other layers");
        linkLists_ = linkLists_new;

        max_elements_ = new_max_elements;
    }

    size_t indexFileSize() const {
        size_t size = 0;
        size += sizeof(offsetLevel0_);
        size += sizeof(max_elements_);
        size += sizeof(cur_element_count);
        size += sizeof(size_data_per_element_);
        size += sizeof(label_offset_);
        size += sizeof(offsetData_);
        size += sizeof(maxlevel_);
        size += sizeof(enterpoint_node_);
        size += sizeof(maxM_);

        size += sizeof(maxM0_);
        size += sizeof(M_);
        size += sizeof(mult_);
        size += sizeof(ef_construction_);

        size += cur_element_count * size_data_per_element_;

        for (size_t i = 0; i < cur_element_count; i++) {
            unsigned int linkListSize = element_levels_[i] > 0 ? size_links_per_element_ * element_levels_[i] : 0;
            size += sizeof(linkListSize);
            size += linkListSize;
        }
        return size;
    }

    void saveIndex(const std::string &location) {
        std::ofstream output(location, std::ios::binary);
        std::streampos position;

        writeBinaryPOD(output, offsetLevel0_);
        writeBinaryPOD(output, max_elements_);
        writeBinaryPOD(output, cur_element_count);
        writeBinaryPOD(output, size_data_per_element_);
        writeBinaryPOD(output, label_offset_);
        writeBinaryPOD(output, offsetData_);
        writeBinaryPOD(output, maxlevel_);
        writeBinaryPOD(output, enterpoint_node_);
        writeBinaryPOD(output, maxM_);

        writeBinaryPOD(output, maxM0_);
        writeBinaryPOD(output, M_);
        writeBinaryPOD(output, mult_);
        writeBinaryPOD(output, ef_construction_);

        output.write(data_level0_memory_, cur_element_count * size_data_per_element_);

        for (size_t i = 0; i < cur_element_count; i++) {
            unsigned int linkListSize = element_levels_[i] > 0 ? size_links_per_element_ * element_levels_[i] : 0;
            writeBinaryPOD(output, linkListSize);
            if (linkListSize)
                output.write(linkLists_[i], linkListSize);
        }
        output.close();
    }

    void loadIndex(const std::string &location, SpaceInterface<dist_t> *s, size_t max_elements_i = 0) {
        std::ifstream input(location, std::ios::binary);

        if (!input.is_open())
            throw std::runtime_error("Cannot open file");

        clear();
        input.seekg(0, input.end);
        std::streampos total_filesize = input.tellg();
        input.seekg(0, input.beg);

        readBinaryPOD(input, offsetLevel0_);
        readBinaryPOD(input, max_elements_);
        readBinaryPOD(input, cur_element_count);

        size_t max_elements = max_elements_i;
        if (max_elements < cur_element_count)
            max_elements = max_elements_;
        max_elements_ = max_elements;
        readBinaryPOD(input, size_data_per_element_);
        readBinaryPOD(input, label_offset_);
        readBinaryPOD(input, offsetData_);
        readBinaryPOD(input, maxlevel_);
        readBinaryPOD(input, enterpoint_node_);

        readBinaryPOD(input, maxM_);
        readBinaryPOD(input, maxM0_);
        readBinaryPOD(input, M_);
        readBinaryPOD(input, mult_);
        readBinaryPOD(input, ef_construction_);

        data_size_ = s->get_data_size();
        fstdistfunc_ = s->get_dist_func();
        dist_func_param_ = s->get_dist_func_param();

        auto pos = input.tellg();

        input.seekg(cur_element_count * size_data_per_element_, input.cur);
        for (size_t i = 0; i < cur_element_count; i++) {
            if (input.tellg() < 0 || input.tellg() >= total_filesize) {
                throw std::runtime_error("Index seems to be corrupted or unsupported");
            }

            unsigned int linkListSize;
            readBinaryPOD(input, linkListSize);
            if (linkListSize != 0) {
                input.seekg(linkListSize, input.cur);
            }
        }

        if (input.tellg() != total_filesize)
            throw std::runtime_error("Index seems to be corrupted or unsupported");

        input.clear();

        input.seekg(pos, input.beg);
        if(data_level0_memory_!=nullptr)free(data_level0_memory_);
        data_level0_memory_ = (char *) malloc(max_elements * size_data_per_element_);
        if (data_level0_memory_ == nullptr)
            throw std::runtime_error("Not enough memory: loadIndex failed to allocate level0");
        input.read(data_level0_memory_, cur_element_count * size_data_per_element_);

        size_links_per_element_ = maxM_ * sizeof(tableint) + sizeof(linklistsizeint);

        size_links_level0_ = maxM0_ * sizeof(tableint) + sizeof(linklistsizeint);
        std::vector<std::mutex>(max_elements).swap(link_list_locks_);
        std::vector<std::mutex>(MAX_LABEL_OPERATION_LOCKS).swap(label_op_locks_);

        visited_list_pool_.reset(new VisitedListPool(1, max_elements));

        linkLists_ = (char **) malloc(sizeof(void *) * max_elements);
        if (linkLists_ == nullptr)
            throw std::runtime_error("Not enough memory: loadIndex failed to allocate linklists");
        element_levels_ = std::vector<int>(max_elements);
        revSize_ = 1.0 / mult_;
        ef_ = 10;
        for (size_t i = 0; i < cur_element_count; i++) {
            label_lookup_[getExternalLabel(i)] = i;
            unsigned int linkListSize;
            readBinaryPOD(input, linkListSize);
            if (linkListSize == 0) {
                element_levels_[i] = 0;
                linkLists_[i] = nullptr;
            } else {
                element_levels_[i] = linkListSize / size_links_per_element_;
                linkLists_[i] = (char *) malloc(linkListSize);
                if (linkLists_[i] == nullptr)
                    throw std::runtime_error("Not enough memory: loadIndex failed to allocate linklist");
                input.read(linkLists_[i], linkListSize);
            }
        }

        for (size_t i = 0; i < cur_element_count; i++) {
            if (isMarkedDeleted(i)) {
                num_deleted_ += 1;
                if (allow_replace_deleted_) deleted_elements.insert(i);
            }
        }

        input.close();

        return;
    }

    template<typename data_t>
    std::vector<data_t> getDataByLabel(labeltype label) const {
        std::unique_lock <std::mutex> lock_label(getLabelOpMutex(label));

        std::unique_lock <std::mutex> lock_table(label_lookup_lock);
        auto search = label_lookup_.find(label);
        if (search == label_lookup_.end() || isMarkedDeleted(search->second)) {
            throw std::runtime_error("Label not found 1");
        }
        tableint internalId = search->second;
        lock_table.unlock();

        char* data_ptrv = getDataByInternalId(internalId);
        size_t dim = *((size_t *) dist_func_param_);
        std::vector<data_t> data;
        data_t* data_ptr = (data_t*) data_ptrv;
        for (size_t i = 0; i < dim; i++) {
            data.push_back(*data_ptr);
            data_ptr += 1;
        }
        return data;
    }

    void markDelete(labeltype label) {
        std::unique_lock <std::mutex> lock_label(getLabelOpMutex(label));

        std::unique_lock <std::mutex> lock_table(label_lookup_lock);
        auto search = label_lookup_.find(label);
        if (search == label_lookup_.end()) {
            std::cout<<label<<std::endl;
            std::cout<<cur_element_count-num_deleted_<<std::endl;
            throw std::runtime_error("Label not found 2");
        }
        tableint internalId = search->second;
        lock_table.unlock();

        markDeletedInternal(internalId);
    }

    void markDeletedInternal(tableint internalId) {
        if (!isMarkedDeleted(internalId)) {
            unsigned char *ll_cur = ((unsigned char *)get_linklist0(internalId))+2;
            *ll_cur |= DELETE_MARK;
            num_deleted_ += 1;
            if (allow_replace_deleted_) {
                std::unique_lock <std::mutex> lock_deleted_elements(deleted_elements_lock);
                deleted_elements.insert(internalId);
            }
        } else {
            throw std::runtime_error("The requested to delete element is already deleted");
        }
    }

    void unmarkDelete(labeltype label) {
        std::unique_lock <std::mutex> lock_label(getLabelOpMutex(label));

        std::unique_lock <std::mutex> lock_table(label_lookup_lock);
        auto search = label_lookup_.find(label);
        if (search == label_lookup_.end()) {
            throw std::runtime_error("Label not found 2");
        }
        tableint internalId = search->second;
        lock_table.unlock();

        unmarkDeletedInternal(internalId);
    }

    void unmarkDeletedInternal(tableint internalId) {
        assert(internalId < max_elements_);
        if (isMarkedDeleted(internalId)) {
            unsigned char *ll_cur = ((unsigned char *)get_linklist0(internalId)) + 2;
            *ll_cur &= ~DELETE_MARK;
            num_deleted_ -= 1;
            if (allow_replace_deleted_) {
                std::unique_lock <std::mutex> lock_deleted_elements(deleted_elements_lock);
                deleted_elements.erase(internalId);
            }
        } else {
            throw std::runtime_error("The requested to undelete element is not deleted");
        }
    }

    inline bool isMarkedDeleted(tableint internalId) const {
        unsigned char *ll_cur = ((unsigned char*)get_linklist0(internalId)) + 2;
        return *ll_cur & DELETE_MARK;
    }

    inline unsigned short int getListCount(linklistsizeint * ptr) const {
        return *((unsigned short int *)ptr);
    }

    inline void setListCount(linklistsizeint * ptr, unsigned short int size) const {
        *((unsigned short int*)(ptr))=*((unsigned short int *)&size);
    }

    // hnsw style add point
    void addPoint(const void *data_point, labeltype label, bool replace_deleted = false) {
        if ((allow_replace_deleted_ == false) && (replace_deleted == true)) {
            throw std::runtime_error("Replacement of deleted elements is disabled in constructor");
        }

        std::unique_lock <std::mutex> lock_label(getLabelOpMutex(label));
        if (!replace_deleted) {
            addPoint(data_point, label, -1);
            return;
        }
        tableint internal_id_replaced;
        std::unique_lock <std::mutex> lock_deleted_elements(deleted_elements_lock);
        bool is_vacant_place = !deleted_elements.empty();
        if (is_vacant_place) {
            std::unique_lock <std::mutex> lock_label(label_lookup_lock);
            if(label_lookup_.find(label)!=label_lookup_.end()){
                internal_id_replaced = label_lookup_[label];
            }
            else
                internal_id_replaced = *deleted_elements.begin();
            deleted_elements.erase(internal_id_replaced);
        }
        lock_deleted_elements.unlock();

        if (!is_vacant_place) {
            addPoint(data_point, label, -1);
        } else {
            labeltype label_replaced = getExternalLabel(internal_id_replaced);
            setExternalLabel(internal_id_replaced, label);

            std::unique_lock <std::mutex> lock_table(label_lookup_lock);
            label_lookup_.erase(label_replaced);
            label_lookup_[label] = internal_id_replaced;
            lock_table.unlock();

            unmarkDeletedInternal(internal_id_replaced);
            updatePoint(data_point, internal_id_replaced, 1.0);
        }
    }

    void get_one_hop_neigh(tableint node,int & layer, std::unordered_set<tableint> & one_hop_neighs) {
        auto one_hop_neis = getConnectionsWithLock(node, layer);
        for(auto && one_hop_neigh:one_hop_neis){
            if(!isMarkedDeleted(one_hop_neigh))one_hop_neighs.insert(one_hop_neigh);
        }
    }

    void get_one_hop_neigh(tableint node,int & layer, std::vector<tableint> & one_hop_neighs) {
        auto one_hop_neis = getConnectionsWithLock(node, layer);
        for(auto && one_hop_neigh:one_hop_neis){
            if(!isMarkedDeleted(one_hop_neigh))one_hop_neighs.emplace_back(one_hop_neigh);
        }
    }

    void get_two_hop_neigh(tableint node,int & layer, std::unordered_set<tableint> & two_hop_neighs) {
        auto one_hop_neis = getConnectionsWithLock(node, layer);
        for(auto && one_hop_neigh:one_hop_neis){
            if(isMarkedDeleted(one_hop_neigh))continue;
            two_hop_neighs.insert(one_hop_neigh);
            auto thns = getConnectionsWithLock(one_hop_neigh, layer);
            for(auto && thn:thns){
                if(isMarkedDeleted(thn))continue;
                two_hop_neighs.insert(thn);
            }
        }
    }

    void updatePoint(const void *dataPoint, tableint internalId, float updateNeighborProbability) {
        memcpy(getDataByInternalId(internalId), dataPoint, data_size_);

        int maxLevelCopy = maxlevel_;
        tableint entryPointCopy = enterpoint_node_;
        if (entryPointCopy == internalId && cur_element_count == 1)
            return;

        int elemLevel = element_levels_[internalId];
        std::uniform_real_distribution<float> distribution(0.0, 1.0);
        for (int layer = 0; layer <= elemLevel; layer++) {
            std::unordered_set<tableint> sCand;
            std::unordered_set<tableint> sNeigh;
            std::vector<tableint> listOneHop = getConnectionsWithLock(internalId, layer);
            if (listOneHop.size() == 0)
                continue;

            sCand.insert(internalId);

            for (auto&& elOneHop : listOneHop) {
                sCand.insert(elOneHop);

                if (distribution(update_probability_generator_) > updateNeighborProbability)
                    continue;

                sNeigh.insert(elOneHop);
                std::vector<tableint> listTwoHop = getConnectionsWithLock(elOneHop, layer);
                for (auto&& elTwoHop : listTwoHop) {
                    sCand.insert(elTwoHop);
                }
            }

            for (auto&& neigh : sNeigh) {

                std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst> candidates;
                size_t size = sCand.find(neigh) == sCand.end() ? sCand.size() : sCand.size() - 1;  
                size_t elementsToKeep = std::min(ef_construction_, size);
                for (auto&& cand : sCand) {
                    if (cand == neigh)
                        continue;

                    dist_t distance = fstdistfunc_(getDataByInternalId(neigh), getDataByInternalId(cand), dist_func_param_);
                    if (candidates.size() < elementsToKeep) {
                        candidates.emplace(distance, cand);
                    } else {
                        if (distance < candidates.top().first) {
                            candidates.pop();
                            candidates.emplace(distance, cand);
                        }
                    }
                }

                getNeighborsByHeuristic2(candidates, layer == 0 ? maxM0_ : maxM_);
                {
                    std::unique_lock <std::mutex> lock(link_list_locks_[neigh]);
                    linklistsizeint *ll_cur;
                    ll_cur = get_linklist_at_level(neigh, layer);
                    size_t candSize = candidates.size();
                    setListCount(ll_cur, candSize);
                    tableint *data = (tableint *) (ll_cur + 1);
                    for (size_t idx = 0; idx < candSize; idx++) {
                        data[idx] = candidates.top().second;
                        candidates.pop();
                    }
                }
            }
        }

        repairConnectionsForUpdate(dataPoint, entryPointCopy, internalId, elemLevel, maxLevelCopy);
    }

    void repairConnectionsForUpdate(
        const void *dataPoint,
        tableint entryPointInternalId,
        tableint dataPointInternalId,
        int dataPointLevel,
        int maxLevel) {
        tableint currObj = entryPointInternalId;
        if (dataPointLevel < maxLevel) {
            dist_t curdist = fstdistfunc_(dataPoint, getDataByInternalId(currObj), dist_func_param_);
            for (int level = maxLevel; level > dataPointLevel; level--) {
                bool changed = true;
                while (changed) {
                    changed = false;
                    unsigned int *data;
                    std::unique_lock <std::mutex> lock(link_list_locks_[currObj]);
                    data = get_linklist_at_level(currObj, level);
                    int size = getListCount(data);
                    tableint *datal = (tableint *) (data + 1);
#ifdef USE_SSE
                    _mm_prefetch(getDataByInternalId(*datal), _MM_HINT_T0);
#endif
                    for (int i = 0; i < size; i++) {
#ifdef USE_SSE
                        _mm_prefetch(getDataByInternalId(*(datal + i + 1)), _MM_HINT_T0);
#endif
                        tableint cand = datal[i];
                        dist_t d = fstdistfunc_(dataPoint, getDataByInternalId(cand), dist_func_param_);
                        if (d < curdist) {
                            curdist = d;
                            currObj = cand;
                            changed = true;
                        }
                    }
                }
            }
        }

        if (dataPointLevel > maxLevel)
            throw std::runtime_error("Level of item to be updated cannot be bigger than max level");

        for (int level = dataPointLevel; level >= 0; level--) {
            std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst> topCandidates = searchBaseLayer(
                    currObj, dataPoint, level);

            std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst> filteredTopCandidates;
            while (topCandidates.size() > 0) {
                if (topCandidates.top().second != dataPointInternalId)
                    filteredTopCandidates.push(topCandidates.top());

                topCandidates.pop();
            }


            if (filteredTopCandidates.size() > 0) {
                bool epDeleted = isMarkedDeleted(entryPointInternalId);
                if (epDeleted) {
                    filteredTopCandidates.emplace(fstdistfunc_(dataPoint, getDataByInternalId(entryPointInternalId), dist_func_param_), entryPointInternalId);
                    if (filteredTopCandidates.size() > ef_construction_)
                        filteredTopCandidates.pop();
                }

                currObj = mutuallyConnectNewElement(dataPoint, dataPointInternalId, filteredTopCandidates, level, true);
            }
        }
    }

    std::vector<tableint> getConnectionsWithLock(tableint internalId, int level) {
        std::unique_lock <std::mutex> lock(link_list_locks_[internalId]);
        unsigned int *data = get_linklist_at_level(internalId, level);
        int size = getListCount(data);
        std::vector<tableint> result(size);
        tableint *ll = (tableint *) (data + 1);
        memcpy(result.data(), ll, size * sizeof(tableint));
        return result;
    }

    tableint addPoint(const void *data_point, labeltype label, int level) {
        tableint cur_c = 0;
        {
            std::unique_lock <std::mutex> lock_table(label_lookup_lock);
            auto search = label_lookup_.find(label);
            if (search != label_lookup_.end()) {
                tableint existingInternalId = search->second;
                if (allow_replace_deleted_) {
                    if (isMarkedDeleted(existingInternalId)) {
                        throw std::runtime_error("Can't use addPoint to update deleted elements if replacement of deleted elements is enabled.");
                    }
                }
                lock_table.unlock();

                if (isMarkedDeleted(existingInternalId)) {
                    unmarkDeletedInternal(existingInternalId);
                }
                updatePoint(data_point, existingInternalId, 1.0);

                return existingInternalId;
            }

            if (cur_element_count >= max_elements_) {
                std::cout<<cur_element_count<<" "<<max_elements_<<std::endl;
                throw std::runtime_error("The number of elements exceeds the specified limit");
            }

            cur_c = cur_element_count;
            cur_element_count++;
            label_lookup_[label] = cur_c;
        }

        std::unique_lock <std::mutex> lock_el(link_list_locks_[cur_c]);
        int curlevel = getRandomLevel(mult_);
        if (level > 0)
            curlevel = level;

        element_levels_[cur_c] = curlevel;

        std::unique_lock <std::mutex> templock(global);
        int maxlevelcopy = maxlevel_;
        if (curlevel <= maxlevelcopy)
            templock.unlock();
        tableint currObj = enterpoint_node_;
        tableint enterpoint_copy = enterpoint_node_;

        memset(data_level0_memory_ + cur_c * size_data_per_element_ + offsetLevel0_, 0, size_data_per_element_);

        memcpy(getExternalLabeLp(cur_c), &label, sizeof(labeltype));
        memcpy(getDataByInternalId(cur_c), data_point, data_size_);

        if (curlevel) {
            linkLists_[cur_c] = (char *) malloc(size_links_per_element_ * curlevel + 1);
            if (linkLists_[cur_c] == nullptr)
                throw std::runtime_error("Not enough memory: addPoint failed to allocate linklist");
            memset(linkLists_[cur_c], 0, size_links_per_element_ * curlevel + 1);
        }

        if ((signed)currObj != -1) {
            if (curlevel < maxlevelcopy) {
                dist_t curdist = fstdistfunc_(data_point, getDataByInternalId(currObj), dist_func_param_);
                for (int level = maxlevelcopy; level > curlevel; level--) {
                    bool changed = true;
                    while (changed) {
                        changed = false;
                        unsigned int *data;
                        std::unique_lock <std::mutex> lock(link_list_locks_[currObj]);
                        data = get_linklist(currObj, level);
                        int size = getListCount(data);

                        tableint *datal = (tableint *) (data + 1);
                        for (int i = 0; i < size; i++) {
                            tableint cand = datal[i];
                            dist_t d = fstdistfunc_(data_point, getDataByInternalId(cand), dist_func_param_);
                            if (d < curdist) {
                                curdist = d;
                                currObj = cand;
                                changed = true;
                            }
                        }
                    }
                }
            }

            bool epDeleted = isMarkedDeleted(enterpoint_copy);
            for (int level = std::min(curlevel, maxlevelcopy); level >= 0; level--) {
                std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst> top_candidates = searchBaseLayer(
                        currObj, data_point, level);
                if (epDeleted) {
                    top_candidates.emplace(fstdistfunc_(data_point, getDataByInternalId(enterpoint_copy), dist_func_param_), enterpoint_copy);
                    if (top_candidates.size() > ef_construction_)
                        top_candidates.pop();
                }
                currObj = mutuallyConnectNewElement(data_point, cur_c, top_candidates, level, false);
            }
        } else {
            enterpoint_node_ = 0;
            maxlevel_ = curlevel;
        }

        if (curlevel > maxlevelcopy) {
            enterpoint_node_ = cur_c;
            maxlevel_ = curlevel;
        }
        return cur_c;
    }

    std::priority_queue<std::pair<dist_t, labeltype >>
    searchKnn(const void *query_data, size_t k, BaseFilterFunctor* isIdAllowed = nullptr) const {
        std::priority_queue<std::pair<dist_t, labeltype >> result;
        if (cur_element_count == 0) return result;

        tableint currObj = enterpoint_node_;
        dist_t curdist = fstdistfunc_(query_data, getDataByInternalId(enterpoint_node_), dist_func_param_);

        for (int level = maxlevel_; level > 0; level--) {
            bool changed = true;
            while (changed) {
                changed = false;
                unsigned int *data;

                data = (unsigned int *) get_linklist(currObj, level);
                int size = getListCount(data);

                tableint *datal = (tableint *) (data + 1);
                for (int i = 0; i < size; i++) {
                    tableint cand = datal[i];
                    dist_t d = fstdistfunc_(query_data, getDataByInternalId(cand), dist_func_param_);

                    if (d < curdist) {
                        curdist = d;
                        currObj = cand;
                        changed = true;
                    }
                }
            }
        }
        std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst> top_candidates;
        bool bare_bone_search = !num_deleted_ && !isIdAllowed;
        if (bare_bone_search) {
            top_candidates = searchBaseLayerST<true>(
                    currObj, query_data, std::max(ef_, k), isIdAllowed);
        } else {
            top_candidates = searchBaseLayerST<false>(
                    currObj, query_data, std::max(ef_, k), isIdAllowed);
        }
        while (top_candidates.size() > k) {
            top_candidates.pop();
        }
        while (top_candidates.size() > 0) {
            std::pair<dist_t, tableint> rez = top_candidates.top();
            result.push(std::pair<dist_t, labeltype>(rez.first, getExternalLabel(rez.second)));
            top_candidates.pop();
        }
        return result;
    }

    void searchKnn_with_visited_lists(const void *query_data, size_t k , std::vector<std::pair<labeltype,dist_t>>& visited_list, BaseFilterFunctor* isIdAllowed = nullptr) const {

        tableint currObj = enterpoint_node_;
        dist_t curdist = fstdistfunc_(query_data, getDataByInternalId(enterpoint_node_), dist_func_param_);

        for (int level = maxlevel_; level > 0; level--) {
            bool changed = true;
            while (changed) {
                changed = false;
                unsigned int *data;

                data = (unsigned int *) get_linklist(currObj, level);
                int size = getListCount(data);
                tableint *datal = (tableint *) (data + 1);
                for (int i = 0; i < size; i++) {
                    tableint cand = datal[i];
                    dist_t d = fstdistfunc_(query_data, getDataByInternalId(cand), dist_func_param_);

                    if (d < curdist) {
                        curdist = d;
                        currObj = cand;
                        changed = true;
                    }
                }
            }
        }
        std::vector<std::pair<tableint,dist_t>> visited_list___;

        searchBaseLayer_with_visited_node(currObj,query_data,k,visited_list___);
        for(auto && node:visited_list___){
            visited_list.emplace_back(getExternalLabel(node.first),node.second);
        }
        std::partial_sort(visited_list.begin(),visited_list.begin()+k,visited_list.end(),[&](std::pair<mergegraph::labeltype, dist_t> a,std::pair<mergegraph::labeltype, dist_t> b){return a.second<b.second;});
        visited_list.resize(k);

    }

    std::vector<std::pair<dist_t, labeltype >>
    searchStopConditionClosest(
        const void *query_data,
        BaseSearchStopCondition<dist_t>& stop_condition,
        BaseFilterFunctor* isIdAllowed = nullptr) const {
        std::vector<std::pair<dist_t, labeltype >> result;
        if (cur_element_count == 0) return result;

        tableint currObj = enterpoint_node_;
        dist_t curdist = fstdistfunc_(query_data, getDataByInternalId(enterpoint_node_), dist_func_param_);

        for (int level = maxlevel_; level > 0; level--) {
            bool changed = true;
            while (changed) {
                changed = false;
                unsigned int *data;

                data = (unsigned int *) get_linklist(currObj, level);
                int size = getListCount(data);
                metric_hops++;
                metric_distance_computations+=size;

                tableint *datal = (tableint *) (data + 1);
                for (int i = 0; i < size; i++) {
                    tableint cand = datal[i];
                    dist_t d = fstdistfunc_(query_data, getDataByInternalId(cand), dist_func_param_);

                    if (d < curdist) {
                        curdist = d;
                        currObj = cand;
                        changed = true;
                    }
                }
            }
        }

        std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst> top_candidates;
        top_candidates = searchBaseLayerST<false>(currObj, query_data, 0, isIdAllowed, &stop_condition);

        size_t sz = top_candidates.size();
        result.resize(sz);
        while (!top_candidates.empty()) {
            result[--sz] = top_candidates.top();
            top_candidates.pop();
        }

        stop_condition.filter_results(result);

        return result;
    }

    void checkIntegrity() {
        int connections_checked = 0;
        std::vector <int > inbound_connections_num(cur_element_count, 0);
        for (int i = 0; i < cur_element_count; i++) {
            for (int l = 0; l <= element_levels_[i]; l++) {
                linklistsizeint *ll_cur = get_linklist_at_level(i, l);
                int size = getListCount(ll_cur);
                tableint *data = (tableint *) (ll_cur + 1);
                std::unordered_set<tableint> s;
                for (int j = 0; j < size; j++) {
                    assert(data[j] < cur_element_count);
                    assert(data[j] != i);
                    inbound_connections_num[data[j]]++;
                    s.insert(data[j]);
                    connections_checked++;
                }
                assert(s.size() == size);
            }
        }
        if (cur_element_count > 1) {
            int min1 = inbound_connections_num[0], max1 = inbound_connections_num[0];
            for (int i=0; i < cur_element_count; i++) {
                assert(inbound_connections_num[i] > 0);
                min1 = std::min(inbound_connections_num[i], min1);
                max1 = std::max(inbound_connections_num[i], max1);
            }
            std::cout << "Min inbound: " << min1 << ", Max inbound:" << max1 << "\n";
        }
        std::cout << "integrity ok, checked " << connections_checked << " connections\n";
    }

    // quick check zero indegree node size and total link size
    uint64_t quick_check() {
        size_t total_size = cur_element_count;
        unsigned int num_threads = std::thread::hardware_concurrency();
        if (num_threads == 0) num_threads = 4;

        std::vector<std::vector<uint64_t>> local_inbound(num_threads, std::vector<uint64_t>(total_size, 0));

        ParallelFor(0, cur_element_count, 0, [&](size_t i, size_t threadId) {
            if (isMarkedDeleted(i))
                return;
            for (int level = 0; level <= element_levels_[i]; level++) {
                linklistsizeint* ll = get_linklist_at_level(i, level);
                if (!ll)
                    continue;
                int count = getListCount(ll);
                tableint* data = reinterpret_cast<tableint*>(ll + 1);
                for (int j = 0; j < count; j++) {
                    int target = data[j];
                    if (target >= total_size || target == static_cast<int>(i))
                        continue;
                    local_inbound[threadId][target]++;
                }
            }
        });

        uint64_t total_inbound = 0;
        std::vector<uint64_t> inbound(total_size, 0);
        for (unsigned int t = 0; t < num_threads; t++) {
            for (size_t i = 0; i < total_size; i++) {
                inbound[i] += local_inbound[t][i];
                total_inbound += local_inbound[t][i];
            }
        }
        uint64_t zero_inbound_count = 0;
        #pragma omp parallel for reduction(+:zero_inbound_count) \
                    schedule(dynamic)
        for(int i=0;i<total_size;i++){
            if (inbound[i] <= 0&&!isMarkedDeleted(i)) {
                zero_inbound_count++;
            }
        }
        std::cout<<"max indegree is:"<<*std::max_element(inbound.begin(),inbound.end())<<std::endl;
        std::cout<<"max indegree node label is :"<<std::max_element(inbound.begin(),inbound.end())-inbound.begin()<<std::endl;
        std::cout<<"check "<<total_inbound<<" connections"<<std::endl;
        std::cout<<"zero indegree node size is: "<<zero_inbound_count<<std::endl;
        return total_inbound;
    }

    // mutually link after prune
    void symmetrizeAndPruneAllLevelsMT(size_t num_threads = 0, size_t shards = 0) {
        alpha = 1.2f;
        const tableint N = static_cast<tableint>(cur_element_count);
        if (N == 0) return;

        if (num_threads == 0) num_threads = std::max<size_t>(1, std::thread::hardware_concurrency());
        if (shards == 0) shards = num_threads;

        auto shard_of = [&](tableint v) -> size_t {
            size_t B = (N + shards - 1) / shards;
            size_t s = std::min<size_t>(v / B, shards - 1);
            return s;
        };

        struct Edge { tableint v; tableint u; }; 

        int Lmax = maxlevel_;
        if (Lmax < 0) return;

        std::vector<std::vector<std::vector<Edge>>> tls; 
        tls.resize(num_threads);
        for (auto &v : tls) v.resize(shards);

        auto collect_level = [&](int level, size_t tid, tableint beg, tableint end) {
            auto &my_bins = tls[tid];
            std::vector<tableint> neigh;
            for (tableint u = beg; u < end && u < N; ++u) {
                if (isMarkedDeleted(u)) continue;
                if (element_levels_[u] < level) continue;

                linklistsizeint* llu;
                {
                    std::unique_lock<std::mutex> lu(link_list_locks_[u]);
                    llu = get_linklist_at_level(u, level);
                    int su = getListCount(llu);
                    tableint* datau = (tableint*)(llu + 1);
                    neigh.assign(datau, datau + su);
                }

                for (tableint v : neigh) {
                    if (v == u) continue;
                    if (v >= N) continue;
                    if (isMarkedDeleted(v)) continue;
                    size_t s = shard_of(v);
                    my_bins[s].push_back({v, u});
                }
            }
        };

        auto prune_and_write = [&](tableint u, int level,
                                const tableint* rev_begin, const tableint* rev_end)
        {
            linklistsizeint* llu = get_linklist_at_level(u, level);
            tableint* datau = (tableint*)(llu + 1);
            size_t su = getListCount(llu);

            std::vector<tableint> cand;
            cand.reserve(su + (size_t)(rev_end - rev_begin));
            for (size_t j = 0; j < su; ++j) {
                tableint w = datau[j];
                if (w == u || w >= N || isMarkedDeleted(w)) continue;
                cand.push_back(w);
            }
            for (auto p = rev_begin; p != rev_end; ++p) {
                tableint w = *p;
                if (w == u || w >= N || isMarkedDeleted(w)) continue;
                cand.push_back(w);
            }

            if (cand.empty()) {
                setListCount(llu, 0);
                return;
            }

            std::sort(cand.begin(), cand.end());
            cand.erase(std::unique(cand.begin(), cand.end()), cand.end());

            const size_t Mcurmax = (level == 0) ? maxM0_ : maxM_;

            if (cand.size() > Mcurmax) {
                std::priority_queue<std::pair<dist_t, tableint>,
                                    std::vector<std::pair<dist_t, tableint>>,
                                    CompareByFirst> pq;
                pq = decltype(pq)(); 
                for (tableint w : cand) {
                    dist_t d = fstdistfunc_(getDataByInternalId(w),
                                            getDataByInternalId(u),
                                            dist_func_param_);
                    pq.emplace(d, w);
                }
                getNeighborsByHeuristic2_VamanaStyle(pq, Mcurmax);
                size_t k = pq.size();
                for (size_t i = 0; i < k; ++i) {
                    datau[i] = pq.top().second;
                    pq.pop();
                }
                setListCount(llu, static_cast<unsigned short>(k));
            } else {
                for (size_t i = 0; i < cand.size(); ++i) datau[i] = cand[i];
                setListCount(llu, static_cast<unsigned short>(cand.size()));
            }
        };

        for (int level = Lmax; level >= 0; --level) {
            const size_t chunk = (N + num_threads - 1) / num_threads;
            std::vector<std::thread> th;
            th.reserve(num_threads);
            for (size_t t = 0; t < num_threads; ++t) {
                tableint beg = static_cast<tableint>(t * chunk);
                tableint end = static_cast<tableint>(std::min((t + 1) * chunk, (size_t)N));
                th.emplace_back(collect_level, level, t, beg, end);
            }
            for (auto &x : th) x.join();

            const size_t B = (N + shards - 1) / shards;

            auto process_shard = [&](size_t s) {
                tableint v0 = static_cast<tableint>(s * B);
                tableint v1 = static_cast<tableint>(std::min((s + 1) * B, (size_t)N));
                const size_t span = v1 - v0;
                if (span == 0) return;

                std::vector<Edge> edges;
                size_t total = 0;
                for (size_t t = 0; t < num_threads; ++t) total += tls[t][s].size();
                edges.reserve(total);
                for (size_t t = 0; t < num_threads; ++t) {
                    auto &bin = tls[t][s];
                    edges.insert(edges.end(), bin.begin(), bin.end());
                    bin.clear();
                    bin.shrink_to_fit();
                }
                if (edges.empty()) {
                    for (tableint u = v0; u < v1; ++u) {
                        if (isMarkedDeleted(u) || element_levels_[u] < level) continue;
                        std::unique_lock<std::mutex> lu(link_list_locks_[u]);
                        prune_and_write(u, level, nullptr, nullptr);
                    }
                    return;
                }

                std::vector<uint32_t> counts(span, 0);
                for (const auto &e : edges) {
                    tableint v = e.v;
                    if (v < v0 || v >= v1) continue; 
                    counts[v - v0]++;
                }
                std::vector<uint32_t> prefix(span + 1, 0);
                for (size_t i = 0; i < span; ++i) prefix[i + 1] = prefix[i] + counts[i];

                std::vector<tableint> rev(span ? prefix[span] : 0);
                std::vector<uint32_t> cursor = prefix;

                for (const auto &e : edges) {
                    tableint v = e.v;
                    if (v < v0 || v >= v1) continue;
                    uint32_t &pos = cursor[v - v0];
                    rev[pos++] = e.u;
                }

                for (tableint u = v0; u < v1; ++u) {
                    if (isMarkedDeleted(u) || element_levels_[u] < level) continue;

                    const tableint* rv_begin = nullptr;
                    const tableint* rv_end   = nullptr;
                    if (span) {
                        uint32_t a = prefix[u - v0];
                        uint32_t b = prefix[u - v0 + 1];
                        rv_begin = rev.data() + a;
                        rv_end   = rev.data() + b;
                    }

                    std::unique_lock<std::mutex> lu(link_list_locks_[u]);
                    prune_and_write(u, level, rv_begin, rv_end);
                }
            };

            std::vector<std::thread> shard_threads;
            shard_threads.reserve(shards);
            for (size_t s = 0; s < shards; ++s)
                shard_threads.emplace_back(process_shard, s);
            for (auto &x : shard_threads) x.join();
        }
    }

    // NN descent function
    std::vector<std::vector<Neighbor>>
    update_neighbors(float cap_factor=1.0,float sample_rate = 1.0f,
                    unsigned iter_max = 30,
                    float threshold = 0.002f);

    // NN descent and prune
    void fgim(float sample_rate=0.3,float cap_factor = 1.0){
        alpha = 1.2f;
        auto start = get_time_now();
        std::vector<std::vector<Neighbor>> final_neighbors = update_neighbors(cap_factor,sample_rate);
        int count = 0 ; 
        for(int i = 0 ; i < final_neighbors.size() ; ++i){
            count+=final_neighbors[i].size();
        }
        std::cout<<"num for final neighbors is: "<<count<<std::endl;
        print_using_time("NN descent",start);
        start = get_time_now();
        #pragma omp parallel for
        for(int i = 0 ; i < final_neighbors.size() ; ++i){
            if(final_neighbors[i].size()<maxM0_){
                std::unique_lock <std::mutex> lock(link_list_locks_[i]);
                linklistsizeint *ll_cur;
                ll_cur = get_linklist_at_level(i, 0);
                size_t candSize = final_neighbors[i].size();
                setListCount(ll_cur, candSize);
                tableint *data = (tableint *) (ll_cur + 1);
                for (size_t idx = 0; idx < candSize; idx++) {
                    data[idx] = final_neighbors[i][idx].id;
                }
                continue;
            }
            std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst> top_candidates;
            for(auto && nei:final_neighbors[i]){
                top_candidates.emplace(nei.distance,nei.id);
            }
            getNeighborsByHeuristic2_VamanaStyle(top_candidates,maxM0_,false);

            std::unique_lock <std::mutex> lock(link_list_locks_[i]);
            linklistsizeint *ll_cur;
            ll_cur = get_linklist_at_level(i, 0);
            size_t candSize = top_candidates.size();
            setListCount(ll_cur, candSize);
            tableint *data = (tableint *) (ll_cur + 1);
            for (size_t idx = 0; idx < candSize; idx++) {
                data[idx] = top_candidates.top().second;
                top_candidates.pop();
            }
        }
        print_using_time("prune",start);
    }

    void clear_zero_layer_edge(){
        #pragma omp parallel for schedule(dynamic)
        for (tableint i = 0; i < max_elements_; ++i) {
            for(int layer = 0 ; layer <element_levels_[i] ; ++i){
                linklistsizeint* ll_cur = get_linklist_at_level(i,layer);
                setListCount(ll_cur, (unsigned short)0);
            }
        }
    }

    void rebuild_upper_layers_mt(size_t num_threads = std::thread::hardware_concurrency()) {
        if (cur_element_count == 0) return;
        if (num_threads == 0) num_threads = 1;


        int new_maxlevel = 0;
        for (tableint i = 0; i < cur_element_count; ++i) {
            if (!isMarkedDeleted(i)) new_maxlevel = std::max(new_maxlevel, element_levels_[i]);
        }
        if (new_maxlevel <= 0) { 
            maxlevel_ = 0;
            if (cur_element_count > 0) {
                for (tableint i = 0; i < cur_element_count; ++i) {
                    if (!isMarkedDeleted(i)) { enterpoint_node_ = i; break; }
                }
            }
            return;
        }

        std::vector<std::vector<tableint>> nodes_by_level(new_maxlevel + 1);
        nodes_by_level.reserve(new_maxlevel + 1);
        for (tableint i = 0; i < cur_element_count; ++i) {
            if (isMarkedDeleted(i)) continue;
            int lev = element_levels_[i];
            for (int l = 1; l <= lev; ++l) nodes_by_level[l].push_back(i);
        }

        for (tableint i = 0; i < cur_element_count; ++i) {
            int lev = element_levels_[i];
            if (lev <= 0) continue;
            std::unique_lock<std::mutex> lk(link_list_locks_[i]);
            if (linkLists_[i] == nullptr) {
                linkLists_[i] = (char*)std::malloc(size_links_per_element_ * lev + 1);
                if (!linkLists_[i]) throw std::runtime_error("Not enough memory: rebuild alloc linklist");
            }
            std::memset(linkLists_[i], 0, size_links_per_element_ * lev + 1);
        } 

        for (int level = new_maxlevel; level >= 1; --level) {
            auto &nodes = nodes_by_level[level];
            if (nodes.empty()) continue;

            tableint seed = static_cast<tableint>(-1);
            if (enterpoint_node_ != static_cast<tableint>(-1) &&
                !isMarkedDeleted(enterpoint_node_) &&
                element_levels_[enterpoint_node_] >= level) {
                seed = enterpoint_node_;
            } else {
                seed = nodes.front();
            }

            std::atomic<size_t> cursor{0};

            auto worker = [&]() {
                for (;;) {
                    size_t idx = cursor.fetch_add(1, std::memory_order_relaxed);
                    if (idx >= nodes.size()) return;
                    tableint u = nodes[idx];
                    if (u == seed) continue; 

                    std::unique_lock<std::mutex> lock_u(link_list_locks_[u]);
                    linklistsizeint* llu = get_linklist(u, level);
                    if (getListCount(llu) != 0) setListCount(llu, 0);

                    const void* datap = (const void*)getDataByInternalId(u);
                    tableint currObj = seed;

                    auto topcands = searchBaseLayer(currObj, datap, level);
                    if (isMarkedDeleted(seed)) {
                        topcands.emplace(fstdistfunc_(datap, getDataByInternalId(seed), dist_func_param_), seed);
                        if (topcands.size() > ef_construction_) topcands.pop();
                    }

                    mutuallyConnectNewElement(datap, u, topcands, level, false);
                }
            };

            std::vector<std::thread> threads;
            threads.reserve(num_threads);
            for (size_t t = 0; t < num_threads; ++t) threads.emplace_back(worker);
            for (auto &th : threads) th.join();
        }

        tableint new_ep = enterpoint_node_;
        if (new_ep == static_cast<tableint>(-1) || isMarkedDeleted(new_ep) || element_levels_[new_ep] < new_maxlevel) {
            new_ep = static_cast<tableint>(-1);
            for (tableint i = 0; i < cur_element_count; ++i) {
                if (!isMarkedDeleted(i) && element_levels_[i] == new_maxlevel) { new_ep = i; break; }
            }
            int fallback_level = new_maxlevel;
            while (new_ep == static_cast<tableint>(-1) && fallback_level > 0) {
                --fallback_level;
                for (tableint i = 0; i < cur_element_count; ++i) {
                    if (!isMarkedDeleted(i) && element_levels_[i] == fallback_level) { new_ep = i; break; }
                }
                if (new_ep != static_cast<tableint>(-1)) new_maxlevel = fallback_level;
            }
            if (new_ep != static_cast<tableint>(-1)) {
                enterpoint_node_ = new_ep;
            }
        }
        maxlevel_ = new_maxlevel;
    }


    void random_high_level(){
        #pragma omp parallel for schedule(dynamic)
        for(int cur_c = 0 ; cur_c < cur_element_count ; ++ cur_c ){
            int curlevel = getRandomLevel(mult_);

            element_levels_[cur_c] = curlevel;

            std::unique_lock <std::mutex> templock(global);
            int maxlevelcopy = maxlevel_;
            if (curlevel <= maxlevelcopy)
                templock.unlock();
            tableint currObj = enterpoint_node_;
            tableint enterpoint_copy = enterpoint_node_;
            if (curlevel) {
                std::unique_lock <std::mutex> lock(link_list_locks_[cur_c],std::defer_lock);
                if(linkLists_[cur_c]!=nullptr){
                    lock.lock();
                    free(linkLists_[cur_c]);
                }
                linkLists_[cur_c] = (char *) malloc(size_links_per_element_ * curlevel + 1);
                memset(linkLists_[cur_c], 0, size_links_per_element_ * curlevel + 1);
            }
            if (curlevel > maxlevelcopy) {
                enterpoint_node_ = cur_c;
                maxlevel_ = curlevel;
            }
        }
    }

    // reallocate memory for new degree
    void resize_M(int new_M0) {
        if (new_M0 <= 0) {
            throw std::runtime_error("new_M0 must be > 0");
        }
        if ((size_t)new_M0 == maxM0_) {
            return; 
        }
        if ((size_t)new_M0 > std::numeric_limits<unsigned short>::max()) {
            throw std::runtime_error("new_M0 exceeds 16-bit list count limit");
        }

        char*  old_mem                  = data_level0_memory_;
        size_t old_size_data_per_elem   = size_data_per_element_;
        size_t old_offset_data          = offsetData_;
        size_t old_label_offset         = label_offset_;

        size_t new_size_links_level0    = (size_t)new_M0 * sizeof(tableint) + sizeof(linklistsizeint);
        size_t new_size_data_per_elem   = new_size_links_level0 + data_size_ + sizeof(labeltype);
        size_t new_offset_data          = new_size_links_level0;
        size_t new_label_offset         = new_size_links_level0 + data_size_;

        char* new_mem = (char*) malloc(max_elements_ * new_size_data_per_elem);
        if (new_mem == nullptr) {
            throw std::runtime_error("Not enough memory: resize_M failed to allocate level0");
        }

        for (tableint i = 0; i < cur_element_count; ++i) {
            linklistsizeint* old_ll = get_linklist0(i, old_mem);
            unsigned short old_cnt  = getListCount(old_ll);
            unsigned short copy_cnt = (unsigned short)std::min<size_t>(old_cnt, (size_t)new_M0);

            unsigned char old_flags = *(((unsigned char*)old_ll) + 2);

            char* dst = new_mem + (size_t)i * new_size_data_per_elem;
            linklistsizeint* new_ll = (linklistsizeint*)(dst + offsetLevel0_);

            memset(new_ll, 0, sizeof(linklistsizeint) + (size_t)new_M0 * sizeof(tableint));
            setListCount(new_ll, copy_cnt);
            *(((unsigned char*)new_ll) + 2) = old_flags; 

            if (copy_cnt) {
                tableint* old_links = (tableint*)(old_ll + 1);
                tableint* new_links = (tableint*)(new_ll + 1);
                memcpy(new_links, old_links, (size_t)copy_cnt * sizeof(tableint));
            }

            memcpy(dst + new_offset_data,
                old_mem + (size_t)i * old_size_data_per_elem + old_offset_data,
                data_size_);
            memcpy(dst + new_label_offset,
                old_mem + (size_t)i * old_size_data_per_elem + old_label_offset,
                sizeof(labeltype));
        }

        free(old_mem);
        data_level0_memory_  = new_mem;

        maxM0_               = (size_t)new_M0;
        size_links_level0_   = new_size_links_level0;
        size_data_per_element_ = new_size_data_per_elem;
        offsetData_          = new_offset_data;
        label_offset_        = new_label_offset;
    }

    // function for vamana build
    void vamana_refine(int L = 50,float vamana_alpha = 1.0){
        float tmp_alpha = alpha;
        alpha = vamana_alpha;
        auto start = std::chrono::high_resolution_clock::now();
        int count = 0 ;

        std::vector<std::vector<tableint>> now_link(cur_element_count);
        #pragma omp parallel for schedule(dynamic)
        for(tableint row = 0 ; row < cur_element_count ; ++row){
            if(isMarkedDeleted(row))continue;
            if(row%100000==0){
                std::cout<<"refine: "<<float(row)*100/cur_element_count<<"%, "<<"now time use: "<<std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::high_resolution_clock::now()-start).count()/1e9<<"s"<<std::endl;
            }
            std::vector<std::pair<tableint,dist_t>> visited_nodes;
            searchBaseLayer_with_visited_node(enterpoint_node_,getDataByInternalId(row),L,visited_nodes);
            std::priority_queue<std::pair<dist_t, tableint>, std::vector<std::pair<dist_t, tableint>>, CompareByFirst> link;
            for(auto && el:visited_nodes){
                if(el.second==0)continue;
                link.emplace(el.second,el.first);
            }

            mutuallyConnectNewElement_(getDataByInternalId(row), row, link, 0,true);

        }

        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> elapsed = end - start;
        std::cout << "Refinement completed in " << elapsed.count() << " seconds." << std::endl;
        alpha = tmp_alpha;
    }

    // vamana build function
    void vamana_build(int L = 50,float vamana_alpha=1.2){
        enterpoint_node_ = get_navi_point(true);

        if (element_levels_[enterpoint_node_]<maxlevel_){
            std::cout<<"element_levels_[enterpoint_node_]<maxlevel_"<<std::endl;
            std::cout<<"max level is: "<<maxlevel_<<", ep level is: "<<element_levels_[enterpoint_node_]<<std::endl;
            std::cout<<"ep is: "<<enterpoint_node_<<std::endl;
            element_levels_[enterpoint_node_] = maxlevel_;
            if(linkLists_[enterpoint_node_] != nullptr)
                free(linkLists_[enterpoint_node_]);
            linkLists_[enterpoint_node_] = (char *) malloc(size_links_per_element_ * element_levels_[enterpoint_node_] + 1);
            if (linkLists_[enterpoint_node_] == nullptr)
                throw std::runtime_error("Not enough memory: addPoint failed to allocate linklist");
            memset(linkLists_[enterpoint_node_], 0, size_links_per_element_ * element_levels_[enterpoint_node_] + 1);
        }

        clear_zero_layer_edge();

        vamana_refine(L,vamana_alpha);

    }

    bool is_exist(labeltype label){
        if(label_lookup_.find(label)==label_lookup_.end())return false;
        return true;
    }

    // if index build, use ANN search to calculate centroid, else parallel brute search for centroid
    tableint get_navi_point(bool brute = false) {
        size_t l = ef_construction_ ;
        size_t m = maxM0_ ;

        size_t dim = this->data_size_ / sizeof(dist_t);
        std::vector<dist_t> centroid(dim, 0.0f);
        #pragma omp parallel
        {
            std::vector<dist_t> local_centroid(dim, 0.0f); 

            #pragma omp for nowait
            for (tableint i = 0; i < this->cur_element_count; ++i) {
                if(isMarkedDeleted(i))continue;
                dist_t* data_ptr = (dist_t*)this->getDataByInternalId(i);
                for (size_t d = 0; d < dim; ++d) {
                    local_centroid[d] += data_ptr[d];
                }
            }

            #pragma omp critical
            {
                for (size_t d = 0; d < dim; ++d) {
                    centroid[d] += local_centroid[d];
                }
            }
        }

        for (size_t d = 0; d < dim; ++d) {
            centroid[d] /= static_cast<dist_t>(this->cur_element_count);
        }

        std::uniform_int_distribution<int> distribution(0, this->cur_element_count - 1);
        tableint r = distribution(this->level_generator_);

        tableint new_ep = 0;
        if(!brute){
            auto top_candidate_for_n = this->searchBaseLayerST<true>(r, centroid.data(), 100);
            while(top_candidate_for_n.size()>1)top_candidate_for_n.pop(); 
            new_ep = top_candidate_for_n.top().second;
        }
        if(brute){
            float min = fstdistfunc_(centroid.data(),getDataByInternalId(0),dist_func_param_);
            #pragma omp parallel for
            for(tableint i = 1 ; i < cur_element_count ; ++i){
                float dist = fstdistfunc_(centroid.data(),getDataByInternalId(i),dist_func_param_);
                #pragma omp critical
                {
                    if(dist<min) {
                        min = dist;
                        new_ep = i;
                    }
                }
            }
        }
        std::cout<<"ep: "<<new_ep<<std::endl;

        return new_ep;
    }

    // delete high level link to flatten hnsw to nsw
    void flatten_to_level0() {

        if (this->maxlevel_ == 0) {
            std::cout << " max level is 0 "<<std::endl;
            std::cout << "need not flatten " << std::endl;
            return;
        }

        for (tableint i = 0; i < this->cur_element_count; ++i) {
            if (this->element_levels_[i] > 0) {

                if (this->linkLists_[i] != nullptr) {
                    free(this->linkLists_[i]);
                    this->linkLists_[i] = nullptr; 
                }

                this->element_levels_[i] = 0;
            }
        }

        this->maxlevel_ = 0;

        std::cout << "  success" << std::endl;
    }
};

}

namespace mergegraph{

template<typename dist_t>
std::vector<std::vector<typename HierarchicalNSW<dist_t>::Neighbor>>
HierarchicalNSW<dist_t>::update_neighbors(float cap_factor,float sample_rate, 
                                          unsigned iter_max,
                                          float threshold) {
    using Nbr = typename HierarchicalNSW<dist_t>::Neighbor;
    using Pool = typename HierarchicalNSW<dist_t>::Neighborhood;

    const size_t N = getCurrentElementCount();
    if (N == 0) return {};

    const size_t cap = cap_factor * maxM0_;
    size_t now_cap = cap;

    std::vector<Pool> pools(N);
#pragma omp parallel for schedule(static)
    for (int u = 0; u < (int)N; ++u) {
        auto& pu = pools[u];
        pu.candidates_.reserve(cap);

        linklistsizeint* ll = get_linklist0((tableint)u);
        const size_t deg = getListCount(ll);
        tableint* ll_data = (tableint*)(ll + 1);

#ifdef USE_SSE
        if (deg) _mm_prefetch(getDataByInternalId(ll_data[0]), _MM_HINT_T0);
#endif
        char* u_ptr = getDataByInternalId((tableint)u);
        for (size_t j = 0; j < deg; ++j) {
            tableint v = ll_data[j];
#ifdef USE_SSE
            if (j + 1 < deg) _mm_prefetch(getDataByInternalId(ll_data[j + 1]), _MM_HINT_T0);
#endif
            dist_t d = fstdistfunc_(u_ptr, getDataByInternalId(v), dist_func_param_);
            pu.candidates_.emplace_back(v, d, true);
        }
        std::make_heap(pu.candidates_.begin(), pu.candidates_.end());
    }


        size_t iter = 0;
        while (++iter && iter <= iter_max) {
            size_t cnt_new_edges = 0;

    #pragma omp parallel
            {
                std::vector<Neighbor> new_cand; new_cand.reserve(2 * cap);
                std::vector<Neighbor> old_cand; old_cand.reserve(2 * cap);

    #pragma omp for reduction(+ : cnt_new_edges) schedule(dynamic, 128)
                for (int u = 0; u < (int)N; ++u) {
                    new_cand.clear();
                    old_cand.clear();

                    auto& pu = pools[u];
                    for (auto& neighbor : pu.candidates_) {
                        if (neighbor.flag) {
                            new_cand.push_back(neighbor);
                        } else {
                            old_cand.push_back(neighbor);
                        }
                        neighbor.flag = false; 
                    }


                    for (const auto& p1 : new_cand) {
                        for (const auto& p2 : new_cand) {
                            if (p1.id >= p2.id) continue;
                            dist_t d = fstdistfunc_(getDataByInternalId(p1.id), getDataByInternalId(p2.id), dist_func_param_);
                                cnt_new_edges += pools[p1.id].pushHeap(p2.id, d);
                                cnt_new_edges += pools[p2.id].pushHeap(p1.id, d);
                        }
                        for (const auto& p2 : old_cand) {
                            if (p1.id == p2.id) continue;
                            dist_t d = fstdistfunc_(getDataByInternalId(p1.id), getDataByInternalId(p2.id), dist_func_param_);
                                cnt_new_edges += pools[p1.id].pushHeap(p2.id, d);
                                cnt_new_edges += pools[p2.id].pushHeap(p1.id, d);
                        }
                    }
                } 
            } 

            const unsigned convergence = (unsigned)std::lround(threshold * (double)N * (double)cap);
            if (cnt_new_edges <= (int)convergence) break;
        } 

    std::vector<std::vector<Nbr>> result(N);
#pragma omp parallel for schedule(dynamic, 256)
    for (int u = 0; u < (int)N; ++u) {
        auto cand = pools[u].candidates_;
        result[u] = std::move(cand);
    }

    return result;
}

}
