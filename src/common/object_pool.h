/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

/*!
 * Copyright (c) 2015 by Contributors
 */
#ifndef MXNET_COMMON_OBJECT_POOL_H_
#define MXNET_COMMON_OBJECT_POOL_H_
#include <dmlc/logging.h>
#include <dmlc/concurrentqueue.h>
#include <cstdlib>
#include <mutex>
#include <utility>
#include <vector>

#define LOCKFREE_OBJECT_POOL

namespace mxnet {
namespace common {

template <typename N>
struct FreeListNode {
  FreeListNode()
    : freeListRefs_(0)
      , next_(nullptr) {}

  std::atomic<std::uint32_t> freeListRefs_;
  std::atomic<N *> next_;
};

// A simple CAS-based lock-free free list. Not the fastest thing in the world under heavy contention,
// but simple and correct (assuming nodes are never freed until after the free list is destroyed),
// and fairly speedy under low contention.
template<typename N>    // N must inherit FreeListNode or have the same fields (and initialization)
struct FreeList
{
  FreeList() : freeListHead_(nullptr) { }

  inline void add(N* node) {
    // We know that the should-be-on-freelist bit is 0 at this point, so it's safe to
    // set it using a fetch_add
    if (node->freeListRefs.fetch_add(SHOULD_BE_ON_FREELIST, std::memory_order_release) == 0) {
      // Oh look! We were the last ones referencing this node, and we know
      // we want to add it to the free list, so let's do it!
      add_knowing_refcount_is_zero(node);
    }
  }

  inline N* try_get() {
    auto head = freeListHead_.load(std::memory_order_acquire);
    while (head != nullptr) {
      auto prevHead = head;
      auto refs = head->freeListRefs.load(std::memory_order_relaxed);
      if ((refs & REFS_MASK) == 0
          || !head->freeListRefs.compare_exchange_strong(refs, refs + 1,
                                                         std::memory_order_acquire,
                                                         std::memory_order_relaxed)) {
        head = freeListHead_.load(std::memory_order_acquire);
        continue;
      }

      // Good, reference count has been incremented (it wasn't at zero), which means
      // we can read the next and not worry about it changing between now and the time
      // we do the CAS
      auto next = head->freeListNext.load(std::memory_order_relaxed);
      if (freeListHead_.compare_exchange_strong(head, next,
                                                std::memory_order_acquire,
                                                std::memory_order_relaxed)) {
        // Yay, got the node. This means it was on the list, which means
        // shouldBeOnFreeList must be false no matter the refcount (because
        // nobody else knows it's been taken off yet, it can't have been put back on).
        assert((head->freeListRefs.load(std::memory_order_relaxed) &
                SHOULD_BE_ON_FREELIST) == 0);

        // Decrease refcount twice, once for our ref, and once for the list's ref
        head->freeListRefs.fetch_add(-2, std::memory_order_relaxed);

        return head;
      }

      // OK, the head must have changed on us, but we still need to decrease the refcount we
      // increased
      refs = prevHead->freeListRefs.fetch_add(-1, std::memory_order_acq_rel);
      if (refs == SHOULD_BE_ON_FREELIST + 1) {
        add_knowing_refcount_is_zero(prevHead);
      }
    }

    return nullptr;
  }

  // Useful for traversing the list when there's no contention (e.g. to destroy remaining nodes)
  N* head_unsafe() const { return freeListHead_.load(std::memory_order_relaxed); }

 private:
  inline void add_knowing_refcount_is_zero(N* node) {
    // Since the refcount is zero, and nobody can increase it once it's zero (except us, and we
    // run only one copy of this method per node at a time, i.e. the single thread case), then we
    // know we can safely change the next pointer of the node; however, once the refcount is back
    // above zero, then other threads could increase it (happens under heavy contention, when the
    // refcount goes to zero in between a load and a refcount increment of a node in try_get, then
    // back up to something non-zero, then the refcount increment is done by the other thread) --
    // so, if the CAS to add the node to the actual list fails, decrease the refcount and leave
    // the add operation to the next thread who puts the refcount back at zero (which could be us,
    // hence the loop).
    auto head = freeListHead_.load(std::memory_order_relaxed);
    while (true) {
      node->freeListNext.store(head, std::memory_order_relaxed);
      node->freeListRefs.store(1, std::memory_order_release);
      if (!freeListHead_.compare_exchange_strong(head, node,
                                                std::memory_order_release,
                                                 std::memory_order_relaxed)) {
        // Hmm, the add failed, but we can only try again when the refcount goes back to zero
        if (node->freeListRefs.fetch_add(SHOULD_BE_ON_FREELIST - 1,
                                         std::memory_order_release) == 1) {
          continue;
        }
      }
      return;
    }
  }

 private:
  static const std::uint32_t REFS_MASK = 0x7FFFFFFF;
  static const std::uint32_t SHOULD_BE_ON_FREELIST = 0x80000000;

  // Implemented like a stack, but where node order doesn't matter (nodes are
  // inserted out of order under contention)
  std::atomic<N*> freeListHead_;
};

/*!
 * \brief Object pool for fast allocation and deallocation.
 */
template <typename T>
class ObjectPool {
 public:
  /*!
   * \brief Destructor.
   */
  ~ObjectPool();
  /*!
   * \brief Create new object.
   * \return Pointer to the new object.
   */
  template <typename... Args>
  T* New(Args&&... args);
  /*!
   * \brief Delete an existing object.
   * \param ptr The pointer to delete.
   *
   * Make sure the pointer to delete is allocated from this pool.
   */
  void Delete(T* ptr);

  /*!
   * \brief Get singleton instance of pool.
   * \return Object Pool.
   */
  static ObjectPool* Get();

  /*!
   * \brief Get a shared ptr of the singleton instance of pool.
   * \return Shared pointer to the Object Pool.
   */
  static std::shared_ptr<ObjectPool> _GetSharedRef();

 private:
#ifndef LOCKFREE_OBJECT_POOL
  /*!
   * \brief Internal structure to hold pointers.
   */
  struct LinkedList {
#if defined(_MSC_VER)
    T t;
    LinkedList* next{nullptr};
#else
    union {
      T t;
      LinkedList* next{nullptr};
    };
#endif
  };
#endif
  /*!
   * \brief Page size of allocation.
   *
   * Currently defined to be 4KB.
   */
  constexpr static std::size_t kPageSize = 1 << 12;
#ifdef LOCKFREE_OBJECT_POOL
  dmlc::moodycamel::ConcurrentQueue<T *> queue_;
#else
  /*! \brief internal mutex */
  std::mutex m_;
  /*!
   * \brief Head of free list.
   */
  LinkedList* head_{nullptr};
#endif
  /*! \brief Pages allocated mutex */
  std::mutex allocated_mutex_;
  /*!
   * \brief Pages allocated.
   */
  std::vector<void*> allocated_;
  /*!
   * \brief Private constructor.
   */
  ObjectPool();
  /*!
   * \brief Allocate a page of raw objects.
   *
   * This function is not protected and must be called with caution.
   */
  void AllocateChunk();
  DISALLOW_COPY_AND_ASSIGN(ObjectPool);
};  // class ObjectPool

/*!
 * \brief Helper trait class for easy allocation and deallocation.
 */
template <typename T>
struct ObjectPoolAllocatable {
  /*!
   * \brief Create new object.
   * \return Pointer to the new object.
   */
  template <typename... Args>
  static T* New(Args&&... args);
  /*!
   * \brief Delete an existing object.
   * \param ptr The pointer to delete.
   *
   * Make sure the pointer to delete is allocated from this pool.
   */
  static void Delete(T* ptr);
};  // struct ObjectPoolAllocatable

template <typename T>
ObjectPool<T>::~ObjectPool() {
  // TODO(hotpxl): mind destruction order
  // for (auto i : allocated_) {
  //   free(i);
  // }
}

template <typename T>
template <typename... Args>
T* ObjectPool<T>::New(Args&&... args) {
#ifdef LOCKFREE_OBJECT_POOL
  T *ret;
  while (!queue_.try_dequeue(ret)) {
    AllocateChunk();
  }
  return new (static_cast<void*>(ret)) T(std::forward<Args>(args)...);
#else
  LinkedList* ret;
  {
    std::lock_guard<std::mutex> lock{m_};
    if (head_->next == nullptr) {
      AllocateChunk();
    }
    ret = head_;
    head_ = head_->next;
  }
  return new (static_cast<void*>(ret)) T(std::forward<Args>(args)...);
#endif
}

template <typename T>
void ObjectPool<T>::Delete(T* ptr) {
  ptr->~T();
#ifdef LOCKFREE_OBJECT_POOL
  queue_.enqueue(ptr);
#else
  auto linked_list_ptr = reinterpret_cast<LinkedList*>(ptr);
  {
    std::lock_guard<std::mutex> lock{m_};
    linked_list_ptr->next = head_;
    head_ = linked_list_ptr;
  }
#endif
}

template <typename T>
ObjectPool<T>* ObjectPool<T>::Get() {
  return _GetSharedRef().get();
}

template <typename T>
std::shared_ptr<ObjectPool<T> > ObjectPool<T>::_GetSharedRef() {
  static std::shared_ptr<ObjectPool<T> > inst_ptr(new ObjectPool<T>());
  return inst_ptr;
}

template <typename T>
ObjectPool<T>::ObjectPool() {
  AllocateChunk();
}

template <typename T>
void ObjectPool<T>::AllocateChunk() {
#ifdef LOCKFREE_OBJECT_POOL
  static_assert(sizeof(T) <= kPageSize, "Object too big.");
  static_assert(sizeof(T) % alignof(T) == 0, "ObjectPool Invariant");
  static_assert(alignof(T) % alignof(T) == 0, "ObjectPool Invariant");
  static_assert(kPageSize % alignof(T) == 0, "ObjectPool Invariant");
  void* new_chunk_ptr;
#ifdef _MSC_VER
  new_chunk_ptr = _aligned_malloc(kPageSize, kPageSize);
  CHECK(new_chunk_ptr != NULL) << "Allocation failed";
#else
  int ret = posix_memalign(&new_chunk_ptr, kPageSize, kPageSize);
  CHECK_EQ(ret, 0) << "Allocation failed";
#endif
  do {
    std::unique_lock<std::mutex> lk(allocated_mutex_);
    allocated_.emplace_back(new_chunk_ptr);
  } while (false);
  auto size = kPageSize / sizeof(T);
  auto *new_chunk = static_cast<T *>(new_chunk_ptr);
  for (std::size_t i = 0; i < size; ++i) {
    queue_.enqueue(&new_chunk[i]);
  }
#else
  static_assert(sizeof(LinkedList) <= kPageSize, "Object too big.");
  static_assert(sizeof(LinkedList) % alignof(LinkedList) == 0, "ObjectPooll Invariant");
  static_assert(alignof(LinkedList) % alignof(T) == 0, "ObjectPooll Invariant");
  static_assert(kPageSize % alignof(LinkedList) == 0, "ObjectPooll Invariant");
  void* new_chunk_ptr;
#ifdef _MSC_VER
  new_chunk_ptr = _aligned_malloc(kPageSize, kPageSize);
  CHECK(new_chunk_ptr != NULL) << "Allocation failed";
#else
  int ret = posix_memalign(&new_chunk_ptr, kPageSize, kPageSize);
  CHECK_EQ(ret, 0) << "Allocation failed";
#endif
  allocated_.emplace_back(new_chunk_ptr);
  auto new_chunk = static_cast<LinkedList*>(new_chunk_ptr);
  auto size = kPageSize / sizeof(LinkedList);
  for (std::size_t i = 0; i < size - 1; ++i) {
    new_chunk[i].next = &new_chunk[i + 1];
  }
  new_chunk[size - 1].next = head_;
  head_ = new_chunk;
#endif
}

template <typename T>
template <typename... Args>
T* ObjectPoolAllocatable<T>::New(Args&&... args) {
  return ObjectPool<T>::Get()->New(std::forward<Args>(args)...);
}

template <typename T>
void ObjectPoolAllocatable<T>::Delete(T* ptr) {
  ObjectPool<T>::Get()->Delete(ptr);
}

}  // namespace common
}  // namespace mxnet
#endif  // MXNET_COMMON_OBJECT_POOL_H_
