// Copyright (c) 2024-present, Trail of Bits, Inc.

#pragma once

#include <atomic>
#include <condition_variable>
#include <deque>
#include <mutex>
#include <stdexcept>
#include <unordered_map>

namespace vast::server {
    class execution_stopped : public std::runtime_error
    {
      public:
        execution_stopped(const char *what) : std::runtime_error(what) {}
    };

    template<
        typename Key, typename Value, typename BackingStore = std::unordered_map< Key, Value > >
    class sync_map
    {
        std::mutex mutex;
        std::condition_variable cv;
        BackingStore data;

        std::atomic_bool stopped = false;

      public:
        void stop() {
            stopped = true;
            std::lock_guard< std::mutex > lock(mutex);
            cv.notify_all();
        }

        void insert(Key k, Value v) {
            if (stopped) {
                throw execution_stopped("User requested stop");
            }
            std::lock_guard< std::mutex > lock(mutex);
            data[k] = v;
            cv.notify_all();
        }

        Value get(Key k) {
            std::unique_lock< std::mutex > lock(mutex);
            cv.wait(lock, [k, this]() { return data.find(k) != data.end() || stopped; });

            if (stopped) {
                throw execution_stopped("User requested stop");
            }

            auto it = data.find(k);
            VAST_ASSERT(it != data.end());
            auto response = it->second;
            data.erase(it);
            return response;
        }
    };

    template< typename Value, typename BackingStore = std::deque< Value > >
    class sync_queue
    {
        std::mutex mutex;
        std::condition_variable cv;
        BackingStore data;

        std::atomic_bool stopped = false;

      public:
        void stop() {
            stopped = true;
            std::lock_guard< std::mutex > lock(mutex);
            cv.notify_all();
        }

        void enqueue(const Value &v) {
            if (stopped) {
                throw execution_stopped("User requested stop");
            }
            std::lock_guard< std::mutex > lock(mutex);
            data.push_back(v);
            cv.notify_one();
        }

        template< typename... Args >
        void enqueue(Args &&...args) {
            std::lock_guard< std::mutex > lock(mutex);
            data.emplace_back(std::forward< Args && >(args)...);
            cv.notify_one();
        }

        Value dequeue() {
            std::unique_lock< std::mutex > lock(mutex);
            cv.wait(lock, [this]() { return data.begin() != data.end() || stopped; });

            if (stopped) {
                throw execution_stopped("User requested stop");
            }

            auto res = data.front();
            data.pop_front();
            return res;
        }
    };
} // namespace vast::server
