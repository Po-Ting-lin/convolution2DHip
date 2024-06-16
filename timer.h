#pragma once
#include <chrono>
#include <utility>
#define duration(a) std::chrono::duration_cast<std::chrono::microseconds>(a).count()
#define timeNow() std::chrono::high_resolution_clock::now()


template<typename F, typename... Args>
double funcTime(int repeatTime, F func, Args&&... args) {
    std::chrono::high_resolution_clock::time_point t1 = timeNow();
    for (int i = 0; i < repeatTime; i++)
        func(std::forward<Args>(args)...);
    return (double)duration(timeNow() - t1) / repeatTime / 1000.0;
}
