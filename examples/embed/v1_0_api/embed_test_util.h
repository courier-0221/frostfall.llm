#pragma once

#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

inline double embed_l2_norm(const std::vector<float>& v) {
    double s = 0.0;
    for (float x : v) s += (double) x * (double) x;
    return std::sqrt(s);
}

inline double embed_dot(const std::vector<float>& a, const std::vector<float>& b) {
    const size_t n = a.size() < b.size() ? a.size() : b.size();
    double s = 0.0;
    for (size_t i = 0; i < n; ++i) s += (double) a[i] * (double) b[i];
    return s;
}

inline void embed_print_vector_summary(const char* name, const std::vector<float>& v) {
    const int k = (int) (v.size() < 8 ? v.size() : 8);
    std::printf("%s: dim=%zu norm=%.6f v8=[", name, v.size(), embed_l2_norm(v));
    for (int i = 0; i < k; ++i) {
        std::printf("%s%.6f", i == 0 ? "" : ", ", v[(size_t) i]);
    }
    std::printf("]\n");
}
