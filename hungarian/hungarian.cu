#include <cassert>
#include <iostream>
#include <limits>
#include <vector>

bool ckmin(float &a, const float &b) { return b < a ? a = b, 1 : 0; }

std::vector<int> hungarian(const std::vector<std::vector<float>> &cost)
{
    std::cout << "cost.size(): " << cost.size() << std::endl;
    const auto n = cost.size();
    // job[w] = job assigned to w-th worker, or -1 if no job assigned
    // note: a W-th worker was added for convenience
    std::vector<int> job(n + 1, -1);
    std::vector<float> ys(n), yt(n + 1); // potentials
    const auto inf = std::numeric_limits<float>::max();
    for (int j_cur = 0; j_cur < n; ++j_cur)
    { // assign j_cur-th job
        int w_cur = n;
        job[w_cur] = j_cur;
        // min reduced cost over edges from Z to worker w
        std::vector<float> min_to(n + 1, inf);
        std::vector<int> prv(n + 1, -1); // previous worker on alternating path
        std::vector<bool> in_Z(n + 1);   // whether worker is in Z
        while (job[w_cur] != -1)
        { // runs at most j_cur + 1 times
            in_Z[w_cur] = true;
            const int j = job[w_cur];
            float delta = inf;
            int w_next;
            for (int w = 0; w < n; ++w)
            {
                if (!in_Z[w])
                {
                    if (ckmin(min_to[w], cost[j][w] - ys[j] - yt[w]))
                        prv[w] = w_cur;
                    if (ckmin(delta, min_to[w]))
                        w_next = w;
                }
            }
            // delta will always be non-negative,
            // except possibly during the first time this loop runs
            // if any entries of cost[j_cur] are negative
            for (int w = 0; w <= n; ++w)
            {
                if (in_Z[w])
                    ys[job[w]] += delta, yt[w] -= delta;
                else
                    min_to[w] -= delta;
            }
            w_cur = w_next;
        }
        // update assignments along alternating path
        for (int w; w_cur != n; w_cur = w)
            job[w_cur] = job[w = prv[w_cur]];
    }
    return job;
}