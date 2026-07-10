// This file implements a progress bar.

#pragma once

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <functional>
#include <iostream>
#include <string>

namespace zombie {

class ProgressBar {
public:
    // constructor
    ProgressBar(int totalWork, int displayWidth=80):
                completedWork(0),
                totalWork(totalWork),
                displayWidth(displayWidth) {
        startTime = std::chrono::high_resolution_clock::now();
    }

    // reports progress
    void report(int newWorkCompleted, int threadId) {
        completedWork += newWorkCompleted;
        if (threadId == 0) draw();
    }

    // finishes progress bar
    void finish() {
        completedWork = totalWork;
        draw();
        std::cout << std::endl;
        auto endTime = std::chrono::high_resolution_clock::now();
        int nSeconds = std::chrono::duration_cast<std::chrono::seconds>(endTime - startTime).count();
        std::cout << "Finished in " <<  nSeconds << " seconds." << std::endl;
    }

protected:
    // draws progress bar
    void draw() {
        float progress = std::min(completedWork / float(totalWork), 1.0f);
        int pos = displayWidth * progress;
        std::string bar = (pos > 0 ? std::string(pos, '=') + ">" : ">") +
                          (displayWidth - pos > 1 ? std::string(displayWidth - pos - 1, ' ') : "");
        printf("[%s] %3d%%\r", bar.c_str(), int(progress * 100));
        std::cout << std::flush;
    }

    // members
    std::atomic_int completedWork;
    int totalWork;
    int displayWidth;
    std::chrono::time_point<std::chrono::high_resolution_clock> startTime;
};

inline std::function<void(int, int)> getReportProgressCallback(ProgressBar& pb)
{
    return [&pb](int i, int tid) -> void { pb.report(i, tid); };
}

} // zombie
