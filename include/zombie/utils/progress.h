// This file implements a progress bar.

#pragma once

#include <atomic>
#include <chrono>

typedef std::chrono::high_resolution_clock Clock;

class ProgressBar {
public:
    ProgressBar(int totalWork, int displayWidth=80):
                completedWork(0),
                totalWork(totalWork),
                displayWidth(displayWidth),
                startTime(Clock::now()) {}

    void report(int newWorkCompleted, int threadId) {
        completedWork += newWorkCompleted;
        if (threadId == 0) draw();
    }

    void finish() {
        completedWork = totalWork;
        draw();
        std::cout << std::endl;
        int nSeconds = std::chrono::duration_cast<std::chrono::seconds>(Clock::now() - startTime).count();
        std::cout << "Finished in " <<  nSeconds << " seconds." << std::endl;
    }

protected:
    void draw() {
        float progress = std::min(completedWork / float(totalWork), 1.0f);
        int pos = displayWidth * progress;
        std::string bar = (pos > 0 ? std::string(pos, '=') + ">" : ">") +
                          (displayWidth - pos > 1 ? std::string(displayWidth - pos - 1, ' ') : "");
        printf("[%s] %3d%%\r", bar.c_str(), int(progress * 100));
        std::cout << std::flush;
    }

    bool finished;
    int displayWidth;
    int totalWork;
    std::atomic_int completedWork;
    std::chrono::time_point<std::chrono::high_resolution_clock> startTime;
};

std::function<void(int, int)> getReportProgressCallback(ProgressBar& pb)
{
    return [&pb](int i, int tid) -> void { pb.report(i, tid); };
}
