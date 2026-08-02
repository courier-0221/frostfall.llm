// frostfall 轻量日志接口 —— 替代 glog，仅提供 src 代码所需的最小功能。
//
// 用法与 glog 兼容（流式）：
//   LOG(INFO)    << "message";
//   LOG(WARNING) << "message";
//   LOG(ERROR)   << "message";
//   LOG(FATAL)   << "message";   // 打印后 abort()
//
// 输出格式（写到 stderr）：
//   I0708 12:34:56.123456 file.cpp:42] message
//   首字母 I/W/E/F 对应 INFO/WARNING/ERROR/FATAL。
//
// 头文件内实现（inline），无需单独编译单元，也无需初始化。

#pragma once

#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <sstream>
#include <string>

namespace frostfall {

enum LogSeverity { INFO = 0, WARNING = 1, ERROR = 2, FATAL = 3 };

class LogMessage {
public:
    LogMessage(const char * file, int line, LogSeverity severity)
        : severity_(severity) {
        // 只保留文件名（去掉目录），与 glog 风格一致
        const char * base = file;
        for (const char * p = file; *p; ++p) {
            if (*p == '/' || *p == '\\') {
                base = p + 1;
            }
        }
        file_ = base;
        line_ = line;
    }

    ~LogMessage() {
        static const char kLevel[] = {'I', 'W', 'E', 'F'};

        std::timespec ts;
        std::timespec_get(&ts, TIME_UTC);
        std::tm tm_buf;
        localtime_r(&ts.tv_sec, &tm_buf);

        char header[64];
        std::snprintf(header, sizeof(header),
                      "%c%02d%02d %02d:%02d:%02d.%06ld %s:%d] ",
                      kLevel[severity_],
                      tm_buf.tm_mon + 1, tm_buf.tm_mday,
                      tm_buf.tm_hour, tm_buf.tm_min, tm_buf.tm_sec,
                      ts.tv_nsec / 1000,
                      file_.c_str(), line_);

        std::fputs(header, stderr);
        std::fputs(stream_.str().c_str(), stderr);
        std::fputc('\n', stderr);
        std::fflush(stderr);

        if (severity_ == FATAL) {
            std::abort();
        }
    }

    std::ostringstream & stream() { return stream_; }

    LogMessage(const LogMessage &) = delete;
    LogMessage & operator=(const LogMessage &) = delete;

private:
    std::ostringstream stream_;
    std::string        file_;
    int                line_ = 0;
    LogSeverity        severity_;
};

} // namespace frostfall

// glog 兼容宏：LOG(INFO) / LOG(WARNING) / LOG(ERROR) / LOG(FATAL)
#define FROSTFALL_LOG_INFO \
    ::frostfall::LogMessage(__FILE__, __LINE__, ::frostfall::INFO).stream()
#define FROSTFALL_LOG_WARNING \
    ::frostfall::LogMessage(__FILE__, __LINE__, ::frostfall::WARNING).stream()
#define FROSTFALL_LOG_ERROR \
    ::frostfall::LogMessage(__FILE__, __LINE__, ::frostfall::ERROR).stream()
#define FROSTFALL_LOG_FATAL \
    ::frostfall::LogMessage(__FILE__, __LINE__, ::frostfall::FATAL).stream()

#define LOG(severity) FROSTFALL_LOG_##severity
