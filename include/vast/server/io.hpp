// Copyright (c) 2024-present, Trail of Bits, Inc.

#pragma once

#include <memory>
#include <span>
#include <stdexcept>
#include <string>

#include <stdio.h>

#include <gap/core/crtp.hpp>

#include "vast/Util/Warnings.hpp"

namespace vast::server {
    class connection_closed : public std::runtime_error
    {
      public:
        connection_closed() : std::runtime_error("Connection closed") {}

        connection_closed(const char *what) : std::runtime_error(what) {}
    };

    struct io_adapter
    {
        virtual ~io_adapter() = default;

        virtual void close() {}

        virtual size_t read_some(std::span< char > dst)        = 0;
        virtual size_t write_some(std::span< const char > dst) = 0;

        // Upon completion, `dst` is filled with data.
        void read_all(std::span< char > dst) {
            while (!dst.empty()) {
                size_t nread = read_some(dst);
                dst          = dst.subspan(nread);
            }
        }

        // Upon completion, all of the data in `src` is written to the client..
        void write_all(std::span< const char > src) {
            while (!src.empty()) {
                size_t nwritten = write_some(src);
                src             = src.subspan(nwritten);
            }
        }

        char read() {
            char res[1];
            read_all(res);
            return res[0];
        }
    };

    class file_adapter final : public io_adapter
    {
        FILE *ifd;
        FILE *ofd;

      public:
        file_adapter(FILE *ifd = stdin, FILE *ofd = stdout) : ifd(ifd), ofd(ofd) {
            VAST_ASSERT(ifd != nullptr);
            VAST_ASSERT(ofd != nullptr);

            setvbuf(ofd, NULL, _IONBF, 0);
        }

        size_t read_some(std::span< char > dst) override {
            size_t nread = fread(dst.data(), 1, dst.size_bytes(), ifd);
            if (nread == 0 && (feof(ifd) || ferror(ifd))) {
                throw connection_closed{};
            }
            return nread;
        }

        size_t write_some(std::span< const char > src) override {
            size_t nwritten = fwrite(src.data(), 1, src.size_bytes(), ofd);
            if (src.size() != 0 && nwritten == 0) {
                throw connection_closed{};
            }
            return nwritten;
        }
    };

    class sock_adapter final : public io_adapter
    {
      public:
        struct impl;

        size_t read_some(std::span< char > dst) override;
        size_t write_some(std::span< const char > src) override;
        ~sock_adapter();
        void close() override;

        static std::unique_ptr< sock_adapter > create_unix_socket(const std::string &path);
        static std::unique_ptr< sock_adapter >
        create_tcp_server_socket(uint32_t host, uint16_t port);

      private:
        std::unique_ptr< struct impl > pimpl;
        sock_adapter(std::unique_ptr< impl > pimpl);
    };
} // namespace vast::server
