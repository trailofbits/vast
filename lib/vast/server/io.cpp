#include "vast/server/io.hpp"

#include <stdexcept>
#include <system_error>

#include <netinet/in.h>
#include <sys/socket.h>
#include <sys/un.h>
#include <unistd.h>

namespace vast::server {
    union addr {
        sockaddr base;
        sockaddr_un unix;
        sockaddr_in net;
    };

    struct descriptor
    {
        int fd;

        explicit descriptor() : fd(-1) {}

        explicit descriptor(int fd) : fd(fd) {
            if (fd < 0) {
                throw std::system_error(errno, std::generic_category());
            }
        }

        descriptor(const descriptor &)            = delete;
        descriptor &operator=(const descriptor &) = delete;

        descriptor(descriptor &&other) : fd(other.fd) { other.fd = -1; }

        descriptor &operator=(descriptor &&other) {
            if (fd >= 0) {
                close(fd);
            }
            fd       = other.fd;
            other.fd = -1;
            return *this;
        }

        operator int() { return fd; }

        ~descriptor() {
            if (fd >= 0) {
                close(fd);
            }
        }
    };

    struct sock_adapter::impl
    {
        descriptor serverd;
        descriptor clientd;
    };

    sock_adapter::sock_adapter(std::unique_ptr< impl > pimpl) : pimpl(std::move(pimpl)) {}

    sock_adapter::~sock_adapter() = default;

    void sock_adapter::close() {
        pimpl->clientd = descriptor{};
        pimpl->serverd = descriptor{};
    }

    size_t sock_adapter::read_some(std::span< char > dst) {
        auto res = ::read(pimpl->clientd, dst.data(), dst.size_bytes());
        if (res <= 0) {
            throw connection_closed{};
        }
        return static_cast< size_t >(res);
    }

    size_t sock_adapter::write_some(std::span< const char > src) {
        auto res = ::write(pimpl->clientd, src.data(), src.size_bytes());
        if (res == -1) {
            throw connection_closed{};
        }
        return static_cast< size_t >(res);
    }

    static descriptor bind_and_accept(
        descriptor &serverd, addr &sockaddr_server, size_t socklen_server,
        sockaddr *sockaddr_client, socklen_t *socklen_client
    ) {
        int rc = bind(serverd, &sockaddr_server.base, static_cast< socklen_t >(socklen_server));
        if (rc < 0) {
            throw std::system_error(errno, std::generic_category());
        }

        rc = listen(serverd, 1);
        if (rc < 0) {
            throw std::system_error(errno, std::generic_category());
        }

        return descriptor{ accept(serverd, sockaddr_client, socklen_client) };
    }

    std::unique_ptr< sock_adapter > sock_adapter::create_unix_socket(const std::string &path) {
        if (path.size() > (sizeof(sockaddr_un::sun_path) - 1)) {
            throw std::runtime_error("Unix socket pathname is too long");
        }

        descriptor serverd{ socket(AF_UNIX, SOCK_STREAM, 0) };

        addr sock_addr{};
        sock_addr.unix.sun_family = AF_UNIX;
        std::copy(path.begin(), path.end(), sock_addr.unix.sun_path);

        // Here we pre-emptively try to delete an old socket in case it was used before,
        // in order to avoid having to check for existence and running into time-of-check to
        // time-of-use issues.
        // This also means that it's possible that we're tring to delete a socket that does
        // not yet exist, so we ignore ENOENT errors.
        if (unlink(path.c_str()) < 0 && errno != ENOENT) {
            throw std::system_error(errno, std::generic_category());
        }

        auto clientd =
            bind_and_accept(serverd, sock_addr, sizeof(sock_addr.unix), nullptr, nullptr);

        return std::unique_ptr< sock_adapter >(new sock_adapter{
            std::make_unique< impl >(std::move(serverd), std::move(clientd)) });
    }

    std::unique_ptr< sock_adapter >
    sock_adapter::create_tcp_server_socket(uint32_t host, uint16_t port) {
        descriptor serverd{ socket(AF_INET, SOCK_STREAM, 0) };

        addr sock_addr{};
        sock_addr.net.sin_family      = AF_INET;
        sock_addr.net.sin_addr.s_addr = htonl(host);
        sock_addr.net.sin_port        = htons(port);

        addr client_addr{};
        socklen_t client_addr_size = sizeof(client_addr.net);

        auto clientd = bind_and_accept(
            serverd, sock_addr, sizeof(sock_addr.net), &client_addr.base, &client_addr_size
        );

        return std::unique_ptr< sock_adapter >(new sock_adapter{
            std::make_unique< impl >(std::move(serverd), std::move(clientd)) });
    }
} // namespace vast::server
