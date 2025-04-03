// Copyright (c) 2024-present, Trail of Bits, Inc.

#pragma once

#include <limits>
#include <mutex>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <tuple>
#include <utility>

#include "io.hpp"
#include "sync_collections.hpp"
#include "types.hpp"
#include "util.hpp"

namespace vast::server {
    class protocol_error : public std::runtime_error
    {
      public:
        protocol_error(const char *what) : std::runtime_error(what) {}
    };

    enum JSONRPC_ERRORS {
        JSONRPC_PARSE_ERROR      = -32700,
        JSONRPC_INVALID_REQUEST  = -32600,
        JSONRPC_METHOD_NOT_FOUND = -32601,
        JSONRPC_INVALID_PARAMS   = -32602,
        JSONRPC_INTERNAL_ERROR   = -32603,
    };

    template< request_like request >
    class ticket
    {
        friend class server_base;
        size_t id;
        bool is_valid        = true;
        bool has_been_waited = false;

        ticket(size_t id) : id(id) {}

      public:
        ticket(const ticket &) = delete;

        ticket(ticket &&other)
            : id(other.id), is_valid(other.is_valid), has_been_waited(other.has_been_waited) {
            other.is_valid = false;
        }

        ticket &operator=(const ticket &) = delete;

        ticket &operator=(ticket &&other) {
            id              = other.id;
            is_valid        = other.is_valid;
            has_been_waited = other.has_been_waited;

            other.is_valid = false;
        }

        ~ticket() {
            if (is_valid) {
                VAST_ASSERT(has_been_waited);
            }
        }
    };

    class server_base
    {
        std::atomic_size_t progressive_id = 0;

      public:
        virtual ~server_base() = default;

        using json = nlohmann::json;

      protected:
        virtual json wait_message(size_t id)     = 0;
        virtual void send_message(const json &j) = 0;

      public:
        void send_error(int64_t code, const std::string &message, const json &id) {
            send_message({
                { "jsonrpc","2.0"                            },
                {      "id",    id },
                {   "error",
                 {
                 { "code", code },
                 { "message", message },
                 }                }
            });
        }

        void
        send_error(int64_t code, const std::string &message, const json &id, const json &data) {
            send_message({
                { "jsonrpc","2.0"                            },
                {      "id",    id },
                {   "error",
                 {
                 { "code", code },
                 { "message", message },
                 { "data", data },
                 }                }
            });
        }

        void send_response(const json &id, const json &data) {
            send_message({
                { "jsonrpc", "2.0" },
                {      "id",    id },
                {  "result",  data }
            });
        }

        template< request_with_error_like request >
        void send_result(const json &id, const result_type< request > &result) {
            if (auto res = std::get_if< typename request::response_type >(&result)) {
                send_response(id, *res);
            } else if (auto err = std::get_if< error< request > >(&result)) {
                send_error(err->code, err->message, id, err->body);
            }
        }

        template< request_like request >
        void send_result(const json &id, const result_type< request > &result) {
            if (auto res = std::get_if< typename request::response_type >(&result)) {
                send_response(id, *res);
            } else if (auto err = std::get_if< error< request > >(&result)) {
                send_error(err->code, err->message, id);
            }
        }

        template< notification_like notification >
        void send_notification(const notification &noti) {
            send_message({
                { "jsonrpc",                "2.0" },
                {  "method", notification::method },
                {  "params",                 noti },
            });
        }

        template< request_like request >
        [[nodiscard]] ticket< request > send_request_nonblock(const request &req) {
            size_t id = progressive_id++;
            send_message({
                { "jsonrpc",           "2.0" },
                {  "method", request::method },
                {      "id",              id },
                {  "params",             req },
            });

            return ticket< request >(id);
        }

        template< request_like request >
        [[nodiscard]] result_type< request > wait_request(ticket< request > ticket) {
            VAST_ASSERT(ticket.is_valid && !ticket.has_been_waited);
            ticket.has_been_waited = true;
            json response          = wait_message(ticket.id);

            if (response.find("error") != response.end()) {
                error< request > err = response["error"];
                return err;
            } else {
                typename request::response_type body = response["result"];
                return body;
            }
        }

        template< request_like... requests >
        [[nodiscard]] std::tuple< result_type< requests >... >
        wait_requests(ticket< requests >... tickets) {
            return std::make_tuple(wait_request(std::move(tickets))...);
        }

        template< request_like request >
        [[nodiscard]] result_type< request > send_request(const request &req) {
            return wait_request(send_request_nonblock(req));
        }
    };

    namespace detail {
        template< message_like... message_types >
        struct dispatch_handler;

        template<>
        struct dispatch_handler<>
        {
            template< typename handler >
            void operator()(handler &, server_base &server, const nlohmann::json &req) {
                nlohmann::json id = req.find("id") != req.end() ? req["id"] : nullptr;
                server.send_error(JSONRPC_METHOD_NOT_FOUND, "Unsupported method", id);
            }
        };

        template< request_like message_type, message_like... messages >
        struct dispatch_handler< message_type, messages... >
        {
            template< typename handler >
            void operator()(handler &h, server_base &server, const nlohmann::json &j) {
                if (j["method"] == message_type::method) {
                    if (j.find("id") == j.end()) {
                        server.send_error(
                            JSONRPC_INVALID_REQUEST, "ID was expected but not found", nullptr
                        );
                        return;
                    }

                    server.send_result(j["id"], h(server, j["params"]));
                } else {
                    dispatch_handler< messages... > dispatcher;
                    return dispatcher(h, server, j);
                }
            }
        };

        template< notification_like message_type, message_like... messages >
        struct dispatch_handler< message_type, messages... >
        {
            template< typename handler >
            void operator()(handler &h, server_base &server, const nlohmann::json &j) {
                if (j["method"] == message_type::method) {
                    if (j.find("id") != j.end()) {
                        server.send_error(
                            JSONRPC_INVALID_REQUEST, "ID found but not expected", nullptr
                        );
                        return;
                    }

                    h(server, j["params"]);
                } else {
                    dispatch_handler< messages... > dispatcher;
                    return dispatcher(h, server, j);
                }
            }
        };
    } // namespace detail

    template< typename message_handler, message_like... message_types >
    class server final : public server_base
    {
        std::mutex write_mutex;
        std::thread reader_thread;
        std::vector< std::thread > request_threads;

        std::unique_ptr< io_adapter > adapter;

        message_handler handler;

        sync_map< size_t, json > responses;
        sync_queue< json > requests;

        void read_lit(char lit) {
            if (adapter->read() != lit) {
                throw protocol_error("Invalid literal");
            }
        }

        std::pair< std::string, std::string > read_header() {
            std::stringstream header_name;
            std::stringstream header_value;
            do {
                char c = adapter->read();
                if (c == '\r') {
                    break;
                }

                for (; c != ':'; c = adapter->read()) {
                    header_name << c;
                }

                // Skip whitespace between : and header value
                do {
                    c = adapter->read();
                } while (c == ' ' || c == '\t');

                for (; c != '\r'; c = adapter->read()) {
                    header_value << c;
                }
            } while (false);
            read_lit('\n');
            return std::make_pair(header_name.str(), header_value.str());
        }

        size_t read_headers() {
            std::unordered_map< std::string, std::string, ci_hash<>, ci_comparison<> > headers;

            while (true) {
                auto [name, value] = read_header();
                if (name == "") {
                    break;
                }

                headers[name] = value;
            }

            auto content_length = headers.find("content-length");
            if (content_length == headers.end()) {
                throw protocol_error{ "Missing Content-Length header" };
            }

            size_t size = 0;
            for (char c : content_length->second) {
                if (c < '0' || c > '9') {
                    throw protocol_error{ "Invalid Content-Length value" };
                }
                if (size >= std::numeric_limits< size_t >::max() / 10) {
                    throw protocol_error{ "Content-Length too large" };
                }
                size         *= 10;
                size_t digit  = static_cast< size_t >(c - '0');
                if (size >= std::numeric_limits< size_t >::max() - digit) {
                    throw protocol_error{ "Content-Length too large" };
                }
                size += digit;
            }

            return size;
        }

        json receive_message() {
            auto body_size = read_headers();
            std::string body;
            body.resize(body_size);
            adapter->read_all(body);
            return json::parse(body);
        }

        void receive_msg_with_id(const json &msg) {
            // Message is either a response or a request
            json id = msg["id"];

            if (msg.find("method") != msg.end() && msg.find("params") != msg.end()) {
                requests.enqueue(msg);
            } else if (msg.find("result") != msg.end() || msg.find("error") != msg.end()) {
                if (id == nullptr) {
                    VAST_FATAL("JSONRPC Error: {0}", msg.dump(2));
                }
                responses.insert(id, msg);
            } else {
                send_error(JSONRPC_INVALID_REQUEST, "Invalid message", id);
            }
        }

        void reader_thread_routine() {
            try {
                while (true) {
                    json msg = receive_message();

                    if (msg.find("id") != msg.end()) {
                        receive_msg_with_id(msg);
                    } else if (msg.find("method") != msg.end()
                               && msg.find("params") != msg.end())
                    {
                        // Message is a notification
                        requests.enqueue(msg);
                    } else {
                        send_error(JSONRPC_INVALID_REQUEST, "Invalid request", nullptr);
                    }
                }
            } catch (const execution_stopped &) {
            } catch (const connection_closed &) {
                responses.stop();
                requests.stop();
            } catch (const json::parse_error &err) {
                send_error(JSONRPC_PARSE_ERROR, err.what(), nullptr);
            }
        }

        void request_thread_routine() {
            try {
                detail::dispatch_handler< message_types... > dispatcher;
                while (true) {
                    auto req = requests.dequeue();
                    dispatcher(this->handler, *this, req);
                }
            } catch (const execution_stopped &stop) {
            }
        }

      protected:
        virtual void send_message(const json &j) override {
            std::unique_lock< std::mutex > lock(write_mutex);
            // The final \r\n is not necessary, but makes things easier to
            // read when debugging from communication dumps
            std::string data = j.dump() + "\r\n";

            std::stringstream ss;
            ss << "Content-Type: application/json;charset=utf-8\r\n";
            ss << "Content-Length: " << data.size() << "\r\n\r\n";
            ss << data;

            std::string res = ss.str();
            adapter->write_all(res);
        }

        virtual json wait_message(size_t id) override { return responses.get(id); }

      public:
        server(
            std::unique_ptr< io_adapter > adapter, int num_request_threads = 1,
            const message_handler &handler = {}
        )
            : reader_thread(&server::reader_thread_routine, this)
            , adapter(std::move(adapter))
            , handler(handler) {
            for (int i = 0; i < num_request_threads; ++i) {
                request_threads.emplace_back(&server::request_thread_routine, this);
            }
        }

        virtual ~server() override {
            responses.stop();
            requests.stop();
            adapter->close();
            reader_thread.join();
            for (auto &thread : request_threads) {
                thread.join();
            }
        }
    };
} // namespace vast::server
