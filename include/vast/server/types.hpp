// Copyright (c) 2024-present, Trail of Bits, Inc.

#pragma once

#include <concepts>
#include <cstdint>
#include <string>
#include <variant>

#include <nlohmann/json.hpp>

namespace vast::server {
    template< typename T >
    concept json_convertible = requires(T obj, nlohmann::json &json) {
        {
            nlohmann::to_json(json, obj)
        };
        {
            nlohmann::from_json(json, obj)
        };
    };

    template< typename T >
    concept message_like = json_convertible< T > && requires {
        {
            T::is_notification
        } -> std::convertible_to< bool >;
        {
            T::method
        } -> std::convertible_to< std::string >;
    };

    template< typename message >
    concept notification_like = message_like< message > && message::is_notification;

    template< typename message >
    concept request_like = message_like< message > && !message::is_notification
        && json_convertible< typename message::response_type >;

    template< typename message >
    concept request_with_error_like =
        request_like< message > && json_convertible< typename message::error_type >;

    template< typename T >
    struct error;

    template< request_with_error_like request >
    struct error< request >
    {
        int64_t code;
        std::string message;
        typename request::error_type body;
    };

    template< request_like request >
    struct error< request >
    {
        int64_t code;
        std::string message;
    };

    template< request_with_error_like request >
    void to_json(nlohmann::json &json, const error< request > &err) {
        json["code"]    = err.code;
        json["message"] = err.message;
        to_json(json["data"], err.body);
    }

    template< request_like request >
    void to_json(nlohmann::json &json, const error< request > &err) {
        json["code"]    = err.code;
        json["message"] = err.message;
    }

    template< request_with_error_like request >
    void from_json(const nlohmann::json &json, error< request > &err) {
        from_json(json["code"], err.code);
        from_json(json["message"], err.message);
        from_json(json["data"], err.body);
    }

    template< request_like request >
    void from_json(const nlohmann::json &json, error< request > &err) {
        from_json(json["code"], err.code);
        from_json(json["message"], err.message);
    }

    template< request_like request >
    using result_type = std::variant< typename request::response_type, error< request > >;

    struct input_request
    {
        static constexpr const char *method   = "input";
        static constexpr bool is_notification = false;

        nlohmann::json type;
        std::string text;
        NLOHMANN_DEFINE_TYPE_INTRUSIVE(input_request, type, text)

        struct response_type
        {
            nlohmann::json value;
            NLOHMANN_DEFINE_TYPE_INTRUSIVE(response_type, value)
        };
    };

    enum class message_kind {
        info,
        warn,
        err,
    };

    NLOHMANN_JSON_SERIALIZE_ENUM(
        message_kind,
        {
            { message_kind::info, "info" },
            { message_kind::warn, "warn" },
            {  message_kind::err,  "err" },
    }
    )

    struct message_notification
    {
        static constexpr const char *method   = "message";
        static constexpr bool is_notification = true;

        message_kind kind;
        std::string text;
        NLOHMANN_DEFINE_TYPE_INTRUSIVE(message_notification, kind, text)
    };

    enum class console_severity { trace, debug, info, warn, err };

    NLOHMANN_JSON_SERIALIZE_ENUM(
        console_severity,
        {
            { console_severity::trace, "trace" },
            { console_severity::debug, "debug" },
            {  console_severity::info,  "info" },
            {  console_severity::warn,  "warn" },
            {   console_severity::err,   "err" },
    }
    )

    struct console_notification
    {
        static constexpr const char *method   = "console";
        static constexpr bool is_notification = true;

        console_severity severity;
        std::string message;
        nlohmann::json params;
        NLOHMANN_DEFINE_TYPE_INTRUSIVE(console_notification, severity, message, params)
    };
} // namespace vast::server
