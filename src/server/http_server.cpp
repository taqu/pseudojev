#include "http_server.h"
#include <httplib.h>
#include <nlohmann/json.hpp>
#include <spdlog/spdlog.h>

namespace pjev {
namespace {

static std::string json_error(const std::string& code, const std::string& msg) {
    nlohmann::json e;
    e["error"]["code"]    = code;
    e["error"]["message"] = msg;
    return e.dump();
}

} // namespace

struct PjevHttpServer::Impl {
    httplib::Server svr;
    ServerConfig    cfg;
};

PjevHttpServer::PjevHttpServer()  : impl_(std::make_unique<Impl>()) {}
PjevHttpServer::~PjevHttpServer() = default;

void PjevHttpServer::configure(const ServerConfig& cfg) {
    impl_->cfg = cfg;
    impl_->svr.set_read_timeout (cfg.read_timeout_secs, 0);
    impl_->svr.set_write_timeout(cfg.read_timeout_secs, 0);
    impl_->svr.set_payload_max_length(cfg.request_body_limit);
    // Return structured JSON for HTTP-layer errors (404, 405, 413, etc.)
    impl_->svr.set_error_handler([](const httplib::Request& req, httplib::Response& res) {
        std::string code, msg;
        if      (res.status == 413) { code = "request_too_large";     msg = "Request body exceeds the maximum allowed size"; }
        else if (res.status == 404) { code = "not_found";             msg = "Unknown endpoint: " + req.path; }
        else if (res.status == 405) { code = "method_not_allowed";    msg = "Method not allowed"; }
        else                        { code = "internal_error";        msg = "Server error"; }
        res.set_content(json_error(code, msg), "application/json");
    });
    impl_->svr.set_exception_handler([](const httplib::Request&, httplib::Response& res, std::exception_ptr ep) {
        try { if (ep) std::rethrow_exception(ep); }
        catch (const std::exception& e) { spdlog::error("unhandled exception in request: {}", e.what()); }
        catch (...)                     { spdlog::error("unknown exception in request"); }
        res.status = 500;
        res.set_content(json_error("internal_error", "Unexpected server error"), "application/json");
    });
}

void PjevHttpServer::add_get(const std::string& path, GetHandler h) {
    impl_->svr.Get(path.c_str(), [h](const httplib::Request&, httplib::Response& res) {
        int32_t status = 500;
        std::string body;
        h(status, body);
        res.status = status;
        res.set_content(body, "application/json");
    });
}

void PjevHttpServer::add_post(const std::string& path, PostHandler h) {
    impl_->svr.Post(path.c_str(), [h](const httplib::Request& req, httplib::Response& res) {
        int32_t status = 500;
        std::string body;
        h(req.body, status, body);
        res.status = status;
        res.set_content(body, "application/json");
    });
}

bool PjevHttpServer::start() {
    const auto& cfg = impl_->cfg;
    spdlog::info("pjev listening on {}:{}", cfg.host, (int)cfg.port);
    return impl_->svr.listen(cfg.host.c_str(), (int)cfg.port);
}

void PjevHttpServer::stop() {
    impl_->svr.stop();
}

} // namespace pjev
