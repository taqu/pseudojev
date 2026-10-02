#include "http_server.h"
#include <httplib.h>
#include <cstdio>

namespace pjev
{
bool run_http_server(const std::string& host,
                     uint16_t port,
                     const std::string& path,
                     PostHandler handler)
{
    httplib::Server svr;
    svr.Post(path.c_str(), [handler](const httplib::Request& req,
                                      httplib::Response& res) {
        int32_t status = 500;
        std::string body;
        handler(req.body, status, body);
        res.status = status;
        res.set_content(body, "application/json");
    });

    fprintf(stderr, "pseudojev server listening on %s:%d\n", host.c_str(), (int)port);
    if (!svr.listen(host.c_str(), (int)port)) {
        fprintf(stderr, "failed to start server on %s:%d\n", host.c_str(), (int)port);
        return false;
    }
    return true;
}
}

