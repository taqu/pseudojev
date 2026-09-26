// Package args parses and exposes command-line configuration.
package args

import (
	"flag"
)

var context Context

// Context contains command-line configuration shared by the HTTP handler.
type Context struct {
	Bind  string // Bind is the requested HTTP listen address.
	Llama string // Llama is the base URL of the upstream llama-server.
}

// ParseArgs parses command-line flags and stores the resulting application
// configuration.
func ParseArgs() {
	var (
		bind  = flag.String("bind", "127.0.0.1:8080", "Bind address")
		llama = flag.String("llama", "http://127.0.0.1:8081", "llama-server's endpoint")
	)
	flag.Parse()
	context = Context{
		Bind:  *bind,
		Llama: *llama,
	}
}

// AppContext returns the parsed application configuration.
func AppContext() *Context {
	return &context
}
