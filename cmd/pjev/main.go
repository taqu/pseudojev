package main

import (
	"encoding/json"
	"fmt"
	"log/slog"
	"net/http"
	"os"

	"github.com/taqu/pseudojev/internal/args"
	"github.com/taqu/pseudojev/internal/jev"
	"github.com/taqu/pseudojev/internal/llama"
)

func main() {
	// Install a process-wide structured logger before starting the service.
	logger := slog.New(slog.NewTextHandler(os.Stdout, nil))
	slog.SetDefault(logger)

	args.ParseArgs()

	// Register the Jev-compatible endpoint and serve requests until shutdown.
	http.HandleFunc("/api/alpha/decisions", jevToLlamaHandler)

	fmt.Println("Starting Jev Mock Server on :8080...")
	if err := http.ListenAndServe(":8080", nil); err != nil {
		slog.Error("Server failed to start: %v", err)
	}
}

// jevToLlamaHandler validates a Jev request, delegates classification to
// llama-server, and translates the result back into the Jev response format.
func jevToLlamaHandler(w http.ResponseWriter, r *http.Request) {
	// This endpoint only accepts classification requests encoded as JSON.
	if r.Method != http.MethodPost {
		http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
		return
	}

	var request jev.JevRequest
	if err := json.NewDecoder(r.Body).Decode(&request); err != nil {
		http.Error(w, "Invalid JSON payload", http.StatusBadRequest)
		return
	}
	slog.Info("jev request:", "request", request)

	// Constrain the model output to one of the choices supplied by the client.
	schemaDefinition := llama.BuildJevJSONSchema(&request)
	if schemaDefinition == nil {
		http.Error(w, "Invalid request", http.StatusBadRequest)
		return
	}
	llamaRequest := llama.BuildJevLlamaRequest(&request, schemaDefinition)
	slog.Info("llama-server:", "request", llamaRequest)

	// Treat transport and decoding failures as upstream service errors.
	httpResponse := llama.PostLlamaRequest(args.AppContext().Llama, llamaRequest)
	if httpResponse == nil {
		http.Error(w, "llama-server unavailable", http.StatusInternalServerError)
		return
	}
	llamaResponse := llama.ParseLlamaResponse(httpResponse)
	if llamaResponse == nil {
		http.Error(w, "Failed to parse llama response", http.StatusInternalServerError)
		return
	}
	slog.Info("llama-server:", "response", llamaResponse)

	// Convert the chosen label and token evidence into the Jev result shape.
	decision, err := llama.ParseDecision(llamaResponse)
	if err != 0 {
		http.Error(w, "No response from model", http.StatusInternalServerError)
	}
	confidence, probabilities := llama.CalcProbabilities(decision, &request, llamaResponse)

	llama.ReplyResponse(w, decision, confidence, probabilities)
}
