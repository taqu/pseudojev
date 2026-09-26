// Package llama translates between Jev requests and llama-server's chat API.
package llama

import (
	"bytes"
	"encoding/json"
	"fmt"
	"math"
	"net/http"
	"net/url"
	"strconv"
	"strings"

	"github.com/taqu/pseudojev/internal/jev"
	"github.com/taqu/pseudojev/internal/util"
)

// LlamaChatRequest describes the subset of a chat-completion request used for schema-constrained classification.
type LlamaChatRequest struct {
	Model          string              `json:"model"`
	Messages       []LlamaMessage      `json:"messages"`
	Temperature    float64             `json:"temperature"`
	MaxTokens      int                 `json:"max_tokens"`
	Logprobs       bool                `json:"logprobs"`
	TopLogprobs    int                 `json:"top_logprobs"`
	ResponseFormat LlamaResponseFormat `json:"response_format"`
}

// LlamaMessage is a single role-tagged message in a llama chat request.
type LlamaMessage struct {
	Role    string `json:"role"`
	Content string `json:"content"`
}

// LlamaResponseFormat requests JSON output that conforms to a supplied schema.
type LlamaResponseFormat struct {
	Type       string         `json:"type"`
	JSONSchema map[string]any `json:"json_schema"`
}

// LlamaChatResponse contains the generated decision and token probabilities needed to produce a Jev response.
type LlamaChatResponse struct {
	Choices []struct {
		Message struct {
			Content string `json:"content"`
		} `json:"message"`
		Logprobs struct {
			Content []struct {
				Token       string  `json:"token"`
				Logprob     float64 `json:"logprob"`
				TopLogprobs []struct {
					Token   string  `json:"token"`
					Logprob float64 `json:"logprob"`
				} `json:"top_logprobs"`
			} `json:"content"`
		} `json:"logprobs"`
	} `json:"choices"`
}

func getTargetChoices(jevRequest *jev.JevRequest) []string {
	switch jevRequest.Schema.Type {
	case "choice":
		return jevRequest.Schema.Choices
	case "noul":
		return []string{"YES", "NO"}
	case "score":
		if jevRequest.Schema.Min == nil || jevRequest.Schema.Max == nil {
			return nil
		}
		min := *jevRequest.Schema.Min
		max := *jevRequest.Schema.Max
		var choices []string
		for i := min; i <= max; i++ {
			choices = append(choices, strconv.Itoa(i))
		}
		return choices
	default:
		return nil
	}
}

// BuildJevJSONSchema converts the allowed Jev choices into a JSON schema for the model's decision.
func BuildJevJSONSchema(jevRequest *jev.JevRequest) map[string]any {
	var enumChoices []string = getTargetChoices(jevRequest)
	if enumChoices == nil {
		return nil
	}

	schemaDefinition := map[string]interface{}{
		"type": "object",
		"properties": map[string]interface{}{
			"decision": map[string]interface{}{
				"type": "string",
				"enum": enumChoices,
			},
		},
		"required": []string{"decision"},
	}
	return schemaDefinition
}

func buildPromptForJevRequest(request *jev.JevRequest) (string, int) {
	var enumChoices []string = getTargetChoices(request)
	if enumChoices == nil {
		return "", -1
	}
}

// BuildJevLlamaRequest builds a schema-constrained classification prompt from a Jev request.
func BuildJevLlamaRequest(jevRequest *jev.JevRequest, schemaDefinition map[string]any) *LlamaChatRequest {
	prompt := fmt.Sprintf("Analyze the following text and classify it perfectly according to the schema:\n\n%s", jevRequest.Text)
	llamaRequest := LlamaChatRequest{
		Model:       "local-model",
		Temperature: 0.1,
		MaxTokens:   50,
		Logprobs:    true,
		TopLogprobs: 10,
		Messages: []LlamaMessage{
			{Role: "system", Content: "You are a precise classification engine. Respond ONLY with the requested JSON schema format."},
			{Role: "user", Content: prompt},
		},
		ResponseFormat: LlamaResponseFormat{
			Type: "json_schema",
			JSONSchema: map[string]interface{}{
				"name":   "jev_decision",
				"schema": schemaDefinition,
			},
		},
	}
	return &llamaRequest
	/*
		prompt := fmt.Sprintf("Analyze the following text and classify it perfectly according to the schema:\n\n%s", jevRequest.Text)
		llamaRequest := LlamaChatRequest{
			Model:       "local-model",
			Temperature: 0.1,
			MaxTokens:   50,
			Logprobs:    true,
			TopLogprobs: 10,
			Messages: []LlamaMessage{
				{Role: "system", Content: "You are a precise classification engine. Respond ONLY with the requested JSON schema format."},
				{Role: "user", Content: prompt},
			},
			ResponseFormat: LlamaResponseFormat{
				Type: "json_schema",
				JSONSchema: map[string]interface{}{
					"name":   "jev_decision",
					"schema": schemaDefinition,
				},
			},
		}
		return &llamaRequest
	*/
}

// PostLlamaRequest sends a chat-completion request to the configured
// llama-server endpoint. It returns nil when the server cannot be reached.
func PostLlamaRequest(endpoint string, request *LlamaChatRequest) *http.Response {
	requestBody, _ := json.Marshal(request)
	url, err := url.JoinPath(endpoint, "v1", "chat", "completions")
	if err != nil {
		panic(err)
	}
	response, err := http.Post(url, "application/json", bytes.NewBuffer(requestBody))
	if err != nil {
		return nil
	}
	return response
}

// ParseLlamaResponse decodes a llama-server response and closes its body. It
// returns nil when the response is not valid JSON.
func ParseLlamaResponse(response *http.Response) *LlamaChatResponse {
	defer response.Body.Close()
	var llamaResponse LlamaChatResponse
	if err := json.NewDecoder(response.Body).Decode(&llamaResponse); err != nil {
		return nil
	}
	return &llamaResponse
}

// ParseDecision extracts the schema decision from the first model choice. The
// status is -1 when no choice is available and zero otherwise.
func ParseDecision(response *LlamaChatResponse) (string, int) {
	if len(response.Choices) == 0 {
		return "", -1
	}

	var parsedDecision map[string]string
	json.Unmarshal([]byte(response.Choices[0].Message.Content), &parsedDecision)
	finalDecition := parsedDecision["decision"]
	return finalDecition, 0
}

// CalcProbabilities derives a normalized Jev probability distribution from
// llama-server token log probabilities.
func CalcProbabilities(finalDecision string, request *jev.JevRequest, response *LlamaChatResponse) (float64, map[string]float64) {
	probabilities := make(map[string]float64)

	// Use a stable distribution when the model provides no usable token data.
	if response == nil || len(response.Choices) == 0 || len(response.Choices[0].Logprobs.Content) == 0 {
		return 0.0, nil
	}

	var targetChoices []string = getTargetChoices(request)
	if targetChoices == nil {
		return 0.0, nil
	}

	rawProbabilities := make(map[string]float64)
	var totalRawProb float64

	// Include every allowed choice, even if it never appears among the tokens.
	for _, choice := range targetChoices {
		rawProbabilities[choice] = 0.0
	}

	// Accumulate probability mass from candidate tokens that overlap a choice.
	for _, content := range response.Choices[0].Logprobs.Content {
		for _, top := range content.TopLogprobs {
			cleanToken := cleanTokenString(top.Token)
			if cleanToken == "" {
				continue
			}

			for _, choice := range targetChoices {
				isMatch := false
				if request.Schema.Type == "score" {
					isMatch = (choice == cleanToken)
				} else {
					isMatch = strings.Contains(strings.ToLower(choice), strings.ToLower(cleanToken)) ||
						strings.Contains(strings.ToLower(cleanToken), strings.ToLower(choice))
				}
				if isMatch {
					prob := math.Exp(top.Logprob)
					rawProbabilities[choice] += prob
					break // Count each token toward at most one choice.
				}
			}
		}
	}

	for _, prob := range rawProbabilities {
		totalRawProb += prob
	}

	// Avoid normalizing an empty distribution when no tokens matched a choice.
	if totalRawProb == 0 {
		return 0.0, nil
	}

	// Normalize the collected mass across choices for a client-friendly result.
	for choice, rawProb := range rawProbabilities {
		normalizedProb := rawProb / totalRawProb
		probabilities[choice] = util.Round(normalizedProb, 4)
	}

	// Confidence is the normalized probability assigned to the final decision.
	confidence := probabilities[finalDecision]
	if confidence == 0 {
		// Preserve a usable confidence when the parsed decision was not matched.
		confidence = 0.50
	}

	return confidence, probabilities
}

// cleanTokenString removes JSON punctuation and surrounding whitespace before
// a token is compared with a schema choice.
func cleanTokenString(token string) string {
	t := strings.TrimSpace(token)
	t = strings.ReplaceAll(t, "\"", "")
	t = strings.ReplaceAll(t, "\\", "")
	t = strings.ReplaceAll(t, "\n", "")
	t = strings.ReplaceAll(t, "\r", "")
	return t
}

// fallbackProbabilities builds a conservative distribution when token evidence cannot be used.
func fallbackProbabilities(finalDecision string, request *jev.JevRequest) (float64, map[string]float64) {
	probabilities := make(map[string]float64)
	confidence := 0.50

	for _, choice := range request.Schema.Choices {
		if choice == finalDecision {
			probabilities[choice] = confidence
		} else {
			probabilities[choice] = util.Round((1.0-confidence)/float64(len(request.Schema.Choices)-1), 4)
		}
	}
	return confidence, probabilities
}

// ReplyResponse writes a Jev-compatible JSON response to the client.
func ReplyResponse(w http.ResponseWriter, finalDecision string, confidence float64, probabilities map[string]float64) {
	jevRes := jev.JevResponse{
		Decision:      finalDecision,
		Confidence:    confidence,
		Probabilities: probabilities,
	}

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(jevRes)
}
