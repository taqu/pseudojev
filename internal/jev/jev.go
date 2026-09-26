// Package jev defines the public request and response payloads served by the
// pseudo Jev API.
package jev

// JevRequest asks the service to classify text according to a choice schema.
type JevRequest struct {
	Model  string `json:"model"`
	Text   string `json:"text"`
	Schema struct {
		Type    string   `json:"type"`
		Choices []string `json:"choices,omitempty"`
		Min     *int     `json:"min,omitempty"`
		Max     *int     `json:"max,omitempty"`
	} `json:"schema"`
}

// JevResponse reports the selected choice, its confidence, and the full
// probability distribution.
type JevResponse struct {
	Decision      string             `json:"decision"`
	Confidence    float64            `json:"confidence"`
	Probabilities map[string]float64 `json:"probabilities"`
}
