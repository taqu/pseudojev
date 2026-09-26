// Package util contains small helpers shared across the service.
package util

// Round rounds val to the requested number of decimal places.
func Round(val float64, precision int) float64 {
	p := 1.0
	for range precision {
		p *= 10
	}
	return float64(int(val*p+0.5)) / p
}
