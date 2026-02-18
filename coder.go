package main

import (
	"bufio"
	"context"
	"errors"
	"flag"
	"fmt"
	"io"
	"log"
	"net/http"
	"os"
	"strings"
	"time"

	"github.com/anthropics/anthropic-sdk-go"
	"github.com/anthropics/anthropic-sdk-go/option"
)

const (
	defaultModelID   = "claude-sonnet-4-6"
	defaultModelName = "Sonnet 4.6"
	defaultMaxTokens = int64(2048)
	requestTimeout   = 120 * time.Second

	userColor   = "\x1b[38;2;102;178;255m"
	claudeColor = "\x1b[38;2;217;119;6m"
	colorReset  = "\x1b[0m"
)

type Config struct {
	APIKey      string
	ModelID     string
	ModelName   string
	Verbose     bool
	ColorOutput bool
}

type ToolUse struct {
	ID    string
	Name  string
	Input string
}

func main() {
	cfg, err := loadConfig()
	if err != nil {
		fmt.Fprintln(os.Stderr, "Error:", err)
		os.Exit(1)
	}

	configureLogging(cfg.Verbose)
	debugf("startup init model_id=%q model_name=%q api_key_present=%t color_output=%t", cfg.ModelID, cfg.ModelName, cfg.APIKey != "", cfg.ColorOutput)

	client := anthropic.NewClient(option.WithAPIKey(cfg.APIKey))
	if err := runChatLoop(cfg, &client); err != nil {
		fmt.Fprintln(os.Stderr, "Error:", err)
		os.Exit(1)
	}
}

func loadConfig() (Config, error) {
	verbose := flag.Bool("verbose", false, "Enable verbose debug logs")
	modelID := flag.String("model", defaultModelID, "Anthropic model ID")
	flag.Parse()

	apiKey := strings.TrimSpace(os.Getenv("ANTHROPIC_API_KEY"))
	if apiKey == "" {
		return Config{}, errors.New("ANTHROPIC_API_KEY is not set")
	}

	selectedModel := strings.TrimSpace(*modelID)
	if selectedModel == "" {
		selectedModel = defaultModelID
	}

	return Config{
		APIKey:      apiKey,
		ModelID:     selectedModel,
		ModelName:   modelDisplayName(selectedModel),
		Verbose:     *verbose,
		ColorOutput: supportsColor(os.Stdout),
	}, nil
}

func configureLogging(verbose bool) {
	if !verbose {
		log.SetOutput(io.Discard)
		return
	}
	log.SetOutput(os.Stderr)
	log.SetFlags(log.LstdFlags | log.Lshortfile)
	log.SetPrefix("DEBUG ")
}

func debugf(format string, args ...any) {
	_ = log.Output(2, fmt.Sprintf(format, args...))
}

func runChatLoop(cfg Config, client *anthropic.Client) error {
	scanner := bufio.NewScanner(os.Stdin)
	history := make([]anthropic.MessageParam, 0, 16)
	turn := 0

	for {
		fmt.Fprint(os.Stdout, userPrefix(cfg.ColorOutput))
		if !scanner.Scan() {
			if err := scanner.Err(); err != nil {
				return fmt.Errorf("failed to read input: %w", err)
			}
			fmt.Fprintln(os.Stdout)
			debugf("shutdown end_of_loop reason=%q", "stdin_eof")
			return nil
		}

		prompt := strings.TrimSpace(scanner.Text())
		if prompt == "" {
			continue
		}
		if prompt == "/quit" || prompt == "/exit" {
			debugf("shutdown end_of_loop reason=%q command=%q", "user_command", prompt)
			return nil
		}

		turn++
		history = append(history, anthropic.NewUserMessage(anthropic.NewTextBlock(prompt)))
		debugf("user_input_received turn=%d prompt_chars=%d conversation_len=%d", turn, len(prompt), len(history))
		debugf("api_call_start turn=%d model_id=%q conversation_len=%d", turn, cfg.ModelID, len(history))

		start := time.Now()
		ctx, cancel := context.WithTimeout(context.Background(), requestTimeout)
		message, requestID, err := sendAnthropicMessage(ctx, client, cfg.ModelID, history)
		cancel()

		if err != nil {
			debugf("api_call_result turn=%d ok=false latency_ms=%d request_id=%q error=%q", turn, time.Since(start).Milliseconds(), requestID, err.Error())
			fmt.Fprintf(os.Stderr, "API error: %v\n", err)
			continue
		}

		text, toolUses := parseContent(message.Content)
		if text == "" {
			text = "(no text content returned)"
		}

		history = append(history, message.ToParam())
		fmt.Fprintf(os.Stdout, "%s%s\n", assistantPrefix(cfg.ModelName, cfg.ColorOutput), text)

		debugf(
			"api_call_result turn=%d ok=true latency_ms=%d request_id=%q message_id=%q response_model=%q stop_reason=%q input_tokens=%d output_tokens=%d tool_use_count=%d",
			turn,
			time.Since(start).Milliseconds(),
			requestID,
			message.ID,
			message.Model,
			message.StopReason,
			message.Usage.InputTokens,
			message.Usage.OutputTokens,
			len(toolUses),
		)
		if len(toolUses) == 0 {
			debugf("api_response_tool_use_none turn=%d", turn)
			continue
		}
		for i, tool := range toolUses {
			debugf("api_response_tool_use turn=%d index=%d tool_id=%q tool_name=%q tool_input=%q", turn, i, tool.ID, tool.Name, tool.Input)
		}
	}
}

func sendAnthropicMessage(ctx context.Context, client *anthropic.Client, modelID string, history []anthropic.MessageParam) (*anthropic.Message, string, error) {
	var rawResp *http.Response
	message, err := client.Messages.New(
		ctx,
		anthropic.MessageNewParams{
			Model:     anthropic.Model(modelID),
			MaxTokens: defaultMaxTokens,
			Messages:  history,
		},
		option.WithResponseInto(&rawResp),
	)

	requestID := ""
	if rawResp != nil {
		requestID = rawResp.Header.Get("request-id")
	}
	if err != nil {
		if requestID != "" {
			return nil, requestID, fmt.Errorf("%w (request_id=%s)", err, requestID)
		}
		return nil, requestID, err
	}
	return message, requestID, nil
}

func parseContent(blocks []anthropic.ContentBlockUnion) (string, []ToolUse) {
	var text strings.Builder
	tools := make([]ToolUse, 0)

	for _, block := range blocks {
		switch block.Type {
		case "text":
			text.WriteString(block.Text)
		case "tool_use":
			input := strings.TrimSpace(string(block.Input))
			if input == "" {
				input = "{}"
			}
			tools = append(tools, ToolUse{ID: block.ID, Name: block.Name, Input: input})
		}
	}

	return strings.TrimSpace(text.String()), tools
}

func userPrefix(colorEnabled bool) string {
	if !colorEnabled {
		return "User: "
	}
	return userColor + "User: " + colorReset
}

func assistantPrefix(modelName string, colorEnabled bool) string {
	prefix := fmt.Sprintf("Claude (%s): ", modelName)
	if !colorEnabled {
		return prefix
	}
	return claudeColor + prefix + colorReset
}

func modelDisplayName(modelID string) string {
	if modelID == defaultModelID {
		return defaultModelName
	}
	return modelID
}

func supportsColor(output *os.File) bool {
	if output == nil || os.Getenv("NO_COLOR") != "" {
		return false
	}
	term := strings.ToLower(strings.TrimSpace(os.Getenv("TERM")))
	if term == "" || term == "dumb" {
		return false
	}
	info, err := output.Stat()
	if err != nil {
		return false
	}
	return (info.Mode() & os.ModeCharDevice) != 0
}
