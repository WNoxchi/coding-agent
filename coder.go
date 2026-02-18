package main

import (
	"bufio"
	"context"
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"io"
	"io/fs"
	"log"
	"net/http"
	"os"
	"path/filepath"
	"sort"
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

	defaultListFilesMaxEntries = 500
	hardListFilesMaxEntries    = 2000

	userColor   = "\x1b[38;2;102;178;255m"
	claudeColor = "\x1b[38;2;217;119;6m"
	toolColor   = "\x1b[96m"
	resultColor = "\x1b[92m"
	errorColor  = "\x1b[91m"
	colorReset  = "\x1b[0m"
)

var errListLimitReached = errors.New("list_files entry limit reached")

type Config struct {
	APIKey      string
	ModelID     string
	ModelName   string
	Verbose     bool
	ColorOutput bool
}

type ToolDefinition struct {
	Name        string
	Description string
	InputSchema anthropic.ToolInputSchemaParam
	Function    func(input json.RawMessage) (string, error)
}

type ToolUse struct {
	ID    string
	Name  string
	Input json.RawMessage
}

type ListFilesInput struct {
	Path       string `json:"path,omitempty"`
	Recursive  *bool  `json:"recursive,omitempty"`
	MaxEntries int    `json:"max_entries,omitempty"`
}

func main() {
	cfg, err := loadConfig()
	if err != nil {
		fmt.Fprintln(os.Stderr, "Error:", err)
		os.Exit(1)
	}

	toolDefs := registeredTools()
	toolMap, anthropicTools, err := buildToolRegistry(toolDefs)
	if err != nil {
		fmt.Fprintln(os.Stderr, "Error:", err)
		os.Exit(1)
	}

	configureLogging(cfg.Verbose)
	debugf(
		"startup init model_id=%q model_name=%q api_key_present=%t color_output=%t tool_count=%d",
		cfg.ModelID,
		cfg.ModelName,
		cfg.APIKey != "",
		cfg.ColorOutput,
		len(toolDefs),
	)

	client := anthropic.NewClient(option.WithAPIKey(cfg.APIKey))
	if err := runChatLoop(cfg, &client, toolMap, anthropicTools); err != nil {
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

func runChatLoop(cfg Config, client *anthropic.Client, toolMap map[string]ToolDefinition, anthropicTools []anthropic.ToolUnionParam) error {
	scanner := bufio.NewScanner(os.Stdin)
	history := make([]anthropic.MessageParam, 0, 32)
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

		call := 0
		callFailed := false
		for {
			call++
			start := time.Now()
			debugf(
				"api_call_start turn=%d call=%d model_id=%q conversation_len=%d tool_count=%d",
				turn,
				call,
				cfg.ModelID,
				len(history),
				len(anthropicTools),
			)

			ctx, cancel := context.WithTimeout(context.Background(), requestTimeout)
			message, requestID, err := sendAnthropicMessage(ctx, client, cfg.ModelID, history, anthropicTools)
			cancel()
			latencyMs := time.Since(start).Milliseconds()

			if err != nil {
				debugf("api_call_result turn=%d call=%d ok=false latency_ms=%d request_id=%q error=%q", turn, call, latencyMs, requestID, err.Error())
				fmt.Fprintf(os.Stderr, "API error: %v\n", err)
				callFailed = true
				break
			}

			history = append(history, message.ToParam())
			text, toolUses := parseContent(message.Content)

			debugf(
				"api_call_result turn=%d call=%d ok=true latency_ms=%d request_id=%q message_id=%q response_model=%q stop_reason=%q input_tokens=%d output_tokens=%d tool_use_count=%d",
				turn,
				call,
				latencyMs,
				requestID,
				message.ID,
				message.Model,
				message.StopReason,
				message.Usage.InputTokens,
				message.Usage.OutputTokens,
				len(toolUses),
			)

			if text != "" {
				fmt.Fprintf(os.Stdout, "%s%s\n", assistantPrefix(cfg.ModelName, cfg.ColorOutput), text)
			}

			if len(toolUses) == 0 {
				if text == "" {
					fmt.Fprintf(os.Stdout, "%s%s\n", assistantPrefix(cfg.ModelName, cfg.ColorOutput), "(no text content returned)")
				}
				debugf("api_response_tool_use_none turn=%d call=%d", turn, call)
				break
			}

			toolResults := make([]anthropic.ContentBlockParamUnion, 0, len(toolUses))
			for i, tool := range toolUses {
				debugf("api_response_tool_use turn=%d call=%d index=%d tool_id=%q tool_name=%q tool_input=%q", turn, call, i, tool.ID, tool.Name, string(tool.Input))

				fmt.Fprintf(os.Stdout, "%s: %s(%s)\n", colorLabel("tool", toolColor, cfg.ColorOutput), tool.Name, string(tool.Input))
				resultText, isError := runTool(toolMap, tool)
				if isError {
					fmt.Fprintf(os.Stdout, "%s: %s\n", colorLabel("error", errorColor, cfg.ColorOutput), resultText)
				} else {
					fmt.Fprintf(os.Stdout, "%s: %s\n", colorLabel("result", resultColor, cfg.ColorOutput), resultText)
				}
				toolResults = append(toolResults, anthropic.NewToolResultBlock(tool.ID, resultText, isError))
			}

			history = append(history, anthropic.NewUserMessage(toolResults...))
			debugf("tool_results_submitted turn=%d call=%d result_count=%d conversation_len=%d", turn, call, len(toolResults), len(history))
		}

		if callFailed {
			continue
		}
	}
}

func sendAnthropicMessage(
	ctx context.Context,
	client *anthropic.Client,
	modelID string,
	history []anthropic.MessageParam,
	tools []anthropic.ToolUnionParam,
) (*anthropic.Message, string, error) {
	var rawResp *http.Response
	message, err := client.Messages.New(
		ctx,
		anthropic.MessageNewParams{
			Model:     anthropic.Model(modelID),
			MaxTokens: defaultMaxTokens,
			Messages:  history,
			Tools:     tools,
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
			input := json.RawMessage(append([]byte(nil), block.Input...))
			if strings.TrimSpace(string(input)) == "" {
				input = json.RawMessage([]byte("{}"))
			}
			tools = append(tools, ToolUse{ID: block.ID, Name: block.Name, Input: input})
		}
	}

	return strings.TrimSpace(text.String()), tools
}

func runTool(toolMap map[string]ToolDefinition, toolUse ToolUse) (string, bool) {
	tool, ok := toolMap[toolUse.Name]
	if !ok {
		errMsg := fmt.Sprintf("unknown tool: %s", toolUse.Name)
		debugf("tool_call_result tool_name=%q ok=false error=%q", toolUse.Name, errMsg)
		return errMsg, true
	}

	debugf("tool_call_start tool_name=%q", toolUse.Name)
	result, err := tool.Function(toolUse.Input)
	if err != nil {
		errMsg := err.Error()
		debugf("tool_call_result tool_name=%q ok=false error=%q", toolUse.Name, errMsg)
		return errMsg, true
	}
	debugf("tool_call_result tool_name=%q ok=true result_chars=%d", toolUse.Name, len(result))
	return result, false
}

func registeredTools() []ToolDefinition {
	return []ToolDefinition{
		{
			Name:        "list_files",
			Description: "List files and directories in the current workspace. Use this to inspect the filesystem before reading or editing files.",
			InputSchema: listFilesInputSchema(),
			Function:    listFiles,
		},
	}
}

func buildToolRegistry(defs []ToolDefinition) (map[string]ToolDefinition, []anthropic.ToolUnionParam, error) {
	toolMap := make(map[string]ToolDefinition, len(defs))
	anthropicTools := make([]anthropic.ToolUnionParam, 0, len(defs))

	for _, def := range defs {
		if strings.TrimSpace(def.Name) == "" {
			return nil, nil, errors.New("tool name cannot be empty")
		}
		if _, exists := toolMap[def.Name]; exists {
			return nil, nil, fmt.Errorf("duplicate tool name: %s", def.Name)
		}

		toolMap[def.Name] = def
		anthropicTools = append(anthropicTools, anthropic.ToolUnionParam{
			OfTool: &anthropic.ToolParam{
				Name:        def.Name,
				Description: anthropic.String(def.Description),
				InputSchema: def.InputSchema,
			},
		})
	}

	return toolMap, anthropicTools, nil
}

func listFilesInputSchema() anthropic.ToolInputSchemaParam {
	return anthropic.ToolInputSchemaParam{
		Properties: map[string]any{
			"path": map[string]any{
				"type":        "string",
				"description": "Optional relative directory path. Defaults to current directory.",
			},
			"recursive": map[string]any{
				"type":        "boolean",
				"description": "Whether to recursively include nested files and directories. Defaults to true.",
			},
			"max_entries": map[string]any{
				"type":        "integer",
				"description": fmt.Sprintf("Maximum number of entries to return. Defaults to %d, capped at %d.", defaultListFilesMaxEntries, hardListFilesMaxEntries),
				"minimum":     1,
				"maximum":     hardListFilesMaxEntries,
			},
		},
		ExtraFields: map[string]any{
			"additionalProperties": false,
		},
	}
}

func listFiles(input json.RawMessage) (string, error) {
	args := ListFilesInput{}
	raw := strings.TrimSpace(string(input))
	if raw == "" {
		raw = "{}"
	}
	if err := json.Unmarshal([]byte(raw), &args); err != nil {
		return "", fmt.Errorf("invalid list_files input: %w", err)
	}

	recursive := true
	if args.Recursive != nil {
		recursive = *args.Recursive
	}

	maxEntries := defaultListFilesMaxEntries
	if args.MaxEntries > 0 {
		maxEntries = args.MaxEntries
	}
	if maxEntries > hardListFilesMaxEntries {
		maxEntries = hardListFilesMaxEntries
	}

	absDir, displayPath, err := resolveWorkspaceDir(args.Path)
	if err != nil {
		return "", err
	}

	entries, truncated, err := collectFileEntries(absDir, recursive, maxEntries)
	if err != nil {
		return "", err
	}

	if truncated {
		fmt.Fprintf(os.Stdout, "Searched %s\nListed %d files (truncated at max_entries=%d)\n", displayPath, len(entries), maxEntries)
	} else {
		fmt.Fprintf(os.Stdout, "Searched %s\nListed %d files\n", displayPath, len(entries))
	}

	encoded, err := json.Marshal(entries)
	if err != nil {
		return "", fmt.Errorf("failed to encode list_files output: %w", err)
	}

	return string(encoded), nil
}

func resolveWorkspaceDir(pathArg string) (string, string, error) {
	cwd, err := os.Getwd()
	if err != nil {
		return "", "", fmt.Errorf("failed to resolve working directory: %w", err)
	}

	pathArg = strings.TrimSpace(pathArg)
	if pathArg == "" {
		pathArg = "."
	}
	if filepath.IsAbs(pathArg) {
		return "", "", errors.New("path must be relative to the current workspace")
	}

	clean := filepath.Clean(pathArg)
	if clean == ".." || strings.HasPrefix(clean, ".."+string(filepath.Separator)) {
		return "", "", errors.New("path escapes the current workspace")
	}

	abs := filepath.Join(cwd, clean)
	abs, err = filepath.Abs(abs)
	if err != nil {
		return "", "", fmt.Errorf("failed to resolve absolute path: %w", err)
	}

	rel, err := filepath.Rel(cwd, abs)
	if err != nil {
		return "", "", fmt.Errorf("failed to resolve relative path: %w", err)
	}
	if rel == ".." || strings.HasPrefix(rel, ".."+string(filepath.Separator)) {
		return "", "", errors.New("path escapes the current workspace")
	}

	info, err := os.Stat(abs)
	if err != nil {
		return "", "", fmt.Errorf("failed to access path %q: %w", clean, err)
	}
	if !info.IsDir() {
		return "", "", fmt.Errorf("path is not a directory: %s", filepath.ToSlash(rel))
	}

	display := filepath.ToSlash(rel)
	if display == "" || display == "." {
		display = "."
	}

	return abs, display, nil
}

func collectFileEntries(dir string, recursive bool, maxEntries int) ([]string, bool, error) {
	if maxEntries < 1 {
		maxEntries = defaultListFilesMaxEntries
	}

	entries := make([]string, 0, min(maxEntries, 128))
	truncated := false

	if recursive {
		err := filepath.WalkDir(dir, func(path string, d fs.DirEntry, walkErr error) error {
			if walkErr != nil {
				return walkErr
			}
			if path == dir {
				return nil
			}

			rel, err := filepath.Rel(dir, path)
			if err != nil {
				return err
			}
			rel = filepath.ToSlash(rel)
			if d.IsDir() {
				rel += "/"
			}
			entries = append(entries, rel)

			if len(entries) >= maxEntries {
				truncated = true
				return errListLimitReached
			}
			return nil
		})
		if err != nil && !errors.Is(err, errListLimitReached) {
			return nil, false, err
		}
	} else {
		dirEntries, err := os.ReadDir(dir)
		if err != nil {
			return nil, false, err
		}
		for _, entry := range dirEntries {
			name := entry.Name()
			if entry.IsDir() {
				name += "/"
			}
			entries = append(entries, filepath.ToSlash(name))
			if len(entries) >= maxEntries {
				truncated = true
				break
			}
		}
	}

	sort.Strings(entries)
	return entries, truncated, nil
}

func min(a, b int) int {
	if a < b {
		return a
	}
	return b
}

func colorLabel(label, color string, colorEnabled bool) string {
	if !colorEnabled {
		return label
	}
	return color + label + colorReset
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
