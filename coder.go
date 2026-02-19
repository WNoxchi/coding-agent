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
	"os/exec"
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
	defaultMaxTokens = int64(8192)
	defaultTemp      = 0.2
	requestTimeout   = 120 * time.Second

	defaultListFilesMaxEntries = 500
	hardListFilesMaxEntries    = 2000
	defaultReadFilesMaxBytes   = 32_000
	hardReadFilesMaxBytes      = 256_000
	defaultBashTimeoutSeconds  = 30
	hardBashTimeoutSeconds     = 120
	defaultBashMaxOutputBytes  = 32_000
	hardBashMaxOutputBytes     = 256_000
	maxToolRoundsPerTurn       = 16
	maxRepeatedToolFailures    = 2

	toolUseSystemPrompt = `You are a coding agent that can use filesystem and shell tools.
Use tools with strict JSON inputs that match each schema exactly.
- For creating a new file or replacing an entire file, use write_file.
- For targeted edits, use edit_file or edit_files with path, old_str, and new_str.
- Never call bash without a non-empty "command" field.
- If a tool returns an input-validation error, fix the JSON and retry with corrected arguments.`

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

type ReadFilesInput struct {
	Path     *string `json:"path"`
	MaxBytes int     `json:"max_bytes,omitempty"`
}

type BashInput struct {
	Command        *string `json:"command"`
	Cmd            *string `json:"cmd,omitempty"`
	TimeoutSeconds int     `json:"timeout_seconds,omitempty"`
	MaxOutputBytes int     `json:"max_output_bytes,omitempty"`
}

type EditFilesInput struct {
	Path   *string `json:"path"`
	OldStr *string `json:"old_str"`
	NewStr *string `json:"new_str"`
}

type WriteFileInput struct {
	Path      *string `json:"path"`
	Content   *string `json:"content"`
	Text      *string `json:"text,omitempty"`
	Body      *string `json:"body,omitempty"`
	NewStr    *string `json:"new_str,omitempty"`
	Overwrite *bool   `json:"overwrite,omitempty"`
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
		lastFailureSignature := ""
		repeatedFailureCount := 0
		for {
			if call >= maxToolRoundsPerTurn {
				stopMsg := fmt.Sprintf("Stopped after %d tool rounds in this turn to prevent a tool loop. Please provide corrected instructions and try again.", maxToolRoundsPerTurn)
				fmt.Fprintf(os.Stdout, "%s%s\n", assistantPrefix(cfg.ModelName, cfg.ColorOutput), stopMsg)
				debugf("tool_loop_stop turn=%d reason=%q call=%d", turn, "max_tool_rounds", call)
				break
			}

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
			allToolsFailed := true
			failureSig := make([]string, 0, len(toolUses))
			hasValidationError := false
			for i, tool := range toolUses {
				debugf("api_response_tool_use turn=%d call=%d index=%d tool_id=%q tool_name=%q tool_input=%q", turn, call, i, tool.ID, tool.Name, string(tool.Input))
				failureSig = append(failureSig, tool.Name+"="+strings.TrimSpace(string(tool.Input)))

				fmt.Fprintf(os.Stdout, "%s: %s(%s)\n", colorLabel("tool", toolColor, cfg.ColorOutput), tool.Name, string(tool.Input))
				resultText, isError := runTool(toolMap, tool)
				if !isError {
					allToolsFailed = false
				}
				if isError && isToolInputValidationError(resultText) {
					hasValidationError = true
				}
				if isError {
					fmt.Fprintf(os.Stdout, "%s: %s\n", colorLabel("error", errorColor, cfg.ColorOutput), resultText)
				} else {
					fmt.Fprintf(os.Stdout, "%s: %s\n", colorLabel("result", resultColor, cfg.ColorOutput), resultText)
				}
				toolResults = append(toolResults, anthropic.NewToolResultBlock(tool.ID, resultText, isError))
			}

			if hasValidationError {
				toolResults = append(toolResults, anthropic.NewTextBlock(
					"One or more tool calls had invalid JSON input. Retry with exact required fields from each error message. For full file contents, use write_file with path and content. Do not call bash unless command is non-empty.",
				))
			}

			history = append(history, anthropic.NewUserMessage(toolResults...))
			debugf("tool_results_submitted turn=%d call=%d result_count=%d conversation_len=%d", turn, call, len(toolResults), len(history))

			if allToolsFailed {
				signature := strings.Join(failureSig, "|")
				if signature == lastFailureSignature {
					repeatedFailureCount++
				} else {
					lastFailureSignature = signature
					repeatedFailureCount = 1
				}
				if repeatedFailureCount >= maxRepeatedToolFailures {
					stopMsg := "Stopping tool loop after repeated identical tool failures. I need corrected tool inputs to continue."
					fmt.Fprintf(os.Stdout, "%s%s\n", assistantPrefix(cfg.ModelName, cfg.ColorOutput), stopMsg)
					debugf("tool_loop_stop turn=%d reason=%q call=%d repeat_count=%d signature=%q", turn, "repeated_tool_failures", call, repeatedFailureCount, signature)
					break
				}
			} else {
				lastFailureSignature = ""
				repeatedFailureCount = 0
			}
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
			Model:       anthropic.Model(modelID),
			MaxTokens:   defaultMaxTokens,
			Temperature: anthropic.Float(defaultTemp),
			Messages:    history,
			System:      []anthropic.TextBlockParam{{Text: toolUseSystemPrompt}},
			Tools:       tools,
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
			Name:        "write_file",
			Description: "Create or overwrite a text file in the current workspace. Use this to write full file contents in one call.",
			InputSchema: writeFileInputSchema(),
			Function:    writeFile,
		},
		{
			Name: "edit_file",
			Description: `Apply a targeted edit to an existing text file.
If old_str is empty and the file exists, new_str is appended.
If old_str is non-empty, it must match exactly once and will be replaced by new_str.`,
			InputSchema: editFilesInputSchema(),
			Function:    editFiles,
		},
		{
			Name: "edit_files",
			Description: `Apply a targeted edit to an existing text file.
If old_str is empty and the file exists, new_str is appended.
If old_str is non-empty, it must match exactly once and will be replaced by new_str.`,
			InputSchema: editFilesInputSchema(),
			Function:    editFiles,
		},
		{
			Name:        "bash",
			Description: "Execute a bash command in the current workspace and return combined stdout/stderr output. Always include a non-empty command field.",
			InputSchema: bashInputSchema(),
			Function:    bashTool,
		},
		{
			Name:        "read_file",
			Description: "Read a file in the current workspace. Use this to inspect exact file contents.",
			InputSchema: readFilesInputSchema(),
			Function:    readFiles,
		},
		{
			Name:        "read_files",
			Description: "Read the contents of a file in the current workspace. Use this to inspect specific files after discovering paths with list_files.",
			InputSchema: readFilesInputSchema(),
			Function:    readFiles,
		},
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

func writeFileInputSchema() anthropic.ToolInputSchemaParam {
	return anthropic.ToolInputSchemaParam{
		Properties: map[string]any{
			"path": map[string]any{
				"type":        "string",
				"description": "Relative file path within the current workspace.",
			},
			"content": map[string]any{
				"type":        "string",
				"description": "Full text content to write to the file.",
			},
			"overwrite": map[string]any{
				"type":        "boolean",
				"description": "Whether to overwrite an existing file. Defaults to false.",
			},
		},
		Required: []string{"path", "content"},
		ExtraFields: map[string]any{
			"additionalProperties": false,
		},
	}
}

func editFilesInputSchema() anthropic.ToolInputSchemaParam {
	return anthropic.ToolInputSchemaParam{
		Properties: map[string]any{
			"path": map[string]any{
				"type":        "string",
				"description": "Relative file path within the current workspace.",
			},
			"old_str": map[string]any{
				"type":        "string",
				"description": "Text to replace. Use an empty string to create a new file or append to an existing file.",
			},
			"new_str": map[string]any{
				"type":        "string",
				"description": "Replacement text, or content to create/append when old_str is empty.",
			},
		},
		Required: []string{"path", "old_str", "new_str"},
		ExtraFields: map[string]any{
			"additionalProperties": false,
		},
	}
}

func bashInputSchema() anthropic.ToolInputSchemaParam {
	return anthropic.ToolInputSchemaParam{
		Properties: map[string]any{
			"command": map[string]any{
				"type":        "string",
				"description": "The bash command to execute.",
			},
			"cmd": map[string]any{
				"type":        "string",
				"description": "Alias of command. Prefer command.",
			},
			"timeout_seconds": map[string]any{
				"type":        "integer",
				"description": fmt.Sprintf("Optional timeout in seconds. Defaults to %d, capped at %d.", defaultBashTimeoutSeconds, hardBashTimeoutSeconds),
				"minimum":     1,
				"maximum":     hardBashTimeoutSeconds,
			},
			"max_output_bytes": map[string]any{
				"type":        "integer",
				"description": fmt.Sprintf("Maximum bytes of command output to return. Defaults to %d, capped at %d.", defaultBashMaxOutputBytes, hardBashMaxOutputBytes),
				"minimum":     1,
				"maximum":     hardBashMaxOutputBytes,
			},
		},
		Required: []string{"command"},
		ExtraFields: map[string]any{
			"additionalProperties": false,
		},
	}
}

func readFilesInputSchema() anthropic.ToolInputSchemaParam {
	return anthropic.ToolInputSchemaParam{
		Properties: map[string]any{
			"path": map[string]any{
				"type":        "string",
				"description": "Relative file path within the current workspace.",
			},
			"max_bytes": map[string]any{
				"type":        "integer",
				"description": fmt.Sprintf("Maximum bytes to read from the file. Defaults to %d, capped at %d.", defaultReadFilesMaxBytes, hardReadFilesMaxBytes),
				"minimum":     1,
				"maximum":     hardReadFilesMaxBytes,
			},
		},
		Required: []string{"path"},
		ExtraFields: map[string]any{
			"additionalProperties": false,
		},
	}
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

func toolInputValidationError(toolName, reason, expected string) error {
	if expected == "" {
		return fmt.Errorf("invalid %s input: %s", toolName, reason)
	}
	return fmt.Errorf("invalid %s input: %s. expected input like %s", toolName, reason, expected)
}

func isToolInputValidationError(resultText string) bool {
	lower := strings.ToLower(strings.TrimSpace(resultText))
	return strings.HasPrefix(lower, "invalid ")
}

func requireToolString(toolName, fieldName string, value *string, allowEmpty bool, expected string) (string, error) {
	if value == nil {
		return "", toolInputValidationError(toolName, fmt.Sprintf("missing required field %q", fieldName), expected)
	}
	if !allowEmpty && strings.TrimSpace(*value) == "" {
		return "", toolInputValidationError(toolName, fmt.Sprintf("field %q cannot be empty", fieldName), expected)
	}
	return *value, nil
}

func writeFile(input json.RawMessage) (string, error) {
	const expected = `{"path":"src/main.py","content":"print(\"hello\")","overwrite":true}`

	args := WriteFileInput{}
	raw := strings.TrimSpace(string(input))
	if raw == "" {
		raw = "{}"
	}
	if err := json.Unmarshal([]byte(raw), &args); err != nil {
		return "", toolInputValidationError("write_file", err.Error(), expected)
	}

	pathValue, err := requireToolString("write_file", "path", args.Path, false, expected)
	if err != nil {
		return "", err
	}
	contentSource := args.Content
	if contentSource == nil {
		contentSource = args.Text
	}
	if contentSource == nil {
		contentSource = args.Body
	}
	if contentSource == nil {
		contentSource = args.NewStr
	}
	if contentSource == nil {
		return "", toolInputValidationError(
			"write_file",
			`missing required field "content" (accepted aliases: "text", "body", "new_str"); include the full file contents`,
			expected,
		)
	}
	content, err := requireToolString("write_file", "content", contentSource, true, expected)
	if err != nil {
		return "", err
	}
	pathValue = strings.TrimSpace(pathValue)

	overwrite := false
	if args.Overwrite != nil {
		overwrite = *args.Overwrite
	}

	absFile, displayPath, err := resolveWorkspaceFileForWrite(pathValue)
	if err != nil {
		return "", err
	}

	exists := false
	info, statErr := os.Stat(absFile)
	if statErr == nil {
		exists = true
		if info.IsDir() {
			return "", fmt.Errorf("path is a directory: %s", displayPath)
		}
	} else if !os.IsNotExist(statErr) {
		return "", fmt.Errorf("failed to access path %q: %w", displayPath, statErr)
	}

	if exists && !overwrite {
		return "", toolInputValidationError("write_file", fmt.Sprintf("file already exists: %s (set overwrite=true to replace it)", displayPath), expected)
	}
	if err := os.MkdirAll(filepath.Dir(absFile), 0o755); err != nil {
		return "", fmt.Errorf("failed to create parent directory for %q: %w", displayPath, err)
	}
	if err := os.WriteFile(absFile, []byte(content), 0o644); err != nil {
		return "", fmt.Errorf("failed to write file %q: %w", displayPath, err)
	}

	if exists {
		fmt.Fprintf(os.Stdout, "Overwrote %s (%d bytes)\n", displayPath, len(content))
	} else {
		fmt.Fprintf(os.Stdout, "Created %s (%d bytes)\n", displayPath, len(content))
	}
	return fmt.Sprintf("wrote file %s", displayPath), nil
}

func editFiles(input json.RawMessage) (string, error) {
	const expected = `{"path":"src/main.py","old_str":"before","new_str":"after"}`

	args := EditFilesInput{}
	raw := strings.TrimSpace(string(input))
	if raw == "" {
		raw = "{}"
	}
	if err := json.Unmarshal([]byte(raw), &args); err != nil {
		return "", toolInputValidationError("edit_files", err.Error(), expected)
	}

	pathValue, err := requireToolString("edit_files", "path", args.Path, false, expected)
	if err != nil {
		return "", err
	}
	oldStr, err := requireToolString("edit_files", "old_str", args.OldStr, true, expected)
	if err != nil {
		return "", err
	}
	newStr, err := requireToolString("edit_files", "new_str", args.NewStr, true, expected)
	if err != nil {
		return "", err
	}
	pathValue = strings.TrimSpace(pathValue)

	if oldStr == newStr {
		return "", toolInputValidationError("edit_files", `"old_str" and "new_str" must be different`, expected)
	}

	absFile, displayPath, err := resolveWorkspaceFileForWrite(pathValue)
	if err != nil {
		return "", err
	}

	info, statErr := os.Stat(absFile)
	if statErr != nil {
		if !os.IsNotExist(statErr) {
			return "", fmt.Errorf("failed to access path %q: %w", displayPath, statErr)
		}
		if oldStr != "" {
			return "", fmt.Errorf("file does not exist: %s (old_str must be empty to create it; otherwise use write_file)", displayPath)
		}
		if err := os.MkdirAll(filepath.Dir(absFile), 0o755); err != nil {
			return "", fmt.Errorf("failed to create parent directory for %q: %w", displayPath, err)
		}
		if err := os.WriteFile(absFile, []byte(newStr), 0o644); err != nil {
			return "", fmt.Errorf("failed to create file %q: %w", displayPath, err)
		}
		fmt.Fprintf(os.Stdout, "Created %s (%d bytes)\n", displayPath, len(newStr))
		return fmt.Sprintf("created file %s", displayPath), nil
	}

	if info.IsDir() {
		return "", fmt.Errorf("path is a directory: %s", displayPath)
	}

	contentBytes, err := os.ReadFile(absFile)
	if err != nil {
		return "", fmt.Errorf("failed to read file %q: %w", displayPath, err)
	}
	content := string(contentBytes)

	var newContent string
	switch {
	case oldStr == "":
		newContent = content + newStr
	case strings.Count(content, oldStr) == 0:
		return "", fmt.Errorf("old_str not found in file: %s", displayPath)
	case strings.Count(content, oldStr) > 1:
		return "", fmt.Errorf("old_str appears multiple times in file: %s; provide more specific text", displayPath)
	default:
		newContent = strings.Replace(content, oldStr, newStr, 1)
	}

	if err := os.WriteFile(absFile, []byte(newContent), 0o644); err != nil {
		return "", fmt.Errorf("failed to write file %q: %w", displayPath, err)
	}

	fmt.Fprintf(os.Stdout, "Edited %s\n", displayPath)
	return fmt.Sprintf("edited file %s", displayPath), nil
}

func bashTool(input json.RawMessage) (string, error) {
	const expected = `{"command":"python3 app.py","timeout_seconds":30}`

	args := BashInput{}
	raw := strings.TrimSpace(string(input))
	if raw == "" {
		raw = "{}"
	}
	if err := json.Unmarshal([]byte(raw), &args); err != nil {
		return "", toolInputValidationError("bash", err.Error(), expected)
	}

	command := ""
	if args.Command != nil {
		command = *args.Command
	}
	if strings.TrimSpace(command) == "" && args.Cmd != nil {
		command = *args.Cmd
	}
	command = strings.TrimSpace(command)
	if command == "" {
		return "", toolInputValidationError("bash", `missing required field "command"`, expected)
	}

	timeoutSeconds := defaultBashTimeoutSeconds
	if args.TimeoutSeconds > 0 {
		timeoutSeconds = args.TimeoutSeconds
	}
	if timeoutSeconds > hardBashTimeoutSeconds {
		timeoutSeconds = hardBashTimeoutSeconds
	}

	maxOutputBytes := defaultBashMaxOutputBytes
	if args.MaxOutputBytes > 0 {
		maxOutputBytes = args.MaxOutputBytes
	}
	if maxOutputBytes > hardBashMaxOutputBytes {
		maxOutputBytes = hardBashMaxOutputBytes
	}

	cwd, err := os.Getwd()
	if err != nil {
		return "", fmt.Errorf("failed to resolve working directory: %w", err)
	}

	debugf("bash_tool_start command=%q timeout_seconds=%d max_output_bytes=%d", command, timeoutSeconds, maxOutputBytes)

	ctx, cancel := context.WithTimeout(context.Background(), time.Duration(timeoutSeconds)*time.Second)
	defer cancel()

	cmd := exec.CommandContext(ctx, "bash", "-lc", command)
	cmd.Dir = cwd
	output, runErr := cmd.CombinedOutput()

	truncatedOutput, wasTruncated := truncateOutput(output, maxOutputBytes)
	trimmedOutput := strings.TrimSpace(truncatedOutput)

	if ctx.Err() == context.DeadlineExceeded {
		msg := fmt.Sprintf("Command timed out after %d seconds.", timeoutSeconds)
		if trimmedOutput != "" {
			msg += "\n\nPartial output:\n" + trimmedOutput
		}
		if wasTruncated {
			msg += fmt.Sprintf("\n\n(output truncated at max_output_bytes=%d)", maxOutputBytes)
		}
		return msg, nil
	}

	if runErr != nil {
		var exitErr *exec.ExitError
		if errors.As(runErr, &exitErr) {
			msg := fmt.Sprintf("Command exited with code %d.", exitErr.ExitCode())
			if trimmedOutput != "" {
				msg += "\n\nOutput:\n" + trimmedOutput
			}
			if wasTruncated {
				msg += fmt.Sprintf("\n\n(output truncated at max_output_bytes=%d)", maxOutputBytes)
			}
			return msg, nil
		}
		return "", fmt.Errorf("failed to execute command: %w", runErr)
	}

	if trimmedOutput == "" {
		return "Command completed successfully with no output.", nil
	}
	if wasTruncated {
		return trimmedOutput + fmt.Sprintf("\n\n(output truncated at max_output_bytes=%d)", maxOutputBytes), nil
	}
	return trimmedOutput, nil
}

func readFiles(input json.RawMessage) (string, error) {
	const expected = `{"path":"main.py","max_bytes":32000}`

	args := ReadFilesInput{}
	raw := strings.TrimSpace(string(input))
	if raw == "" {
		raw = "{}"
	}
	if err := json.Unmarshal([]byte(raw), &args); err != nil {
		return "", toolInputValidationError("read_files", err.Error(), expected)
	}

	pathValue, err := requireToolString("read_files", "path", args.Path, false, expected)
	if err != nil {
		return "", err
	}
	pathValue = strings.TrimSpace(pathValue)

	maxBytes := defaultReadFilesMaxBytes
	if args.MaxBytes > 0 {
		maxBytes = args.MaxBytes
	}
	if maxBytes > hardReadFilesMaxBytes {
		maxBytes = hardReadFilesMaxBytes
	}

	absFile, displayPath, err := resolveWorkspaceFile(pathValue)
	if err != nil {
		return "", err
	}

	content, err := os.ReadFile(absFile)
	if err != nil {
		return "", fmt.Errorf("failed to read file %q: %w", displayPath, err)
	}

	truncated := false
	if len(content) > maxBytes {
		content = content[:maxBytes]
		truncated = true
	}

	if truncated {
		fmt.Fprintf(os.Stdout, "Read %s (%d bytes, truncated at max_bytes=%d)\n", displayPath, len(content), maxBytes)
	} else {
		fmt.Fprintf(os.Stdout, "Read %s (%d bytes)\n", displayPath, len(content))
	}

	return string(content), nil
}

func truncateOutput(output []byte, maxBytes int) (string, bool) {
	if maxBytes < 1 {
		maxBytes = defaultBashMaxOutputBytes
	}
	if len(output) <= maxBytes {
		return string(output), false
	}
	return string(output[:maxBytes]), true
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

func resolveWorkspaceFileForWrite(pathArg string) (string, string, error) {
	cwd, err := os.Getwd()
	if err != nil {
		return "", "", fmt.Errorf("failed to resolve working directory: %w", err)
	}

	pathArg = strings.TrimSpace(pathArg)
	if pathArg == "" {
		return "", "", errors.New("path is required")
	}
	if filepath.IsAbs(pathArg) {
		return "", "", errors.New("path must be relative to the current workspace")
	}

	clean := filepath.Clean(pathArg)
	if clean == "." {
		return "", "", errors.New("path must point to a file")
	}
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

	return abs, filepath.ToSlash(rel), nil
}

func resolveWorkspaceFile(pathArg string) (string, string, error) {
	cwd, err := os.Getwd()
	if err != nil {
		return "", "", fmt.Errorf("failed to resolve working directory: %w", err)
	}

	pathArg = strings.TrimSpace(pathArg)
	if pathArg == "" {
		return "", "", errors.New("path is required")
	}
	if filepath.IsAbs(pathArg) {
		return "", "", errors.New("path must be relative to the current workspace")
	}

	clean := filepath.Clean(pathArg)
	if clean == "." {
		return "", "", errors.New("path must point to a file")
	}
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
	if info.IsDir() {
		return "", "", fmt.Errorf("path is a directory: %s", filepath.ToSlash(rel))
	}

	display := filepath.ToSlash(rel)
	return abs, display, nil
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
