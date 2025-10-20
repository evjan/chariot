package main

import (
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"os"
	"path/filepath"

	"github.com/invopop/jsonschema"
	orderedmap "github.com/wk8/go-ordered-map/v2"
)

func main() {
	client := NewOllamaClient("http://localhost:11434", "qwen3:8b")

	scanner := bufio.NewScanner(os.Stdin)
	getUserMessage := func() (string, bool) {
		if !scanner.Scan() {
			return "", false
		}
		return scanner.Text(), true
	}

	tools := []ToolDefinition{ReadFileDefinition, ListFilesDefinition}

	agent := NewAgent(client, getUserMessage, tools)
	err := agent.Run(context.TODO())
	if err != nil {
		fmt.Printf("Error: %s\n", err.Error())
	}
}

// OllamaClient wraps HTTP calls to Ollama
type OllamaClient struct {
	baseURL string
	model   string
	client  *http.Client
}

func NewOllamaClient(baseURL, model string) *OllamaClient {
	return &OllamaClient{
		baseURL: baseURL,
		model:   model,
		client:  &http.Client{},
	}
}

// Ollama API structures
type OllamaMessage struct {
	Role      string     `json:"role"`
	Content   string     `json:"content"`
	ToolCalls []ToolCall `json:"tool_calls,omitempty"`
}

type ToolCall struct {
	Function struct {
		Name      string          `json:"name"`
		Arguments json.RawMessage `json:"arguments"`
	} `json:"function"`
}

type OllamaTool struct {
	Type     string `json:"type"`
	Function struct {
		Name        string                 `json:"name"`
		Description string                 `json:"description"`
		Parameters  map[string]interface{} `json:"parameters"`
	} `json:"function"`
}

type OllamaRequest struct {
	Model    string          `json:"model"`
	Messages []OllamaMessage `json:"messages"`
	Tools    []OllamaTool    `json:"tools,omitempty"`
	Stream   bool            `json:"stream"`
}

type OllamaResponse struct {
	Message OllamaMessage `json:"message"`
	Done    bool          `json:"done"`
}

func (c *OllamaClient) Chat(ctx context.Context, messages []OllamaMessage, tools []OllamaTool) (*OllamaResponse, error) {
	reqBody := OllamaRequest{
		Model:    c.model,
		Messages: messages,
		Tools:    tools,
		Stream:   false,
	}

	jsonData, err := json.Marshal(reqBody)
	if err != nil {
		return nil, err
	}

	req, err := http.NewRequestWithContext(ctx, "POST", c.baseURL+"/api/chat", bytes.NewBuffer(jsonData))
	if err != nil {
		return nil, err
	}
	req.Header.Set("Content-Type", "application/json")

	resp, err := c.client.Do(req)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()

	body, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, err
	}

	var ollamaResp OllamaResponse
	err = json.Unmarshal(body, &ollamaResp)
	if err != nil {
		return nil, fmt.Errorf("failed to parse response: %w, body: %s", err, string(body))
	}

	return &ollamaResp, nil
}

func NewAgent(client *OllamaClient, getUserMessage func() (string, bool), tools []ToolDefinition) *Agent {
	return &Agent{
		client:         client,
		getUserMessage: getUserMessage,
		tools:          tools,
	}
}

type Agent struct {
	client         *OllamaClient
	getUserMessage func() (string, bool)
	tools          []ToolDefinition
}

func (a *Agent) Run(ctx context.Context) error {
	conversation := []OllamaMessage{}

	fmt.Println("Chat with Ollama (use 'ctrl-c' to quit)")

	readUserInput := true
	for {
		if readUserInput {
			fmt.Print("\u001b[94mYou\u001b[0m: ")
			userInput, ok := a.getUserMessage()
			if !ok {
				break
			}

			conversation = append(conversation, OllamaMessage{
				Role:    "user",
				Content: userInput,
			})
		}

		response, err := a.runInference(ctx, conversation)
		if err != nil {
			return err
		}

		conversation = append(conversation, response.Message)

		// Check if there are tool calls
		if len(response.Message.ToolCalls) > 0 {
			readUserInput = false
			toolResults := []string{}

			for _, toolCall := range response.Message.ToolCalls {
				result := a.executeTool(toolCall.Function.Name, toolCall.Function.Arguments)
				toolResults = append(toolResults, fmt.Sprintf("Tool %s result: %s", toolCall.Function.Name, result))
			}

			// Add tool results as a user message
			conversation = append(conversation, OllamaMessage{
				Role:    "user",
				Content: fmt.Sprintf("Tool results: %v", toolResults),
			})
		} else {
			// No tool calls, just print the response
			fmt.Printf("\u001b[93mOllama\u001b[0m: %s\n", response.Message.Content)
			readUserInput = true
		}
	}

	return nil
}

func (a *Agent) executeTool(name string, input json.RawMessage) string {
	var toolDef ToolDefinition
	var found bool
	for _, tool := range a.tools {
		if tool.Name == name {
			toolDef = tool
			found = true
			break
		}
	}
	if !found {
		return "tool not found"
	}

	fmt.Printf("\u001b[92mtool\u001b[0m: %s(%s)\n", name, input)
	response, err := toolDef.Function(input)
	if err != nil {
		return err.Error()
	}
	return response
}

func (a *Agent) runInference(ctx context.Context, conversation []OllamaMessage) (*OllamaResponse, error) {
	ollamaTools := []OllamaTool{}

	for _, tool := range a.tools {
		ollamaTool := OllamaTool{
			Type: "function",
		}
		ollamaTool.Function.Name = tool.Name
		ollamaTool.Function.Description = tool.Description

		// Convert OrderedMap to regular map
		properties := make(map[string]interface{})
		if tool.InputSchema.Properties != nil {
			for pair := tool.InputSchema.Properties.Oldest(); pair != nil; pair = pair.Next() {
				properties[pair.Key] = pair.Value
			}
		}

		// Convert the schema to Ollama format
		params := map[string]interface{}{
			"type":       "object",
			"properties": properties,
			"required":   []string{}, // Add required fields if needed
		}
		ollamaTool.Function.Parameters = params

		ollamaTools = append(ollamaTools, ollamaTool)
	}

	return a.client.Chat(ctx, conversation, ollamaTools)
}

type ToolDefinition struct {
	Name        string               `json:"name"`
	Description string               `json:"description"`
	InputSchema ToolInputSchemaParam `json:"input_schema"`
	Function    func(input json.RawMessage) (string, error)
}

type ToolInputSchemaParam struct {
	Properties *orderedmap.OrderedMap[string, *jsonschema.Schema]
}

var ReadFileDefinition = ToolDefinition{
	Name:        "read_file",
	Description: "Read the contents of a given relative file path. Use this when you want to see what's inside a file. Do not use this with directory names.",
	InputSchema: ReadFileInputSchema,
	Function:    ReadFile,
}

type ReadFileInput struct {
	Path string `json:"path" jsonschema_description:"The relative path of a file in the working directory."`
}

var ReadFileInputSchema = GenerateSchema[ReadFileInput]()

func ReadFile(input json.RawMessage) (string, error) {
	readFileInput := ReadFileInput{}
	err := json.Unmarshal(input, &readFileInput)
	if err != nil {
		panic(err)
	}

	content, err := os.ReadFile(readFileInput.Path)
	if err != nil {
		return "", err
	}
	return string(content), nil
}

func GenerateSchema[T any]() ToolInputSchemaParam {
	reflector := jsonschema.Reflector{
		AllowAdditionalProperties: false,
		DoNotReference:            true,
	}
	var v T

	schema := reflector.Reflect(v)

	return ToolInputSchemaParam{
		Properties: schema.Properties,
	}
}

var ListFilesDefinition = ToolDefinition{
	Name:        "list_files",
	Description: "List files and directories at a given path. If no path is provided, lists files in the current directory.",
	InputSchema: ListFilesInputSchema,
	Function:    ListFiles,
}

type ListFilesInput struct {
	Path string `json:"path,omitempty" jsonschema_description:"Optional relative path to list files from. Defaults to current directory if not provided."`
}

var ListFilesInputSchema = GenerateSchema[ListFilesInput]()

func ListFiles(input json.RawMessage) (string, error) {
	listFilesInput := ListFilesInput{}
	err := json.Unmarshal(input, &listFilesInput)
	if err != nil {
		panic(err)
	}

	dir := "."
	if listFilesInput.Path != "" {
		dir = listFilesInput.Path
	}

	var files []string
	err = filepath.Walk(dir, func(path string, info os.FileInfo, err error) error {
		if err != nil {
			return err
		}

		relPath, err := filepath.Rel(dir, path)
		if err != nil {
			return err
		}

		if relPath != "." {
			if info.IsDir() {
				files = append(files, relPath+"/")
			} else {
				files = append(files, relPath)
			}
		}
		return nil
	})

	if err != nil {
		return "", err
	}

	result, err := json.Marshal(files)
	if err != nil {
		return "", err
	}

	return string(result), nil
}
