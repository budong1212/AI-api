const DEFAULT_UPSTREAM_BASE_URL = "https://unlimited.surf";
const DEFAULT_OPENAI_MODEL = "gateway-gpt-5-5";
const DEFAULT_CLAUDE_MODEL = "claude-opus-4-7-20260101";

const CORS_HEADERS = {
  "Access-Control-Allow-Origin": "*",
  "Access-Control-Allow-Methods": "GET,POST,PUT,PATCH,DELETE,OPTIONS",
  "Access-Control-Allow-Headers": "authorization,content-type,x-api-key,anthropic-api-key,anthropic-version,anthropic-beta,openai-beta",
  "Access-Control-Expose-Headers": "content-type,request-id,x-request-id",
};

export default {
  async fetch(request, env) {
    if (request.method === "OPTIONS") {
      return new Response(null, { status: 204, headers: CORS_HEADERS });
    }

    const url = new URL(request.url);
    const path = normalizePath(url.pathname);

    try {
      const authError = validateWorkerApiKey(request, env);
      if (authError) return authError;

      if (path === "/" || path === "/health") {
        return jsonResponse(serviceInfo(request, env));
      }

      if (path.startsWith("/api/")) {
        return proxyUpstream(request, env, path);
      }

      if (path === "/mcp" || path === "/v1/mcp" || path === "/anthropic/mcp" || path === "/anthropic/v1/mcp") {
        return jsonResponse(mcpInfo(request));
      }

      if (path === "/codex" || path === "/v1/codex" || path === "/anthropic/codex" || path === "/anthropic/v1/codex") {
        return textResponse(codexSetup(request), "text/plain; charset=utf-8");
      }

      if (path === "/v1/setup" || path === "/anthropic/setup" || path === "/anthropic/v1/setup") {
        return textResponse(agentSetup(request), "text/plain; charset=utf-8");
      }

      if (path === "/v1/messages" || (path === "/v1/models" && looksLikeAnthropicRequest(request)) || path.startsWith("/anthropic/")) {
        return handleAnthropic(request, env, path);
      }

      if (path.startsWith("/v1/")) {
        return handleOpenAI(request, env, path);
      }

      return errorResponse(404, "not_found", `No route for ${path}`);
    } catch (error) {
      return errorResponse(500, "internal_error", error && error.message ? error.message : String(error));
    }
  },
};

async function handleOpenAI(request, env, path) {
  if ((path === "/v1/key" || path === "/v1/auth-key" || path === "/v1/usage") && request.method === "GET") {
    const rawPath = path === "/v1/usage" ? "/api/usage" : "/api/key";
    return proxyUpstream(request, env, rawPath);
  }

  if (path === "/v1/models" && request.method === "GET") {
    if (looksLikeAnthropicRequest(request)) {
      return anthropicModels(request, env);
    }
    return openAIModels(request, env);
  }

  if (path === "/v1/search" && request.method === "POST") {
    const body = await readJson(request);
    return openAIDirectCapability(request, env, body, "/api/search");
  }

  if (path === "/v1/merge" && request.method === "POST") {
    const body = await readJson(request);
    return openAIDirectCapability(request, env, body, "/api/merge");
  }

  if (path === "/v1/chat/completions" && request.method === "POST") {
    const body = await readJson(request);
    return openAIChatCompletions(request, env, body);
  }

  if (path === "/v1/responses" && request.method === "POST") {
    const body = await readJson(request);
    return openAIResponses(request, env, body);
  }

  if (path === "/v1/files" && request.method === "GET") {
    return jsonResponse({ object: "list", data: [], has_more: false });
  }

  if (path === "/v1/files" && request.method === "POST") {
    return openAIFileUpload(request, env);
  }

  if ((path === "/v1/files/extract" || path === "/v1/attachments/extract") && request.method === "POST") {
    const body = await readJson(request);
    const extracted = await callUnlimitedJson(request, env, "/api/attachments/extract", body);
    return jsonResponse(extracted);
  }

  if (path.startsWith("/v1/files/") && request.method === "GET") {
    return errorResponse(404, "not_found", "This Worker is stateless. Bind KV/R2 if you need persisted OpenAI file retrieval.");
  }

  if (path === "/v1/embeddings" || path.startsWith("/v1/audio/") || path.startsWith("/v1/images/")) {
    return errorResponse(501, "unsupported_endpoint", `${path} is not exposed by unlimited.surf and cannot be emulated faithfully.`);
  }

  return errorResponse(404, "not_found", `Unsupported OpenAI-compatible route ${path}`);
}

async function openAIDirectCapability(request, env, body, route) {
  const model = body.model || env.DEFAULT_MODEL || DEFAULT_OPENAI_MODEL;
  const created = nowSeconds();
  const id = `chatcmpl_${randomId()}`;
  const payload = buildUnlimitedPayload({ ...body, web_search: route === "/api/search", merge: route === "/api/merge" }, route);

  if (body.stream !== false && !hasEmulatableTools(body.tools)) {
    const upstream = await callUnlimitedStream(request, env, route, payload);
    return sseResponse(streamOpenAIChat(upstream, { id, created, model }));
  }

  const result = finalizeToolResult(await collectUnlimitedText(request, env, route, payload), body);

  if (body.stream !== false) {
    return sseResponse(streamSyntheticOpenAIChat(result, {
      id,
      created,
      model,
      systemFingerprint: `unlimited-surf-worker:${route}`,
      inputText: payload.message || payload.query || "",
    }));
  }

  return jsonResponse(buildOpenAIChatCompletion(result, {
    id,
    created,
    model,
    inputText: payload.message || payload.query || "",
    systemFingerprint: `unlimited-surf-worker:${route}`,
  }));
}

async function openAIChatCompletions(request, env, body) {
  const model = body.model || env.DEFAULT_MODEL || DEFAULT_OPENAI_MODEL;
  const created = nowSeconds();
  const id = `chatcmpl_${randomId()}`;
  const route = chooseUnlimitedRoute(body);
  const payload = buildUnlimitedPayload(body, route);

  if (body.stream && !hasEmulatableTools(body.tools)) {
    const upstream = await callUnlimitedStream(request, env, route, payload);
    return sseResponse(streamOpenAIChat(upstream, { id, created, model }));
  }

  const result = finalizeToolResult(await collectUnlimitedText(request, env, route, payload), body);

  if (body.stream) {
    return sseResponse(streamSyntheticOpenAIChat(result, {
      id,
      created,
      model,
      systemFingerprint: "unlimited-surf-worker",
      inputText: payload.message || "",
    }));
  }

  return jsonResponse(buildOpenAIChatCompletion(result, {
    id,
    created,
    model,
    inputText: payload.message || "",
    systemFingerprint: "unlimited-surf-worker",
  }));
}

async function openAIResponses(request, env, body) {
  const model = body.model || env.DEFAULT_MODEL || DEFAULT_OPENAI_MODEL;
  const created = nowSeconds();
  const id = `resp_${randomId()}`;
  const syntheticChatBody = responsesToChatBody(body, model);
  const route = chooseUnlimitedRoute(syntheticChatBody);
  const payload = buildUnlimitedPayload(syntheticChatBody, route);

  if (body.stream && !hasEmulatableTools(body.tools)) {
    const upstream = await callUnlimitedStream(request, env, route, payload);
    return sseResponse(streamOpenAIResponses(upstream, { id, created, model }));
  }

  const result = finalizeToolResult(await collectUnlimitedText(request, env, route, payload), body);

  if (body.stream) {
    return sseResponse(streamSyntheticOpenAIResponse(result, {
      id,
      created,
      model,
      body,
      inputText: payload.message || "",
    }));
  }

  return jsonResponse(buildOpenAIResponse(result, {
    id,
    created,
    model,
    body,
    inputText: payload.message || "",
  }));
}

async function handleAnthropic(request, env, path) {
  const anthPath = path.startsWith("/anthropic/") ? normalizePath(path.slice("/anthropic".length) || "/") : path;

  if ((anthPath === "/v1/key" || anthPath === "/key" || anthPath === "/v1/auth-key" || anthPath === "/auth-key") && request.method === "GET") {
    return proxyUpstream(request, env, "/api/key");
  }

  if ((anthPath === "/v1/usage" || anthPath === "/usage") && request.method === "GET") {
    return proxyUpstream(request, env, "/api/usage");
  }

  if ((anthPath === "/v1/models" || anthPath === "/models") && request.method === "GET") {
    return anthropicModels(request, env);
  }

  if ((anthPath === "/v1/messages" || anthPath === "/messages") && request.method === "POST") {
    const body = await readJson(request);
    return anthropicMessages(request, env, body);
  }

  if ((anthPath === "/v1/search" || anthPath === "/search") && request.method === "POST") {
    const body = await readJson(request);
    return anthropicDirectCapability(request, env, body, "/api/search");
  }

  if ((anthPath === "/v1/merge" || anthPath === "/merge") && request.method === "POST") {
    const body = await readJson(request);
    return anthropicDirectCapability(request, env, body, "/api/merge");
  }

  if (anthPath === "/v1/setup" || anthPath === "/setup") {
    return textResponse(agentSetup(request), "text/plain; charset=utf-8");
  }

  return errorResponse(404, "not_found", `Unsupported Anthropic-compatible route ${path}`);
}

async function anthropicDirectCapability(request, env, body, route) {
  const requestedModel = body.model || env.DEFAULT_CLAUDE_MODEL || DEFAULT_CLAUDE_MODEL;
  const payload = buildAnthropicUnlimitedPayload({ ...body, web_search: route === "/api/search", merge: route === "/api/merge" }, route);
  const id = `msg_${randomId()}`;

  if (body.stream !== false && !hasEmulatableTools(body.tools)) {
    const upstream = await callUnlimitedStream(request, env, route, payload);
    return sseResponse(streamAnthropicMessages(upstream, { id, model: requestedModel }));
  }

  const result = finalizeToolResult(await collectUnlimitedText(request, env, route, payload), body);

  if (body.stream !== false) {
    return sseResponse(streamSyntheticAnthropicMessage(result, {
      id,
      model: requestedModel,
      inputText: payload.message || payload.query || "",
    }));
  }

  return jsonResponse(buildAnthropicMessage(result, {
    id,
    model: requestedModel,
    inputText: payload.message || payload.query || "",
  }));
}

async function anthropicMessages(request, env, body) {
  const requestedModel = body.model || env.DEFAULT_CLAUDE_MODEL || DEFAULT_CLAUDE_MODEL;
  const route = chooseUnlimitedRoute(body);
  const payload = buildAnthropicUnlimitedPayload(body, route);
  const id = `msg_${randomId()}`;

  if (body.stream && !hasEmulatableTools(body.tools)) {
    const upstream = await callUnlimitedStream(request, env, route, payload);
    return sseResponse(streamAnthropicMessages(upstream, { id, model: requestedModel }));
  }

  const result = finalizeToolResult(await collectUnlimitedText(request, env, route, payload), body);

  if (body.stream) {
    return sseResponse(streamSyntheticAnthropicMessage(result, {
      id,
      model: requestedModel,
      inputText: payload.message || "",
    }));
  }

  return jsonResponse(buildAnthropicMessage(result, {
    id,
    model: requestedModel,
    inputText: payload.message || "",
  }));
}

async function openAIModels(request, env) {
  const catalog = await getModelCatalog(request, env);
  return jsonResponse({
    object: "list",
    data: catalog.map((model) => ({
      id: model.id,
      object: "model",
      created: 0,
      owned_by: model.provider || "unlimited.surf",
      permission: [],
      root: model.id,
      parent: null,
    })),
  });
}

async function anthropicModels(request, env) {
  const catalog = await getModelCatalog(request, env);
  const claudeModels = catalog
    .filter((model) => /claude|anthropic/i.test(`${model.id} ${model.name || ""} ${model.provider || ""}`))
    .map((model) => toAnthropicModel(model));

  return jsonResponse({
    data: claudeModels.length ? claudeModels : [toAnthropicModel({ id: DEFAULT_CLAUDE_MODEL, name: "Claude Opus 4.7" })],
    has_more: false,
    first_id: claudeModels[0] ? claudeModels[0].id : DEFAULT_CLAUDE_MODEL,
    last_id: claudeModels[claudeModels.length - 1] ? claudeModels[claudeModels.length - 1].id : DEFAULT_CLAUDE_MODEL,
  });
}

async function openAIFileUpload(request, env) {
  const contentType = request.headers.get("content-type") || "";
  if (!contentType.includes("multipart/form-data")) {
    return errorResponse(400, "invalid_request_error", "OpenAI file upload expects multipart/form-data with a file field.");
  }

  const form = await request.formData();
  const file = form.get("file");
  if (!file || typeof file === "string") {
    return errorResponse(400, "invalid_request_error", "Missing multipart file field named file.");
  }

  const bytes = new Uint8Array(await file.arrayBuffer());
  const payload = {
    name: file.name || "upload.bin",
    type: file.type || "application/octet-stream",
    data: bytesToBase64(bytes),
  };
  const extracted = await callUnlimitedJson(request, env, "/api/attachments/extract", payload);
  const id = `file_${randomId()}`;
  return jsonResponse({
    id,
    object: "file",
    bytes: bytes.byteLength,
    created_at: nowSeconds(),
    filename: payload.name,
    purpose: form.get("purpose") || "assistants",
    status: extracted && extracted.success === false ? "error" : "processed",
    status_details: null,
    unlimited_extract: extracted,
  });
}

function chooseUnlimitedRoute(body) {
  if (body.models && Array.isArray(body.models) && body.models.length >= 2) return "/api/merge";
  if (body.merge || body.merge_ai) return "/api/merge";
  if (body.query || body.web_search || body.web_search_options || hasWebSearchTool(body.tools)) return "/api/search";
  return "/api/chat";
}

function buildUnlimitedPayload(body, route) {
  if (route === "/api/search") {
    return {
      ...body,
      query: body.query || latestUserText(body.messages) || inputToText(body.input) || body.prompt || "",
      model: mapUpstreamModel(body.model),
      effort: body.effort || reasoningEffort(body),
    };
  }

  const message = applyToolPrompt(
    body.message || messagesToText(body.messages) || inputToText(body.input) || body.prompt || "",
    body
  );
  const payload = {
    ...body,
    message,
    model: mapUpstreamModel(body.model),
    effort: body.effort || reasoningEffort(body),
  };

  if (route === "/api/merge") {
    payload.models = Array.isArray(body.models) && body.models.length ? body.models.map(mapUpstreamModel) : undefined;
  }

  return payload;
}

function buildAnthropicUnlimitedPayload(body, route) {
  if (route === "/api/search") {
    return {
      ...body,
      query: latestUserText(body.messages) || body.query || "",
      model: mapUpstreamModel(body.model),
      effort: body.effort || reasoningEffort(body),
    };
  }

  const prompt = applyToolPrompt(anthropicMessagesToText(body), body);
  const payload = {
    ...body,
    message: prompt,
    model: mapUpstreamModel(body.model),
    effort: body.effort || reasoningEffort(body),
  };

  if (route === "/api/merge") {
    payload.models = Array.isArray(body.models) && body.models.length ? body.models.map(mapUpstreamModel) : undefined;
  }

  return payload;
}

function responsesToChatBody(body, fallbackModel) {
  const messages = [];
  if (body.instructions) messages.push({ role: "system", content: body.instructions });
  messages.push(...responsesInputToMessages(body.input));

  return {
    ...body,
    model: body.model || fallbackModel,
    messages,
    stream: body.stream,
  };
}

function responsesInputToMessages(input) {
  if (!input) return [];
  if (typeof input === "string") return [{ role: "user", content: input }];
  if (!Array.isArray(input)) return [{ role: "user", content: contentToText(input) }];

  const messages = [];
  for (const item of input) {
    if (typeof item === "string") {
      messages.push({ role: "user", content: item });
      continue;
    }

    if (!item || typeof item !== "object") continue;

    if (item.type === "message" || item.role) {
      messages.push({ role: item.role || "user", content: item.content || item.text || "" });
      continue;
    }

    if (item.type === "function_call") {
      messages.push({
        role: "assistant",
        content: "",
        tool_calls: [
          {
            id: item.call_id || item.id || `call_${randomId()}`,
            type: "function",
            function: {
              name: item.name || item.function?.name || "tool",
              arguments: typeof item.arguments === "string" ? item.arguments : JSON.stringify(item.arguments || {}),
            },
          },
        ],
      });
      continue;
    }

    if (item.type === "function_call_output") {
      messages.push({
        role: "tool",
        tool_call_id: item.call_id || item.id || "",
        name: item.name || item.tool_name || "tool",
        content: item.output || item.content || "",
      });
      continue;
    }

    if (item.type === "input_text" || item.type === "output_text") {
      messages.push({ role: "user", content: item.text || "" });
      continue;
    }
  }

  return messages;
}

function finalizeToolResult(result, body) {
  if (result.toolCalls && result.toolCalls.length > 0) return result;
  const parsedToolCalls = parseToolCallsFromText(result.text, body.tools);
  if (!parsedToolCalls.length) return result;

  return {
    ...result,
    text: stripToolCallMarkup(result.text),
    toolCalls: parsedToolCalls,
    finishReason: "tool_calls",
  };
}

function buildOpenAIChatCompletion(result, meta) {
  const messageObj = {
    role: "assistant",
    content: result.text || "",
  };
  if (result.toolCalls && result.toolCalls.length > 0) {
    messageObj.tool_calls = result.toolCalls;
    if (!result.text) messageObj.content = "";
  }

  return {
    id: meta.id,
    object: "chat.completion",
    created: meta.created,
    model: meta.model,
    choices: [
      {
        index: 0,
        message: messageObj,
        logprobs: null,
        finish_reason: result.finishReason || (result.toolCalls ? "tool_calls" : "stop"),
      },
    ],
    usage: usageFromText(meta.inputText || "", result.text || ""),
    system_fingerprint: meta.systemFingerprint,
  };
}

function buildOpenAIResponse(result, meta) {
  const output = [];

  if (result.text) {
    output.push({
      id: `msg_${randomId()}`,
      type: "message",
      status: "completed",
      role: "assistant",
      content: [{ type: "output_text", text: result.text, annotations: [] }],
    });
  }

  if (result.toolCalls && result.toolCalls.length > 0) {
    for (const toolCall of result.toolCalls) {
      output.push({
        id: toolCall.id,
        type: "function_call",
        status: "completed",
        call_id: toolCall.id,
        name: toolCall.function.name,
        arguments: toolCall.function.arguments,
      });
    }
  }

  if (!output.length) {
    output.push({
      id: `msg_${randomId()}`,
      type: "message",
      status: "completed",
      role: "assistant",
      content: [{ type: "output_text", text: "", annotations: [] }],
    });
  }

  return {
    id: meta.id,
    object: "response",
    created_at: meta.created,
    status: "completed",
    error: null,
    incomplete_details: null,
    instructions: meta.body.instructions || null,
    max_output_tokens: meta.body.max_output_tokens || meta.body.max_tokens || null,
    model: meta.model,
    output,
    output_text: result.text || "",
    parallel_tool_calls: true,
    previous_response_id: meta.body.previous_response_id || null,
    reasoning: meta.body.reasoning || null,
    store: meta.body.store || false,
    temperature: meta.body.temperature || null,
    text: meta.body.text || { format: { type: "text" } },
    tool_choice: meta.body.tool_choice || "auto",
    tools: meta.body.tools || [],
    top_p: meta.body.top_p || null,
    truncation: meta.body.truncation || "disabled",
    usage: responseUsageFromText(meta.inputText || "", result.text || ""),
    user: meta.body.user || null,
  };
}

function buildAnthropicMessage(result, meta) {
  const content = [];
  if (result.text) content.push({ type: "text", text: result.text });
  if (result.toolCalls && result.toolCalls.length > 0) {
    for (const toolCall of result.toolCalls) {
      content.push({
        type: "tool_use",
        id: toolCall.id,
        name: toolCall.function.name,
        input: safeJsonParse(toolCall.function.arguments, {}),
      });
    }
  }
  if (!content.length) content.push({ type: "text", text: "" });

  return {
    id: meta.id,
    type: "message",
    role: "assistant",
    model: meta.model,
    content,
    stop_reason: result.toolCalls && result.toolCalls.length > 0 ? "tool_use" : anthropicStopReason(result.finishReason),
    stop_sequence: null,
    usage: anthropicUsageFromText(meta.inputText || "", result.text || ""),
  };
}

function streamSyntheticOpenAIChat(result, meta) {
  return new ReadableStream({
    start(controller) {
      writeSse(controller, {
        id: meta.id,
        object: "chat.completion.chunk",
        created: meta.created,
        model: meta.model,
        choices: [{ index: 0, delta: { role: "assistant", content: "" }, finish_reason: null }],
      });

      if (result.text) {
        writeSse(controller, {
          id: meta.id,
          object: "chat.completion.chunk",
          created: meta.created,
          model: meta.model,
          choices: [{ index: 0, delta: { content: result.text }, finish_reason: null }],
        });
      }

      if (result.toolCalls && result.toolCalls.length > 0) {
        writeSse(controller, {
          id: meta.id,
          object: "chat.completion.chunk",
          created: meta.created,
          model: meta.model,
          choices: [{ index: 0, delta: { tool_calls: result.toolCalls }, finish_reason: null }],
        });
      }

      writeSse(controller, {
        id: meta.id,
        object: "chat.completion.chunk",
        created: meta.created,
        model: meta.model,
        choices: [{ index: 0, delta: {}, finish_reason: result.toolCalls && result.toolCalls.length > 0 ? "tool_calls" : "stop" }],
      });
      writeRawSse(controller, "data: [DONE]\n\n");
      controller.close();
    },
  });
}

function streamSyntheticOpenAIResponse(result, meta) {
  return new ReadableStream({
    start(controller) {
      writeSseEvent(controller, "response.created", {
        type: "response.created",
        response: {
          id: meta.id,
          object: "response",
          created_at: meta.created,
          status: "in_progress",
          model: meta.model,
          output: [],
        },
      });

      if (result.text) {
        const outputId = `msg_${randomId()}`;
        writeSseEvent(controller, "response.output_item.added", {
          type: "response.output_item.added",
          output_index: 0,
          item: { id: outputId, type: "message", status: "in_progress", role: "assistant", content: [] },
        });
        writeSseEvent(controller, "response.content_part.added", {
          type: "response.content_part.added",
          item_id: outputId,
          output_index: 0,
          content_index: 0,
          part: { type: "output_text", text: "", annotations: [] },
        });
        writeSseEvent(controller, "response.output_text.delta", {
          type: "response.output_text.delta",
          item_id: outputId,
          output_index: 0,
          content_index: 0,
          delta: result.text,
        });
        writeSseEvent(controller, "response.output_text.done", {
          type: "response.output_text.done",
          item_id: outputId,
          output_index: 0,
          content_index: 0,
          text: result.text,
        });
        writeSseEvent(controller, "response.content_part.done", {
          type: "response.content_part.done",
          item_id: outputId,
          output_index: 0,
          content_index: 0,
          part: { type: "output_text", text: result.text, annotations: [] },
        });
        writeSseEvent(controller, "response.output_item.done", {
          type: "response.output_item.done",
          output_index: 0,
          item: { id: outputId, type: "message", status: "completed", role: "assistant", content: [] },
        });
      }

      if (result.toolCalls && result.toolCalls.length > 0) {
        result.toolCalls.forEach((toolCall, index) => {
          const outputIndex = result.text ? index + 1 : index;
          const item = {
            id: toolCall.id,
            type: "function_call",
            status: "completed",
            call_id: toolCall.id,
            name: toolCall.function.name,
            arguments: toolCall.function.arguments,
          };
          writeSseEvent(controller, "response.output_item.added", {
            type: "response.output_item.added",
            output_index: outputIndex,
            item,
          });
          writeSseEvent(controller, "response.output_item.done", {
            type: "response.output_item.done",
            output_index: outputIndex,
            item,
          });
        });
      }

      writeSseEvent(controller, "response.completed", {
        type: "response.completed",
        response: {
          id: meta.id,
          object: "response",
          created_at: meta.created,
          status: "completed",
          model: meta.model,
        },
      });
      writeRawSse(controller, "data: [DONE]\n\n");
      controller.close();
    },
  });
}

function streamSyntheticAnthropicMessage(result, meta) {
  return new ReadableStream({
    start(controller) {
      writeSseEvent(controller, "message_start", {
        type: "message_start",
        message: {
          id: meta.id,
          type: "message",
          role: "assistant",
          model: meta.model,
          content: [],
          stop_reason: null,
          stop_sequence: null,
          usage: anthropicUsageFromText(meta.inputText || "", result.text || ""),
        },
      });

      let index = 0;
      if (result.text) {
        writeSseEvent(controller, "content_block_start", {
          type: "content_block_start",
          index,
          content_block: { type: "text", text: "" },
        });
        writeSseEvent(controller, "content_block_delta", {
          type: "content_block_delta",
          index,
          delta: { type: "text_delta", text: result.text },
        });
        writeSseEvent(controller, "content_block_stop", { type: "content_block_stop", index });
        index += 1;
      }

      if (result.toolCalls && result.toolCalls.length > 0) {
        for (const toolCall of result.toolCalls) {
          writeSseEvent(controller, "content_block_start", {
            type: "content_block_start",
            index,
            content_block: {
              type: "tool_use",
              id: toolCall.id,
              name: toolCall.function.name,
              input: safeJsonParse(toolCall.function.arguments, {}),
            },
          });
          writeSseEvent(controller, "content_block_stop", { type: "content_block_stop", index });
          index += 1;
        }
      }

      writeSseEvent(controller, "message_delta", {
        type: "message_delta",
        delta: {
          stop_reason: result.toolCalls && result.toolCalls.length > 0 ? "tool_use" : anthropicStopReason(result.finishReason),
          stop_sequence: null,
        },
        usage: anthropicUsageFromText(meta.inputText || "", result.text || ""),
      });
      writeSseEvent(controller, "message_stop", { type: "message_stop" });
      controller.close();
    },
  });
}

function hasEmulatableTools(tools) {
  return normalizeEmulatableTools(tools).length > 0;
}

function normalizeEmulatableTools(tools) {
  if (!Array.isArray(tools)) return [];
  return tools.map((tool) => {
    if (!tool || typeof tool !== "object") return null;
    if (tool.type === "function" && tool.function && tool.function.name) {
      return {
        name: tool.function.name,
        description: tool.function.description || "",
        parameters: tool.function.parameters || {},
      };
    }
    if (tool.name) {
      return {
        name: tool.name,
        description: tool.description || "",
        parameters: tool.input_schema || tool.parameters || {},
      };
    }
    return null;
  }).filter(Boolean);
}

function applyToolPrompt(prompt, body) {
  const tools = normalizeEmulatableTools(body.tools);
  if (!tools.length) return prompt;

  const toolChoice = stringifyToolChoice(body.tool_choice);
  const toolGuide = [
    "You can call client-side tools exposed by the agent or IDE.",
    "When a tool is needed, respond with only one fenced code block labeled tool_calls and valid JSON.",
    "Required format:",
    "```tool_calls",
    '[{"name":"tool_name","arguments":{"example":"value"}}]',
    "```",
    "Do not add prose before or after the tool_calls block.",
    "If tool results are already present in the conversation, use them and continue normally unless another tool call is still required.",
    `tool_choice: ${toolChoice}`,
    `available_tools: ${JSON.stringify(tools)}`,
  ].join("\n");

  return [toolGuide, prompt].filter(Boolean).join("\n\n");
}

function stringifyToolChoice(toolChoice) {
  if (!toolChoice) return "auto";
  if (typeof toolChoice === "string") return toolChoice;
  if (toolChoice.function && toolChoice.function.name) return `function:${toolChoice.function.name}`;
  if (toolChoice.name) return `tool:${toolChoice.name}`;
  return JSON.stringify(toolChoice);
}

function parseToolCallsFromText(text, tools) {
  if (!text) return [];
  const availableNames = new Set(normalizeEmulatableTools(tools).map((tool) => tool.name));
  if (!availableNames.size) return [];

  const candidates = [];
  const fenceRegex = /```tool_calls\s*([\s\S]*?)```/gi;
  for (const match of text.matchAll(fenceRegex)) {
    candidates.push(match[1].trim());
  }

  const xmlRegex = /<tool_call>([\s\S]*?)<\/tool_call>/gi;
  for (const match of text.matchAll(xmlRegex)) {
    candidates.push(match[1].trim());
  }

  if (!candidates.length && /^[\s\r\n]*[\[{]/.test(text)) {
    candidates.push(text.trim());
  }

  for (const candidate of candidates) {
    const parsed = safeJsonParse(candidate, null);
    const normalized = normalizeParsedToolCalls(parsed, availableNames);
    if (normalized.length) return normalized;
  }

  return [];
}

function normalizeParsedToolCalls(value, availableNames) {
  const rawCalls = Array.isArray(value)
    ? value
    : Array.isArray(value && value.tool_calls)
      ? value.tool_calls
      : value && typeof value === "object"
        ? [value]
        : [];

  return rawCalls.map((call) => normalizeParsedToolCall(call, availableNames)).filter(Boolean);
}

function normalizeParsedToolCall(call, availableNames) {
  if (!call || typeof call !== "object") return null;
  const name = call.name || call.tool_name || (call.function && call.function.name);
  if (!name || (availableNames && !availableNames.has(name))) return null;

  let argumentsValue =
    call.arguments != null ? call.arguments :
      call.input != null ? call.input :
        call.parameters != null ? call.parameters :
          call.args != null ? call.args :
            (call.function && call.function.arguments != null ? call.function.arguments : {});

  if (typeof argumentsValue === "string") {
    const parsed = safeJsonParse(argumentsValue, argumentsValue);
    argumentsValue = parsed;
  }

  const argumentsText = typeof argumentsValue === "string" ? argumentsValue : JSON.stringify(argumentsValue || {});

  return {
    id: call.id || call.call_id || `call_${randomId()}`,
    type: "function",
    function: {
      name,
      arguments: argumentsText,
    },
  };
}

function stripToolCallMarkup(text) {
  return String(text || "")
    .replace(/```tool_calls\s*[\s\S]*?```/gi, "")
    .replace(/<tool_call>[\s\S]*?<\/tool_call>/gi, "")
    .trim();
}

function safeJsonParse(value, fallback) {
  if (typeof value !== "string") return value == null ? fallback : value;
  try {
    return JSON.parse(value);
  } catch (_) {
    return fallback;
  }
}

async function proxyUpstream(request, env, path) {
  const upstreamUrl = new URL(path + new URL(request.url).search, upstreamBase(env));
  const headers = new Headers(request.headers);
  const key = optionalUpstreamApiKey(request, env);
  if (key) headers.set("authorization", `Bearer ${key}`);
  headers.delete("host");

  const init = {
    method: request.method,
    headers,
    body: request.method === "GET" || request.method === "HEAD" ? undefined : request.body,
    redirect: "manual",
  };

  const response = await fetch(upstreamUrl, init);
  return addCors(response);
}

async function callUnlimitedJson(request, env, path, payload) {
  const response = await fetch(new URL(path, upstreamBase(env)), {
    method: "POST",
    headers: upstreamHeaders(request, env, false),
    body: JSON.stringify(payload || {}),
  });

  if (!response.ok) {
    throw new Error(`upstream ${path} failed: ${response.status} ${await response.text()}`);
  }

  return response.json();
}

async function callUnlimitedStream(request, env, path, payload) {
  const response = await fetch(new URL(path, upstreamBase(env)), {
    method: "POST",
    headers: upstreamHeaders(request, env, true),
    body: JSON.stringify(payload || {}),
  });

  if (!response.ok) {
    throw new Error(`upstream ${path} failed: ${response.status} ${await response.text()}`);
  }

  return response;
}

async function collectUnlimitedText(request, env, path, payload) {
  const response = await callUnlimitedStream(request, env, path, payload);
  const events = await readUnlimitedEvents(response);
  let text = "";
  let finishReason = "stop";
  const annotations = [];
  let toolCalls = null;

  for (const event of events) {
    if (typeof event.delta === "string") text += event.delta;
    if (event.results) annotations.push(event.results);
    if (event.finish && event.reason) finishReason = event.reason;
    if (event.choices && event.choices[0]) {
      const choice = event.choices[0];
      if (choice.message && choice.message.content) text += choice.message.content;
      if (choice.delta && choice.delta.content) text += choice.delta.content;
      if (choice.finish_reason) finishReason = choice.finish_reason;
      if (choice.message && choice.message.tool_calls) toolCalls = choice.message.tool_calls;
      if (choice.delta && choice.delta.tool_calls) {
        if (!toolCalls) toolCalls = choice.delta.tool_calls;
      }
    }
  }

  return { text, finishReason, annotations, rawEvents: events, toolCalls };
}

async function getModelCatalog(request, env) {
  try {
    const headers = new Headers();
    const key = optionalUpstreamApiKey(request, env);
    if (key) headers.set("Authorization", `Bearer ${key}`);
    const response = await fetch(new URL("/api/models", upstreamBase(env)), { headers });
    if (!response.ok) throw new Error(`models failed: ${response.status}`);
    const data = await response.json();
    const models = Array.isArray(data) ? data : Array.isArray(data.data) ? data.data : [];
    return models.map((model) => ({
      id: model.id || model.name || String(model),
      name: model.name || model.id || String(model),
      provider: model.provider || providerFromModel(model.id || model.name || ""),
      tier: model.tier || undefined,
    })).filter((model) => model.id);
  } catch (_) {
    return fallbackModels();
  }
}

function streamOpenAIChat(upstream, meta) {
  return streamUnlimitedEvents(upstream, {
    start(controller) {
      writeSse(controller, {
        id: meta.id,
        object: "chat.completion.chunk",
        created: meta.created,
        model: meta.model,
        choices: [{ index: 0, delta: { role: "assistant", content: "" }, finish_reason: null }],
      });
    },
    delta(controller, text, parsed) {
      if (parsed && parsed.choices) {
        writeSse(controller, {
          id: meta.id,
          object: "chat.completion.chunk",
          created: meta.created,
          model: meta.model,
          choices: parsed.choices
        });
        return;
      }

      const deltaObj = {};
      if (text) deltaObj.content = text;
      if (parsed && parsed.tool_calls) deltaObj.tool_calls = parsed.tool_calls;
      if (parsed && parsed.delta && typeof parsed.delta === "object" && parsed.delta.tool_calls) {
        deltaObj.tool_calls = parsed.delta.tool_calls;
      }

      if (Object.keys(deltaObj).length > 0) {
        writeSse(controller, {
          id: meta.id,
          object: "chat.completion.chunk",
          created: meta.created,
          model: meta.model,
          choices: [{ index: 0, delta: deltaObj, finish_reason: null }],
        });
      }
    },
    finish(controller, reason, parsed) {
      if (parsed && parsed.choices && parsed.choices[0] && parsed.choices[0].finish_reason) {
        writeSse(controller, {
          id: meta.id,
          object: "chat.completion.chunk",
          created: meta.created,
          model: meta.model,
          choices: [{ index: 0, delta: {}, finish_reason: parsed.choices[0].finish_reason }],
        });
        writeRawSse(controller, "data: [DONE]\n\n");
        return;
      }

      writeSse(controller, {
        id: meta.id,
        object: "chat.completion.chunk",
        created: meta.created,
        model: meta.model,
        choices: [{ index: 0, delta: {}, finish_reason: openAIStopReason(reason) }],
      });
      writeRawSse(controller, "data: [DONE]\n\n");
    },
  });
}

function streamOpenAIResponses(upstream, meta) {
  const outputId = `msg_${randomId()}`;
  return streamUnlimitedEvents(upstream, {
    start(controller) {
      writeSseEvent(controller, "response.created", {
        type: "response.created",
        response: {
          id: meta.id,
          object: "response",
          created_at: meta.created,
          status: "in_progress",
          model: meta.model,
          output: [],
        },
      });
      writeSseEvent(controller, "response.output_item.added", {
        type: "response.output_item.added",
        output_index: 0,
        item: { id: outputId, type: "message", status: "in_progress", role: "assistant", content: [] },
      });
      writeSseEvent(controller, "response.content_part.added", {
        type: "response.content_part.added",
        item_id: outputId,
        output_index: 0,
        content_index: 0,
        part: { type: "output_text", text: "", annotations: [] },
      });
    },
    delta(controller, text) {
      writeSseEvent(controller, "response.output_text.delta", {
        type: "response.output_text.delta",
        item_id: outputId,
        output_index: 0,
        content_index: 0,
        delta: text,
      });
    },
    finish(controller) {
      writeSseEvent(controller, "response.output_text.done", {
        type: "response.output_text.done",
        item_id: outputId,
        output_index: 0,
        content_index: 0,
        text: "",
      });
      writeSseEvent(controller, "response.content_part.done", {
        type: "response.content_part.done",
        item_id: outputId,
        output_index: 0,
        content_index: 0,
        part: { type: "output_text", text: "", annotations: [] },
      });
      writeSseEvent(controller, "response.output_item.done", {
        type: "response.output_item.done",
        output_index: 0,
        item: { id: outputId, type: "message", status: "completed", role: "assistant", content: [] },
      });
      writeSseEvent(controller, "response.completed", {
        type: "response.completed",
        response: { id: meta.id, object: "response", created_at: meta.created, status: "completed", model: meta.model },
      });
      writeRawSse(controller, "data: [DONE]\n\n");
    },
  });
}

function streamAnthropicMessages(upstream, meta) {
  return streamUnlimitedEvents(upstream, {
    start(controller) {
      writeSseEvent(controller, "message_start", {
        type: "message_start",
        message: {
          id: meta.id,
          type: "message",
          role: "assistant",
          model: meta.model,
          content: [],
          stop_reason: null,
          stop_sequence: null,
          usage: { input_tokens: 0, output_tokens: 0 },
        },
      });
      writeSseEvent(controller, "content_block_start", {
        type: "content_block_start",
        index: 0,
        content_block: { type: "text", text: "" },
      });
    },
    delta(controller, text, parsed) {
      if (parsed && (parsed.type === "content_block_delta" || parsed.type === "message_delta" || parsed.type === "content_block_start" || parsed.type === "content_block_stop")) {
        writeSseEvent(controller, parsed.type, parsed);
        return;
      }

      if (text) {
        writeSseEvent(controller, "content_block_delta", {
          type: "content_block_delta",
          index: 0,
          delta: { type: "text_delta", text },
        });
      }
    },
    finish(controller, reason, parsed) {
      if (parsed && parsed.type === "message_stop") {
        writeSseEvent(controller, "message_stop", parsed);
        return;
      }

      writeSseEvent(controller, "content_block_stop", { type: "content_block_stop", index: 0 });
      writeSseEvent(controller, "message_delta", {
        type: "message_delta",
        delta: { stop_reason: anthropicStopReason(reason), stop_sequence: null },
        usage: { output_tokens: 0 },
      });
      writeSseEvent(controller, "message_stop", { type: "message_stop" });
    },
  });
}

function streamUnlimitedEvents(upstream, handlers) {
  const encoder = new TextEncoder();
  const decoder = new TextDecoder();

  return new ReadableStream({
    async start(controller) {
      let finished = false;
      handlers.start && handlers.start(controller);

      try {
        const reader = upstream.body.getReader();
        let buffer = "";

        while (true) {
          const { done, value } = await reader.read();
          if (done) break;
          buffer += decoder.decode(value, { stream: true });
          const lines = buffer.split(/\r?\n/);
          buffer = lines.pop() || "";

          for (const line of lines) {
            if (!line.startsWith("data:")) continue;
            const parsed = parseSseJson(line.slice(5).trim());
            if (!parsed) continue;

            if (parsed.error) {
              controller.enqueue(encoder.encode(`event: error\ndata: ${JSON.stringify({ error: parsed.error.message || parsed.error })}\n\n`));
              finished = true;
              break;
            }

            let hasDelta = false;
            let deltaText = "";

            if (typeof parsed.delta === "string" && parsed.delta.length) {
              deltaText = parsed.delta;
              hasDelta = true;
            } else if (parsed.delta != null || parsed.choices != null || parsed.tool_calls != null || parsed.type != null) {
              hasDelta = true;
            }

            if (hasDelta) {
              handlers.delta && handlers.delta(controller, deltaText, parsed);
            }

            if (parsed.finish || parsed.done || (parsed.choices && parsed.choices[0] && parsed.choices[0].finish_reason)) {
              finished = true;
              let reason = parsed.reason || "stop";
              if (parsed.choices && parsed.choices[0] && parsed.choices[0].finish_reason) reason = parsed.choices[0].finish_reason;
              handlers.finish && handlers.finish(controller, reason, parsed);
            }
          }
        }

        if (!finished) handlers.finish && handlers.finish(controller, "stop", {});
      } catch (error) {
        controller.enqueue(encoder.encode(`event: error\ndata: ${JSON.stringify({ error: error.message || String(error) })}\n\n`));
      } finally {
        controller.close();
      }
    },
  });
}

async function readUnlimitedEvents(response) {
  const decoder = new TextDecoder();
  const reader = response.body.getReader();
  const events = [];
  let buffer = "";

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });
    const lines = buffer.split(/\r?\n/);
    buffer = lines.pop() || "";
    for (const line of lines) {
      if (!line.startsWith("data:")) continue;
      const parsed = parseSseJson(line.slice(5).trim());
      if (parsed) events.push(parsed);
    }
  }

  if (buffer.startsWith("data:")) {
    const parsed = parseSseJson(buffer.slice(5).trim());
    if (parsed) events.push(parsed);
  }

  return events;
}

function writeSse(controller, data) {
  writeRawSse(controller, `data: ${JSON.stringify(data)}\n\n`);
}

function writeSseEvent(controller, event, data) {
  writeRawSse(controller, `event: ${event}\ndata: ${JSON.stringify(data)}\n\n`);
}

function writeRawSse(controller, chunk) {
  controller.enqueue(new TextEncoder().encode(chunk));
}

function sseResponse(body) {
  return new Response(body, {
    headers: {
      ...CORS_HEADERS,
      "Content-Type": "text/event-stream; charset=utf-8",
      "Cache-Control": "no-cache, no-transform",
      "Connection": "keep-alive",
      "X-Accel-Buffering": "no",
    },
  });
}

function jsonResponse(data, init = {}) {
  return new Response(JSON.stringify(data, null, 2), {
    ...init,
    headers: {
      ...CORS_HEADERS,
      "Content-Type": "application/json; charset=utf-8",
      "Cache-Control": "no-store",
      ...(init.headers || {}),
    },
  });
}

function textResponse(text, contentType, init = {}) {
  return new Response(text, {
    ...init,
    headers: {
      ...CORS_HEADERS,
      "Content-Type": contentType,
      "Cache-Control": "no-store",
      ...(init.headers || {}),
    },
  });
}

function errorResponse(status, code, message) {
  return jsonResponse({
    error: {
      message,
      type: code,
      code,
    },
  }, { status });
}

function addCors(response) {
  const headers = new Headers(response.headers);
  for (const [key, value] of Object.entries(CORS_HEADERS)) headers.set(key, value);
  return new Response(response.body, { status: response.status, statusText: response.statusText, headers });
}

async function readJson(request) {
  if (!request.body) return {};
  const text = await request.text();
  if (!text.trim()) return {};
  try {
    return JSON.parse(text);
  } catch (_) {
    throw new Error("Request body must be valid JSON.");
  }
}

function upstreamHeaders(request, env, wantsStream) {
  const headers = new Headers();
  headers.set("Authorization", `Bearer ${upstreamApiKey(request, env)}`);
  headers.set("Content-Type", "application/json");
  if (wantsStream) headers.set("Accept", "text/event-stream");
  return headers;
}

function upstreamApiKey(request, env) {
  const key = optionalUpstreamApiKey(request, env);
  if (key) return key;

  if (env.WORKER_API_KEY) {
    throw new Error("Missing upstream API key. Set UNLIMITED_SURF_API_KEY when WORKER_API_KEY is enabled.");
  }

  throw new Error("Missing upstream API key. Set UNLIMITED_SURF_API_KEY or pass Authorization: Bearer <key> / x-api-key: <key>.");
}

function optionalUpstreamApiKey(request, env) {
  const configured = env.UNLIMITED_SURF_API_KEY || env.API_KEY || env.AUTH_KEY;
  if (configured) return configured;

  if (env.WORKER_API_KEY) return "";

  return clientApiKey(request);
}

function validateWorkerApiKey(request, env) {
  const expected = env.WORKER_API_KEY;
  if (!expected) return null;

  const actual = clientApiKey(request);
  if (actual && constantTimeEqual(actual, expected)) return null;

  return jsonResponse({
    error: {
      message: "Invalid or missing Worker API key.",
      type: "authentication_error",
      code: "invalid_api_key",
    },
  }, { status: 401, headers: { "WWW-Authenticate": "Bearer" } });
}

function clientApiKey(request) {
  const auth = request.headers.get("authorization") || "";
  if (/^bearer\s+/i.test(auth)) return auth.replace(/^bearer\s+/i, "").trim();

  const xKey = request.headers.get("x-api-key") || request.headers.get("anthropic-api-key");
  return xKey ? xKey.trim() : "";
}

function constantTimeEqual(actual, expected) {
  const actualText = String(actual || "");
  const expectedText = String(expected || "");
  if (actualText.length !== expectedText.length) return false;

  let diff = 0;
  for (let i = 0; i < actualText.length; i += 1) {
    diff |= actualText.charCodeAt(i) ^ expectedText.charCodeAt(i);
  }
  return diff === 0;
}

function upstreamBase(env) {
  return stripTrailingSlash(env.UPSTREAM_BASE_URL || DEFAULT_UPSTREAM_BASE_URL) + "/";
}

function normalizePath(path) {
  if (!path || path === "") return "/";
  const normalized = path.replace(/\/+/g, "/");
  return normalized.length > 1 ? normalized.replace(/\/+$/, "") : normalized;
}

function messagesToText(messages) {
  if (!Array.isArray(messages)) return "";
  return messages.map(messageToTranscript).filter(Boolean).join("\n\n");
}

function anthropicMessagesToText(body) {
  const parts = [];
  if (body.system) parts.push(`system: ${contentToText(body.system)}`);
  if (Array.isArray(body.messages)) parts.push(messagesToText(body.messages));
  return parts.filter(Boolean).join("\n\n");
}

function inputToText(input) {
  if (!input) return "";
  if (typeof input === "string") return input;
  if (!Array.isArray(input)) return contentToText(input);

  return input.map((item) => {
    if (typeof item === "string") return item;
    if (item.type === "message") return messageToTranscript(item);
    if (item.role) return messageToTranscript(item);
    if (item.type === "function_call") {
      return `assistant tool_call: ${JSON.stringify({
        id: item.call_id || item.id || "",
        name: item.name || item.function?.name || "tool",
        arguments: item.arguments || item.function?.arguments || {},
      })}`;
    }
    if (item.type === "function_call_output") {
      return `tool (${item.name || item.tool_name || "tool"}) result for ${item.call_id || item.id || ""}: ${contentToText(item.output || item.content)}`;
    }
    if (item.type === "input_file") return formatFileReference(item);
    if (item.type === "input_text" || item.type === "output_text") return item.text || "";
    return contentToText(item);
  }).filter(Boolean).join("\n\n");
}

function contentToText(content) {
  if (content == null) return "";
  if (typeof content === "string") return content;
  if (Array.isArray(content)) {
    return content.map((part) => contentToText(part)).filter(Boolean).join("\n");
  }
  if (typeof content === "object") {
    if (typeof content.text === "string") return content.text;
    if (typeof content.input_text === "string") return content.input_text;
    if (content.type === "text" && typeof content.text === "string") return content.text;
    if (content.type === "input_text" && typeof content.text === "string") return content.text;
    if (content.type === "input_file" || content.type === "file" || content.file_id || content.filename || content.file_name) {
      return formatFileReference(content);
    }
    if (content.type === "image_url") {
      let url = content.image_url && content.image_url.url ? content.image_url.url : "attached";
      if (url.startsWith("data:") && url.length > 200) {
        url = url.substring(0, 100) + "...[truncated to prevent payload overflow]";
      }
      return `[image: ${url}]`;
    }
    if (content.type === "image") return "[image attached]";
    if (content.type === "tool_result") return `[tool_result ${content.tool_use_id || ""}] ${contentToText(content.content)}`;
    if (content.type === "tool_use") return `[tool_use ${content.name || "tool"}] ${JSON.stringify(content.input || {})}`;
    if (content.type === "function_call_output") return `[tool_result ${content.call_id || content.id || ""}] ${contentToText(content.output || content.content)}`;
    if (content.type === "function_call") {
      return `[tool_use ${content.name || content.function?.name || "tool"}] ${JSON.stringify(content.arguments || content.function?.arguments || {})}`;
    }
    if (content.type) return `[${content.type}] ${JSON.stringify(content)}`;
  }
  return String(content);
}

function messageToTranscript(message) {
  if (!message || typeof message !== "object") return "";
  const role = message.role || "user";
  const parts = [];
  const contentText = contentToText(message.content);

  if (role === "tool") {
    return `tool (${message.name || message.tool_name || "tool"}) result for ${message.tool_call_id || ""}: ${contentText}`;
  }

  if (contentText) parts.push(`${role}: ${contentText}`);

  if (role === "assistant") {
    if (message.function_call) {
      parts.push(`assistant tool_call: ${JSON.stringify({
        name: message.function_call.name || "tool",
        arguments: message.function_call.arguments || {},
      })}`);
    }

    if (Array.isArray(message.tool_calls)) {
      for (const toolCall of message.tool_calls) {
        parts.push(`assistant tool_call: ${JSON.stringify({
          id: toolCall.id || "",
          name: toolCall.function?.name || toolCall.name || "tool",
          arguments: toolCall.function?.arguments || toolCall.arguments || {},
        })}`);
      }
    }
  }

  return parts.join("\n");
}

function formatFileReference(fileLike) {
  const fileId = fileLike.file_id || fileLike.id || "";
  const fileName = fileLike.filename || fileLike.file_name || fileLike.name || "";
  const mediaType = fileLike.mime_type || fileLike.type || fileLike.media_type || "";
  const source = fileLike.url || fileLike.file_url || "";
  return `[file id=${fileId || "n/a"} name=${fileName || "unknown"} type=${mediaType || "unknown"} source=${source || "local"}]`;
}

function latestUserText(messages) {
  if (!Array.isArray(messages)) return "";
  for (let i = messages.length - 1; i >= 0; i -= 1) {
    if ((messages[i].role || "user") === "user") return contentToText(messages[i].content);
  }
  return "";
}

function hasWebSearchTool(tools) {
  if (!Array.isArray(tools)) return false;
  return tools.some((tool) => {
    const type = tool && (tool.type || tool.name || (tool.function && tool.function.name));
    return /web.?search|browser|search/i.test(String(type || ""));
  });
}

function reasoningEffort(body) {
  if (body.effort) return body.effort;
  if (typeof body.reasoning_effort === "string") return body.reasoning_effort;
  if (body.reasoning && typeof body.reasoning.effort === "string") return body.reasoning.effort;
  return "medium";
}

function mapUpstreamModel(model) {
  if (!model) return DEFAULT_OPENAI_MODEL;
  if (model.startsWith("gateway-")) return model;
  if (/^claude-/i.test(model)) return `gateway-${model.replace(/-\d{8}$/, "")}`;
  if (/^gpt-/i.test(model)) return `gateway-${model}`;
  if (/^gemini-/i.test(model)) return `gateway-google-${model.replace(/^gemini-/i, "")}`;
  return model;
}

function toAnthropicModel(model) {
  const id = model.id.startsWith("gateway-") ? model.id.replace(/^gateway-/, "") : model.id;
  const versioned = /^claude-.*-\d{8}$/.test(id) ? id : anthropicVersionedId(id);
  return {
    id: versioned,
    type: "model",
    display_name: model.name || versioned,
    created_at: "2026-01-01T00:00:00Z",
  };
}

function anthropicVersionedId(id) {
  if (/^claude-/i.test(id)) return `${id}-20260101`;
  return id;
}

function providerFromModel(model) {
  if (/claude|anthropic/i.test(model)) return "anthropic";
  if (/gemini|google/i.test(model)) return "google";
  if (/gpt|openai/i.test(model)) return "openai";
  return "unlimited.surf";
}

function fallbackModels() {
  return [
    { id: "gateway-gpt-5", name: "GPT-5", provider: "openai", tier: "flagship" },
    { id: "gateway-gpt-5-1", name: "GPT-5.1", provider: "openai", tier: "flagship" },
    { id: "gateway-claude-opus-4-7", name: "Claude Opus 4.7", provider: "anthropic", tier: "flagship" },
    { id: "gateway-google-2.5-pro", name: "Gemini 2.5 Pro", provider: "google", tier: "flagship" },
    { id: "gateway-gemini-3-flash", name: "Gemini 3 Flash", provider: "google", tier: "fast" },
  ];
}

function parseSseJson(data) {
  if (!data || data === "[DONE]") return null;
  try {
    return JSON.parse(data);
  } catch (_) {
    return null;
  }
}

function openAIStopReason(reason) {
  if (!reason) return "stop";
  if (reason === "max_tokens") return "length";
  if (reason === "tool_use") return "tool_calls";
  return reason === "end_turn" ? "stop" : reason;
}

function anthropicStopReason(reason) {
  if (!reason || reason === "stop") return "end_turn";
  if (reason === "length") return "max_tokens";
  if (reason === "tool_calls") return "tool_use";
  return reason;
}

function usageFromText(input, output) {
  const promptTokens = estimateTokens(input);
  const completionTokens = estimateTokens(output);
  return { prompt_tokens: promptTokens, completion_tokens: completionTokens, total_tokens: promptTokens + completionTokens };
}

function responseUsageFromText(input, output) {
  const inputTokens = estimateTokens(input);
  const outputTokens = estimateTokens(output);
  return { input_tokens: inputTokens, output_tokens: outputTokens, total_tokens: inputTokens + outputTokens };
}

function anthropicUsageFromText(input, output) {
  return { input_tokens: estimateTokens(input), output_tokens: estimateTokens(output) };
}

function estimateTokens(text) {
  if (!text) return 0;
  return Math.max(1, Math.ceil(String(text).length / 4));
}

function nowSeconds() {
  return Math.floor(Date.now() / 1000);
}

function randomId() {
  const bytes = new Uint8Array(12);
  crypto.getRandomValues(bytes);
  return Array.from(bytes, (byte) => byte.toString(16).padStart(2, "0")).join("");
}

function stripTrailingSlash(value) {
  return String(value || "").replace(/\/+$/, "");
}

function bytesToBase64(bytes) {
  let binary = "";
  const chunkSize = 0x8000;
  for (let i = 0; i < bytes.length; i += chunkSize) {
    binary += String.fromCharCode(...bytes.subarray(i, i + chunkSize));
  }
  return btoa(binary);
}

function looksLikeAnthropicRequest(request) {
  return request.headers.has("anthropic-version") || request.headers.has("anthropic-beta") || request.headers.has("x-api-key");
}

function serviceInfo(request, env) {
  const origin = new URL(request.url).origin;
  return {
    ok: true,
    service: "unlimited.surf OpenAI/Anthropic compatibility Worker",
    upstream: stripTrailingSlash(env.UPSTREAM_BASE_URL || DEFAULT_UPSTREAM_BASE_URL),
    routes: {
      raw: `${origin}/api/chat, /api/search, /api/merge, /api/models, /api/key, /api/attachments/extract`,
      openai: `${origin}/v1/chat/completions, /v1/responses, /v1/models, /v1/files`,
      anthropic: `${origin}/v1/messages or ${origin}/anthropic/v1/messages`,
      setup: `${origin}/v1/setup, /v1/codex, /v1/mcp`,
    },
  };
}

function agentSetup(request) {
  const origin = new URL(request.url).origin;
  return `Claude Code / Anthropic-compatible setup

PowerShell:
$env:ANTHROPIC_BASE_URL = "${origin}"
$env:ANTHROPIC_AUTH_TOKEN = "<your unlimited.surf key>"
$env:ANTHROPIC_API_KEY = "<your unlimited.surf key>"
$env:ANTHROPIC_MODEL = "${DEFAULT_CLAUDE_MODEL}"
claude

Bash:
export ANTHROPIC_BASE_URL="${origin}"
export ANTHROPIC_AUTH_TOKEN="<your unlimited.surf key>"
export ANTHROPIC_API_KEY="<your unlimited.surf key>"
export ANTHROPIC_MODEL="${DEFAULT_CLAUDE_MODEL}"
claude

Goose / Hermes / other agents:
Provider: Anthropic-compatible
Base URL: ${origin}
API key: <your unlimited.surf key>
Model: ${DEFAULT_CLAUDE_MODEL}

Messages endpoint: POST ${origin}/v1/messages
Models endpoint: GET ${origin}/v1/models

MCP tools run in the client/agent environment. Use this Worker as the model endpoint, then configure MCP servers in your IDE or agent.
`;
}

function codexSetup(request) {
  const origin = new URL(request.url).origin;
  return `Codex custom provider notes

OpenAI-compatible Chat Completions:
base_url = "${origin}/v1"
api_key = "<your unlimited.surf key>"
model = "${DEFAULT_OPENAI_MODEL}"

OpenAI Responses-compatible route for newer agents:
POST ${origin}/v1/responses

Direct smoke test:
curl ${origin}/v1/chat/completions \\
  -H "Authorization: Bearer <your unlimited.surf key>" \\
  -H "Content-Type: application/json" \\
  -d '{"model":"${DEFAULT_OPENAI_MODEL}","messages":[{"role":"user","content":"Write a small test function."}],"stream":true}'

Anthropic-compatible agent route:
POST ${origin}/v1/messages

MCP execution remains client-side; configure MCP servers in Codex or your IDE, and point the model provider at this Worker.
`;
}

function mcpInfo(request) {
  const origin = new URL(request.url).origin;
  return {
    supported: true,
    model_endpoint: origin,
    note: "MCP servers execute inside the client or agent. This Worker supplies OpenAI/Anthropic-compatible model endpoints and does not run local MCP tools in the browser or edge runtime.",
    endpoints: {
      openai_responses: `${origin}/v1/responses`,
      openai_chat_completions: `${origin}/v1/chat/completions`,
      anthropic_messages: `${origin}/v1/messages`,
      setup: `${origin}/v1/setup`,
    },
  };
}
