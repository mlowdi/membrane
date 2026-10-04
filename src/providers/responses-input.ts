import type { ImageContent, ProviderRequest } from '../types/index.js';
import type { OpenAIResponsesInputItem } from './openai-responses-api.js';
import { normalizeImageContent, IMAGE_UNAVAILABLE_TEXT, isVisualImageContent, asImageContent, ASSISTANT_GENERATED_IMAGE_ORIGIN_TEXT,
  projectResponsesGeneratedImage, sameResponsesItem } from '../utils/image-policy.js';

type JsonObject = Record<string, unknown>;

/**
 * Most agent turns arrive already formatted as provider-native Responses
 * items. Internal maintenance calls, however, can bypass that formatter and
 * carry Membrane's normalized `text`/`image`/tool blocks. Normalize at the
 * final transport boundary so every call shape accepted by ProviderAdapter is
 * valid on the Codex Responses endpoint.
 */
export function normalizeResponsesInput(messages: ProviderRequest['messages']): OpenAIResponsesInputItem[] {
  const output: unknown[] = [];

  for (const rawMessage of messages as unknown[]) {
    if (!isObject(rawMessage)) {
      output.push(rawMessage);
      continue;
    }
    if (rawMessage.type !== 'message' && rawMessage.role === undefined) {
      output.push(normalizeStandaloneItem(rawMessage));
      continue;
    }

    // Native messages (including phase, status and developer/system roles)
    // must survive replay verbatim. Only translate normalized content blocks.
    if (Array.isArray(rawMessage.content) && !rawMessage.content.some((block) =>
      isObject(block) && ['text', 'image', 'generated_image', 'tool_use', 'tool_result', 'redacted_thinking'].includes(asString(block.type))
    )) {
      output.push(rawMessage);
      continue;
    }
    const role = typeof rawMessage.role === 'string' ? rawMessage.role : 'user';
    const blocks = Array.isArray(rawMessage.content)
      ? rawMessage.content
      : typeof rawMessage.content === 'string'
        ? [{ type: 'text', text: rawMessage.content }]
        : [];
    let parts: unknown[] = [];
    let seenRawItems: JsonObject[] | undefined;
    const flush = () => {
      if (parts.length === 0) return;
      output.push({
        type: 'message',
        ...rawMessage,
        role,
        content: parts,
      });
      parts = [];
    };

    for (const rawBlock of blocks) {
      if (!isObject(rawBlock)) continue;
      if (rawBlock.type === 'text') {
        parts.push({ type: role === 'assistant' ? 'output_text' : 'input_text', text: asString(rawBlock.text) });
      } else if (isVisualImageContent(rawBlock)) {
        const visual = normalizeImageContent(rawBlock);
        if (visual.type === 'generated_image' && isObject(visual.rawItem) && projectResponsesGeneratedImage(visual.rawItem)) {
          flush();
          const rawItem = visual.rawItem;
          const seen = seenRawItems?.some(item => typeof rawItem.id === 'string'
            ? item.type === rawItem.type && item.id === rawItem.id : sameResponsesItem(item, rawItem));
          if (!seen) { output.push(rawItem); (seenRawItems ??= []).push(rawItem); }
          continue;
        }
        const imageUrl = isVisualImageContent(visual) ? admittedImageUrl(asImageContent(visual)) : undefined;
        const part = imageUrl ? { type: 'input_image', image_url: imageUrl }
          : { type: 'input_text', text: IMAGE_UNAVAILABLE_TEXT };
        if (role !== 'assistant') parts.push(part);
        else if (rawBlock.type === 'generated_image') {
          flush();
          output.push({ type: 'message', role: 'user', content: [
            { type: 'input_text', text: ASSISTANT_GENERATED_IMAGE_ORIGIN_TEXT }, part,
          ] });
        }
      } else if (rawBlock.type === 'tool_use') {
        flush();
        output.push(normalizeStandaloneItem(rawBlock));
      } else if (rawBlock.type === 'tool_result') {
        flush();
        output.push(normalizeStandaloneItem(rawBlock));
      } else if (rawBlock.type === 'redacted_thinking') {
        flush();
        output.push(reasoningInputItem(rawBlock));
      } else {
        // Already-native input_text/output_text/input_image/refusal parts.
        parts.push(rawBlock);
      }
    }
    flush();
  }

  return output as OpenAIResponsesInputItem[];
}

/** Responses tool outputs accept typed image parts. Serializing those parts
 * into a string makes their base64 data ordinary prompt text instead. */
export function responsesToolResultOutput(content: unknown): string | JsonObject[] {
  if (!Array.isArray(content)) {
    return typeof content === 'string' ? content : JSON.stringify(content ?? null);
  }
  return content.flatMap((block): JsonObject[] => {
    if (isObject(block)) {
      if (block.type === 'text') return [{ type: 'input_text', text: asString(block.text) }];
      if (block.type === 'tool_result') {
        const nested = responsesToolResultOutput(block.content);
        return typeof nested === 'string' ? [{ type: 'input_text', text: nested }] : nested;
      }
      if (isVisualImageContent(block) || block.type === 'input_image') {
        const imageUrl = responsesImageUrl(block);
        return [imageUrl ? { type: 'input_image', image_url: imageUrl }
          : { type: 'input_text', text: IMAGE_UNAVAILABLE_TEXT }];
      }
      if (block.type === 'input_text' || block.type === 'input_file') return [block];
    }
    return [{ type: 'input_text', text: JSON.stringify(block) }];
  });
}

function normalizeStandaloneItem(item: JsonObject): unknown {
  if (item.type === 'tool_use') {
    return {
      type: 'function_call',
      call_id: asString(item.id),
      name: asString(item.name),
      arguments: JSON.stringify(isObject(item.input) ? item.input : {}),
    };
  }
  if (item.type === 'tool_result') {
    const content = item.content;
    return {
      type: 'function_call_output',
      call_id: asString(item.toolUseId) || asString(item.tool_use_id),
      output: responsesToolResultOutput(content),
    };
  }
  if (item.type === 'redacted_thinking') {
    return reasoningInputItem(item);
  }
  return item;
}

/** Replay a captured reasoning carrier as a Responses input item.
 *
 * Prefer the provider-native item verbatim when the block still carries it
 * (`rawItem` from response parsing). Otherwise reconstruct the minimum the
 * Responses API accepts: `summary` is a REQUIRED field on reasoning input
 * items (empty array = "no summaries") — omitting it 400s with
 * "Missing required parameter: 'input[N].summary'". */
function reasoningInputItem(block: JsonObject): unknown {
  const raw = block.rawItem;
  if (isObject(raw) && raw.type === 'reasoning') return raw;
  return { type: 'reasoning', summary: [], encrypted_content: asString(block.data) };
}

function responsesImageUrl(block: JsonObject): string | undefined {
  const visual = normalizeImageContent(block);
  if (!isVisualImageContent(visual)) return undefined;
  return admittedImageUrl(asImageContent(visual));
}

function admittedImageUrl(image: ImageContent): string {
  return image.source.type === 'url' ? image.source.url
    : `data:${image.source.mediaType};base64,${image.source.data}`;
}

function asString(value: unknown): string {
  return typeof value === 'string' ? value : '';
}

function isObject(value: unknown): value is JsonObject {
  return Boolean(value) && typeof value === 'object' && !Array.isArray(value);
}
