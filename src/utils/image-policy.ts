import type { ContentBlock, ImageContent, GeneratedImageContent } from '../types/content.js';
import { isAcceptedImageMediaType, resolveImageMediaType } from './image-media.js';

export const IMAGE_TOKEN_ESTIMATE = 1600;
export const DEFAULT_MAX_LIVE_IMAGE_BYTES = 20 * 1024 * 1024;
export const IMAGE_DROPPED_TEXT = '[image dropped from live context]';
export const IMAGE_UNAVAILABLE_TEXT = '[image unavailable: invalid or unsupported image source/MIME]';
export const RESPONSES_ITEMS_KEY = 'openaiResponsesItems';
export const ASSISTANT_GENERATED_IMAGE_ORIGIN_TEXT = 'prior assistant-generated visual context; not user authorship or a new user request';

/** Zero disables a dimension. Omitted bytes use the shared 20 MiB wall;
 * omitted count/depth are unlimited. The estimator is local, never a wire field. */
export interface LiveImagePolicy {
  maxLiveImages?: number;
  maxLiveImageBytes?: number;
  imageStripDepthTokens?: number;
  estimateTokens?: (content: ContentBlock[]) => number;
}

type Item = Record<string, unknown>;
type ImageMessage = { content: ContentBlock[]; metadata?: Record<string, unknown> };
type ImageReference = { type: 'blob_ref'; ref: { originalType: 'image' | 'generated_image'; hash: string }; encodedBytes?: number; tokenEstimate?: number };
/** Legacy inline diagnostics only: not a producer block or inference input. */
export type GeneratedImageMetadata = Omit<GeneratedImageContent, 'data'> & {
  metadataOnly: true;
  encodedBytes: number;
};
type PolicyImage = ImageContent | GeneratedImageContent | GeneratedImageMetadata | ImageReference;
const object = (value: unknown): value is Item => !!value && typeof value === 'object' && !Array.isArray(value);
export const isVisualImageContent = (value: unknown): value is ImageContent | GeneratedImageContent =>
  object(value) && (value.type === 'image' || value.type === 'generated_image');
export const isGeneratedImageMetadata = (value: unknown): value is GeneratedImageMetadata => object(value) &&
  value.type === 'generated_image' && value.metadataOnly === true && typeof value.encodedBytes === 'number' &&
  !Object.hasOwn(value, 'data');
export const isImageReference = (value: unknown): value is ImageReference => object(value) && value.type === 'blob_ref' &&
  object(value.ref) && (value.ref.originalType === 'image' || value.ref.originalType === 'generated_image');

/** Recognized recursive tool media only; opaque carriers and arbitrary fields stay opaque. */
export function hasVisualImageContent(content: readonly unknown[]): boolean {
  for (const block of content) {
    if (isVisualImageContent(block) || isImageReference(block)) return true;
    if (object(block) && block.type === 'tool_result' && Array.isArray(block.content) &&
        hasVisualImageContent(block.content)) return true;
  }
  return false;
}

/** Wire-only view of an admitted image. The encoded string and raw carrier are
 * shared, never decoded/copied or written back over the public generated variant. */
export function asImageContent(image: ImageContent | GeneratedImageContent): ImageContent {
  if (image.type === 'image') return image;
  return { type: 'image', source: { type: 'base64', data: image.data, mediaType: image.mimeType },
    ...(image.tokenEstimate !== undefined ? { tokenEstimate: image.tokenEstimate } : {}),
    ...(image.rawItem !== undefined ? { rawItem: image.rawItem } : {}) };
}
/** Reject nonalphabet characters and illegal padding/length before a
 * permissive decoder can replace the source. RFC 4648 §3.5 does not require
 * rejecting unused pad bits; previously decodable image bytes remain valid. */
export function isValidImageBase64(data: string): boolean {
  return data.length > 0 && data.length % 4 === 0 && /^[A-Za-z0-9+/]+={0,2}$/.test(data);
}

/** MCP inline images require both data and MIME (MCP 2025-11-25 schema,
 * ImageContent); MCPL additionally permits URI-form references. Never infer
 * a MIME from a missing field or serialize invalid image bytes as text. */
export function normalizeImageContent(block: Record<string, unknown>): ContentBlock {
  if (block.type === 'input_image') return responsesImageContent(block);
  const source = block.type !== 'generated_image' && object(block.source) ? block.source : undefined;
  if (source?.type === 'url' && typeof source.url === 'string' && source.url) {
    if (source.url.startsWith('data:')) {
      const image = responsesImageContent({ image_url: source.url });
      return image.type === 'image' ? { ...block, ...image } as unknown as ImageContent : image;
    }
    return block.type === 'image' ? block as unknown as ImageContent
      : { ...block, type: 'image', source: { type: 'url', url: source.url } } as ImageContent;
  }
  const data = source ? source.data : block.data;
  const mime = source ? source.mediaType ?? source.media_type : block.mimeType;
  if ((!source || source.type === 'base64') && typeof data === 'string' && isValidImageBase64(data) &&
      typeof mime === 'string' && isAcceptedImageMediaType(mime)) {
    const mediaType = resolveImageMediaType(data, mime)!;
    if (block.type === 'generated_image') return (block.mimeType === mediaType ? block : { ...block, mimeType: mediaType }) as unknown as GeneratedImageContent;
    if (block.type === 'image' && source?.mediaType === mediaType) return block as unknown as ImageContent;
    return { type: 'image', source: { type: 'base64', data, mediaType },
      ...(typeof block.tokenEstimate === 'number' ? { tokenEstimate: block.tokenEstimate } : {}),
      ...(typeof block.sourceUrl === 'string' ? { sourceUrl: block.sourceUrl } : {}) };
  }
  return { type: 'text', text: IMAGE_UNAVAILABLE_TEXT };
}

/** No decoding or copying of binary media is needed to enforce the wire wall. */
export function imagePayloadBytes(image: PolicyImage): number {
  if (image.type === 'blob_ref') {
    if (image.encodedBytes === undefined) throw new Error('Unresolved image has no known encoded-byte cost');
    return image.encodedBytes;
  }
  if (isGeneratedImageMetadata(image)) return image.encodedBytes;
  if (image.type === 'generated_image') return image.data.length;
  if (image.source.type === 'base64') return image.source.data.length;
  const url = image.source.url;
  const comma = url.startsWith('data:') ? url.indexOf(',') : -1;
  return comma < 0 ? 0 : url.length - comma - 1;
}

export function responsesImageContent(part: Item): ContentBlock {
  const url = part.image_url;
  if (typeof url !== 'string' || !url) return { type: 'text', text: IMAGE_UNAVAILABLE_TEXT };
  if (!url.startsWith('data:')) return { type: 'image', source: { type: 'url', url } };
  const comma = url.indexOf(',');
  const header = comma < 0 ? '' : url.slice(5, comma);
  const data = url.slice(comma + 1);
  const declared = header.slice(0, -7);
  if (!header.endsWith(';base64') || !isAcceptedImageMediaType(declared) || !isValidImageBase64(data)) {
    return { type: 'text', text: IMAGE_UNAVAILABLE_TEXT };
  }
  return { type: 'image', source: { type: 'base64', mediaType: resolveImageMediaType(data, declared)!, data } };
}

/** Typed auxiliary projection, not a serialization of a native output array. */
export function projectResponsesContent(content: unknown): ContentBlock[] {
  if (typeof content === 'string') return [{ type: 'text', text: content }];
  if (!Array.isArray(content)) return [{ type: 'text', text: '' }];
  return content.map((part): ContentBlock => {
    if (!object(part)) return { type: 'text', text: '' };
    if (part.type === 'input_image') return responsesImageContent(part);
    if (typeof part.text === 'string') return { type: 'text', text: part.text };
    if (typeof part.refusal === 'string') return { type: 'text', text: part.refusal };
    // Non-image native parts stay opaque in the authoritative carrier.
    return { type: 'text', text: '' };
  });
}

/** Existing native producer only. An omitted OUTPUT format is not the request
 * default: require a supported explicit format or a recognized raster signature. */
export function projectResponsesGeneratedImage(item: Item): GeneratedImageContent | undefined {
  if (item.type !== 'image_generation_call' || item.status !== 'completed' ||
      typeof item.result !== 'string' || !isValidImageBase64(item.result)) return undefined;
  const format = item.output_format;
  const declared = format === 'png' ? 'image/png' : format === 'jpeg' ? 'image/jpeg'
    : format === 'webp' ? 'image/webp' : undefined;
  if (format !== undefined && format !== null && declared === undefined) return undefined;
  const mimeType = resolveImageMediaType(item.result, declared);
  if (!isAcceptedImageMediaType(mimeType)) return undefined;
  return { type: 'generated_image', data: item.result, mimeType: mimeType!, rawItem: item };
}

/** Lossless carriers for primary replay, typed projections for auxiliary calls. */
export function projectResponsesItem(item: Item): ContentBlock[] {
  const carry = (block: ContentBlock): ContentBlock => ({ ...block, rawItem: item });
  switch (item.type) {
    case 'image_generation_call': {
      const image = projectResponsesGeneratedImage(item);
      return [image ?? carry({ type: 'text', text: '' })];
    }
    case 'message': return projectResponsesContent(item.content).map(carry);
    case 'function_call_output': return [carry({ type: 'tool_result', toolUseId: String(item.call_id ?? ''),
      content: typeof item.output === 'string' ? item.output : projectResponsesContent(item.output) })];
    case 'function_call': {
      let input: Record<string, unknown>;
      try { input = JSON.parse(String(item.arguments ?? '{}')); }
      catch { input = { _rawArguments: item.arguments }; }
      return [carry({ type: 'tool_use', id: String(item.call_id ?? item.id ?? ''), name: String(item.name ?? ''), input })];
    }
    case 'reasoning': return [carry(typeof item.encrypted_content === 'string'
      ? { type: 'redacted_thinking', data: item.encrypted_content }
      : { type: 'thinking', thinking: projectResponsesContent(item.summary).flatMap(b => b.type === 'text' ? [b.text] : []).join('\n') })];
    case 'compaction': return [carry({ type: 'redacted_thinking', data: typeof item.encrypted_content === 'string' ? item.encrypted_content : '' })];
    default: return [carry({ type: 'text', text: '' })];
  }
}

/** Equality without JSON-stringifying (and copying) base64-bearing carriers. */
export function sameResponsesItem(a: unknown, b: unknown): boolean {
  if (a === b) return true;
  if (Array.isArray(a) && Array.isArray(b)) return a.length === b.length && a.every((v, i) => sameResponsesItem(v, b[i]));
  if (!object(a) || !object(b)) return false;
  const keys = Object.keys(a);
  return keys.length === Object.keys(b).length && keys.every(key => Object.hasOwn(b, key) && sameResponsesItem(a[key], b[key]));
}

function mediaParts(item: Item): unknown[] | undefined {
  if (projectResponsesGeneratedImage(item)) return [item];
  const parts = item.type === 'function_call_output' ? item.output : item.type === 'message' ? item.content : undefined;
  return Array.isArray(parts) && parts.some(p => object(p) && p.type === 'input_image') ? parts : undefined;
}

/** Re-project legacy text-only carriers too, but never rewrite opaque items. */
export function projectNativeImageContent(message: ImageMessage): ContentBlock[] {
  const native = message.metadata?.[RESPONSES_ITEMS_KEY];
  if (Array.isArray(native) && native.some(item => object(item) && mediaParts(item))) {
    return native.flatMap(item => object(item) ? projectResponsesItem(item) : []);
  }
  let result: ContentBlock[] | undefined;
  for (let i = 0; i < message.content.length; i++) {
    const block = message.content[i]!;
    const raw = block.rawItem;
    if (!object(raw) || !mediaParts(raw)) {
      result?.push(block);
      continue;
    }
    let end = i + 1;
    while (end < message.content.length && sameResponsesItem(message.content[end]!.rawItem, raw)) end++;
    // Already typed: keep the original blocks, including token estimates.
    const typed = message.content.slice(i, end).some(b => isVisualImageContent(b) ||
      (b.type === 'tool_result' && Array.isArray(b.content)));
    if (typed) result?.push(...message.content.slice(i, end));
    else {
      result ??= message.content.slice(0, i);
      result.push(...projectResponsesItem(raw));
    }
    i = end - 1;
  }
  return result ?? message.content;
}

function mapNativeImages(item: Item, keep: (image: ImageContent | GeneratedImageContent) => boolean): Item {
  const generated = projectResponsesGeneratedImage(item);
  if (generated) return keep(generated) ? item : { type: 'message', role: 'assistant',
    content: [{ type: 'output_text', text: IMAGE_DROPPED_TEXT }] };
  const parts = mediaParts(item);
  if (!parts) return item;
  let mapped: unknown[] | undefined;
  for (let i = parts.length - 1; i >= 0; i--) {
    const part = parts[i];
    if (!object(part) || part.type !== 'input_image') continue;
    const image = responsesImageContent(part);
    if (image.type === 'image' && keep(image)) continue;
    mapped ??= parts.slice();
    mapped[i] = { type: item.role === 'assistant' ? 'output_text' : 'input_text',
      text: image.type === 'image' ? IMAGE_DROPPED_TEXT : IMAGE_UNAVAILABLE_TEXT };
  }
  return mapped ? { ...item, [item.type === 'function_call_output' ? 'output' : 'content']: mapped } : item;
}

/** Newest-first recursive copy-on-write. Raw media carriers are filtered ONCE,
 * then every projection uses the same new carrier; signed/opaque items are untouched. */
function mapContentImages(content: ContentBlock[], keep: (image: PolicyImage) => boolean): ContentBlock[] {
  let mapped: ContentBlock[] | undefined;
  for (let i = content.length - 1; i >= 0; i--) {
    const block = content[i]!;
    const raw = block.rawItem;
    if (object(raw) && mediaParts(raw)) {
      let start = i;
      while (start > 0 && sameResponsesItem(content[start - 1]!.rawItem, raw)) start--;
      const next = mapNativeImages(raw, keep);
      if (next !== raw) {
        mapped ??= content.slice();
        mapped.splice(start, i - start + 1, ...projectResponsesItem(next));
      }
      i = start;
      continue;
    }
    let next = block;
    if (isGeneratedImageMetadata(block)) {
      if (!isAcceptedImageMediaType(block.mimeType)) next = { type: 'text', text: IMAGE_UNAVAILABLE_TEXT };
      else if (!keep(block)) next = { type: 'text', text: IMAGE_DROPPED_TEXT };
    } else if (isVisualImageContent(block)) {
      const image = normalizeImageContent(block as unknown as Item);
      if (!isVisualImageContent(image)) next = image;
      else if (!keep(image)) next = { type: 'text', text: IMAGE_DROPPED_TEXT };
    } else if (isImageReference(block) && !keep(block)) next = { type: 'text', text: IMAGE_DROPPED_TEXT };
    else if (block.type === 'tool_result' && Array.isArray(block.content)) {
      const nested = mapContentImages(block.content, keep);
      if (nested !== block.content) {
        const { rawXml: _rawXml, ...rest } = block;
        next = { ...rest, content: nested };
      }
    }
    if (next !== block) {
      mapped ??= content.slice();
      mapped[i] = next;
    }
  }
  return mapped ?? content;
}

/** Default only for standalone callers; CM supplies its calibrated estimator. */
export function estimateImagePolicyContentTokens(content: ContentBlock[]): number {
  let tokens = 0;
  for (const b of content) {
    if (isVisualImageContent(b) || isImageReference(b)) tokens += b.tokenEstimate ?? IMAGE_TOKEN_ESTIMATE;
    else if (b.type === 'text') tokens += Math.ceil(b.text.length / 4);
    else if (b.type === 'thinking') tokens += Math.max(Math.ceil(b.thinking.length / 4), Math.round((b.signature?.length ?? 0) / 6));
    else if (b.type === 'redacted_thinking') tokens += Math.round(b.data.length / 6);
    else if (b.type === 'tool_use') tokens += Math.ceil(JSON.stringify(b.input).length / 4) + 20;
    else if (b.type === 'tool_result') tokens += typeof b.content === 'string' ? Math.ceil(b.content.length / 4) : estimateImagePolicyContentTokens(b.content);
    else tokens += 1000;
  }
  return tokens;
}

/** The boundary message counts toward depth (same rule as CM's raw timeline). */
export function imageDepthStart<T>(messages: T[], depth: number, estimate: (message: T) => number): number {
  if (depth <= 0) return 0;
  let tokens = 0;
  for (let i = messages.length - 1; i >= 0; i--) {
    tokens += estimate(messages[i]!);
    if (tokens > depth) return i + 1;
  }
  return 0;
}

/** Consume already-projected messages newest-first. Incremental consumers
 * can stop at a budget boundary without inspecting unrelated older media. */
export interface ImageMessageFilter {
  filter<T extends ImageMessage>(message: T, inDepth: boolean): T;
}

export function createImageMessageFilter(
  policy: LiveImagePolicy, onDrop?: () => void, knownLegacyBytes?: (hash: string) => number | undefined,
): ImageMessageFilter {
  const countCap = policy.maxLiveImages ?? 0;
  const byteCap = policy.maxLiveImageBytes ?? DEFAULT_MAX_LIVE_IMAGE_BYTES;
  let count = 0;
  let bytes = 0;
  return { filter<T extends ImageMessage>(message: T, inDepth: boolean): T {
    const keep = (image: PolicyImage): boolean => {
      if (!inDepth || (countCap > 0 && count >= countCap)) {
        onDrop?.();
        return false;
      }
      // A legacy metadata length is consulted only after count/depth
      // eligibility. Lazy selectors supply needed candidate lengths here;
      // absent costs must never be treated as zero-byte images.
      const size = byteCap > 0
        ? image.type === 'blob_ref' && image.encodedBytes === undefined && knownLegacyBytes
          ? knownLegacyBytes(image.ref.hash) : imagePayloadBytes(image)
        : 0;
      if (size === undefined) throw new Error('Unresolved image has no known encoded-byte cost');
      if (byteCap > 0 && bytes + size > byteCap) {
        onDrop?.();
        return false;
      }
      count++;
      bytes += size;
      return true;
    };
    const native = message.metadata?.[RESPONSES_ITEMS_KEY];
    let next = message;
    if (Array.isArray(native) && native.some(item => object(item) && mediaParts(item))) {
      let mapped: unknown[] | undefined;
      for (let n = native.length - 1; n >= 0; n--) {
        const original = native[n];
        if (!object(original)) continue;
        const item = mapNativeImages(original, keep);
        if (item !== original) { mapped ??= native.slice(); mapped[n] = item; }
      }
      if (mapped) next = { ...message, metadata: { ...message.metadata, [RESPONSES_ITEMS_KEY]: mapped },
        content: mapped.flatMap(item => object(item) ? projectResponsesItem(item) : []) };
    } else {
      const content = mapContentImages(message.content, keep);
      if (content !== message.content) next = { ...message, content };
    }
    return next;
  } };
}

export function filterImageMessages<T extends ImageMessage>(
  messages: T[], policy: LiveImagePolicy, eligible?: (message: T, index: number) => boolean,
  onDrop?: () => void, knownLegacyBytes?: (hash: string) => number | undefined,
): T[] {
  let projected: T[] | undefined;
  for (let i = 0; i < messages.length; i++) {
    const message = messages[i]!;
    const content = projectNativeImageContent(message);
    if (content !== message.content) {
      projected ??= messages.slice();
      projected[i] = { ...message, content };
    }
  }
  const source = projected ?? messages;
  const depthStart = eligible ? 0 : imageDepthStart(source, policy.imageStripDepthTokens ?? 0,
    m => (policy.estimateTokens ?? estimateImagePolicyContentTokens)(m.content));
  const filter = createImageMessageFilter(policy, onDrop, knownLegacyBytes);
  let result: T[] | undefined;
  for (let i = source.length - 1; i >= 0; i--) {
    const message = source[i]!;
    const next = filter.filter(message, eligible ? eligible(message, i) : i >= depthStart);
    if (next !== message) { result ??= source.slice(); result[i] = next; }
  }
  return result ?? source;
}
